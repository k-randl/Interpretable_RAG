import os
import json
import torch
from transformers import PreTrainedModel, AutoModel, AutoTokenizer
from typing import Optional, List

#=======================================================================#
# Helper Functions:                                                     #
#=======================================================================#

def checkattr(obj:object, name:str) -> bool:
    if hasattr(obj, name):
        return getattr(obj, name) is not None
    return False

def embedding_backward(model:PreTrainedModel, dh:torch.Tensor):
    # this is very specific for BERT and may need to be updated to support other models:
    # phi(input_ids, token_type_ids, token_ps) = W @ one_hot(input_ids) + f(token_type_ids, token_ps)
    #  => phi'(input_ids, token_type_ids, token_ps) = phi'(input_ids) = W
    #  
    # for f(input_ids, token_type_ids, token_ps) = nn(phi(input_ids, token_type_ids, token_ps)):
    #   => f'(input_ids, token_type_ids, token_ps) = nn'(input_ids, token_type_ids, token_ps) @ W
    dPhi = model.embeddings.word_embeddings.weight.T.detach().to(dh)
    return torch.vmap(lambda grad: grad @ dPhi)(dh)

#=======================================================================#
# Generic Model Class:                                                  #
#=======================================================================#

def ExplainableGenericModel(T:type):
    # make sure T is derived from PreTrainedModel:
    assert issubclass(T, PreTrainedModel)

    # generic class definition:
    class _ExplainableGenericModel(T):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._x, self._y      = None, None
            self._a, self._da     = None, None
            self._h, self._dh     = None, None

        def attentionRollout(self, layer:int=0):
            '''Attention Rollout scores of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default 0).

            Returns:        Importance scores with shape = (bs, n_heads, n_outputs, n_inputs)'''

            # get attention weigths (shape: bs x n_layers x n_heads x n_outputs x n_inputs):
            a = self._a

            # average over heads (shape: bs x n_layers x n_outputs x n_inputs):
            a = a.mean(dim=2)

            # simulate residuals:
            a += (torch.eye(a.shape[-1]) / 2.).view(1, 1, a.shape[2], a.shape[3])

            return torch.vmap(lambda x: torch.linalg.multi_dot([item for item in x]))(a)

        def aGrad(self, layer:int=-1):
            '''AGrad (`-da ⊙ a`) scores of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default -1).

            Returns:        Importance scores with shape = (bs, n_heads, n_outputs, n_inputs)'''

            # compute attention weigths and gradients:
            a  = self._a[:,layer]   # shape: (bs x n_layers x n_heads x n_outputs x n_inputs)
            da = self._da[:,layer]  # shape: (bs x n_layers x n_heads x n_outputs x n_inputs)

            return -da * a

        def repAGrad(self):
            '''RepAGrad scores of the last batch.

            Returns:        Importance scores with shape = (bs, n_heads, n_classes, n_outputs, n_inputs)'''

            raise NotImplementedError()

        def grad(self, layer:int=0):
            '''Gradients towards the inputs of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default 0).

            Returns:        Importance scores with shape = (bs, n_inputs, n_tokens)'''

            # compute importance:
            return embedding_backward(self, self._dh[:,layer])

        def gradIn(self, layer:int=0):
            '''GradIn (`dx ⊙ x`) scores of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default 0).

            Returns:        Importance scores with shape = (bs, n_inputs)'''

            # compute hidden states and gradients:
            dx = self.grad(layer=layer)

            # elementwise multiplication with a one-hot tensor can be replaced with indexing:
            return torch.stack([
                -dx[i, torch.arange(len(tokens)), tokens]
                for i, tokens in enumerate(self._x)
            ])

        def forward(self, input_ids:torch.Tensor, *args, output_hidden_states:bool=True, output_attentions:bool=True, **kwargs):
            # control gradient computation:
            prev_grad = torch.is_grad_enabled()
            torch.set_grad_enabled(True)

            # propagate through model:
            outputs = super().forward(input_ids, *args, output_hidden_states=output_hidden_states, output_attentions=output_attentions, **kwargs)

            has_attentions    = checkattr(outputs, 'attentions')
            has_hidden_states = checkattr(outputs, 'hidden_states')

            # save gradients:
            # register attentions for gradient computation:
            if has_attentions:
                for a in outputs.attentions:
                    a.retain_grad()
                    a.grad = None

            # register hidden states for gradient computation:
            if has_hidden_states:
                for h in outputs.hidden_states:
                    h.retain_grad()
                    h.grad = None

            y = outputs.last_hidden_state.sum()

            # calculate gradients of output:
            y.backward(retain_graph = True)

            # save gradients with regard to attention weights:
            if has_attentions:
                # Tensor of shape (bs x n_layers x n_heads x n_outputs x n_inputs)
                self._a = torch.stack(
                    [a.detach().clone().cpu() for a in outputs.attentions],
                    dim=1
                )
                # Tensor of shape (bs x n_layers x n_heads x n_outputs x n_inputs)
                self._da = torch.stack(
                    [a.grad.detach().clone().cpu() for a in outputs.attentions],
                    dim=1
                )
            else: self._a, self._da = None, None

            # save gradients with regard to hidden states:
            if has_hidden_states:
                # Tensor of shape (bs x n_layers x n_inputs x encoding_size)
                self._h = torch.stack(
                    [h.detach().clone().cpu() for h in outputs.hidden_states],
                    dim=1
                )
                # Tensor of shape (bs x n_layers x n_inputs x encoding_size)
                self._dh = torch.stack(
                    [h.grad.detach().clone().cpu() for h in outputs.hidden_states],
                    dim=1
                )
            else: self._h, self._dh = None, None

            # save last sequence:
            self._x, self._y = input_ids.detach().clone().cpu(), outputs.last_hidden_state.detach().clone().cpu()

            # reset gradient computation:
            torch.set_grad_enabled(prev_grad)

            return outputs

    # return generic class subclassed from T:
    return _ExplainableGenericModel


class ExplainableAutoModel:
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path:str, *model_args, **kwargs):
        model_map_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model_map.json'))
        print(f'Using model mapping from "{model_map_path}".')

        # load model mapping:
        model_map = {}
        if os.path.exists(model_map_path):
            with open(model_map_path, 'r') as file:
                model_map = json.load(file)

        try:
            module, name = model_map[pretrained_model_name_or_path]
            t = getattr(__import__(module), name)

        except KeyError:
            # learn new mapping by instantiating from huggingface:
            t = type(AutoModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs))
            model_map[pretrained_model_name_or_path] = t.__module__, t.__name__

            with open(model_map_path, 'w') as file:
                json.dump(model_map, file)

        # instantiate model:
        return ExplainableGenericModel(t).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class ExplainableAutoModelForRAG(torch.nn.Module):
    @classmethod
    def from_pretrained(cls, query_encoder_name_or_path:str, context_encoder_name_or_path:Optional[str]=None, tokenizer_name_or_path:Optional[str]=None, *args, **kwargs):
        rag = cls()
        
        # load tokenizer:
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = query_encoder_name_or_path
        rag.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, *args, **kwargs
        )

        # load query encoder:
        rag.query_encoder = AutoModel.from_pretrained(
            query_encoder_name_or_path, *args, **kwargs
        )

        # load context encoder (if specified):
        if context_encoder_name_or_path is None:
            rag.context_encoder = rag.query_encoder

        else:
            rag.context_encoder = AutoModel.from_pretrained(
                context_encoder_name_or_path, *args, **kwargs
            )

        return rag
    
    def grad(self, layer:int=0, filter_special_tokens:bool=True):
        '''Gradients towards the inputs of the last batch.

        Args:
            layer:                  Transformer layer to compute the scores for (default 0).
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_inputs, n_tokens)'''

        # compute importance:
        grad = {
            'query':   embedding_backward(self.query_encoder, self._dh['query'][:,layer]),
            'context': embedding_backward(self.context_encoder, self._dh['context'][:,layer])
        }
    
        if filter_special_tokens:
            # set the importance of special tokens to 0.
            grad = {key: grad[key] * self.special_tokens_mask[key][:,:,None] for key in grad}

        return grad

    def aGrad(self, layer:int=-1, filter_special_tokens:bool=True):
        '''AGrad (`-da ⊙ a`) scores of the last batch.

        Args:
            layer:                  Transformer layer to compute the scores for (default -1).
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_heads, n_outputs, n_inputs)'''

        # compute attention weigths and gradients:
        a  = {key:self._a[key][:,layer,:,0,:] for key in self._a}     # shape: (bs x n_layers x n_heads x n_outputs x n_inputs)
        da = {key:self._da[key][:,layer,:,0,:] for key in self._da}   # shape: (bs x n_layers x n_heads x n_outputs x n_inputs)

        # compute importance:
        aGrad = {key:-da[key] * a[key] for key in da}

        if filter_special_tokens:
            # set the importance of special tokens to 0.
            aGrad = {key: aGrad[key] * self.special_tokens_mask[key][:,None,:] for key in aGrad}

        return aGrad

    def repAGrad(self, filter_special_tokens:bool=True):
        '''RepAGrad scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_heads, n_classes, n_outputs, n_inputs)'''

        raise NotImplementedError()

    def gradIn(self, layer:int=0, filter_special_tokens:bool=True):
        '''GradIn (`dx ⊙ x`) scores of the last batch.

        Args:
            layer:                  Transformer layer to compute the scores for (default 0).
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.
            
        Returns:                    Importance scores with shape = (bs, n_inputs)'''

        # compute gradient to input:
        dx = self.grad(layer=layer, filter_special_tokens=False)

        # compute importance:
        # elementwise multiplication with a one-hot tensor can be replaced with indexing:
        gradIn = {key: torch.stack([
                -dx[key][i, torch.arange(len(tokens)), tokens]
                for i, tokens in enumerate(self._x[key])
            ]) for key in dx
        }
    
        if filter_special_tokens:
            # set the importance of special tokens to 0.
            gradIn = {key: gradIn[key] * self._special_tokens_mask[key] for key in gradIn}

        return gradIn

    def forward(self, query:str, contexts:List[str], **kwargs):
        # control gradient computation:
        prev_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # Apply tokenizer
        qry_input = self.tokenizer(query, return_special_tokens_mask=True, return_tensors='pt')
        ctx_input = self.tokenizer(contexts, padding=True, truncation=True, return_special_tokens_mask=True, return_tensors='pt')

        self._x = {  
            'query':   qry_input.input_ids.detach().cpu().numpy(),
            'context': ctx_input.input_ids.detach().cpu().numpy()
        }
        self._special_tokens_mask = {  
            'query':   1. - qry_input.pop('special_tokens_mask').detach().cpu().numpy(),
            'context': 1. - ctx_input.pop('special_tokens_mask').detach().cpu().numpy()
        }
        self._in_tokens = {
            'query':   [[self.tokenizer.decode(token) for token in text] for text in qry_input.input_ids],
            'context': [[self.tokenizer.decode(token) for token in text] for text in ctx_input.input_ids],
        }

        # Compute embeddings: take the last-layer hidden state of the [CLS] token
        qry_output = self.query_encoder(**qry_input, **kwargs)
        ctx_output = self.context_encoder(**ctx_input, **kwargs)

        # Compute dot product:
        similarity = qry_output.last_hidden_state[:, 0, :] @ ctx_output.last_hidden_state[:, 0, :].T

        has_attentions    = checkattr(qry_output, 'attentions') and checkattr(ctx_output, 'attentions')
        has_hidden_states = checkattr(qry_output, 'hidden_states') and checkattr(ctx_output, 'hidden_states')

        # save gradients:
        # register attentions for gradient computation:
        if has_attentions:
            for a in qry_output.attentions + ctx_output.attentions:
                a.retain_grad()
                a.grad = None

        # register hidden states for gradient computation:
        if has_hidden_states:
            for h in qry_output.hidden_states + ctx_output.hidden_states:
                h.retain_grad()
                h.grad = None

        # calculate gradients of output:
        similarity.sum().backward(retain_graph = True)

        # save gradients with regard to attention weights:
        if has_attentions:
            # Tensor of shape (bs x n_layers x n_heads x n_outputs x n_inputs)
            self._a = {
                'query': torch.stack(
                    [a.detach().clone().cpu() for a in qry_output.attentions],
                    dim=1
                ),
                'context': torch.stack(
                    [a.detach().clone().cpu() for a in ctx_output.attentions],
                    dim=1
                )
            }
            # Tensor of shape (bs x n_layers x n_heads x n_outputs x n_inputs)
            self._da = {
                'query': torch.stack(
                    [a.grad.detach().clone().cpu() for a in qry_output.attentions],
                    dim=1
                ),
                'context': torch.stack(
                    [a.grad.detach().clone().cpu() for a in ctx_output.attentions],
                    dim=1
                )
            }
        else: self._a, self._da = None, None

        # save gradients with regard to hidden states:
        if has_hidden_states:
            # Tensor of shape (bs x n_layers x n_inputs x encoding_size)
            self._h = {
                'query': torch.stack(
                    [h.detach().clone().cpu() for h in qry_output.hidden_states],
                    dim=1
                ),
                'context': torch.stack(
                    [h.detach().clone().cpu() for h in ctx_output.hidden_states],
                    dim=1
                )
            }
            # Tensor of shape (bs x n_layers x n_inputs x encoding_size)
            self._dh = {
                'query': torch.stack(
                    [h.grad.detach().clone().cpu() for h in qry_output.hidden_states],
                    dim=1
                ),
                'context': torch.stack(
                    [h.grad.detach().clone().cpu() for h in ctx_output.hidden_states],
                    dim=1
                )
            }
        else: self._h, self._dh = None, None

        # reset gradient computation:
        torch.set_grad_enabled(prev_grad)

        return similarity
