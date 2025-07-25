import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModel, AutoTokenizer
from typing import Optional, List, Dict, Literal

from resources.retrieval import RetrieverExplanationBase

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

class ExplainableAutoModelForRetrieval(torch.nn.Module, RetrieverExplanationBase):
    @classmethod
    def from_pretrained(cls, query_encoder_name_or_path:str, context_encoder_name_or_path:Optional[str]=None, tokenizer_name_or_path:Optional[str]=None, *args, **kwargs):
        retriever = cls()
        
        # load tokenizer:
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = query_encoder_name_or_path
        retriever._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, *args, **kwargs
        )

        # load query encoder:
        retriever._query_encoder = AutoModel.from_pretrained(
            query_encoder_name_or_path, *args, **kwargs
        )

        # load context encoder (if specified):
        if context_encoder_name_or_path is None:
            retriever._context_encoder = retriever.query_encoder

        else:
            retriever._context_encoder = AutoModel.from_pretrained(
                context_encoder_name_or_path, *args, **kwargs
            )

        return retriever
    
    @property
    def in_tokens(self) -> Dict[Literal['query', 'context'], List[List[str]]]:
        """A dicionary containing the following two keys:
        - `'query'`: a list containing the tokenized query
        - `'context'`: a list containing the tokenized contexts"""
        if hasattr(self, '_in_tokens'): return self._in_tokens
        else: return None

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """The tokenizer used by the retriever model."""
        return self._tokenizer

    @property
    def query_encoder(self) -> PreTrainedModel:
        """The query encoder used by the retriever model."""
        return self._query_encoder

    @property
    def context_encoder(self) -> PreTrainedModel:
        """The context encoder used by the retriever model."""
        return self._context_encoder

    @property
    def query_encoder_name_or_path(self) -> str:
        """The huggingface string identifier of the query encoder model."""
        return self._query_encoder.config.name_or_path

    @property
    def context_encoder_name_or_path(self) -> str:
        """The huggingface string identifier of the context encoder model."""
        return self._context_encoder.config.name_or_path
    
    @property
    def special_tokens_mask(self):
        if hasattr(self, '_special_tokens_mask'): return {key:
            ~self._special_tokens_mask[key].astype(bool)
            for key in self._special_tokens_mask
        }
        else: return None

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
            grad = {key: grad[key] * self._special_tokens_mask[key][:,:,None] for key in grad}

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
            aGrad = {key: aGrad[key] * self._special_tokens_mask[key][:,None,:] for key in aGrad}

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

    def forward(self, query:str, contexts:List[str], k:Optional[int]=None, **kwargs):
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
            'query':   [self.tokenizer.convert_ids_to_tokens(text) for text in qry_input.input_ids],
            'context': [self.tokenizer.convert_ids_to_tokens(text) for text in ctx_input.input_ids],
        }

        # Compute embeddings: take the last-layer hidden state of the [CLS] token
        qry_output = self.query_encoder(**qry_input.to(self.query_encoder.device), **kwargs)
        ctx_output = self.context_encoder(**ctx_input.to(self.context_encoder.device), **kwargs)

        # Compute dot product:
        similarity = qry_output.last_hidden_state[:, 0, :] @ ctx_output.last_hidden_state[:, 0, :].T
        if k is not None:
            retrieved_ids = torch.argsort(similarity, dim=1).flip(dims=(1,))[0,:k]
            similarity = similarity[:, retrieved_ids]

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

        if k is None: return similarity.detach()
        else: return retrieved_ids, similarity.detach()