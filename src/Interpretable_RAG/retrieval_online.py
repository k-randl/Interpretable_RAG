import torch
from math import ceil
from tqdm.autonotebook import trange
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModel, AutoTokenizer
from typing import Optional, List, Dict, Literal

from .retrieval import RetrieverExplanationBase

#=======================================================================#
# Helper Functions:                                                     #
#=======================================================================#

def checkattr(obj:object, name:str) -> bool:
    if hasattr(obj, name):
        return getattr(obj, name) is not None
    return False

def embedding_backward(model:PreTrainedModel, dPhi:torch.Tensor):
    # this is very specific for BERT and may need to be updated to support other models:
    # phi(input_ids, token_type_ids, token_ps) = W @ one_hot(input_ids) + f(token_type_ids, token_ps)
    #  => phi'(input_ids, token_type_ids, token_ps) = phi'(input_ids) = W
    #  
    # for f(input_ids, token_type_ids, token_ps) = nn(phi(input_ids, token_type_ids, token_ps)):
    #   => f'(input_ids, token_type_ids, token_ps) = nn'(input_ids, token_type_ids, token_ps) @ W
    w = model.embeddings.word_embeddings.weight.T.detach().to(dPhi)
    return torch.vmap(lambda grad: grad @ w)(dPhi)

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

    def grad(self, filter_special_tokens:bool=True):
        '''Gradients towards the inputs of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_inputs, n_tokens)'''

        # compute importance:
        grad = {
            'query':   embedding_backward(self.query_encoder, self._dPhi['query']),
            'context': embedding_backward(self.context_encoder, self._dPhi['context'])
        }
    
        if filter_special_tokens:
            # set the importance of special tokens to 0.
            grad = {key: grad[key] * self._special_tokens_mask[key][:,:,None] for key in grad}

        return grad

    def aGrad(self, filter_special_tokens:bool=True):
        '''AGrad (`-da ⊙ a`) scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_heads, n_outputs, n_inputs)'''

        # compute attention weigths and gradients:
        a  = {key:self._a[key][:,-1,:,0,:] for key in self._a}     # shape: (bs x n_layers x n_heads x n_outputs x n_inputs)
        da = {key:self._da[key][:,-1,:,0,:] for key in self._da}   # shape: (bs x n_layers x n_heads x n_outputs x n_inputs)

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

    def gradIn(self, filter_special_tokens:bool=True):
        '''GradIn (`dx ⊙ x`) scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.
            
        Returns:                    Importance scores with shape = (bs, n_inputs)'''

        # compute gradient to input:
        dx = self.grad(filter_special_tokens=False)

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

    def intGrad(self, filter_special_tokens:bool=True, num_steps:int=100, batch_size:int=64):
        '''Integrated gradient scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.
            num_steps:              Number of approximation steps for the Rieman approximation
                                    of the integral (default 100).
            batch_size:             Batch size used for calculating the gradients (default 64).
            
        Returns:                    Importance scores with shape = (bs, n_inputs)'''
        # get the embedings:
        in_qry_embeds = self.query_encoder.get_input_embeddings()(torch.tensor(
            self._x['query'],
            device=self.query_encoder.device
        ))
        in_ctx_embeds = self.context_encoder.get_input_embeddings()(torch.tensor(
            self._x['context'],
            device=self.context_encoder.device
        ))

        # get attention masks:
        qry_attention_mask = torch.tensor(self._x['query'] != self.tokenizer.pad_token_id,
                                          device=in_qry_embeds.device, dtype=in_qry_embeds.dtype)
        ctx_attention_mask = torch.tensor(self._x['context'] != self.tokenizer.pad_token_id,
                                          device=in_ctx_embeds.device, dtype=in_ctx_embeds.dtype)

        # baseline is zeros with only special tokens remaining:
        bl_qry_embeds = in_qry_embeds * qry_attention_mask[:,:,None] * torch.tensor(
            (1 - self._special_tokens_mask['query']),
            device=self.query_encoder.device
        )[:,:,None]
        bl_ctx_embeds = in_ctx_embeds * ctx_attention_mask[:,:,None] * torch.tensor(
            (1 - self._special_tokens_mask['context']),
            device=self.context_encoder.device
        )[:,:,None]

        in_qry_embeds -= bl_qry_embeds
        in_ctx_embeds -= bl_ctx_embeds

        # path functions:
        steps = torch.linspace(0., 1., num_steps)

        def qry_path_fn(t:torch.Tensor): return bl_qry_embeds.to(in_qry_embeds) + (t.to(in_qry_embeds) * in_qry_embeds)
        qry_path_fn_batched = torch.vmap(qry_path_fn)

        def ctx_path_fn(t:torch.Tensor): return bl_ctx_embeds.to(in_ctx_embeds) + (t.to(in_ctx_embeds) * in_ctx_embeds)
        ctx_path_fn_batched = torch.vmap(ctx_path_fn)
        
        # chunk wise dot product gradients:
        def calculate_sim(a:torch.Tensor, b:torch.Tensor): return a @ b.T
        calculate_sim_batched = torch.vmap(calculate_sim)

        # get chunk sizes:
        qry_size, ctx_size = [self._x[key].shape[0] for key in ['query', 'context']]
        chunk_size = batch_size//max(qry_size, ctx_size)

        # get original outputs:
        in_qry_output = self._y['query'][None, :, :].to(in_qry_embeds)
        in_ctx_output = self._y['context'][None, :, :].to(in_ctx_embeds)

        # Riemann integrate along path:
        intGrad = {'query': 0., 'context':0.}  # Tensors of shape (n_inputs x encoding_size)
        qry_mult = in_qry_embeds / num_steps
        ctx_mult = in_ctx_embeds / num_steps
        for i in trange(ceil(num_steps/chunk_size)):
            batch = steps[i*chunk_size:(i+1)*chunk_size]
            batch_qry_embeds = qry_path_fn_batched(batch.to(self.query_encoder.device)).flatten(start_dim=0, end_dim=1)
            batch_ctx_embeds = ctx_path_fn_batched(batch.to(self.context_encoder.device)).flatten(start_dim=0, end_dim=1)

            # compute embeddings: take the last-layer hidden state of the [CLS] token
            batch_qry_output = self.query_encoder(
                inputs_embeds=batch_qry_embeds,
                attention_mask=qry_attention_mask.repeat(batch.shape[0], 1)
            ).last_hidden_state[:, 0, :].view(batch.shape[0], qry_size, -1)

            batch_ctx_output = self.context_encoder(
                inputs_embeds=batch_ctx_embeds,
                attention_mask=ctx_attention_mask.repeat(batch.shape[0], 1)
            ).last_hidden_state[:, 0, :].view(batch.shape[0], ctx_size, -1)

            # compute dot product similarity:
            batch_qry_similarity = calculate_sim_batched(batch_qry_output, in_ctx_output.repeat(batch.shape[0], 1, 1))
            batch_ctx_similarity = calculate_sim_batched(in_qry_output.repeat(batch.shape[0], 1, 1), batch_ctx_output)

            # save first and last output:
            if 0 in batch:
                bl_qry_similarity = batch_qry_similarity[batch == 0].detach().cpu().numpy()
                bl_ctx_similarity = batch_ctx_similarity[batch == 0].detach().cpu().numpy()

            if 1 in batch:
                in_qry_similarity = batch_qry_similarity[batch == 1].detach().cpu().numpy()
                in_ctx_similarity = batch_ctx_similarity[batch == 1].detach().cpu().numpy()

            # register input embeddings for gradient computation:
            batch_qry_embeds.retain_grad()
            batch_qry_embeds.grad = None

            batch_ctx_embeds.retain_grad()
            batch_ctx_embeds.grad = None

            # calculate gradients:
            batch_qry_similarity.sum().backward(retain_graph = True)
            batch_ctx_similarity.sum().backward(retain_graph = True)

            # save gradients with regard to hidden states:
            with torch.no_grad():
                intGrad['query']   += (batch_qry_embeds.grad.detach().sum(dim=(0)) * qry_mult).sum(dim=-1).cpu()
                intGrad['context'] += (batch_ctx_embeds.grad.detach().sum(dim=(0)) * ctx_mult).sum(dim=-1).cpu()

        # calculate total attribition scaled by number of used tokens:
        total_qry = (in_qry_similarity - bl_qry_similarity) / qry_attention_mask.detach().cpu().numpy().mean(axis=1)
        total_ctx = (in_ctx_similarity - bl_ctx_similarity) / ctx_attention_mask.detach().cpu().numpy().mean(axis=1)

        # check additivity:
        ratio_qry = intGrad['query'].sum() / total_qry.sum()
        ratio_ctx = intGrad['context'].sum() / total_ctx.sum()
        if ratio_qry < .95: print(f'WARNING: query attributions add up to only {ratio_qry*100.:.1f}% of the score. Please increase number of steps!')
        if ratio_ctx < .95: print(f'WARNING: context attributions add up to only {ratio_ctx*100.:.1f}% of the score. Please increase number of steps!')

        if filter_special_tokens:
            # set the importance of special tokens to 0.
            intGrad = {key: intGrad[key] * self._special_tokens_mask[key] for key in intGrad}

        return intGrad

    def forward(self, query:str, contexts:List[str], k:Optional[int]=None, *, reorder:bool=False, **kwargs):
        # control gradient computation:
        prev_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # apply tokenizer:
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

        # apply embeding:
        qry_embeds = self.query_encoder.get_input_embeddings()(torch.tensor(
            qry_input.pop('input_ids'),
            device=self.query_encoder.device
        ))
        ctx_embeds = self.context_encoder.get_input_embeddings()(torch.tensor(
            ctx_input.pop('input_ids'),
            device=self.context_encoder.device
        ))

        # Compute embeddings: take the last-layer hidden state of the [CLS] token
        qry_output = self.query_encoder(inputs_embeds=qry_embeds, **qry_input.to(self.query_encoder.device), **kwargs)
        ctx_output = self.context_encoder(inputs_embeds=ctx_embeds, **ctx_input.to(self.context_encoder.device), **kwargs)

        # Compute dot product:
        similarity = qry_output.last_hidden_state[:, 0, :] @ ctx_output.last_hidden_state[:, 0, :].T
        similarity, retrieved_ids = torch.sort(similarity, dim=1, descending=True)
        if k is not None:
            similarity    = similarity[:, :k]
            retrieved_ids = retrieved_ids[:, :k]

        has_attentions    = checkattr(qry_output, 'attentions') and checkattr(ctx_output, 'attentions')

        # save gradients:
        # register input embeddings for gradient computation:
        qry_embeds.retain_grad()
        qry_embeds.grad = None

        ctx_embeds.retain_grad()
        ctx_embeds.grad = None

        # register attentions for gradient computation:
        if has_attentions:
            for a in qry_output.attentions + ctx_output.attentions:
                a.retain_grad()
                a.grad = None

        # calculate gradients of output:
        similarity.sum().backward(retain_graph = True)

        # save gradients with regard to input embeddings:
        # Tensor of shape (bs x n_inputs x encoding_size)
        self._phi = {
            'query':   qry_embeds.detach().cpu(),
            'context': ctx_embeds.detach().cpu()
        }
        # Tensor of shape (bs x n_inputs x encoding_size)
        self._dPhi = {
            'query':   qry_embeds.grad.detach().cpu(),
            'context': ctx_embeds.grad.detach().cpu()
        }
        # Tensor of shape (bs x encoding_size)
        self._y = {  
            'query':   qry_output.last_hidden_state[:, 0, :].detach().cpu(),
            'context': ctx_output.last_hidden_state[:, 0, :].detach().cpu()
        }

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

        # reset gradient computation:
        torch.set_grad_enabled(prev_grad)

        # move outputs to cpu:
        similarity = similarity[0].detach().cpu()
        retrieved_ids = retrieved_ids[0].cpu()

        # reorder contexts:
        if reorder:
            self._x['context']                   = self._x['context'][retrieved_ids.numpy()]
            self._phi['context']                 = self._phi['context'][retrieved_ids]
            self._dPhi['context']                = self._dPhi['context'][retrieved_ids]
            if has_attentions:
                self._a['context']               = self._a['context'][retrieved_ids]
                self._da['context']              = self._da['context'][retrieved_ids]
            self._special_tokens_mask['context'] = self._special_tokens_mask['context'][retrieved_ids]
            self._in_tokens['context']           = [self._in_tokens['context'][i] for i in retrieved_ids]

        if k is None: return similarity
        else: return retrieved_ids, similarity