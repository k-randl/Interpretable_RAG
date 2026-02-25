import torch
import numpy as np
from math import ceil
from tqdm.autonotebook import trange
from sklearn.linear_model import Ridge
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModel, AutoTokenizer
from typing import Optional, List, Literal, Union, Callable, Any

from .retrieval import RetrieverExplanationBase, List_t, Array_t, Tensor_t
from .utils import sample_perturbations

#=======================================================================#
# Helper Functions:                                                     #
#=======================================================================#

def checkattr(obj:object, name:str) -> bool:
    if hasattr(obj, name):
        return getattr(obj, name) is not None
    return False

def embedding_backward(model:PreTrainedModel, dPhi:torch.Tensor) -> torch.Tensor:
    # this is very specific for BERT and may need to be updated to support other models:
    # phi(input_ids, token_type_ids, token_ps) = W @ one_hot(input_ids) + f(token_type_ids, token_ps)
    #  => phi'(input_ids, token_type_ids, token_ps) = phi'(input_ids) = W
    #  
    # for f(input_ids, token_type_ids, token_ps) = nn(phi(input_ids, token_type_ids, token_ps)):
    #   => f'(input_ids, token_type_ids, token_ps) = nn'(input_ids, token_type_ids, token_ps) @ W
    w = model.embeddings.word_embeddings.weight.T.detach().to(dPhi)
    return torch.vmap(lambda grad: grad @ w)(dPhi)

def append_tensor_t(obj:Optional[Tensor_t], append:bool, qry:torch.Tensor, ctx:torch.Tensor,
        pad_val:Any, is_grad:bool=False) -> Tensor_t:
    if obj is not None and append:
        if is_grad: qry += obj['query']
        else: assert torch.equal(obj['query'], qry)

        old_shape = obj['context'].shape
        new_shape = tuple([old_shape[0] + ctx.shape[0],] + [max(*s) for s in zip(old_shape[1:], ctx.shape[1:], strict=True)])

        new_c = torch.full(new_shape, pad_val,
                    dtype=obj['context'].dtype, device=obj['context'].device)
        idx = tuple([slice(0, old_shape[0])] + [slice(0, s) for s in old_shape[1:]])
        new_c[idx] = obj['context']
        idx = tuple([slice(old_shape[0], None)] + [slice(0, s) for s in ctx.shape[1:]])
        new_c[idx] = ctx

        ctx = new_c
    
    return {'query': qry, 'context': ctx}

#=======================================================================#
# Generic Model Class:                                                  #
#=======================================================================#

class ExplainableAutoModelForRetrieval(torch.nn.Module, RetrieverExplanationBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._x:Optional[Tensor_t]=None
        self._y:Optional[Tensor_t]=None
        self._phi:Optional[Tensor_t]=None
        self._dPhi:Optional[Tensor_t]=None
        self._a:Optional[Tensor_t]=None
        self._da:Optional[Tensor_t]=None

        self._special_tokens_mask:Optional[Tensor_t]=None

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
    def in_tokens(self) -> Optional[List_t]:
        """A dicionary containing the following two keys:
        - `'query'`: a list containing the tokenized query
        - `'context'`: a list containing the tokenized contexts"""
        if self._x is not None: return {key:
            [self.tokenizer.convert_ids_to_tokens(text) for text in value]
            for key, value in self._x.items()
        }
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
    def special_tokens_mask(self) -> Optional[Tensor_t]:
        if self._special_tokens_mask is not None: return {key:
            ~value.detach().to(device='cpu', dtype=torch.bool, copy=True)
            for key, value in self._special_tokens_mask.items()
        }
        else: return None

    def grad(self, filter_special_tokens:bool=True) -> Tensor_t:
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

    def aGrad(self, filter_special_tokens:bool=True) -> Tensor_t:
        '''AGrad (`da ⊙ a`) scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_heads, n_outputs, n_inputs)'''

        # compute attention weigths and gradients:
        a  = {key:self._a[key][:,-1,:,0,:] for key in self._a}     # shape: (bs x n_layers x n_heads x n_outputs x n_inputs)
        da = {key:self._da[key][:,-1,:,0,:] for key in self._da}   # shape: (bs x n_layers x n_heads x n_outputs x n_inputs)

        # compute importance:
        aGrad = {key:da[key] * a[key] for key in da}

        if filter_special_tokens:
            # set the importance of special tokens to 0.
            aGrad = {key: aGrad[key] * self._special_tokens_mask[key][:,None,:] for key in aGrad}

        return aGrad

    def repAGrad(self, filter_special_tokens:bool=True) -> Tensor_t:
        '''RepAGrad scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_heads, n_classes, n_outputs, n_inputs)'''

        raise NotImplementedError()

    def gradIn(self, filter_special_tokens:bool=True) -> Tensor_t:
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

    def intGrad(self, filter_special_tokens:bool=True, num_steps:int=100, batch_size:int=64, *,
            base:Optional[Literal['pos', 'mask', 'pad', 'unk']]=None,
            output_offset:bool=False,
            output_coverage:bool=False,
            verbose:bool=True) -> Tensor_t:
        '''Integrated gradient scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.
            num_steps:              Number of approximation steps for the Rieman approximation
                                    of the integral (default 100).
            batch_size:             Batch size used for calculating the gradients (default 64).
            base:                   Optional value for the baseline input. Possible values are ...
                                    - `'pos'` positional encodings only
                                    - `'mask'` positional encodings + mask tokens
                                    - `'pad'` positional encodings + pad tokens
                                    - `None` zeroed embeddings (default)
            output_offset:          If `True` returns a tuple for which the first element are the attribution scores
                                    and the second element are the baseline predictions as a dictionary (default: `False`).
            output_coverage:        If `True` returns a tuple for which the first element are the attribution scores
                                    and the last element are the additivity coverage ratios as a dictionary (default: `False`).
            verbose:                If `True`, shows a progress bar.
            
        Returns:                    Importance scores with shape = (bs, n_inputs)'''
        # get the embedings:
        in_qry_embeds_fn = self.query_encoder.get_input_embeddings()
        in_qry_embeds = in_qry_embeds_fn(torch.tensor(self._x['query'], device=self.query_encoder.device))

        in_ctx_embeds_fn = self.context_encoder.get_input_embeddings()
        in_ctx_embeds = in_ctx_embeds_fn(torch.tensor(self._x['context'], device=self.context_encoder.device))

        # get attention masks:
        qry_attention_mask = torch.tensor(self._x['query'] != self.tokenizer.pad_token_id,
                                          device=in_qry_embeds.device, dtype=in_qry_embeds.dtype)
        ctx_attention_mask = torch.tensor(self._x['context'] != self.tokenizer.pad_token_id,
                                          device=in_ctx_embeds.device, dtype=in_ctx_embeds.dtype)

        # get baseline embeddings:
        bl_qry_embeds_fn = lambda t: in_qry_embeds_fn(torch.full(self._x['query'].shape, torch.tensor(t), device=self.query_encoder.device))
        bl_ctx_embeds_fn = lambda t: in_ctx_embeds_fn(torch.full(self._x['context'].shape, t, device=self.context_encoder.device))
        if base is None:
            bl_qry_embeds = torch.zeros_like(in_qry_embeds)
            bl_ctx_embeds = torch.zeros_like(in_ctx_embeds)

        elif base == 'pos':
            bl_qry_embeds = self.query_encoder.get_position_embeddings()
            bl_ctx_embeds = self.context_encoder.get_position_embeddings()

        elif base == 'mask':
            bl_qry_embeds = bl_qry_embeds_fn(self.tokenizer.mask_token_id)
            bl_ctx_embeds = bl_ctx_embeds_fn(self.tokenizer.mask_token_id)

        elif base == 'pad':
            bl_qry_embeds = bl_qry_embeds_fn(self.tokenizer.pad_token_id)
            bl_ctx_embeds = bl_ctx_embeds_fn(self.tokenizer.pad_token_id)

        elif base == 'unk':
            bl_qry_embeds = bl_qry_embeds_fn(self.tokenizer.unk_token_id)
            bl_ctx_embeds = bl_ctx_embeds_fn(self.tokenizer.unk_token_id)

        else: raise ValueError(f'Parameter `base` must be one of `\'pos\'`, `\'mask\'`, `\'pad\'`, `\'unk\'`, or `None` but is {base}.')

        # baseline is baseline tokens + special tokens:
        bl_qry_mask = qry_attention_mask[:,:,None] * torch.tensor(
            self._special_tokens_mask['query'],
            device=self.query_encoder.device
        )[:,:,None]
        bl_qry_embeds = torch.where(bl_qry_mask.to(torch.bool), bl_qry_embeds, in_qry_embeds)

        bl_ctx_mask = ctx_attention_mask[:,:,None] * torch.tensor(
            self._special_tokens_mask['context'],
            device=self.context_encoder.device
        )[:,:,None]
        bl_ctx_embeds = torch.where(bl_ctx_mask.to(torch.bool), bl_ctx_embeds, in_ctx_embeds)

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

        # get relevant tokens:
        tkns_qry = self._x['query'].flatten()
        tkns_ctx = self._x['context'].flatten()

        if base == 'mask':
            np.append(tkns_qry, self.tokenizer.mask_token_id)
            np.append(tkns_ctx, self.tokenizer.mask_token_id)

        elif base == 'pad':
            np.append(tkns_qry, self.tokenizer.pad_token_id)
            np.append(tkns_ctx, self.tokenizer.pad_token_id)

        elif base == 'unk':
            np.append(tkns_qry, self.tokenizer.unk_token_id)
            np.append(tkns_ctx, self.tokenizer.unk_token_id)

        tkns_qry = np.unique(tkns_qry)
        tkns_ctx = np.unique(tkns_ctx)

        # get embedding weights:
        #w_qry = self.query_encoder.embeddings.word_embeddings.weight[tkns_qry].detach().T
        #w_ctx = self.context_encoder.embeddings.word_embeddings.weight[tkns_ctx].detach().T

        # Riemann integrate along path:
        intGrad = {'query': 0., 'context':0.}  # Tensors of shape (n_inputs x encoding_size)
        qry_mult = in_qry_embeds / num_steps
        ctx_mult = in_ctx_embeds / num_steps

        #intGrad = {'query': [], 'context':[]}  # Tensors of shape (n_inputs x encoding_size)
        #qry_mult = torch.tensor(self._x['query'][:,:,None] == tkns_qry[None,None,:]).to(in_qry_embeds) / num_steps
        #ctx_mult = torch.tensor(self._x['context'][:,:,None] == tkns_ctx[None,None,:]).to(in_ctx_embeds)  / num_steps

        iterator = None
        if verbose: iterator = trange(ceil(num_steps/chunk_size))
        else:       iterator = range(ceil(num_steps/chunk_size))

        for i in iterator:
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

            with torch.no_grad():
                # get gradients with regard to hidden states:
                batch_qry_grads = batch_qry_embeds.grad.detach().reshape((-1,) + in_qry_embeds.shape)
                batch_ctx_grads = batch_ctx_embeds.grad.detach().reshape((-1,) + in_ctx_embeds.shape)

                # backpropagate to input:
                #batch_qry_grads = batch_qry_grads @ w_qry
                #batch_ctx_grads = batch_ctx_grads @ w_ctx

                # compute partial Riemann integral for the batch:
                intGrad['query']   += (batch_qry_grads.sum(dim=(0)) * qry_mult).sum(dim=-1).cpu()
                intGrad['context'] += (batch_ctx_grads.sum(dim=(0)) * ctx_mult).sum(dim=-1).cpu()

            # free some memory:
            del batch_qry_embeds
            del batch_ctx_embeds
            del batch_qry_similarity
            del batch_ctx_similarity
            del batch_qry_grads
            del batch_ctx_grads

        # check additivity:
        coverage_qry = intGrad['query'].sum(axis=-1) / (in_qry_similarity - bl_qry_similarity).sum()
        coverage_ctx = intGrad['context'].sum(axis=-1) / (in_ctx_similarity - bl_ctx_similarity).squeeze()
        if (coverage_qry < .95).any(): print(f'WARNING: query attributions add up to only {coverage_qry.min()*100.:.1f}% of the score. Please increase number of steps!')
        if (coverage_ctx < .95).any(): print(f'WARNING: context attributions add up to only {coverage_ctx.min()*100.:.1f}% of the score. Please increase number of steps!')

        if filter_special_tokens:
            # set the importance of special tokens to 0.
            intGrad = {key: intGrad[key] * self._special_tokens_mask[key] for key in intGrad}

        if not (output_offset or output_coverage):
            return intGrad

        result = (intGrad, )
        if output_offset:   result = result + ({'query': bl_qry_similarity.sum(), 'context': bl_ctx_similarity.squeeze()},)
        if output_coverage: result = result + ({'query': coverage_qry, 'context': coverage_ctx},)
        return result

    def forward(self, query:str, contexts:List[str], k:Optional[int]=None, *,
            reorder:bool=False,
            batch_size:Optional[int]=None,
            max_length:Optional[int]=None,
            **kwargs
        ):
        
        # control gradient computation:
        prev_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # create tokenizer arguments:
        tokenizer_args = {'padding':True, 'truncation':True, 'return_special_tokens_mask':True, 'return_tensors':'pt'}
        if max_length is not None:
            tokenizer_args['padding'] = 'max_length'
            tokenizer_args['max_length'] = max_length

        qry_embed_fcn = self.query_encoder.get_input_embeddings()
        ctx_embed_fcn = self.context_encoder.get_input_embeddings()

        if batch_size is None: batch_size = len(contexts)
        for batch in trange(0, len(contexts), batch_size, desc='Retrieving documents'):
            ctx_batch = contexts[batch:batch+batch_size]
            len_batch = len(ctx_batch)
            idx_batch = torch.arange(len_batch, dtype=int)
            append = batch > 0

            # apply tokenizer:
            qry_input = self.tokenizer(query, **tokenizer_args)
            ctx_input = self.tokenizer(ctx_batch, **tokenizer_args)

            self._x = append_tensor_t(self._x, append,
                qry_input.input_ids.detach().cpu(),
                ctx_input.input_ids.detach().cpu(),
                self.tokenizer.pad_token_id
            )
            self._special_tokens_mask = append_tensor_t(self._special_tokens_mask, append,
                1. - qry_input.pop('special_tokens_mask').detach().cpu(),
                1. - ctx_input.pop('special_tokens_mask').detach().cpu(),
                0.
            )

            # apply embeding:
            qry_embeds = qry_embed_fcn(qry_input.pop('input_ids').to(self.query_encoder.device))
            ctx_embeds = ctx_embed_fcn(ctx_input.pop('input_ids').to(self.query_encoder.device))

            # Compute embeddings: take the last-layer hidden state of the [CLS] token
            qry_output = self.query_encoder(inputs_embeds=qry_embeds, **qry_input.to(self.query_encoder.device), **kwargs)
            ctx_output = self.context_encoder(inputs_embeds=ctx_embeds, **ctx_input.to(self.context_encoder.device), **kwargs)

            # Compute dot product:
            sim_batch = qry_output.last_hidden_state[:, 0, :] @ ctx_output.last_hidden_state[:, 0, :].T

            has_attentions = checkattr(qry_output, 'attentions') and checkattr(ctx_output, 'attentions')

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
            sim_batch.sum().backward(retain_graph = True)

            # save gradients with regard to input embeddings:
            # Tensor of shape (bs x n_inputs x encoding_size)
            self._phi = append_tensor_t(self._phi, append,
                qry_embeds.detach().cpu(),
                ctx_embeds.detach().cpu(),
                torch.nan
            )
            # Tensor of shape (bs x n_inputs x encoding_size)
            self._dPhi = append_tensor_t(self._dPhi, append,
                qry_embeds.grad.detach().cpu(),
                ctx_embeds.grad.detach().cpu(),
                0.,
                is_grad=True
            )
            # Tensor of shape (bs x encoding_size)
            self._y = append_tensor_t(self._y, append,
                qry_output.last_hidden_state[:, 0, :].detach().cpu(),
                ctx_output.last_hidden_state[:, 0, :].detach().cpu(),
                torch.nan
            )

            # save gradients with regard to attention weights:
            if has_attentions:
                # Tensor of shape (bs x n_layers x n_heads x n_outputs x n_inputs)
                self._a = append_tensor_t(self._a, append,
                    torch.stack(
                        [a.detach().clone().cpu() for a in qry_output.attentions],
                        dim=1
                    ),
                    torch.stack(
                        [a.detach().clone().cpu() for a in ctx_output.attentions],
                        dim=1
                    ),
                    torch.nan
                )
                # Tensor of shape (bs x n_layers x n_heads x n_outputs x n_inputs)
                self._da = append_tensor_t(self._da, append,
                    torch.stack(
                        [a.grad.detach().clone().cpu() for a in qry_output.attentions],
                        dim=1
                    ),
                    torch.stack(
                        [a.grad.detach().clone().cpu() for a in ctx_output.attentions],
                        dim=1
                    ),
                    0.,
                    is_grad=True
                )
            else: self._a, self._da = None, None

            # compute similarity and context ids:
            if not append:
                similarity    = sim_batch[0].detach().cpu()
                retrieved_ids = idx_batch

            else:
                similarity    = torch.concatenate([similarity, sim_batch[0].detach().cpu()])
                retrieved_ids = torch.concatenate([retrieved_ids, idx_batch + batch])

            similarity, ret_ids_batch = torch.sort(similarity, dim=0, descending=True)
            retrieved_ids             = retrieved_ids[ret_ids_batch]

            # apply top-k:
            if k is not None:
                similarity    = similarity[:k]
                retrieved_ids = retrieved_ids[:k]
                ret_ids_batch = ret_ids_batch[:k]

            # reorder contexts:
            if reorder:
                self._x['context']                   = self._x['context'][ret_ids_batch]
                self._phi['context']                 = self._phi['context'][ret_ids_batch]
                self._dPhi['context']                = self._dPhi['context'][ret_ids_batch]
                if has_attentions:
                    self._a['context']               = self._a['context'][ret_ids_batch]
                    self._da['context']              = self._da['context'][ret_ids_batch]
                self._special_tokens_mask['context'] = self._special_tokens_mask['context'][ret_ids_batch]

        # reset gradient computation:
        torch.set_grad_enabled(prev_grad)

        if k is None: return similarity
        else: return retrieved_ids, similarity