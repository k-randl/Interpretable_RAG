import os
import pickle
import torch
from tqdm.autonotebook import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModel, AutoTokenizer
from typing import Optional, List, Union, Dict, Tuple

from .retrieval import RetrieverExplanationBase, List_t, Tensor_t

#=======================================================================#
# Helper Functions:                                                     #
#=======================================================================#

def checkattr(obj:object, name:str) -> bool:
    if hasattr(obj, name):
        return getattr(obj, name) is not None
    return False

def embedding_backward(model:PreTrainedModel, dPhi:List[torch.Tensor]) -> List[torch.Tensor]:
    # this is very specific for BERT and may need to be updated to support other models:
    # phi(input_ids, token_type_ids, token_ps) = W @ one_hot(input_ids) + f(token_type_ids, token_ps)
    #  => phi'(input_ids, token_type_ids, token_ps) = phi'(input_ids) = W
    #  
    # for f(input_ids, token_type_ids, token_ps) = nn(phi(input_ids, token_type_ids, token_ps)):
    #   => f'(input_ids, token_type_ids, token_ps) = nn'(input_ids, token_type_ids, token_ps) @ W
    w = model.embeddings.word_embeddings.weight.T.detach().to(dPhi[0])
    return [grad @ w for grad in dPhi]

def metadata_to_pkl(dir:str, doc_ids:List[Union[int, str]], metadata:Dict[str, Union[List, torch.Tensor]]):
    keys = list(metadata.keys())
    for i, values in enumerate(zip(*[metadata[key] for key in keys])):
        metadata_doc = {}
        for key, value in zip(keys, values):
            # if value is tensor:
            if torch.is_tensor(value):
                # detach gradients:
                value = value.detach().cpu()

            # add to document: 
            metadata_doc[key] = value

        # save to disk:
        with open(os.path.join(dir, 'meta_data', f'{doc_ids[i]}.pkl'), 'wb') as file:
            pickle.dump(metadata_doc, file)

def metadata_from_pkl(dir:str, doc_ids:List[Union[int, str]]) -> Dict[str, List]:
    metadata = None
    for doc_id in doc_ids:
        with open(os.path.join(dir, 'meta_data', f'{doc_id}.pkl'), 'rb') as file:
            md = pickle.load(file)

        if metadata is not None:
            assert set(metadata.keys()) == set(md.keys())
            for key in metadata: metadata[key].append(md[key])

        else: metadata = {key:[md[key]] for key in md}

    return metadata

#=======================================================================#
# Generic Model Class:                                                  #
#=======================================================================#

class ExplainableAutoModelForContextEncoding(torch.nn.Module):
    @classmethod
    def from_pretrained(cls, model_name_or_path:str, tokenizer_name_or_path:Optional[str]=None, **kwargs):
        encoder = cls()

        # load tokenizer:
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path
        encoder._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, **kwargs
        )

        # load encoder:
        encoder._context_encoder = AutoModel.from_pretrained(
            model_name_or_path, **kwargs
        )

        return encoder

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """The tokenizer used by the retriever model."""
        return self._tokenizer

    @property
    def context_encoder(self) -> PreTrainedModel:
        """The context encoder used by the retriever model."""
        return self._context_encoder

    def forward(self, contexts:List[str], *, max_length:Optional[int]=None, **kwargs):
        # control gradient computation:
        prev_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # set attention implementation to eager if attention-
        # based explantions are active:
        if kwargs.get('output_attentions', False):
            self.context_encoder.set_attn_implementation('eager')

        # create tokenizer arguments:
        tokenizer_args = {'padding':True, 'truncation':True, 'return_special_tokens_mask':True, 'return_tensors':'pt'}
        if max_length is not None:
            tokenizer_args['padding'] = 'max_length'
            tokenizer_args['max_length'] = max_length

        # apply tokenizer:
        ctx_input = self.tokenizer(contexts, **tokenizer_args)
        token_mask = ctx_input.pop('special_tokens_mask') == 0.
        output = {
            'texts':                contexts,
            'input_ids':            [ctx_input.input_ids[i, m].detach().cpu() for i, m in enumerate(token_mask)],
            'in_tokens':            [self.tokenizer.convert_ids_to_tokens(t[m]) for t, m in zip(ctx_input.input_ids, token_mask)]
        }

        # apply embeding:
        ctx_embeds = self.context_encoder.get_input_embeddings()(torch.tensor(
            ctx_input.pop('input_ids'),
            device=self.context_encoder.device
        ))

        # register gradient computation:
        self.context_encoder.get_input_embeddings().requires_grad_(True)

        # Compute embeddings: take the last-layer hidden state of the [CLS] token
        ctx_output = self.context_encoder(inputs_embeds=ctx_embeds, **ctx_input.to(self.context_encoder.device), **kwargs)
        output['embedding'] = ctx_output.last_hidden_state[:, 0, :].detach().cpu()
        embedding_size = output['embedding'].shape[-1]
        has_attentions = checkattr(ctx_output, 'attentions')

        # register gradient computation for embedings:
        # register retain_grad:
        ctx_embeds.retain_grad()

        # list of tensors of shape (n_inputs x embedding_size)
        output['phi'] = [ctx_embeds[i,m,:].detach().bfloat16().cpu() for i, m in enumerate(token_mask)]

        # list of tensors of shape (n_inputs x embedding_size x embedding_size)
        output['dPhi'] = [torch.BFloat16Tensor(item.shape + (embedding_size,), device='cpu') for item in output['phi']]

        # register gradient computation for attention weights:
        if has_attentions:
            # tensor of shape (bs x n_heads x n_outputs x n_inputs)
            a = ctx_output.attentions[-1]

            # register retain_grad:
            a.retain_grad()

            # list of tensors of shape (n_heads x n_inputs)
            output['a'] = [a[i,:,0,m].detach().bfloat16().cpu() for i, m in enumerate(token_mask)]

            # list of tensors of shape (n_heads x n_inputs x embedding_size)
            output['da'] = [torch.BFloat16Tensor(item.shape + (embedding_size,), device='cpu') for item in output['a']]

        for i in tqdm(range(embedding_size), total=embedding_size):
            # reset old gradients:
            self.context_encoder.zero_grad()

            # calculate gradients of output:
            ctx_output.last_hidden_state[:, 0, i].sum().backward(retain_graph = True)

            # save gradients with regard to input embedding:
            for j, m in enumerate(token_mask):
                output['dPhi'][j][:,:,i] = ctx_embeds.grad[j,m,:].detach().bfloat16().cpu()

            # save gradients with regard to attention weights:
            if has_attentions:
                a = ctx_output.attentions[-1]
                for j, m in enumerate(token_mask):
                    output['da'][j][:,:,i] = a.grad[j,:,0,m].detach().bfloat16().cpu()

        # reset gradient computation:
        torch.set_grad_enabled(prev_grad)

        return output

    def save_index(self, contexts:List[str], batch_size:int, dir:str='embeddings', **kwargs):
        embeddings = []

        # create dir for saving the embeddings:
        os.makedirs(os.path.join(dir, 'meta_data'), exist_ok=True)

        # iterate through texts:
        indices = list(range(0, len(contexts), batch_size)) + [len(contexts)]
        for batch, (i, j) in enumerate(zip(indices[:-1], indices[1:])):
            print(f'Batch {batch+1:d}/{len(indices)-1:d}:')
            if i == j: continue

            # compute embeddings:
            outputs = self(contexts[i:j], **kwargs)

            # extend embeddings:
            embeddings.append(outputs.pop('embedding'))

            # save meta-data:
            metadata_to_pkl(dir, range(i, j), outputs)

        with open(os.path.join(dir, 'embeddings.pt'), 'wb') as file:
            torch.save(torch.concatenate(embeddings, dim=0), file)

class ExplainableAutoModelForRetrieval(torch.nn.Module, RetrieverExplanationBase):
    @classmethod
    def from_pretrained(cls, query_encoder_name_or_path:str, *,
            tokenizer_name_or_path:Optional[str]=None,
            dir:str='embeddings',
            index:Optional[torch.FloatTensor]=None,
            **kwargs
        ) -> 'ExplainableAutoModelForRetrieval':
        encoder = cls()

        # load tokenizer:
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = query_encoder_name_or_path
        encoder._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, **kwargs
        )

        # load encoder:
        encoder._query_encoder = AutoModel.from_pretrained(
            query_encoder_name_or_path, **kwargs
        )

        # get index dir:
        encoder._index_dir = dir

        # load index from disk if not specified:
        if index is None:
            with open(os.path.join(encoder.index_dir, 'embeddings.pt'), 'rb') as file:
                encoder._index = torch.load(file).to(encoder.query_encoder.device)
        else: encoder._index = index.to(encoder.query_encoder.device)

        return encoder

    @property
    def index(self) -> Optional[torch.FloatTensor]:
        """A tensor of document embeddings."""
        if hasattr(self, '_index'): return self._index
        else: return None

    @property
    def index_dir(self) -> Optional[str]:
        """The papth of the index directory storing the metadata."""
        if hasattr(self, '_index_dir'): return self._index_dir
        else: return None

    @property
    def in_tokens(self) -> Optional[List_t]:
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
    def query_encoder_name_or_path(self) -> str:
        """The huggingface string identifier of the query encoder model."""
        return self._query_encoder.config.name_or_path

    def grad(self, filter_special_tokens:bool=True) -> List_t:
        '''Gradients towards the inputs of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_inputs, n_tokens)'''

        # compute importance:
        grad = {
            'query':   embedding_backward(self.query_encoder, self._dPhi['query']),
            'context': embedding_backward(self.query_encoder, self._dPhi['context'])
        }
    
        if not filter_special_tokens: print('WARNING: `filter_special_tokens` is not used in offline retrieval.')

        return grad

    def aGrad(self, filter_special_tokens:bool=True) -> List_t:
        '''AGrad (`da ⊙ a`) scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_heads, n_inputs)'''

        # compute attention weigths and gradients:
        a  = {key:self._a[key] for key in self._a}     # shape: (bs x n_heads x n_inputs)
        da = {key:self._da[key] for key in self._da}   # shape: (bs x n_heads x n_inputs)

        # compute importance:
        aGrad = {key:[_da * _a for _da, _a in zip(da[key], a[key], strict=True)] for key in da}

        if not filter_special_tokens: print('WARNING: `filter_special_tokens` is not used in offline retrieval.')

        return aGrad

    def repAGrad(self, filter_special_tokens:bool=True) -> List_t:
        '''RepAGrad scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_heads, n_classes, n_inputs)'''

        raise NotImplementedError()

    def gradIn(self, filter_special_tokens:bool=True) -> List_t:
        '''GradIn (`dx ⊙ x`) scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.
            
        Returns:                    Importance scores with shape = (bs, n_inputs)'''

        # compute gradient to input:
        dx = self.grad(filter_special_tokens=False)

        # compute importance:
        # elementwise multiplication with a one-hot tensor can be replaced with indexing:
        gradIn = {key: [
                -dx[key][i][torch.arange(len(tokens)), tokens]
                for i, tokens in enumerate(self._x[key])
            ] for key in dx
        }

        if not filter_special_tokens: print('WARNING: `filter_special_tokens` is not used in offline retrieval.')

        return gradIn

    def forward(self, query:str, k:int, *,
            reorder:bool=False,
            output_texts:bool=False,
            max_length:Optional[int]=None,
            **kwargs
        ) -> Tuple[Union[List[str],torch.IntTensor],torch.FloatTensor]:
        # control gradient computation:
        prev_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # set attention implementation to eager if attention-
        # based explantions are active:
        if kwargs.get('output_attentions', False):
            self.query_encoder.set_attn_implementation('eager')

        # create tokenizer arguments:
        tokenizer_args = {'padding':True, 'truncation':True, 'return_special_tokens_mask':True, 'return_tensors':'pt'}
        if max_length is not None:
            tokenizer_args['padding'] = 'max_length'
            tokenizer_args['max_length'] = max_length

        # apply tokenizer:
        qry_input = self.tokenizer(query, **tokenizer_args)
        token_mask = qry_input.pop('special_tokens_mask') == 0.
        
        self._x = {'query': [qry_input.input_ids[i, m].detach().cpu() for i, m in enumerate(token_mask)]}
        self._in_tokens = {'query': [self.tokenizer.convert_ids_to_tokens(t[m]) for t, m in zip(qry_input.input_ids, token_mask)]}

        # apply embeding:
        qry_embeds = self.query_encoder.get_input_embeddings()(torch.tensor(
            qry_input.pop('input_ids'),
            device=self.query_encoder.device
        ))

        # register gradient computation:
        self.query_encoder.get_input_embeddings().requires_grad_(True)

        # register index for gradient computation:
        self.index.requires_grad_(True)
        self.index.retain_grad()
        self.index.grad = None

        # Compute embeddings: take the last-layer hidden state of the [CLS] token
        qry_output = self.query_encoder(inputs_embeds=qry_embeds, **qry_input.to(self.query_encoder.device), **kwargs)
        has_attentions = checkattr(qry_output, 'attentions')

        # register embedding for gradient computation:
        qry_embeds.retain_grad()

        # register attentions for gradient computation:
        if has_attentions:
            a = qry_output.attentions[-1]
            a.retain_grad()

        # Compute dot product:
        similarity = qry_output.last_hidden_state[:, 0, :] @ self.index.to(qry_output.last_hidden_state).T
        similarity, retrieved_ids = torch.sort(similarity, dim=1, descending=True)
        similarity    = similarity[:, :k]
        retrieved_ids = retrieved_ids[:, :k]

        # calculate gradients of output:
        similarity.sum().backward()

        # load retrieved docs (for first query):
        retrieved = metadata_from_pkl(self.index_dir, retrieved_ids[0])

        self._x['context'] = retrieved['input_ids']
        self._in_tokens['context'] = retrieved['in_tokens']

        # save gradients with regard to phi:
        assert 'phi' in retrieved
        assert 'dPhi' in retrieved
        # Tensor of shape (bs x n_inputs x encoding_size)
        self._phi = {
            'query': [qry_embeds[i,m,:].detach().cpu()  for i, m in enumerate(token_mask)],
            'context': retrieved['phi']
        }
        # Tensor of shape (bs x n_inputs x encoding_size)
        self._dPhi = {
            'query': [qry_embeds.grad[i,m,:].detach().cpu() for i, m in enumerate(token_mask)],
            'context': [
                (ctx.to(qry)@qry.T).detach().cpu()
                for ctx, qry
                in zip(retrieved['dPhi'], self.index.grad[retrieved_ids[0]])
            ]
        }
        # Tensor of shape (bs x encoding_size)
        self._y = {  
            'query':   qry_output.last_hidden_state[:, 0, :].detach().cpu(),
            'context': self.index[retrieved_ids].detach().cpu()
        }

        # save gradients with regard to attention weights:
        if has_attentions:
            assert 'a' in retrieved
            assert 'da' in retrieved
            # Tensor of shape (bs x n_heads x n_outputs x n_inputs)
            a = qry_output.attentions[-1]
            # Tensor of shape (bs x n_heads x n_inputs)
            self._a = {
                'query': [a[i,:,0,m].detach().cpu() for i, m in enumerate(token_mask)],
                'context': retrieved['a']
            }
            # Tensor of shape (bs x n_heads x n_outputs x n_inputs)
            self._da = {
                'query': [a.grad[i,:,0,m].detach().cpu() for i, m in enumerate(token_mask)],
                'context': [
                    (ctx.to(qry)@qry.T).detach().cpu()
                    for ctx, qry
                    in zip(retrieved['da'], self.index.grad[retrieved_ids[0]])
                ]
            }

        # reset gradient computation:
        torch.set_grad_enabled(prev_grad)

        # move outputs to cpu:
        similarity = similarity[0].detach().cpu()
        retrieved_ids = retrieved_ids[0].cpu()
        retrieved_texts = retrieved['texts']

        # reorder contexts to orginial document order:
        if not reorder:
            sorted_ids = torch.argsort(retrieved_ids)
            self._x['context']                   = [self._x['context'][i] for i in sorted_ids]
            self._phi['context']                 = [self._phi['context'][i] for i in sorted_ids]
            self._dPhi['context']                = [self._dPhi['context'][i] for i in sorted_ids]
            if has_attentions:
                self._a['context']               = [self._a['context'][i] for i in sorted_ids]
                self._da['context']              = [self._da['context'][i] for i in sorted_ids]
            self._in_tokens['context']           = [self._in_tokens['context'][i] for i in sorted_ids]

            retrieved_texts                      = [retrieved_texts[i] for i in sorted_ids]

        if output_texts: return retrieved_texts, similarity
        else:            return retrieved_ids, similarity