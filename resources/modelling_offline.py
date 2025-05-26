import os
import pickle
import torch
import numpy as np
from torch.func import jacrev
from tqdm.autonotebook import tqdm
from transformers import PreTrainedModel, AutoModel, AutoTokenizer
from typing import Optional, List, Union, Dict

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

def metadata_to_json(dir:str, doc_ids:List[Union[int, str]], metadata:Dict[str, Union[List, torch.Tensor]]):
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

def metadata_from_json(dir:str, doc_ids:List[Union[int, str]]) -> Dict[str, List]:
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
    def from_pretrained(cls, model_name_or_path:str, tokenizer_name_or_path:Optional[str]=None, *args, **kwargs):
        encoder = cls()

        # load tokenizer:
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path
        encoder.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, *args, **kwargs
        )

        # load encoder:
        encoder.context_encoder = AutoModel.from_pretrained(
            model_name_or_path, *args, **kwargs
        )

        return encoder
    
    def forward(self, contexts:List[str], **kwargs):
        # control gradient computation:
        prev_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # Apply tokenizer
        ctx_input = self.tokenizer(contexts, padding=True, truncation=True, return_special_tokens_mask=True, return_tensors='pt')
        output = {
            'texts':                contexts,
            'input_ids':            ctx_input.input_ids.detach().cpu(),
            'special_tokens_mask':  ctx_input.pop('special_tokens_mask').detach().cpu(),
            'in_tokens':            [[self.tokenizer.decode(token) for token in text] for text in ctx_input.input_ids]
        }

        # register gradient computation:
        self.context_encoder.get_input_embeddings().requires_grad_(True)

        # Compute embeddings: take the last-layer hidden state of the [CLS] token
        ctx_output = self.context_encoder(**ctx_input, **kwargs)
        output['embedding'] = ctx_output.last_hidden_state[:, 0, :].detach().cpu()
        embedding_size = output['embedding'].shape[-1]

        has_attentions    = checkattr(ctx_output, 'attentions')
        has_hidden_states = checkattr(ctx_output, 'hidden_states')

        # register gradient computation for attention weights:
        if has_attentions:
            # Tensor of shape (bs x n_heads x n_outputs x n_inputs)
            a = ctx_output.attentions[-1]

            # Tensor of shape (bs x n_heads x n_inputs)
            output['a'] = a[:,:,0,:].detach().clone().cpu()

            # register retain_grad:
            a.retain_grad()

            # Tensor of shape (bs x n_heads x n_inputs x embedding_size)
            output['da'] = torch.FloatTensor(a[:,:,0,:].shape + (embedding_size,), device='cpu')

        # register gradient computation for hidden states:
        if has_hidden_states:
            # Tensor of shape (bs x n_inputs x embedding_size)
            h0 = ctx_output.hidden_states[0]

            # Tensor of shape (bs x n_inputs x embedding_size)
            output['h0'] = h0.detach().clone().cpu()

            # register retain_grad:
            h0.retain_grad()

            # Tensor of shape (bs x n_inputs x embedding_size x embedding_size)
            output['dh0'] = torch.FloatTensor(h0.shape + (embedding_size,), device='cpu')

        for i in tqdm(range(embedding_size), total=embedding_size):
            # reset old gradients:
            self.context_encoder.zero_grad()

            # calculate gradients of output:
            ctx_output.last_hidden_state[:, 0, i].sum().backward(retain_graph = True)

            # save gradients with regard to attention weights:
            if has_attentions:
                a = ctx_output.attentions[-1]
                output['da'][:,:,:,i] = a.grad[:,:,0,:].detach().clone().cpu()

            # save gradients with regard to hidden states:
            if has_hidden_states:
                h0 = ctx_output.hidden_states[0]
                output['dh0'][:,:,:,i] = h0.grad.detach().clone().cpu()

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
            metadata_to_json(dir, range(i, j), outputs)

        with open(os.path.join(dir, 'embeddings.pt'), 'wb') as file:
            torch.save(torch.concatenate(embeddings, dim=0), file)

class ExplainableAutoModelForRAG(torch.nn.Module):
    @classmethod
    def from_pretrained(cls, query_encoder_name_or_path:str, tokenizer_name_or_path:Optional[str]=None, *args, **kwargs):
        encoder = cls()

        # load tokenizer:
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = query_encoder_name_or_path
        encoder.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, *args, **kwargs
        )

        # load encoder:
        encoder.query_encoder = AutoModel.from_pretrained(
            query_encoder_name_or_path, *args, **kwargs
        )

        return encoder

    @property
    def in_tokens(self):
        if hasattr(self, '_in_tokens'): return self._in_tokens
        else: return None

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
            'query':   embedding_backward(self.query_encoder, self._dh0['query']),
            'context': embedding_backward(self.query_encoder, self._dh0['context'])
        }
    
        if filter_special_tokens:
            # set the importance of special tokens to 0.
            grad = {key: grad[key] * self._special_tokens_mask[key][:,:,None] for key in grad}

        return grad

    def aGrad(self, filter_special_tokens:bool=True):
        '''AGrad (`-da ⊙ a`) scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_heads, n_inputs)'''

        # compute attention weigths and gradients:
        a  = {key:self._a[key] for key in self._a}     # shape: (bs x n_heads x n_inputs)
        da = {key:self._da[key] for key in self._da}   # shape: (bs x n_heads x n_inputs)

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

        Returns:                    Importance scores with shape = (bs, n_heads, n_classes, n_inputs)'''

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

    def forward(self, query:str, k:int, dir:str='embeddings', index:Optional[torch.FloatTensor]=None, **kwargs):
        # load index from disk if not speciifed:
        if index is None:
            with open(os.path.join(dir, 'embeddings.pt'), 'rb') as file:
                index = torch.load(file)
        
        # control gradient computation:
        prev_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # Apply tokenizer
        qry_input = self.tokenizer(query, return_special_tokens_mask=True, return_tensors='pt')

        self._x = {'query': qry_input.input_ids.detach().cpu().numpy()}
        self._special_tokens_mask = {'query':   1. - qry_input.pop('special_tokens_mask').detach().cpu().numpy()}
        self._in_tokens = {'query': [[self.tokenizer.decode(token) for token in text] for text in qry_input.input_ids]}

        # register gradient computation:
        self.query_encoder.get_input_embeddings().requires_grad_(True)

        # register index for gradient computation:
        index.requires_grad_(True)
        index.retain_grad()
        index.grad = None

        # Compute embeddings: take the last-layer hidden state of the [CLS] token
        qry_output = self.query_encoder(**qry_input, **kwargs)

        has_attentions    = checkattr(qry_output, 'attentions')
        has_hidden_states = checkattr(qry_output, 'hidden_states')

        # register attentions for gradient computation:
        if has_attentions:
            a = qry_output.attentions[-1]
            a.retain_grad()

        # register hidden states for gradient computation:
        if has_hidden_states:
            h0 = qry_output.hidden_states[0]
            h0.retain_grad()

        # Compute dot product:
        similarity = qry_output.last_hidden_state[:, 0, :] @ index.to(qry_output.last_hidden_state).T
        retrieved_ids = torch.argsort(similarity, dim=1, descending=True)[:, :k]

        # calculate gradients of output:
        similarity.sum().backward()

        # load retrieved docs (for first query):
        retrieved = metadata_from_json(dir, retrieved_ids[0])

        self._x['context'] = np.array(retrieved['input_ids'])
        self._special_tokens_mask['context'] = 1 - np.array(retrieved['special_tokens_mask'])
        self._in_tokens['context'] = retrieved['in_tokens']

        # save gradients with regard to attention weights:
        if has_attentions:
            assert 'a' in retrieved
            assert 'da' in retrieved
            # Tensor of shape (bs x n_heads x n_outputs x n_inputs)
            a = qry_output.attentions[-1]
            # Tensor of shape (bs x n_heads x n_inputs)
            self._a = {
                'query': a[:,:,0,:].detach().clone().cpu(),
                'context': torch.stack(retrieved['a'])
            }
            # Tensor of shape (bs x n_heads x n_outputs x n_inputs)
            self._da = {
                'query': a.grad[:,:,0,:].detach().clone().cpu(),
                'context': torch.stack([
                    ctx.to(qry)@qry.T
                    for ctx, qry
                    in zip(retrieved['da'], index.grad[retrieved_ids[0]])
                ])
            }

        # save gradients with regard to hidden states:
        if has_hidden_states:
            assert 'h0' in retrieved
            assert 'dh0' in retrieved
            # Tensor of shape (bs x n_inputs x encoding_size)
            h0 = qry_output.hidden_states[0]
            # Tensor of shape (bs x n_inputs x encoding_size)
            self._h0 = {
                'query': h0.detach().clone().cpu(),
                'context': retrieved['h0']
            }
            # Tensor of shape (bs x n_inputs x encoding_size)
            self._dh0 = {
                'query': h0.grad.detach().clone().cpu(),
                'context': torch.stack([
                    ctx.to(qry)@qry.T
                    for ctx, qry
                    in zip(retrieved['dh0'], index.grad[retrieved_ids[0]])
                ])
            }

        # reset gradient computation:
        torch.set_grad_enabled(prev_grad)

        return retrieved_ids[0], similarity[0, retrieved_ids[0]]