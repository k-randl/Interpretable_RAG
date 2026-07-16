import os
import pickle
import torch

from transformers import PreTrainedTokenizer, AutoTokenizer
from torch import Tensor
from numpy.typing import NDArray
from typing import Union, List, Dict, Mapping, Literal, Optional, TypeAlias, Any, Tuple, cast
from .types import FloatTensorOrArray

from abc import ABCMeta, abstractmethod

#=======================================================================#
# Methods:                                                              #
#=======================================================================#

METHODS = ('grad', 'aGrad', 'repAGrad', 'gradIn', 'intGrad', 'lime', 'shap')
RetrieverMethods_t:TypeAlias = Literal['grad', 'aGrad', 'repAGrad', 'gradIn', 'intGrad', 'lime', 'shap']

#=======================================================================#
# Types:                                                                #
#=======================================================================#

RetrieverListDict_t:TypeAlias    = Mapping[Literal['query', 'context'], List[List[str]]]
RetrieverTensorDict_t:TypeAlias  = Mapping[Literal['query', 'context'], Tensor]
RetrieverAttribution_t:TypeAlias = Mapping[Literal['query', 'context'], Union[FloatTensorOrArray, List[FloatTensorOrArray]]]

# some methods (e.g. `intGrad`, `lime`, `shap`) can optionally return extra metadata
# (e.g. baseline predictions, additivity coverage) alongside the attribution itself:
RetrieverAttributionOutput_t:TypeAlias = Union[
    RetrieverAttribution_t,
    Tuple[RetrieverAttribution_t, Dict[str, Any]],
    Tuple[RetrieverAttribution_t, Dict[str, Any], Any],
]

#=======================================================================#
# Helper Functions:                                                     #
#=======================================================================#

def append_tensor_t(obj:Optional[RetrieverTensorDict_t], append:bool, qry:torch.Tensor, ctx:torch.Tensor,
        pad_val:Any, is_grad:bool=False) -> RetrieverTensorDict_t:
    """Build or extend a RetrieverTensorDict_t dict with query and context tensors.

    Args:
        obj:        Existing RetrieverTensorDict_t to extend, or None to create a fresh one.
        append:     If True, extend `obj`; if False, ignore `obj` and return a new dict.
        qry:        Query tensor.
        ctx:        Context tensor to append as new rows.
        pad_val:    Fill value used when expanding the context tensor to the new shape.
        is_grad:    If True, accumulate query gradients instead of asserting equality.

    Returns:
        A RetrieverTensorDict_t dict with keys ``'query'`` and ``'context'``.
    """
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

def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise dot-product similarity between rows of `a` and `b`.

    Args:
        a: Tensor of shape ``(m, d)``.
        b: Tensor of shape ``(n, d)``.

    Returns:
        Similarity matrix of shape ``(m, n)``.
    """
    return a @ b.T
compute_cosine_similarity_batched = torch.vmap(compute_cosine_similarity)
compute_cosine_similarity_batched.__doc__ = """Batched version of `compute_cosine_similarity` via `torch.vmap`.

Args:
    a: Tensor of shape ``(batch, m, d)``.
    b: Tensor of shape ``(batch, n, d)``.

Returns:
    Similarity matrices of shape ``(batch, m, n)``.
"""

def get_retriever_scores(explanation:'RetrieverExplanationBase', method:RetrieverMethods_t, **kwargs) -> RetrieverAttribution_t:
    """Dispatch to a retriever explanation's attribution method by name.

    Handles the `'grad'`/`'aGrad'` methods specially, since their raw per-hidden-state
    output needs an extra mean-reduction to become a per-token saliency score; all other
    methods (e.g. `'intGrad'`, `'shap'`, `'lime'`) are called directly by name.

    Args:
        explanation: The retriever explanation object (any object exposing `grad()`,
                     `aGrad()`, and/or a callable method named `method`).
        method:      Name of the attribution method to use.
        **kwargs:    Additional keyword arguments forwarded to the underlying method.

    Returns:
        A dict with `'query'` and `'context'` keys, each a list of per-document/per-query
        attribution arrays.

    Raises:
        ValueError: If `method` does not name a callable method on `explanation`.
    """
    if   method == 'grad':  return {key: [doc.mean(axis=-1) for doc in docs] for key, docs in explanation.grad(**kwargs).items()}
    elif method == 'aGrad': return {key: [doc.mean(axis=0) for doc in docs] for key, docs in explanation.aGrad(**kwargs).items()}

    method_fn = getattr(explanation, method, None)
    if callable(method_fn): return cast(RetrieverAttribution_t, method_fn(**kwargs))
    raise ValueError(f"`{type(explanation).__name__}` has no callable method named '{method}'")

#=======================================================================#
# Retriever Explanation:                                                #
#=======================================================================#

class RetrieverExplanationBase(metaclass=ABCMeta):
    #===================================================================#
    # Properties:                                                       #
    #===================================================================#

    @property
    @abstractmethod
    def in_tokens(self) -> Union[RetrieverListDict_t, None]:
        """A dicionary containing the following two keys:
        - `'query'`: a list containing the tokenized query
        - `'context'`: a list containing the tokenized contexts.

        `None` if no input has been processed yet."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizer:
        """The tokenizer used by the retriever model."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def query_encoder_name_or_path(self) -> str:
        """The huggingface string identifier of the query encoder model."""
        raise NotImplementedError()

    @property
    def context_encoder_name_or_path(self) -> Union[str,None]:
        """The huggingface string identifier of the context encoder model."""
        return None

    #===================================================================#
    # Methods:                                                          #
    #===================================================================#

    def grad(self, filter_special_tokens:bool=True) -> RetrieverAttribution_t:
        '''Gradients towards the inputs of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_inputs, n_tokens)'''
        raise NotImplementedError()

    def aGrad(self, filter_special_tokens:bool=True) -> RetrieverAttribution_t:
        '''AGrad (`-da ⊙ a`) scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_heads, n_inputs)'''
        raise NotImplementedError()

    def repAGrad(self, filter_special_tokens:bool=True) -> RetrieverAttribution_t:
        '''RepAGrad scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_heads, n_classes, n_inputs)'''
        raise NotImplementedError()

    def gradIn(self, filter_special_tokens:bool=True) -> RetrieverAttribution_t:
        '''GradIn (`dx ⊙ x`) scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.
            
        Returns:                    Importance scores with shape = (bs, n_inputs)'''
        raise NotImplementedError()

    def intGrad(self, filter_special_tokens:bool=True, **kwargs) -> RetrieverAttributionOutput_t:
        '''Integrated gradient scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.
            
        Returns:                    Importance scores with shape = (bs, n_inputs)'''
        raise NotImplementedError()

    def lime(self, filter_special_tokens:bool=True, **kwargs) -> RetrieverAttributionOutput_t:
        '''Lime scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.
            
        Returns:                    Importance scores with shape = (bs, n_inputs)'''
        raise NotImplementedError()

    def shap(self, filter_special_tokens:bool=True, **kwargs) -> RetrieverAttributionOutput_t:
        '''KernelSHAP scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.
            
        Returns:                    Importance scores with shape = (bs, n_inputs)'''
        raise NotImplementedError()

    def save_values(self, path:Optional[str]=None, *,
            methods:Optional[List[RetrieverMethods_t]]=None,
            filter_special_tokens:bool=True,
            num_steps:int=100,
            batch_size:int=64
        ) -> Union[Dict[str, Any], None]:
        """Saves the explanation data to a file.

        Args:
            path (str):                             The path where the values should be saved.
            methods (Lits[str], optional):          List of explanation methods to save. Saves all by default.
            filter_special_tokens (bool, optional): If `True` (default), set the importance of special tokens to 0.
            num_steps (int, optional):              Number of approximation steps for the Rieman approximation
                                                    of the integral in `intGrad` (default 100).
            batch_size (int, optional):             Batch size used for calculating the gradients in `intGrad` (default 64).

        Returns:
            If `path` is not specified, returns the saved data instead.
        """
        data_to_save = {
            'query_encoder_name_or_path': self.query_encoder_name_or_path,
            'context_encoder_name_or_path': self.context_encoder_name_or_path,
            'input': self.in_tokens
        }

        if methods is None:
            methods = list(METHODS)

        if 'grad' in methods:
            try: data_to_save['grad'] = self.grad(filter_special_tokens=filter_special_tokens)
            except NotImplementedError: pass
        
        if 'aGrad' in methods:
            try: data_to_save['aGrad'] = self.aGrad(filter_special_tokens=filter_special_tokens)
            except NotImplementedError: pass

        if 'repAGrad' in methods:
            try: data_to_save['repAGrad'] = self.repAGrad(filter_special_tokens=filter_special_tokens)
            except NotImplementedError: pass

        if 'gradIn' in methods:
            try: data_to_save['gradIn'] = self.gradIn(filter_special_tokens=filter_special_tokens)
            except NotImplementedError: pass

        if 'intGrad' in methods:
            try: data_to_save['intGrad'] = self.intGrad(filter_special_tokens=filter_special_tokens, num_steps=num_steps, batch_size=batch_size)
            except NotImplementedError: pass

        if 'lime' in methods:
            try: data_to_save['lime'] = self.lime(filter_special_tokens=filter_special_tokens, batch_size=batch_size)
            except NotImplementedError: pass

        if 'shap' in methods:
            try: data_to_save['shap'] = self.shap(filter_special_tokens=filter_special_tokens, batch_size=batch_size)
            except NotImplementedError: pass

        if path is None: return data_to_save

        with open(path, 'wb') as f:
            pickle.dump(data_to_save, f)


class RetrieverExplanation(RetrieverExplanationBase):
    _in_tokens:RetrieverListDict_t
    _grad:Union[RetrieverAttribution_t, None]
    _aGrad:Union[RetrieverAttribution_t, None]
    _repAGrad:Union[RetrieverAttribution_t, None]
    _gradIn:Union[RetrieverAttribution_t, None]
    _intGrad:Union[RetrieverAttribution_t, None]
    _lime:Union[RetrieverAttribution_t, None]
    _shap:Union[RetrieverAttribution_t, None]
    _query_encoder_name_or_path:str
    _context_encoder_name_or_path:Union[str, None]
    _tokenizer:PreTrainedTokenizer

    @classmethod
    def load(cls, saved_data:Union[str, dict, list], *,
        query_encoder_name_or_path:Optional[str]=None,
        context_encoder_name_or_path:Optional[str]=None,
        tokenizer:Optional[PreTrainedTokenizer]=None,
    ) -> Union['RetrieverExplanation', List['RetrieverExplanation'], Dict[str, 'RetrieverExplanation']]:
        """Loads the GeneratorExplanation from a file path or dictionary.

        Args:
            saved_data (str | dict | list):     Path to the saved pickle file or the dictionary itself.
            query_encoder_name_or_path (str):   An optional way to specify the query encoder name.
            context_encoder_name_or_path (str): An optional way to specify the context encoder name.
            tokenizer (PreTrainedTokenizer):    An optional way to set the tokenizer to avoid resinstantiation
                                                the same tokenizer multiple times. 
        """
        if isinstance(saved_data, str):
            # load all files in a dir as a list:
            if os.path.isdir(saved_data):
                keys, paths = list(zip(*[(file[:-4], os.path.join(saved_data, file))
                    for file in os.listdir(saved_data)
                    if file.endswith('.pkl')]))

                values = cls.load(paths,
                    query_encoder_name_or_path=query_encoder_name_or_path,
                    context_encoder_name_or_path=context_encoder_name_or_path,
                    tokenizer=tokenizer)
                
                return {k:v for k,v in zip(keys, values)}

            # load file:
            with open(saved_data, 'rb') as f:
                data = pickle.load(f)

        elif isinstance(saved_data, dict):
            data = saved_data

        elif hasattr(saved_data, '__iter__') or hasattr(saved_data, '__len__'):
            result = []

            # load all entries separatelly:
            for item in saved_data:
                try:
                    # load next file:
                    result.append(
                        cls.load(
                            item,
                            query_encoder_name_or_path=query_encoder_name_or_path,
                            context_encoder_name_or_path=context_encoder_name_or_path,
                            tokenizer=tokenizer
                        )
                    )

                    # avoid multiple instances of the tokenizer:
                    if tokenizer is None: tokenizer = result[0].tokenizer

                except Exception as e:
                    print(f'WARNING: Could not load "{item}": {e}')

            return result

        else: raise ValueError("`saved_data` must be a path, an iterable, or a dictionary")

        result = cls()
        result._in_tokens = data['input']
        result._grad = data.get('grad')
        result._aGrad = data.get('aGrad')
        result._repAGrad = data.get('repAGrad')
        result._gradIn = data.get('gradIn')
        result._intGrad = data.get('intGrad')
        result._lime = data.get('lime')
        result._shap = data.get('shap')

        if query_encoder_name_or_path is None: result._query_encoder_name_or_path = data['query_encoder_name_or_path'] 
        else: result._query_encoder_name_or_path = query_encoder_name_or_path
        
        if context_encoder_name_or_path is None: result._context_encoder_name_or_path = data.get('context_encoder_name_or_path')
        else: result._query_encoder_name_or_path = context_encoder_name_or_path

        if tokenizer is None: result._tokenizer = AutoTokenizer.from_pretrained(result._query_encoder_name_or_path)
        else: result._tokenizer =tokenizer

        return result

    @property
    def in_tokens(self) -> RetrieverListDict_t:
        """A dicionary containing the following two keys:
        - `'query'`: a list containing the tokenized query
        - `'context'`: a list containing the tokenized contexts."""
        return self._in_tokens

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """The tokenizer used by the retriever model."""
        return self._tokenizer

    @property
    def query_encoder_name_or_path(self) -> str:
        """The huggingface string identifier of the query encoder model."""
        return self._query_encoder_name_or_path

    @property
    def context_encoder_name_or_path(self) -> Union[str, None]:
        """The huggingface string identifier of the context encoder model."""
        return self._context_encoder_name_or_path

    def grad(self, filter_special_tokens:bool=True) -> RetrieverAttribution_t:
        '''Gradients towards the inputs of the last batch.

        Returns:
            Importance scores with shape = (bs, n_inputs, n_tokens)'''
        if self._grad is None: raise NotImplementedError("No `grad` values were saved.")
        return self._grad

    def aGrad(self, filter_special_tokens:bool=True) -> RetrieverAttribution_t:
        '''AGrad (`-da ⊙ a`) scores of the last batch.

        Returns:
            Importance scores with shape = (bs, n_heads, n_inputs)'''
        if self._aGrad is None: raise NotImplementedError("No `aGrad` values were saved.")
        return self._aGrad

    def repAGrad(self, filter_special_tokens:bool=True) -> RetrieverAttribution_t:
        '''RepAGrad scores of the last batch.

        Returns:
            Importance scores with shape = (bs, n_heads, n_classes, n_inputs)'''
        if self._repAGrad is None: raise NotImplementedError("No `repAGrad` values were saved.")
        return self._repAGrad

    def gradIn(self, filter_special_tokens:bool=True) -> RetrieverAttribution_t:
        '''GradIn (`dx ⊙ x`) scores of the last batch.

        Returns:
            Importance scores with shape = (bs, n_inputs)'''
        if self._gradIn is None: raise NotImplementedError("No `gradIn` values were saved.")
        return self._gradIn
    
    def intGrad(self, filter_special_tokens:bool=True, **kwargs) -> RetrieverAttribution_t:
        '''Integrated gradient scores of the last batch.

        Returns:
            Importance scores with shape = (bs, n_inputs)'''
        if self._intGrad is None: raise NotImplementedError("No `intGrad` values were saved.")
        return self._intGrad

    def lime(self, filter_special_tokens:bool=True, **kwargs) -> RetrieverAttribution_t:
        '''Lime scores of the last batch.
            
        Returns:
            Importance scores with shape = (bs, n_inputs)'''
        if self._lime is None: raise NotImplementedError("No `lime` values were saved.")
        return self._lime

    def shap(self, filter_special_tokens:bool=True, **kwargs) -> RetrieverAttribution_t:
        '''KernelSHAP scores of the last batch.
            
        Returns:
            Importance scores with shape = (bs, n_inputs)'''
        if self._shap is None: raise NotImplementedError("No `shap` values were saved.")
        return self._shap