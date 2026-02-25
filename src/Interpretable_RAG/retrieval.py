import os
import pickle
import numpy as np

from transformers import PreTrainedTokenizer, AutoTokenizer
from torch import Tensor, FloatTensor
from numpy.typing import NDArray
from typing import Union, List, Dict, Literal, Optional, TypeAlias

from abc import ABCMeta, abstractmethod

#=======================================================================#
# Types:                                                                #
#=======================================================================#

List_t:TypeAlias   = Dict[Literal['query', 'context'], List]
Array_t:TypeAlias  = Dict[Literal['query', 'context'], NDArray]
Tensor_t:TypeAlias = Dict[Literal['query', 'context'], Tensor]
Out_t:TypeAlias    = Dict[Literal['query', 'context'], List[Union[NDArray[np.float64], FloatTensor]]]

#=======================================================================#
# Retriever Explanation:                                                #
#=======================================================================#

class RetrieverExplanationBase(metaclass=ABCMeta):
    #===================================================================#
    # Properties:                                                       #
    #===================================================================#

    @property
    @abstractmethod
    def in_tokens(self) -> Dict[Literal['query', 'context'], List[List[str]]]:
        """A dicionary containing the following two keys:
        - `'query'`: a list containing the tokenized query
        - `'context'`: a list containing the tokenized contexts."""
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

    @abstractmethod
    def grad(self, filter_special_tokens:bool=True) -> Out_t:
        '''Gradients towards the inputs of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_inputs, n_tokens)'''
        raise NotImplementedError()

    @abstractmethod
    def aGrad(self, filter_special_tokens:bool=True) -> Out_t:
        '''AGrad (`-da ⊙ a`) scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_heads, n_inputs)'''
        raise NotImplementedError()

    @abstractmethod
    def repAGrad(self, filter_special_tokens:bool=True) -> Out_t:
        '''RepAGrad scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_heads, n_classes, n_inputs)'''
        raise NotImplementedError()

    @abstractmethod
    def gradIn(self, filter_special_tokens:bool=True) -> Out_t:
        '''GradIn (`dx ⊙ x`) scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.
            
        Returns:                    Importance scores with shape = (bs, n_inputs)'''
        raise NotImplementedError()

    @abstractmethod
    def intGrad(self, filter_special_tokens:bool=True, **kwargs) -> Out_t:
        '''Integrated gradient scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.
            
        Returns:                    Importance scores with shape = (bs, n_inputs)'''
        raise NotImplementedError()

    def save_values(self, path:Optional[str]=None, *, filter_special_tokens:bool=True, num_steps:int=100, batch_size:int=64) -> Union[str, None]:
        """Saves the explanation data to a file.

        Args:
            path (str):                             The path where the values should be saved.
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

        try: data_to_save['grad'] = self.grad(filter_special_tokens=filter_special_tokens)
        except NotImplementedError: pass

        try: data_to_save['aGrad'] = self.aGrad(filter_special_tokens=filter_special_tokens)
        except NotImplementedError: pass

        try: data_to_save['repAGrad'] = self.repAGrad(filter_special_tokens=filter_special_tokens)
        except NotImplementedError: pass

        try: data_to_save['gradIn'] = self.gradIn(filter_special_tokens=filter_special_tokens)
        except NotImplementedError: pass

        try: data_to_save['intGrad'] = self.intGrad(filter_special_tokens=filter_special_tokens, num_steps=num_steps, batch_size=batch_size)
        except NotImplementedError: pass

        if path is None: return data_to_save

        with open(path, 'wb') as f:
            pickle.dump(data_to_save, f)


class RetrieverExplanation(RetrieverExplanationBase):
    @classmethod
    def load(cls, saved_data:Union[str, dict, list], *,
        query_encoder_name_or_path:Optional[str]=None,
        context_encoder_name_or_path:Optional[str]=None,
        tokenizer:Optional[PreTrainedTokenizer]=None,
    ) -> Union['RetrieverExplanation', List['RetrieverExplanation']]:
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
                return cls.load([os.path.join(saved_data, file)
                    for file in os.listdir(saved_data)
                    if not file.endswith('.pkl')])

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

        if query_encoder_name_or_path is None: result._query_encoder_name_or_path = data['query_encoder_name_or_path'] 
        else: result._query_encoder_name_or_path = query_encoder_name_or_path
        
        if context_encoder_name_or_path is None: result._context_encoder_name_or_path = data.get('context_encoder_name_or_path')
        else: result._query_encoder_name_or_path = context_encoder_name_or_path

        if tokenizer is None: result._tokenizer = AutoTokenizer.from_pretrained(result._query_encoder_name_or_path)
        else: result._tokenizer =tokenizer

        return result

    @property
    def in_tokens(self) -> Dict[Literal['query', 'context'], List[List[str]]]:
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

    def grad(self, filter_special_tokens:bool=True) -> Out_t:
        '''Gradients towards the inputs of the last batch.

        Returns:
            Importance scores with shape = (bs, n_inputs, n_tokens)'''
        if self._grad is None: raise NotImplementedError("No `grad` values were saved.")
        return self._grad

    def aGrad(self, filter_special_tokens:bool=True) -> Out_t:
        '''AGrad (`-da ⊙ a`) scores of the last batch.

        Returns:
            Importance scores with shape = (bs, n_heads, n_inputs)'''
        if self._aGrad is None: raise NotImplementedError("No `aGrad` values were saved.")
        return self._aGrad

    def repAGrad(self, filter_special_tokens:bool=True) -> Out_t:
        '''RepAGrad scores of the last batch.

        Returns:
            Importance scores with shape = (bs, n_heads, n_classes, n_inputs)'''
        if self._repAGrad is None: raise NotImplementedError("No `repAGrad` values were saved.")
        return self._repAGrad

    def gradIn(self, filter_special_tokens:bool=True) -> Out_t:
        '''GradIn (`dx ⊙ x`) scores of the last batch.

        Returns:
            Importance scores with shape = (bs, n_inputs)'''
        if self._gradIn is None: raise NotImplementedError("No `gradIn` values were saved.")
        return self._gradIn
    
    def intGrad(self, filter_special_tokens:bool=True, **kwargs) -> Out_t:
        '''Integrated gradient scores of the last batch.

        Returns:
            Importance scores with shape = (bs, n_inputs)'''
        if self._intGrad is None: raise NotImplementedError("No `intGrad` values were saved.")
        return self._intGrad