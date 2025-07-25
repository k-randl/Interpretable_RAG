import pickle
import numpy as np

from transformers import PreTrainedTokenizer, AutoTokenizer
from torch import FloatTensor
from numpy.typing import NDArray
from typing import Union, List, Dict, Literal, Optional

from abc import ABCMeta, abstractmethod

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
    def grad(self, filter_special_tokens:bool=True) ->  Dict[Literal['query', 'context'], List[Union[NDArray[np.float_], FloatTensor]]]:
        '''Gradients towards the inputs of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_inputs, n_tokens)'''
        raise NotImplementedError()

    @abstractmethod
    def aGrad(self, filter_special_tokens:bool=True) -> Dict[Literal['query', 'context'], List[Union[NDArray[np.float_], FloatTensor]]]:
        '''AGrad (`-da ⊙ a`) scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_heads, n_inputs)'''
        raise NotImplementedError()

    @abstractmethod
    def repAGrad(self, filter_special_tokens:bool=True) -> Dict[Literal['query', 'context'], List[Union[NDArray[np.float_], FloatTensor]]]:
        '''RepAGrad scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.

        Returns:                    Importance scores with shape = (bs, n_heads, n_classes, n_inputs)'''
        raise NotImplementedError()

    @abstractmethod
    def gradIn(self, filter_special_tokens:bool=True) -> Dict[Literal['query', 'context'], List[Union[NDArray[np.float_], FloatTensor]]]:
        '''GradIn (`dx ⊙ x`) scores of the last batch.

        Args:
            filter_special_tokens:  If `True`, set the importance of special tokens to 0.
            
        Returns:                    Importance scores with shape = (bs, n_inputs)'''
        raise NotImplementedError()

    def save_values(self, path:Optional[str]=None) -> Union[str, None]:
        """Saves the explanation data to a file.

        Args:
            path (str): The path where the values should be saved.

        Returns:
            If `path` is not specified, returns the saved data instead.
        """
        data_to_save = {
            'query_encoder_name_or_path': self.query_encoder_name_or_path,
            'context_encoder_name_or_path': self.context_encoder_name_or_path,
            'input': self.in_tokens
        }

        try: data_to_save['grad'] = self.grad()
        except NotImplementedError: pass

        try: data_to_save['aGrad'] = self.aGrad()
        except NotImplementedError: pass

        try: data_to_save['repAGrad'] = self.repAGrad()
        except NotImplementedError: pass

        try: data_to_save['gradIn'] = self.gradIn()
        except NotImplementedError: pass

        if path is None: return data_to_save

        with open(path, 'wb') as f:
            pickle.dump(data_to_save, f)


class RetrieverExplanation(RetrieverExplanationBase):
    def __init__(self, saved_data: Union[str, dict]):
        """Initializes the RetrieverExplanation from a file path or dictionary.

        Args:
            saved_data (str or dict): Path to the saved pickle file or the dictionary itself.
        """
        if isinstance(saved_data, str):
            with open(saved_data, 'rb') as f:
                data = pickle.load(f)

        elif isinstance(saved_data, dict):
            data = saved_data

        else: raise ValueError("`saved_data` must be a filepath or a dictionary")

        self._query_encoder_name_or_path = data['query_encoder_name_or_path']
        self._context_encoder_name_or_path = data.get('context_encoder_name_or_path')
        self._in_tokens = data['input']
        self._grad = data.get('grad')
        self._aGrad = data.get('aGrad')
        self._repAGrad = data.get('repAGrad')
        self._gradIn = data.get('gradIn')

        self._tokenizer = AutoTokenizer.from_pretrained(self._query_encoder_name_or_path)

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

    def grad(self, filter_special_tokens:bool=True) -> Dict[Literal['query', 'context'], List[Union[NDArray[np.float_], FloatTensor]]]:
        '''Gradients towards the inputs of the last batch.

        Returns:
            Importance scores with shape = (bs, n_inputs, n_tokens)'''
        if self._grad is None: raise NotImplementedError("No `grad` values were saved.")
        return self._grad

    def aGrad(self, filter_special_tokens:bool=True) -> Dict[Literal['query', 'context'], List[Union[NDArray[np.float_], FloatTensor]]]:
        '''AGrad (`-da ⊙ a`) scores of the last batch.

        Returns:
            Importance scores with shape = (bs, n_heads, n_inputs)'''
        if self._aGrad is None: raise NotImplementedError("No `aGrad` values were saved.")
        return self._aGrad

    def repAGrad(self, filter_special_tokens:bool=True) -> Dict[Literal['query', 'context'], List[Union[NDArray[np.float_], FloatTensor]]]:
        '''RepAGrad scores of the last batch.

        Returns:
            Importance scores with shape = (bs, n_heads, n_classes, n_inputs)'''
        if self._repAGrad is None: raise NotImplementedError("No `repAGrad` values were saved.")
        return self._repAGrad

    def gradIn(self, filter_special_tokens:bool=True) -> Dict[Literal['query', 'context'], List[Union[NDArray[np.float_], FloatTensor]]]:
        '''GradIn (`dx ⊙ x`) scores of the last batch.

        Returns:
            Importance scores with shape = (bs, n_inputs)'''
        if self._gradIn is None: raise NotImplementedError("No `gradIn` values were saved.")
        return self._gradIn