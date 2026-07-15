import os
import torch
import inspect
import numpy as np
import transformers
import tqdm
import pickle

from scipy.special import comb
from sklearn.linear_model import Ridge

from .utils import decode_chat_template, get_model_type, generate_permutations, sample_perturbations

from numpy.typing import NDArray
from typing import Union, List, Dict, Tuple, Optional, Literal, Iterable, Callable, TypeAlias, Any, cast

from abc import ABCMeta, abstractmethod

#=======================================================================#
# Declarations:                                                         #
#=======================================================================#

MAXINT = int(2**31 - 1) # Maximum value for signed int32

AGGREGATIONS = ('token', 'sequence', 'bow', 'nucleus')
GeneratorAggregations_t:TypeAlias = Literal['token', 'sequence', 'bow', 'nucleus']

METHODS = ('lime', 'shap')
GeneratorMethods_t:TypeAlias = Literal['lime', 'shap']

# Return type of `shap()`/`lime()`: a per-key value is `None` whenever that key
# (e.g. `'query'`) was not perturbed/computed (e.g. `max_samples_query=0`).
GeneratorAttribution_t:TypeAlias = Union[Dict[Literal['query', 'context'], Union[NDArray[np.float32], None]], NDArray[np.float32], None]

#=======================================================================#
# Helper Functions:                                                     #
#=======================================================================#

def _to_batch(input_ids:torch.Tensor,
              output_ids:torch.Tensor,
              pad_token_id:int,
              batch_size:int) -> Tuple[torch.Tensor, torch.Tensor]:
    
    input_size = input_ids.shape[1]
    output_size = output_ids.shape[1]

    batched_inputs_ids = torch.full(
        (batch_size, input_size + output_size),
        pad_token_id,
        device=input_ids.device,
        dtype=input_ids.dtype
    )

    batched_inputs_ids[-1,-input_size-output_size:-output_size] = input_ids[0]
    batched_inputs_ids[-1,-output_size:] = output_ids[0]

    for i in range(1,batch_size):
        batched_inputs_ids[-i-1,-input_size-output_size+i:-output_size+i] = input_ids[0]
        batched_inputs_ids[-i-1,-output_size+i:] = output_ids[0,:-i]

    return (
        batched_inputs_ids[:,:input_size + batch_size - 1],
        batched_inputs_ids[:,input_size + batch_size - 1:]
    )

def _nucleus_sampling(probs, p=0.9):
    """
    Applies nucleus (top-p) sampling to logits while preserving original order.

    Args:
        logits (torch.Tensor): shape [1, seq_len, vocab_size]
        p (float): cumulative probability threshold

    Returns:
        torch.Tensor: same shape as logits, with only top-p probs kept per token
    """
    #probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create a mask for tokens to keep (top-p)
    sorted_mask = cumulative_probs <= p
    # Ensure at least one token is always included
    sorted_mask[..., 0] = 1

    # Map back to original indices
    unsorted_mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
    batch_size, seq_len, vocab_size = probs.shape

    for i in range(seq_len):
        unsorted_mask[0, i].scatter_(0, sorted_indices[0, i], sorted_mask[0, i])

    # Zero out the rest
    probs_filtered = probs * unsorted_mask

    return probs_filtered

def create_rag_prompt(query:str, contexts:List[str], *, system:Optional[str]=None, max_document_size:Optional[int]=None) -> List[Dict[str,str]]:
    """
    Creates chat-style messages for RAG using LLaMA-style chat template.

    Args:
        query (str):               The user's query.
        contexts (list):           A list of strings representing the retrieved documents.
        system (str):              An optional system prompt.
        max_document_size (int): An optional size limit of context documents in characters.

    Returns:
        A list of messages in chat format suitable for tokenizer.apply_chat_template().
    """

    # System prompt that sets the assistant behavior
    if system is None:
        system = (
            "Use the following retrieved documents, ranked from highest "
            "to lowest relevance, to answer the user's query. "
            "Be thorough and accurate, and cite documents when useful. "
            "Keep the answer under 200 words."
        )

    # Apply size limit (build a new list instead of mutating the caller's `contexts`)
    if max_document_size is not None:
        contexts = [
            doc[:max_document_size - 3] + '...' if len(doc) > max_document_size else doc
            for doc in contexts
        ]

    # Format the context into a single message
    context_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(contexts)])

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{context_text}\n\nQuery: {query}"}
    ]

def logits2probs(logits:torch.Tensor, normalization:Literal['softmax', 'relu', 'offset']='softmax'):
    """Converts raw logits into probability distributions using a specified normalization normalization.

    Args:
        logits (torch.Tensor):  A tensor containing raw, unnormalized prediction scores (logits).
        normalization (str):    The normalization method to use for converting logits to probabilities (default='softmax')
                                - `'softmax'` applies the softmax function along the last dimension.
                                  Ensures outputs are strictly positive and sum to 1.
                                - `'relu'` sets all negative logits to zero (ReLU), then normalizes
                                  by the sum over the last dimension. Can produce sparse outputs.
                                - `'offset'` shifts logits by subtracting the global minimum, then normalizes
                                  by the sum over the last dimension. Ensures non-negative outputs
                                  without enforcing strict positivity.

    Returns:
        probs (torch.Tensor):   A tensor of the same shape as `logits`, containing valid probability
                                distributions over the last dimension.

    Raises:
        ValueError:             If the provided `normalization` is not one of the allowed options.
    """
    if normalization == 'softmax':
        probs = torch.softmax(logits, dim=-1)

    elif normalization == 'relu':
        probs = torch.maximum(logits, torch.tensor(0.))
        probs /= probs.sum(dim=-1, keepdim=True)

    elif normalization == 'offset':
        probs = logits - logits.min()
        probs /= probs.sum(dim=-1, keepdim=True)

    else: raise ValueError(f'Unknown value for parameter `normalization`: "{normalization}".')

    return probs

def get_generator_scores(explanation:'GeneratorExplanationBase', method:GeneratorMethods_t, *,
        key:Union[Literal['query', 'context'], None] = None,
        aggregation:GeneratorAggregations_t = 'token',
        **kwargs
    ) -> GeneratorAttribution_t:
    """Dispatch to a generator explanation's attribution method by name.

    Args:
        explanation: The generator explanation object (any object exposing a callable
                     method named `method`, e.g. `shap()` or `lime()`).
        method:      Name of the attribution method to use (e.g. `'shap'`, `'lime'`).
        key:         Explanation key. Can either be `'query'` or `'context'`.
                     If `None` returns a dictionary of both.
        aggregation: Aggregation method for probabilities (default: `'token'`).
        **kwargs:    Additional keyword arguments forwarded to the underlying method.

    Returns:
        A dicionary containing the following two keys (if `key` is specified) or one of the following:
        - `'query'`: a `numpy.ndarray` containing the attribution values for the query
        - `'context'`: a `numpy.ndarray` containing the attribution values for the contexts.

    Raises:
        ValueError: If `method` does not name a callable method on `explanation`.
    """
    method_fn = getattr(explanation, method, None)
    if callable(method_fn): return cast(GeneratorAttribution_t, method_fn(key, aggregation, **kwargs))
    raise ValueError(f"`{type(explanation).__name__}` has no callable method named '{method}'")

#=======================================================================#
# Generator Explanation:                                                #
#=======================================================================#

class GeneratorExplanationBase(metaclass=ABCMeta):
    #===================================================================#
    # Properties:                                                       #
    #===================================================================#

    @property
    @abstractmethod
    def qry_tokens(self) -> List[str]:
        """The list of query tokens."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def gen_tokens(self) -> List[str]:
        """The list of generated tokens."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def qry_precise(self) -> bool:
        """`True` if the Shapley feature attribution values for the query are not approximated."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def ctx_precise(self) -> bool:
        """`True` if the Shapley feature attribution values for the context are not approximated."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        """The tokenizer used by the generator model."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def model_name_or_path(self) -> str:
        """The huggingface string identifier of the generator model."""
        raise NotImplementedError()

    @property
    def focus(self) -> slice:
        """The focus of the explanation, if set."""
        if hasattr(self, '_focus'): return self._focus
        return slice(None)

    @focus.setter
    def focus(self, value:Union[Tuple[int, int], None]) -> None:
        """Sets the focus of the explanation."""
        if value is None:
            if hasattr(self, '_focus'): del self._focus

        elif isinstance(value, tuple):
            self._focus = slice(*value)

        else: raise TypeError(
            f'`focus` must be a tuple of two integers or `None`, but got `{type(value)}: {value}`'
        )

    #===================================================================#
    # Methods:                                                          #
    #===================================================================#

    @abstractmethod
    def shap(self,
            key:Union[Literal['query', 'context'], None], aggregation:GeneratorAggregations_t='token',
            **kwargs
        ) -> GeneratorAttribution_t:
        """Generates Shapley feature attribution values for the chosen aggregation method.

        Args:
            key (str):          Explanation key. Can either be `'query'` or `'context'`.
                                If `None` returns a dictionary of both.
            aggregation (str):  Aggregation method for probabilities (default: `'token'`).

        Returns:
            A dicionary containing the following two keys (if `key` is specified) or one of the following:
            - `'query'`: a `numpy.ndarray` containing the Shapley values for the query
            - `'context'`: a `numpy.ndarray` containing the Shapley values for the contexts.
        """
        raise NotImplementedError()

    @abstractmethod
    def lime(self, key:Union[Literal['query', 'context'], None], aggregation:GeneratorAggregations_t='token',
            **kwargs
        ) -> GeneratorAttribution_t:
        """Generates LIME feature attribution values for the chosen aggregation method.

        Args:
            key (str):              Explanation key. Can either be `'query'` or `'context'`.
                                    If `None` returns a dictionary of both.
            aggregation (str):      Aggregation method for probabilities (default: `'token'`).

        Returns:
            A dictionary containing the following two keys (if `key` is specified) or one of the following:
            - `'query'`: a `numpy.ndarray` containing the LIME values for the query
            - `'context'`: a `numpy.ndarray` containing the LIME values for the contexts.
        """
        raise NotImplementedError()

    def save_values(self, path:Optional[str]=None, *,
            aggregations:Optional[List[GeneratorAggregations_t]]=None,
        ) -> Union[Dict[str, Any], None]:
        """Saves the explanation data to a file.

        Args:
            path (str):               The path where the values should be saved.
            aggregations (List[str]): List of aggregations to save.
                                      If unspecified, saves all aggregations.

        Returns:
            If `path` is not specified, returns the saved data instead.
        """
        data_to_save = {
            'model_name_or_path': self.model_name_or_path,
            'qry_tokens': self.qry_tokens,
            'gen_tokens': self.gen_tokens,
            'shap_qry_precise': self.qry_precise,
            'shap_ctx_precise': self.ctx_precise
        }

        if aggregations is None:
            aggregations = list(AGGREGATIONS)

        for aggregation in aggregations:
            data_to_save['shapley_values_' + aggregation] = self.shap(None, aggregation)
            data_to_save['lime_' + aggregation] = self.lime(None, aggregation)

        if path is None: return data_to_save

        with open(path, 'wb') as f:
            pickle.dump(data_to_save, f)


class GeneratorExplanation(GeneratorExplanationBase):
    _qry_tokens:List[str]
    _gen_tokens:List[str]
    _qry_precise:bool
    _ctx_precise:bool
    _shapley_attributions:Dict[str, Optional[Dict[Literal['query', 'context'], NDArray[np.float32]]]]
    _lime_attributions:Dict[str, Optional[Dict[Literal['query', 'context'], NDArray[np.float32]]]]
    _model_name_or_path:str
    _tokenizer:transformers.PreTrainedTokenizer

    @classmethod
    def load(cls, saved_data:Union[str, Dict, Iterable], *,
        model_name_or_path:Optional[str]=None,
        tokenizer:Optional[transformers.PreTrainedTokenizer]=None
    ) -> Union['GeneratorExplanation', List['GeneratorExplanation'], Dict[str, 'GeneratorExplanation']]:
        """Loads the GeneratorExplanation from a file path or dictionary.

        Args:
            saved_data (str | dict | list):     Path to the saved pickle file, a directory, or the dictionary itself.
            model_name_or_path (str):           An optional way to specify the model name.
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
                    model_name_or_path=model_name_or_path,
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
                            model_name_or_path=model_name_or_path,
                            tokenizer=tokenizer
                        )
                    )


                    # avoid multiple instances of the tokenizer:
                    if tokenizer is None: tokenizer = result[-1].tokenizer

                except Exception as e:
                    print(f'WARNING: Could not load "{item}": {e}')

            return result

        else: raise ValueError("`saved_data` must be a path, an iterable, or a dictionary")

        result = cls()
        result._qry_tokens = data['qry_tokens']
        result._gen_tokens = data['gen_tokens']
        result._qry_precise = data['shap_qry_precise']
        result._ctx_precise = data['shap_ctx_precise']

        result._shapley_attributions = {}
        for aggregation in ['token', 'sequence', 'bow', 'nucleus']:
            result._shapley_attributions[aggregation] = data.get('shapley_values_' + aggregation)

        result._lime_attributions = {}
        for aggregation in ['token', 'sequence', 'bow', 'nucleus']:
            result._lime_attributions[aggregation] = data.get('lime_' + aggregation)

        if model_name_or_path is None: result._model_name_or_path = data['model_name_or_path'] 
        else: result._model_name_or_path = model_name_or_path

        if tokenizer is None: result._tokenizer = transformers.AutoTokenizer.from_pretrained(result._model_name_or_path)
        else: result._tokenizer = tokenizer

        return result

    @property
    def qry_tokens(self) -> List[str]:
        """The list of query tokens."""
        return self._qry_tokens

    @property
    def gen_tokens(self) -> List[str]:
        """The list of generated tokens."""
        return self._gen_tokens[self.focus]

    @property
    def qry_precise(self) -> bool:
        """`True` if the Shapley feature attribution values for the query are not approximated."""
        return self._qry_precise

    @property
    def ctx_precise(self) -> bool:
        """`True` if the Shapley feature attribution values for the context are not approximated."""
        return self._ctx_precise

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        """The tokenizer used by the generator model."""
        return self._tokenizer

    @property
    def model_name_or_path(self) -> str:
        """The huggingface string identifier of the generator model."""
        return self._model_name_or_path

    def shap(self, key:Union[Literal['query', 'context'], None], aggregation:GeneratorAggregations_t='token',
            **kwargs
        ) -> GeneratorAttribution_t:
        """Generates Shapley feature attribution values for the chosen aggregation method.

        Args:
            key (str):          Explanation key. Can either be `'query'` or `'context'`.
                                If `None` returns a dictionary of both.
            aggregation (str):  Aggregation method for probabilities (default: `'token'`).

        Returns:
            A dicionary containing the following two keys (if `key` is specified) or one of the following:
            - `'query'`: a `numpy.ndarray` containing the Shapley values for the query
            - `'context'`: a `numpy.ndarray` containing the Shapley values for the contexts.
        """
        if kwargs:
            print(f'WARNING: `shap(...)` ignores keyword arguments {list(kwargs.keys())} when loaded from a pickle.')

        if aggregation != 'token' and hasattr(self, '_focus'):
            print(f'WARNING: `focus` is ignored for aggregation "{aggregation}"; only "token" aggregation supports it.')

        if key is None:
            result = self._shapley_attributions[aggregation]
            if aggregation != 'token' or result is None: return result
            return {k: (v[:, self.focus] if v is not None else None) for k, v in result.items()}

        result = self._shapley_attributions[aggregation][key]
        if aggregation != 'token' or result is None: return result
        return result[:, self.focus]

    def lime(self, key:Union[Literal['query', 'context'], None], aggregation:GeneratorAggregations_t='token',
            **kwargs
        ) -> GeneratorAttribution_t:
        """Generates LIME feature attribution values for the chosen aggregation method.

        Args:
            key (str):              Explanation key. Can either be `'query'` or `'context'`.
                                    If `None` returns a dictionary of both.
            aggregation (str):      Aggregation method for probabilities (default: `'token'`).

        Returns:
            A dictionary containing the following two keys (if `key` is specified) or one of the following:
            - `'query'`: a `numpy.ndarray` containing the LIME values for the query
            - `'context'`: a `numpy.ndarray` containing the LIME values for the contexts.
        """
        if kwargs:
            print(f'WARNING: `lime(...)` ignores keyword arguments {list(kwargs.keys())} when loaded from a pickle.')

        if aggregation != 'token' and hasattr(self, '_focus'):
            print(f'WARNING: `focus` is ignored for aggregation "{aggregation}"; only "token" aggregation supports it.')

        if key is None:
            result = self._lime_attributions[aggregation]
            if aggregation != 'token' or result is None: return result
            return {k: (v[:, self.focus] if v is not None else None) for k, v in result.items()}

        result = self._lime_attributions[aggregation][key]
        if aggregation != 'token' or result is None: return result
        return result[:, self.focus]


#=======================================================================#
# Generic Model Class:                                                  #
#=======================================================================#

class ExplainableAutoModelForGeneration(GeneratorExplanationBase, metaclass=ABCMeta):
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path:str, *args, **kwargs) -> 'ExplainableAutoModelForGeneration':
        """ Instantiates an ExplainableAutoModelForGeneration from a pre-trained model configuration.

        Args:
            pretrained_model_name_or_path (str):
                Is either ...
                    - ... a string with the `shortcut name` of a pre-trained model configuration
                      to load from cache or download, e.g.: ``bert-base-uncased``.
                    - ... a string with the `identifier name` of a pre-trained model configuration
                      that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                    - ... a path to a `directory` containing a configuration file saved using the
                      :func:`~transformers.PretrainedConfig.save_pretrained` method, e.g.: ``./my_model_directory/``.
                    - ... a path or url to a saved configuration JSON `file`, e.g.: ``./my_model_directory/configuration.json``.
        """
        # find model type based on model name:
        T = get_model_type(pretrained_model_name_or_path)

        # make sure T is derived from PreTrainedModel:
        assert issubclass(T, transformers.PreTrainedModel)

        # generic class definition:
        class _ExplainableAutoModelForGeneration(T, ExplainableAutoModelForGeneration):
            def __init__(self, config, *inputs, **kwargs):
                super().__init__(config, *inputs, **kwargs)
                self._tokenizer:transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(config.name_or_path)
                self._tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self._explain:bool = False
                self._exp_logits:List[torch.Tensor] = []
                self._exp_logits_buffer:List[torch.Tensor] = []
                self._gen_logits:torch.Tensor = torch.empty((0,), dtype=torch.float32)
                self._gen_logits_buffer:List[torch.Tensor] = []
                self._gen_output:Union[torch.Tensor, None] = None
                self._shap_cache:Union[Dict[Literal['query', 'context'], Optional[Dict[str, Any]]], None] = None

            #===============================================================#
            # Properties:                                                   #
            #===============================================================#
            #>> Pairs of properties named `gen_[name]_probs` and
            #>> `cmp_[name]_probs` will be automatically used by the
            #>> `get_shapley_values(...)` method!

            # Focusing functionailty:
            #---------------------------------------------------------------#
            @property
            def focus(self) -> slice:
                """The focus of the explanation, if set."""
                # generate(...) needs to be called first:
                if self._gen_output is None:
                    raise AttributeError('`generate(...)` needs to be called at least once before accessing `focus`!')

                # return the focus:
                if hasattr(self, '_focus'): return self._focus
                else: return slice(0, self._gen_output.shape[1])
            
            @focus.setter
            def focus(self, value:Union[Tuple[int, int], None]) -> None:
                """Sets the focus of the explanation."""
                # generate(...) needs to be called first:
                if (value is not None) and (len(self._gen_logits) == 0):
                    raise AttributeError('`generate(...)` needs to be called at least once before accessing `focus`!')

                # set the focus:
                if value is None:
                    # if `value` is `None`, delete the private `_focus` attribute,
                    # setting the focus to the whole sequence:
                    if hasattr(self, '_focus'): del self._focus

                elif isinstance(value, tuple):
                    # if `value` is a tuple, interpret it as a slice:
                    self._focus = slice(*value)

                else: raise TypeError(
                    f'`focus` must be a tuple of two integers or `None`, but got `{type(value)}: {value}`'
                )


            # GeneratorExplanationBase properties:
            #---------------------------------------------------------------#
            @property
            def qry_tokens(self) -> List[str]:
                """The list of query tokens."""
                # generate(...) needs to be called first:
                if len(self._gen_logits) == 0:
                    raise AttributeError('`generate(...)` needs to be called at least once before accessing `gen_tokens`!')

                # return query tokens:
                return self._qry_tokens

            @property
            def gen_tokens(self) -> List[str]:
                """The list of generated tokens."""
                # generate(...) needs to be called first:
                if self._gen_output is None:
                    raise AttributeError('`generate(...)` needs to be called at least once before accessing `gen_tokens`!')

                # get focus:
                focus = self.focus

                # get generated token ids:
                output_ids = self._gen_output[0, focus]
                
                # return generated tokens:
                return self.tokenizer.convert_ids_to_tokens(output_ids)

            @property
            def qry_precise(self) -> bool:
                """`True` if the Shapley feature attribution values for the query are not approximated."""
                # generate(...) and get_shapley_values(...) need to be called first:
                if self._shap_cache is None:
                    raise AttributeError(
                        '`generate(...)`, `compare(...)`, and `get_shapley_values(...)` ' +
                        'need to be called at least once before accessing `is_precise`!'
                    )

                if self._shap_cache['query'] is None:
                    return False

                # return probability of each token in the original generation:
                return self._shap_cache['query']['precise']

            @property
            def ctx_precise(self) -> bool:
                """`True` if the Shapley feature attribution values for the context are not approximated."""
                # generate(...) and get_shapley_values(...) need to be called first:
                if self._shap_cache is None:
                    raise AttributeError(
                        '`generate(...)`, `compare(...)`, and `get_shapley_values(...)` ' +
                        'need to be called at least once before accessing `is_precise`!'
                    )

                if self._shap_cache['context'] is None:
                    return False

                # return probability of each token in the original generation:
                return self._shap_cache['context']['precise']

            @property
            def tokenizer(self) -> transformers.PreTrainedTokenizer:
                """The tokenizer used by the generator model."""
                return self._tokenizer

            @property
            def model_name_or_path(self) -> str:
                """The huggingface string identifier of the generator model."""
                return self.config.name_or_path


            # Explanation properties:
            #---------------------------------------------------------------#
            @property
            def gen_token_probs(self) -> NDArray[np.float32]:
                """Probability of each token in the original generation."""
                # generate(...) needs to be called first:
                if self._gen_output is None:
                    raise AttributeError('`generate(...)` needs to be called at least once before accessing `gen_token_probs`!')

                # get focus:
                focus = self.focus

                # get generated token ids:
                output_ids = self._gen_output[:, focus]

                # get probabilities:
                probs = logits2probs(self._gen_logits[:, focus, :], normalization='softmax')

                # return probability of each token in the original generation:
                return np.array([
                    [float(probs[i, j, id]) for j, id  in enumerate(seq)]
                    for i, seq in enumerate(output_ids)
                ])

            @property
            def cmp_token_probs(self) -> List[NDArray[np.float32]]:
                """Probability of each token in the original generation given the compared input."""
                # compare(...) needs to be called first:
                if len(self._exp_logits) == 0 or self._gen_output is None:
                    raise AttributeError(
                        '`generate(...)` and `compare(...)` need to be called ' +
                        'at least once before accessing `cmp_token_probs`!'
                    )

                # get focus:
                focus = self.focus

                # get generated token ids:
                output_ids = self._gen_output[:, focus]

                # get probabilities:
                probs = [logits2probs(t[:, focus, :], normalization='softmax') for t in self._exp_logits]

                # return probability of each token in the original generation:
                return [np.array([
                    [float(t[i, j, id]) for j, id  in enumerate(seq)]
                    for i, seq in enumerate(output_ids)
                ]) for t in probs]


            @property
            def gen_sequence_prob(self) -> NDArray[np.float32]:
                """Total probability of generating the original sequence."""
                # generate(...) needs to be called first:
                if self._gen_output is None:
                    raise AttributeError('`generate(...)` needs to be called at least once before accessing `gen_sequence_prob`!')

                # get focus:
                focus = self.focus

                # get generated token ids:
                output_ids = self._gen_output[:, focus]

                # get probabilities:
                probs = logits2probs(self._gen_logits[:, focus, :], normalization='relu')
                #probs = logits2probs(self._gen_logits[:, focus, :], normalization='offset')

                # return the multiplied probability of each token in the original generation:
                return np.prod([
                    [float(probs[i, j, id]) for j, id  in enumerate(seq)]
                    for i, seq in enumerate(output_ids)
                ], axis=-1)

            @property
            def cmp_sequence_probs(self) -> List[NDArray[np.float32]]:
                """Total probability of generating the original sequence given the compared input."""
                # compare(...) needs to be called first:
                if len(self._exp_logits) == 0 or self._gen_output is None:
                    raise AttributeError(
                        '`generate(...)` and `compare(...)` need to be called ' +
                        'at least once before accessing `cmp_sequence_probs`!'
                    )

                # get focus:
                focus = self.focus

                # get generated token ids:
                output_ids = self._gen_output[:, focus]

                # get probabilities:
                probs = [logits2probs(t[:, focus, :], normalization='relu') for t in self._exp_logits]
                #probs = [logits2probs(t[:, focus, :], normalization='offset') for t in self._exp_logits]

                # return the multiplied probability of each token in the original generation:
                return [np.prod([
                    [float(t[i, j, id]) for j, id  in enumerate(seq)]
                    for i, seq in enumerate(output_ids)
                ], axis=-1) for t in probs]


            @property
            def gen_bow_probs(self) -> NDArray[np.float32]:
                """Average probability of each token in the vocabulary of being generated given the original input."""
                # generate(...) needs to be called first:
                if len(self._gen_logits) == 0:
                    raise AttributeError('`generate(...)` needs to be called at least once before accessing `gen_bow_probs`!')

                # get focus:
                focus = self.focus

                # get probabilities:
                probs = logits2probs(self._gen_logits[:, focus, :], normalization='softmax')

                # return accumulated probability of each token in the vocabulary:
                return probs.mean(dim=1).float().numpy()
            
            @property
            def cmp_bow_probs(self) -> List[NDArray[np.float32]]:
                """Average probability of each token in the vocabulary of being generated given the compared input."""
                # compare(...) needs to be called first:
                if len(self._exp_logits) == 0:
                    raise AttributeError(
                        '`generate(...)` and `compare(...)` need to be called ' +
                        'at least once before accessing `cmp_bow_probs`!'
                    )

                # get focus:
                focus = self.focus

                # get probabilities:
                probs = [logits2probs(t[:, focus, :], normalization='softmax') for t in self._exp_logits]

                # return accumulated probability of each token in the vocabulary:
                return [t.mean(dim=1).float().numpy() for t in probs]
            

            def gen_nucleus_probs(self, p:float=0.9) -> NDArray[np.float32]:
                """Average probability of each token in the vocabulary of being generated given the original input."""
                # generate(...) needs to be called first:
                if len(self._gen_logits) == 0:
                    raise AttributeError('`generate(...)` needs to be called at least once before accessing `gen_nucleus_probs`!')

                # get focus:
                focus = self.focus

                # get probabilities:
                probs = logits2probs(self._gen_logits[:, focus, :], normalization='softmax')

                # return accumulated probability of each token in the vocabulary:
                return _nucleus_sampling(probs.float(),p=p).mean(dim=1).numpy()

            def cmp_nucleus_probs(self, p:float=0.9) -> List[NDArray[np.float32]]:
                """Average probability of each token in the vocabulary of being generated given the compared input."""
                # compare(...) needs to be called first:
                if len(self._exp_logits) == 0:
                    raise AttributeError(
                        '`generate(...)` and `compare(...)` need to be called ' +
                        'at least once before accessing `cmp_nucleus_probs`!'
                    )

                # get focus:
                focus = self.focus

                # get probabilities:
                probs = [logits2probs(t[:, focus, :], normalization='softmax') for t in self._exp_logits]

                # return accumulated probability of each token in the vocabulary:
                return [_nucleus_sampling(t.float(),p=p).mean(dim=1).numpy() for t in probs]

            #===============================================================#
            # Methods:                                                      #
            #===============================================================#

            def forward(self, *args, **kwargs):
                # get token probabilities:
                outputs = super().forward(*args, **kwargs)

                # Fix for Gemma 3 which returns 4D logits [Batch, 1, Seq, Vocab]
                if hasattr(outputs, 'logits') and outputs.logits.dim() == 4 and outputs.logits.shape[1] == 1:
                    outputs.logits = outputs.logits.squeeze(1)

                # save token probabilities:
                if self._explain: self._exp_logits_buffer.append(outputs.logits[:,-1:,:].detach().cpu())
                else:             self._gen_logits_buffer.append(outputs.logits[:,-1:,:].detach().cpu())

                # return token probabilities:
                return outputs

            def generate(self, inputs:List[str], **kwargs) -> List[str]:
                """Generates continuations of the passed input prompt(s).

                Args:
                    inputs:             The string(s) used as a prompt for the generation.
                    generation_config:  The generation configuration to be used as base parametrization for the generation call.
                    stopping_criteria:  Custom stopping criteria that complements the default stopping criteria built from arguments and ageneration config.
                    kwargs:             Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs.

                Returns:
                    List of generated strings.
                """
                # tokenize inputs:
                inputs = self.tokenizer(inputs, truncation=True, return_tensors='pt').to(self.device)

                # deactivate explanation mode:
                self._explain    = False

                # reset focus:
                self.focus       = None

                # reset token probabilities:
                self._gen_logits_buffer = []

                # generate:
                self._gen_output = super().generate(**inputs, **kwargs).sequences[:, inputs.input_ids.shape[-1]:]

                # finalize probabilities:
                self._gen_logits = torch.concatenate(self._gen_logits_buffer, dim=1)
                self._exp_logits = []

                # clear buffer:
                self._gen_logits_buffer.clear()

                # return generated text:
                return self.tokenizer.batch_decode(self._gen_output)

            def compare(self, inputs:List[str], outputs:Optional[Union[List[str], torch.LongTensor, Literal['last']]]=None, batch_size:int=1, **kwargs) -> torch.LongTensor:
                """Calculates the probability `p(outputs) = p(t_0) * p(t_1|t_0) * ... * p(t_n|t_0...t_(n-1))` of
                a specific output `outputs = [t_0, t_1, ..., t_n]` to happen given an input prompt `inputs`.

                Args:
                    inputs:     List of input propmts. If `outputs` is specified, `compare(...)` calculates the
                                probability `p(outputs) = p(t_0) * p(t_1|t_0) * ... * p(t_n|t_0...t_(n-1))` for each
                                token in `outputs = [t_0, t_1, ..., t_n]` given ``. Otherwise, it calculates the
                                unconditional probability (similar to `generate(...)`).
                    outputs:    List of tokens `t_i` or (strings containing those) for which to compute the probability.
                                If set to `'last'`, the last generated sequence will be used (optional).
                    batch_size: Batch size. Ignored if `len(inputs) > 1` or `outputs` not specified (optional).

                Returns:
                    Tensor of generated token ids .
                """

                if batch_size < 1:
                    raise ValueError(f'Parameter batch_size must be a positive integer but got {batch_size:d}.')

                if outputs is None:
                    if batch_size > 1:
                        print('WARNING: when outputs is not specified the parameter batch_size is ignored.')

                    return self.__compare_unconditional(inputs=inputs, **kwargs)
                
                else: return self.__compare_conditional(
                    inputs=inputs,
                    outputs=self._gen_output if outputs == 'last' else outputs,
                    batch_size=batch_size,
                    **kwargs
                )

            def __compare_unconditional(self, inputs:List[str], **kwargs) -> torch.LongTensor:
                # tokenize inputs:
                model_inputs = self.tokenizer(inputs, truncation=True, padding=True, return_tensors='pt')

                # deactivate explanation mode:
                self._explain = True

                # reset token probabilities:
                self._exp_logits_buffer = []

                # generate:
                output = super().generate(**model_inputs, **kwargs).sequences[:, model_inputs.input_ids.shape[-1]:]

                # finalize probabilities:
                self._exp_logits.append(torch.concatenate(self._exp_logits_buffer, dim=1))

                # clear buffer:
                self._exp_logits_buffer.clear()

                # split batch in elements if multiple inputs:
                if len(inputs) > 1:
                    self._exp_logits.extend([t.unsqueeze(0) for t in self._exp_logits.pop(-1)])

                # return generated tokens:
                return output

            def __compare_conditional(self, inputs:List[str], outputs:Union[List[str], torch.Tensor], batch_size:int=1, **kwargs) -> torch.LongTensor:
                # get batch size:
                single_input  = len(inputs) == 1
                single_output = len(outputs) == 1
                assert (len(inputs) == len(outputs)) or single_output
                if not single_input: batch_size = len(inputs)

                # activate explanation mode:
                self._explain = True

                # reset token probabilities:
                self._exp_logits_buffer = []

                # convert string to Iterable of tokens:
                if isinstance(outputs[0], str):
                    outputs = self.tokenizer(outputs, add_special_tokens=False, return_attention_mask=False, return_tensors='pt').input_ids
                outputs = outputs.to(self.device)

                # tokenize input:
                model_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
                input_ids = model_inputs.input_ids.to(self.device)
                attention_mask = model_inputs.attention_mask.to(self.device)

                # batch processing in case of single input:
                if single_input:
                    if single_output:
                        input_ids, outputs = _to_batch(
                            input_ids, outputs, self.tokenizer.pad_token_id, batch_size
                        )
                        attention_mask, _ = _to_batch(
                            attention_mask, torch.ones((1, batch_size)), 0, batch_size
                        )

                    else: raise NotImplementedError()

                # prepare model_kwargs:
                input_ids, _, model_kwargs = self._prepare_model_inputs(input_ids, self.tokenizer.bos_token_id, kwargs)

                # Try to get initial cache position with signature inspection
                if hasattr(self, '_get_initial_cache_position'):
                    sig = inspect.signature(self._get_initial_cache_position)
                    if 'seq_length' in sig.parameters:
                        # Newer API
                        model_kwargs = self._get_initial_cache_position(
                            seq_length=input_ids.shape[1],
                            device=input_ids.device,
                            model_kwargs=model_kwargs,
                        )
                    else:
                        # Older API
                        model_kwargs = self._get_initial_cache_position(
                            input_ids=input_ids, 
                            model_kwargs=model_kwargs
                        )

                # check prepare_inputs_for_generation signature once for both calls:
                _pifg_sig = inspect.signature(self.prepare_inputs_for_generation).parameters
                _supports_next_seq_len = 'next_sequence_length' in _pifg_sig
                _supports_first_iter   = 'is_first_iteration'   in _pifg_sig

                with torch.no_grad():

                    # calculate p(t_0):
                    _prefill_kwargs = {'is_first_iteration': True} if _supports_first_iter else {}
                    model_inputs = self.prepare_inputs_for_generation(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **_prefill_kwargs,
                        **model_kwargs
                    )
                    model_inputs.setdefault('use_cache', True)
                    model_outputs = self.forward(**model_inputs, return_dict=True)
                    model_kwargs = self._update_model_kwargs_for_generation(model_outputs, model_kwargs)
                    torch.cuda.empty_cache()

                    # update inputs:
                    if single_input:    nxt = outputs[:,:batch_size].to(input_ids)
                    elif single_output: nxt = torch.full((batch_size, 1), outputs[0,0], device=input_ids.device, dtype=input_ids.dtype)
                    else:               nxt = torch.unsqueeze(outputs[:,0], dim=1).to(input_ids)

                    input_ids = torch.concatenate((input_ids, nxt), dim=-1)
                    attention_mask = torch.concatenate((attention_mask, torch.full((batch_size, nxt.shape[1]), 1, device=input_ids.device, dtype=input_ids.dtype)), dim=-1)

                    # p(outputs) = p(t_0) * p(t_1|t_0) * ... * p(t_1|t_0...t_(j-1)):
                    step = batch_size if single_input else 1
                    for i in tqdm.tqdm(range(1, outputs.shape[1], step),total=int(outputs.shape[1]/step), desc='Calculating probabilities'):
                        _decode_kwargs = {'next_sequence_length': nxt.shape[1]} if _supports_next_seq_len else {}
                        model_inputs = self.prepare_inputs_for_generation(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **_decode_kwargs,
                            **model_kwargs
                        )
                        model_outputs = self.forward(**model_inputs, return_dict=True)
                        model_kwargs = self._update_model_kwargs_for_generation(model_outputs, model_kwargs, num_new_tokens=nxt.shape[1])
                        torch.cuda.empty_cache()

                        # update inputs:
                        if single_input:    nxt = outputs[:,(i*batch_size):((i+1)*batch_size)].to(input_ids)
                        elif single_output: nxt = torch.full((batch_size, 1), outputs[0,i], device=input_ids.device, dtype=input_ids.dtype)
                        else:               nxt = torch.unsqueeze(outputs[:,i], dim=1).to(input_ids)

                        input_ids = torch.concatenate((input_ids, nxt), dim=-1)
                        attention_mask = torch.concatenate((attention_mask, torch.full((batch_size, nxt.shape[1]), 1, device=input_ids.device, dtype=input_ids.dtype)), dim=-1)

                # finalize probabilities:
                if single_input and single_output:
                    self._exp_logits.append(torch.concatenate(self._exp_logits_buffer, dim=0).transpose(0,1))

                else: self._exp_logits.append(torch.concatenate(self._exp_logits_buffer, dim=1))

                # clear buffer:
                self._exp_logits_buffer.clear()

                # split batch in elements if multiple inputs for the same output:
                if not single_input and single_output:
                    self._exp_logits.extend([t.unsqueeze(0) for t in self._exp_logits.pop(-1)])

                # return generated tokens:
                return torch.argmax(self._exp_logits[-1], dim=-1)


            def explain_generate(self, query:str, contexts:List[str], *,
                    batch_size:int=32,
                    max_samples_query:Union[int, Literal['inf', 'auto']]='auto',
                    max_samples_context:Union[int, Literal['inf', 'auto']]='auto',
                    max_document_size:Optional[int]=None,
                    conditional:bool=True,
                    complementary:Union[bool,Literal['no_mc']]=True,
                    system:Optional[str]=None,
                    **kwargs
                ) -> List[Dict[Literal['role','content'],str]]:
                """Generates continuations of the passed input prompt(s) as well as perturbations for all retrieved documents.

                Args:
                    query (str):                The user's query.
                    contexts (list):            A list of strings representing the retrieved documents.
                    batch_size (int):           The batch size for generating perturbations (default: `32`).
                    max_samples_query (int):    Maximum number of samples used for computing SHAP atribution values for the query.
                                                If `2**len(contexts) <= max_samples`, this automatically computes the precise SHAP values instead of kernel SHAP approximations.
                                                If `inf` is passed, always computes the precise SHAP values.
                                                If `auto` is passed, `max_samples` get's the same value as `batch_size` (default: `auto`).
                    max_samples_context (int):  Maximum number of samples used for computing SHAP atribution values for the context documents.
                                                If `2**len(contexts) <= max_samples`, this automatically computes the precise SHAP values instead of kernel SHAP approximations.
                                                If `inf` is passed, always computes the precise SHAP values.
                                                If `auto` is passed, `max_samples` get's the same value as `batch_size` (default: `auto`).
                    max_document_size (int):    An optional size limit of context documents in characters.
                    conditional (bool):         Whether to compute the compared values conditioned on the original generation (default: `True`).
                    complementary (bool):       If `True` is passed and kernel SHAP approximation is active, samples will be piered complements (default: `True`). 
                    system (str):               An optional system prompt.

                Returns:
                    A list of generated chats.
                """

                # can only deal with up to 64 context documents for indexing reasons:
                if len(contexts) > 64: raise ValueError(
                    f'`explain_generate(...)` accepts only up to 64 context documents but received {len(contexts):d}.'
                )

                # update / override kwargs:
                kwargs['return_dict_in_generate'] = True
                kwargs['output_scores'] = True

                # generate output:
                complete_rag_prompt = create_rag_prompt(query, contexts, system=system, max_document_size=max_document_size)
                output = self.generate(
                    [self.tokenizer.apply_chat_template(complete_rag_prompt, tokenize=False)],
                    **kwargs
                )

                # generate perturbed prompts:
                if   max_samples_query == 'auto': max_samples_query = batch_size//2
                elif max_samples_query == 'inf':  max_samples_query = MAXINT

                if   max_samples_context == 'auto': max_samples_context = batch_size//2
                elif max_samples_context == 'inf':  max_samples_context = MAXINT

                if complementary != False:
                    if (max_samples_query % 2)   > 0: max_samples_query -= 1
                    if (max_samples_context % 2) > 0: max_samples_context -= 1

                self._qry_tokens = query.split()
                self._qry_tokens[1:] = [' ' + t for t in self._qry_tokens[1:]]

                self._shap_cache, perturbed_prompts = {}, {}
                self._shap_cache['query'], perturbed_prompts['query'] = self.__generate_prompts(
                    self._qry_tokens,                                                        # permute the query
                    lambda items:create_rag_prompt(''.join(items), contexts,                 # build a prompt for each permutation
                                                   system=system, max_document_size=max_document_size), 
                    max_samples_query,
                    batch_size,
                    complementary
                )
                self._shap_cache['context'], perturbed_prompts['context'] = self.__generate_prompts(
                    contexts,                                                    # permute the contexts
                    lambda items:create_rag_prompt(query, items,                 # build a prompt for each permutation
                                                   system=system, max_document_size=max_document_size),
                    max_samples_context,
                    batch_size,
                    complementary
                )

                # combine prompt lists comparison output:
                perturbed_prompts_combined = perturbed_prompts['query'][:-1] + perturbed_prompts['context']

                cache  = self._shap_cache['query']
                if cache is not None:
                    offset = len(perturbed_prompts['context']) - 1
                    if cache['precise']: cache['indices'][:,-1] += offset
                    else: cache['indices'][-1] += offset

                cache = self._shap_cache['context']
                if cache is not None:
                    offset = len(perturbed_prompts['query']) - 1
                    if cache['precise']: cache['indices'] += offset
                    else: cache['indices'] += offset

                # generate comparison output:
                num_batches = int(np.ceil(len(perturbed_prompts_combined[:-1]) / batch_size))
                for i in range(num_batches):
                    # print batch number:
                    if num_batches > 1: print(f'Batch {i+1:d} of {num_batches:d}:')

                    # get prompts of this batch:
                    prompts_batch = perturbed_prompts_combined[:-1][i * batch_size:(i+1) * batch_size]

                    # generate probabilities:
                    self.compare(
                        [self.tokenizer.apply_chat_template(prmpt, tokenize=False) for prmpt in prompts_batch],
                        'last' if conditional else None
                    )

                    # print empty line:
                    if num_batches > 1: print()

                return complete_rag_prompt + decode_chat_template(output[0], self.config.name_or_path) 

            def __generate_prompts(self, items, func, max_samples, batch_size, complementary):
                # calculate number of samples needed for precise calculation:
                n = 2 ** len(items)

                # compute minimum number of necessary samples (at least 0.1% of the max):
                #min_samples = max(n//1000, len(items)*10)
                min_samples = len(items)*10
                min_batches = max(1, min_samples//batch_size)
                min_samples = min_batches*batch_size

                # generate prompts:
                if max_samples == 0:
                    # do not generate prompts:
                    perturbed_prompts = [func(items)]
                    cache = None

                elif max_samples >= n or min_samples >= n:
                    max_samples = max(max_samples, min_samples)

                    # generate prompts for perturbed inputs (precise SHAP values):
                    permutations, new_items, perturbed_prompts = generate_permutations(items, func)
                    cache = {'precise': True, 'indices': permutations, 'new_docs': new_items}

                elif max_samples < n:
                    max_samples = max(max_samples, min_samples)

                    # sample prompts for kernel SHAP:
                    perturbations, perturbed_prompts = sample_perturbations(items, func, max_samples, complementary=(complementary!=False))
                    cache = {'precise': False, 'complementary': (complementary==True), 'indices': np.arange(perturbations.shape[0]), 'sets': perturbations}

                else: raise ValueError(f'Unknown value for parameter `max_samples`: {max_samples}')

                return cache, perturbed_prompts


            def shap(self, key:Union[Literal['query', 'context'], None], aggregation:GeneratorAggregations_t='token', *,
                num_samples:int=100,
                sample_size:int=10,
                **kwargs
            ) -> GeneratorAttribution_t:
                """Generates Shapley feature attribution values for the chosen aggregation method.

                Args:
                    key (str):          Explanation key. Can either be `'query'` or `'context'`.
                                        If `None` returns a dictionary of both.
                    aggregation (str):  Aggregation method for probabilities (default: `'token'`).
                    num_samples (int):  Number of samples for Monte-Carlo approximation.
                                        Ignored in case of precise calculation (default: `100`).
                    sample_size (int):  Size of samples for Monte-Carlo approximation.
                                        Ignored in case of precise calculation (default: `10`).

                Returns:
                    A dicionary containing the following two keys (if `key` is specified) or one of the following:
                    - `'query'`: a `numpy.ndarray` containing the Shapley values for the query
                    - `'context'`: a `numpy.ndarray` containing the Shapley values for the contexts.
                """

                # Get the correct probabilities based on the `aggreagtion` parameter:
                if aggregation == 'nucleus':
                    # Flatten each token probability array from compared documents:
                    probs = [p.flatten() for p in self.cmp_nucleus_probs(**kwargs)]

                    # Add the generated token probabilities as the final "player" in the SHAP context:
                    probs.append(self.gen_nucleus_probs(**kwargs).flatten())


                elif aggregation == 'sequence':
                    # Convert the scalar probability from compared documents to a ndarray:
                    probs = [np.array(p) for p in self.cmp_sequence_probs]

                    # Add the generated token probabilities as the final "player" in the SHAP context:
                    probs.append(np.array(self.gen_sequence_prob))


                elif hasattr(self, f'gen_{aggregation}_probs') and hasattr(self, f'cmp_{aggregation}_probs'):
                    # Flatten each token probability array from compared documents:
                    probs = [p.flatten() for p in getattr(self, f'cmp_{aggregation}_probs')]

                    # Add the generated token probabilities as the final "player" in the SHAP context:
                    probs.append(getattr(self, f'gen_{aggregation}_probs').flatten())


                else: raise ValueError(f'Unknown value for parameter `aggregation`: "{aggregation}"')

                # Call actual method:
                assert self._shap_cache is not None, '`explain_generate(...)` must be called before `shap(...)`!'
                result:Dict[Literal['query', 'context'], Optional[NDArray[np.float32]]] = {'query': None, 'context': None} if key is None else {key: None}
                for k in result:
                    if self._shap_cache[k] is None: result[k] = None
                    elif self._shap_cache[k]['precise']: result[k] = self._get_shapley_attributions_precise(probs, **self._shap_cache[k])
                    else: result[k] = self._get_shapley_attributions_monte_carlo(probs, num_samples=num_samples, sample_size=sample_size, **self._shap_cache[k])

                return result if key is None else result[key]

            def _get_shapley_attributions_precise(self, probs:List[NDArray[np.float32]], indices:NDArray[np.int_], new_docs:NDArray[np.int_], precise:bool) -> NDArray[np.float32]:
                assert precise is True, 'Precise SHAP values can only be calculated for precise values!'

                # Get the shape of the permutations matrix: (num_permutations, num_sets)
                num_permutations, num_sets = indices.shape

                # Get the number of documents:
                num_docs = num_sets - 1

                # Initialize array to store marginal contributions for each permutation step
                p_marginal = np.empty((num_permutations, num_docs) + probs[0].shape, dtype=probs[0].dtype)

                # For each permutation, calculate the marginal contributions
                for i in range(num_permutations):
                    for j in range(0, num_docs):
                        # Difference in output probability when adding the j-th document
                        prev = probs[indices[i, j]]
                        curr = probs[indices[i, j + 1]]
                        p_marginal[i, j] = curr - prev

                # Initialize SHAP value container: one entry per document
                p_shap = np.empty((num_docs,) + probs[0].shape, dtype=probs[0].dtype)

                # For each document, aggregate all matching marginal contributions
                for j in range(num_docs):
                    # Mean over all marginal contributions that map to document j
                    p_shap[j] = p_marginal[new_docs == j].mean(0)

                # Return SHAP values for all but the baseline (first one)
                return p_shap

            def _get_shapley_attributions_monte_carlo(self, probs:List[NDArray[np.float32]], indices:NDArray[np.int_], sets:NDArray[np.bool_], precise:bool, complementary:bool, num_samples:int=100, sample_size:int=10) -> NDArray[np.float32]:
                assert precise is False, 'Monte Carlo SHAP values can only be calculated for approximate values!'

                index_size = len(indices)

                # calculate sample size and population:
                if complementary:
                    size = (sample_size // 2) - 1
                    population = (index_size // 2) - 1    # skip the first and last indices

                else:
                    size = sample_size - 2
                    population = index_size - 2           # skip the first and last indices

                # Adjust sample_size to not exceed the number of available coalitions
                if size > population: size = population

                # Generate Shapley values for `num_samples` samples of `sample_size` documents:
                attributions = []
                for _ in range(num_samples):
                    sample = np.random.choice(population, size=size, replace=False) + 1

                    # add complementary examples if necessary:
                    if complementary:
                        sample = np.concatenate([sample, (index_size-1)-sample])

                    attributions.append(
                        self._get_shapley_attributions_kernel(
                            probs   = probs,
                            indices = np.concatenate([indices[:1], indices[sample], indices[-1:]]),
                            sets    = np.concatenate([sets[:1], sets[sample], sets[-1:]]),
                            precise = precise
                        )
                    )

                # Return the mean of the attributions across all samples:
                return np.mean(attributions, axis=0)
            
            def _get_shapley_attributions_kernel(self, probs:List[NDArray[np.float32]], indices:NDArray[np.int_], sets:NDArray[np.bool_], precise:bool, **kwargs) -> NDArray[np.float32]:
                assert precise is False, 'Kernel SHAP values can only be calculated for approximate values!'

                # fit a ridge regressor using the SHAP kernel:
                def _get_shap_weights(z):
                    l, s = len(z), sum(z)
                    denominator = comb(l, s) * s * (l - s)
                    if denominator == 0: return 1e-10
                    else: return (l-1) / denominator
                lr = Ridge(alpha=0.001, solver='cholesky')  # Fast solver with minimal regularization
                x  = sets[1:-1].astype(float)
                y  = np.stack([probs[i] for i in indices[1:-1]])
                w  = [_get_shap_weights(z) for z in x]
                lr.fit(x, y, w)
                # attributions are estimated SHAP values:
                # Ridge collapses `coef_` to 1-D when there is only a single target column
                # (e.g. aggregation='sequence'), so reshape explicitly using the known
                # dimensions of `x`/`y` to guarantee a (num_docs, num_targets) result.
                attributions = np.asarray(lr.coef_).reshape(y.shape[1], x.shape[1]).T

                # rescale attributions to fit prediction:
                return attributions
            
            
            def lime(self, key:Union[Literal['query', 'context'], None], aggregation:GeneratorAggregations_t='token', *,
                    kernel_width:Optional[float]=None,
                    kernel_fn:Optional[Callable]=None,
                    **kwargs
                ) -> GeneratorAttribution_t:
                """Generates LIME feature attribution values for the chosen aggregation method.

                Args:
                    key (str):              Explanation key. Can either be `'query'` or `'context'`.
                                            If `None` returns a dictionary of both.
                    aggregation (str):      Aggregation method for probabilities (default: `'token'`).
                    kernel_width (float):   Width of the exponential similarity kernel (default: `min(25, k/2)`).
                    kernel_fn (callable):   Similarity kernel taking distances and returning weights.
                                            If `None`, defaults to `sqrt(exp(-d^2 / kernel_width^2))`.

                Returns:
                    A dictionary containing the following two keys (if `key` is specified) or one of the following:
                    - `'query'`: a `numpy.ndarray` containing the LIME values for the query
                    - `'context'`: a `numpy.ndarray` containing the LIME values for the contexts.
                """

                # Get the correct probabilities based on the `aggreagtion` parameter:
                if aggregation == 'nucleus':
                    # Flatten each token probability array from compared documents:
                    probs = [p.flatten() for p in self.cmp_nucleus_probs(**kwargs)]

                    # Add the generated token probabilities as the final "player" in the SHAP context:
                    probs.append(self.gen_nucleus_probs(**kwargs).flatten())


                elif aggregation == 'sequence':
                    # Convert the scalar probability from compared documents to a ndarray:
                    probs = [np.array(p) for p in self.cmp_sequence_probs]

                    # Add the generated token probabilities as the final "player" in the SHAP context:
                    probs.append(np.array(self.gen_sequence_prob))


                elif hasattr(self, f'gen_{aggregation}_probs') and hasattr(self, f'cmp_{aggregation}_probs'):
                    # Flatten each token probability array from compared documents:
                    probs = [p.flatten() for p in getattr(self, f'cmp_{aggregation}_probs')]

                    # Add the generated token probabilities as the final "player" in the SHAP context:
                    probs.append(getattr(self, f'gen_{aggregation}_probs').flatten())


                else: raise ValueError(f'Unknown value for parameter `aggregation`: "{aggregation}"')

                # Call actual method:
                assert self._shap_cache is not None, '`explain_generate(...)` must be called before `lime(...)`!'
                result:Dict[Literal['query', 'context'], Optional[NDArray[np.float32]]] = {'query': None, 'context': None} if key is None else {key: None}
                for k in result:
                    if self._shap_cache[k] is None:
                        result[k] = None
                        continue

                    # Default kernel width:
                    if kernel_width is None:
                        if self._shap_cache[k]['precise']:
                            num_docs = self._shap_cache[k]['new_docs'].shape[1]
                        else:
                            num_docs = self._shap_cache[k]['sets'].shape[1]

                        kernel_width = min(25., num_docs/2.)

                    # Default exponential kernel (see https://github.com/marcotcr/lime/blob/master/lime/lime_text.py):
                    # (`kernel_width` is bound as a default arg so the lambda captures its
                    # current, narrowed value instead of a live reference to the outer variable)
                    if kernel_fn is None:
                        kernel_fn = lambda d, kernel_width=kernel_width: np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

                    if self._shap_cache[k]['precise']: result[k] = self._get_lime_attributions_precise(probs, kernel_fn=kernel_fn, **self._shap_cache[k])
                    else: result[k] = self._get_lime_attributions_approx(probs, kernel_fn=kernel_fn, **self._shap_cache[k])

                return result if key is None else result[key]

            def _get_lime_attributions_precise(self, probs:List[NDArray[np.float32]], indices:NDArray[np.int_], new_docs:NDArray[np.int_], precise:bool, kernel_fn:Callable, **kwargs) -> NDArray[np.float32]:
                assert precise is True

                n_permutations, n_steps = indices.shape
                n_features = n_steps - 1

                # Collect unique coalitions and their global probs indices via permutation paths:
                coalition_map = {}
                for i in range(n_permutations):
                    features_so_far = set()
                    for j in range(n_steps):
                        coalition = frozenset(features_so_far)
                        if coalition not in coalition_map:
                            coalition_map[coalition] = int(indices[i, j])
                        if j < n_features:
                            features_so_far.add(int(new_docs[i, j]))

                # Build binary presence matrix (1 = feature present):
                coalitions = sorted(coalition_map.keys(), key=len)
                x = np.array([[1. if f in c else 0. for f in range(n_features)] for c in coalitions], dtype=float)
                y = np.stack([probs[coalition_map[c]] for c in coalitions])

                # Exclude empty and full coalitions:
                mask = (x.sum(axis=1) > 0) & (x.sum(axis=1) < n_features)
                x, y = x[mask], y[mask]

                # Distance = number of absent features (Hamming distance to full instance):
                distances = n_features - x.sum(axis=1)
                w = kernel_fn(distances)

                lr = Ridge(alpha=0.01, fit_intercept=True, solver='cholesky')
                lr.fit(x, y, sample_weight=w)
                # Ridge collapses `coef_` to 1-D when there is only a single target column
                # (e.g. aggregation='sequence'), so reshape explicitly using the known
                # dimensions of `x`/`y` to guarantee a (num_docs, num_targets) result.
                return np.asarray(lr.coef_).reshape(y.shape[1], x.shape[1]).T

            def _get_lime_attributions_approx(self, probs:List[NDArray[np.float32]], indices:NDArray[np.int_], sets:NDArray[np.bool_], precise:bool, complementary:bool, kernel_fn:Callable, **kwargs) -> NDArray[np.float32]:
                assert precise is False

                n_features = sets.shape[1]

                # Exclude empty (row 0) and full (row -1) coalitions:
                x = sets[1:-1].astype(float)
                y = np.stack([probs[i] for i in indices[1:-1]])

                # Distance = number of absent features (Hamming distance to full instance):
                distances = n_features - x.sum(axis=1)
                w = kernel_fn(distances)

                lr = Ridge(alpha=0.01, fit_intercept=True, solver='cholesky')
                lr.fit(x, y, sample_weight=w)
                # Ridge collapses `coef_` to 1-D when there is only a single target column
                # (e.g. aggregation='sequence'), so reshape explicitly using the known
                # dimensions of `x`/`y` to guarantee a (num_docs, num_targets) result.
                return np.asarray(lr.coef_).reshape(y.shape[1], x.shape[1]).T

            def _extract_top_exp_prob(self, top_k = 200):
                """Extracts the top-k probabilities and their corresponding tokens from the generated tensors.

                Args:
                    top_k (int): The number of top probabilities to extract (default: 50).

                Returns:
                    A tuple containing two tensors:
                    - `permutation_top_vals`: A tensor of shape (n_perm, top_k) containing the top-k scores.
                    - `permutation_top_tokens`: A tensor of shape (n_perm, top_k, 2) containing the step and token indices.
                """
                # ensure that we have at least one tensor to process:
                if len(self._exp_logits) == 0: return None, None

                # get the list of tensors:
                tensor_list = self._exp_logits
                all_top_scores = []           # list of (max_new_tokens, top_k)
                all_top_tokens = []           # list of (max_new_tokens, top_k)

                for scores in tensor_list:    # scores: (max_new_tokens, vocab_size)
                    # torch.topk along the *token* dimension (dim=-1)
                    vals, idx = torch.topk(scores, k=top_k, dim=-1)   # both (max_new_tokens, top_k)

                    all_top_scores.append(vals)   # keep the scores
                    all_top_tokens.append(idx)    # keep the token IDs

                # (optional) stack into big tensors: (n_perm, max_new_tokens, top_k)
                return torch.stack(all_top_scores).squeeze(1), torch.stack(all_top_tokens).squeeze(1)

        return _ExplainableAutoModelForGeneration.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            *args, **kwargs
        )

    # Abstract properties for documentation purposes:
    #---------------------------------------------------------------#
    @property
    @abstractmethod
    def gen_token_probs(self) -> NDArray[np.float32]:
        """Probability of each token in the original generation."""
        raise NotImplementedError('ExplainableAutoModelForGeneration objects must be instantiated using the `from_pretrained` method.')

    @property
    @abstractmethod
    def cmp_token_probs(self) -> List[NDArray[np.float32]]:
        """Probability of each token in the original generation given the compared input."""
        raise NotImplementedError('ExplainableAutoModelForGeneration objects must be instantiated using the `from_pretrained` method.')


    @property
    @abstractmethod
    def gen_sequence_prob(self) -> NDArray[np.float32]:
        """Total probability of generating the original sequence."""
        raise NotImplementedError('ExplainableAutoModelForGeneration objects must be instantiated using the `from_pretrained` method.')

    @property
    @abstractmethod
    def cmp_sequence_probs(self) -> List[NDArray[np.float32]]:
        """Total probability of generating the original sequence given the compared input."""
        raise NotImplementedError('ExplainableAutoModelForGeneration objects must be instantiated using the `from_pretrained` method.')


    @property
    @abstractmethod
    def gen_bow_probs(self) -> NDArray[np.float32]:
        """Average probability of each token in the vocabulary of being generated given the original input."""
        raise NotImplementedError('ExplainableAutoModelForGeneration objects must be instantiated using the `from_pretrained` method.')
    
    @property
    @abstractmethod
    def cmp_bow_probs(self) -> List[NDArray[np.float32]]:
        """Average probability of each token in the vocabulary of being generated given the compared input."""
        raise NotImplementedError('ExplainableAutoModelForGeneration objects must be instantiated using the `from_pretrained` method.')
    

    @abstractmethod
    def gen_nucleus_probs(self, p:float=0.9) -> NDArray[np.float32]:
        """Average probability of each token in the vocabulary of being generated given the original input."""
        raise NotImplementedError('ExplainableAutoModelForGeneration objects must be instantiated using the `from_pretrained` method.')

    @abstractmethod
    def cmp_nucleus_probs(self, p:float=0.9) -> List[NDArray[np.float32]]:
        """Average probability of each token in the vocabulary of being generated given the compared input."""
        raise NotImplementedError('ExplainableAutoModelForGeneration objects must be instantiated using the `from_pretrained` method.')

    @property
    @abstractmethod
    def focus(self) -> slice:
        """The focus of the explanation, if set."""
        raise NotImplementedError('ExplainableAutoModelForGeneration objects must be instantiated using the `from_pretrained` method.')

    @focus.setter
    @abstractmethod
    def focus(self, value:Union[Tuple[int, int], None]) -> None:
        """Sets the focus of the explanation."""
        raise NotImplementedError('ExplainableAutoModelForGeneration objects must be instantiated using the `from_pretrained` method.')

    # Abstract methods for documentation purposes:
    #---------------------------------------------------------------#
    @abstractmethod
    def generate(self, inputs:List[str], **kwargs) -> List[str]:
        """Generates continuations of the passed input prompt(s).

        Args:
            inputs:             The string(s) used as a prompt for the generation.
            generation_config:  The generation configuration to be used as base parametrization for the generation call.
            stopping_criteria:  Custom stopping criteria that complements the default stopping criteria built from arguments and ageneration config.
            kwargs:             Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs.

        Returns:
            List of generated strings.
        """
        raise NotImplementedError('ExplainableAutoModelForGeneration objects must be instantiated using the `from_pretrained` method.')
    
    @abstractmethod
    def compare(self, inputs:List[str], outputs:Optional[Union[List[str], torch.LongTensor, Literal['last']]]=None, batch_size:int=1, **kwargs) -> torch.LongTensor:
        """Calculates the probability `p(outputs) = p(t_0) * p(t_1|t_0) * ... * p(t_n|t_0...t_(n-1))` of
        a specific output `outputs = [t_0, t_1, ..., t_n]` to happen given an input prompt `inputs`.

        Args:
            inputs:     List of input propmts. If `outputs` is specified, `compare(...)` calculates the
                        probability `p(outputs) = p(t_0) * p(t_1|t_0) * ... * p(t_n|t_0...t_(n-1))` for each
                        token in `outputs = [t_0, t_1, ..., t_n]` given ``. Otherwise, it calculates the
                        unconditional probability (similar to `generate(...)`).
            outputs:    List of tokens `t_i` or (strings containing those) for which to compute the probability.
                        If set to `'last'`, the last generated sequence will be used (optional).
            batch_size: Batch size. Ignored if `len(inputs) > 1` or `outputs` not specified (optional).

        Returns:
            Tensor of generated token ids .
        """
        raise NotImplementedError('ExplainableAutoModelForGeneration objects must be instantiated using the `from_pretrained` method.')
    
    @abstractmethod
    def explain_generate(self, query:str, contexts:List[str], *,
            batch_size:int=32,
            max_samples_query:Union[int, Literal['inf', 'auto']]='auto',
            max_samples_context:Union[int, Literal['inf', 'auto']]='auto',
            conditional:bool=True,
            complementary:Union[bool,Literal['no_mc']]=True,
            system:Optional[str]=None,
            **kwargs
        ) -> List[Dict[Literal['role','content'],str]]:
        """Generates continuations of the passed input prompt(s) as well as perturbations for all retrieved documents.

        Args:
            query (str):                The user's query.
            contexts (list):            A list of strings representing the retrieved documents.
            batch_size (int):           The batch size for generating perturbations (default: `32`).
            max_samples_query (int):    Maximum number of samples used for computing SHAP atribution values for the query.
                                        If `2**len(contexts) <= max_samples`, this automatically computes the precise SHAP values instead of kernel SHAP approximations.
                                        If `inf` is passed, always computes the precise SHAP values.
                                        If `auto` is passed, `max_samples` get's the same value as `batch_size` (default: `auto`).
            max_samples_context (int):  Maximum number of samples used for computing SHAP atribution values for the context documents.
                                        If `2**len(contexts) <= max_samples`, this automatically computes the precise SHAP values instead of kernel SHAP approximations.
                                        If `inf` is passed, always computes the precise SHAP values.
                                        If `auto` is passed, `max_samples` get's the same value as `batch_size` (default: `auto`).
            conditional (bool):         Whether to compute the compared values conditioned on the original generation (default: `True`).
            complementary (bool):       If `True` is passed and kernel SHAP approximation is active, samples will be piered complements (default: `True`). 
            system (str):               An optional system prompt.

        Returns:
            A list of generated chats.
        """
        raise NotImplementedError('ExplainableAutoModelForGeneration objects must be instantiated using the `from_pretrained` method.')

    @abstractmethod
    def shap(self, key:Union[Literal['query', 'context'], None], aggregation:GeneratorAggregations_t='token', *,
            num_samples:int=100,
            sample_size:int=10,
            **kwargs
        ) -> GeneratorAttribution_t:
        """Generates Shapley feature attribution values for the chosen aggregation method.

        Args:
            key (str):          Explanation key. Can either be `'query'` or `'context'`.
                                If `None` returns a dictionary of both.
            aggregation (str):  Aggregation method for probabilities (default: `'token'`).
            num_samples (int):  Number of samples for Monte-Carlo approximation.
                                Ignored in case of precise calculation (default: `100`).
            sample_size (int):  Size of samples for Monte-Carlo approximation.
                                Ignored in case of precise calculation (default: `10`).

        Returns:
            A dicionary containing the following two keys (if `key` is specified) or one of the following:
            - `'query'`: a `numpy.ndarray` containing the Shapley values for the query
            - `'context'`: a `numpy.ndarray` containing the Shapley values for the contexts.
        """
        raise NotImplementedError('ExplainableAutoModelForGeneration objects must be instantiated using the `from_pretrained` method.')

    @abstractmethod
    def lime(self, key:Union[Literal['query', 'context'], None], aggregation:GeneratorAggregations_t='token', *,
            kernel_width:int=25,
            kernel_fn:Optional[Callable]=None,
            **kwargs
        ) -> GeneratorAttribution_t:
        """Generates LIME feature attribution values for the chosen aggregation method.

        Args:
            key (str):              Explanation key. Can either be `'query'` or `'context'`.
                                    If `None` returns a dictionary of both.
            aggregation (str):      Aggregation method for probabilities (default: `'token'`).
            kernel_width (int):     Width of the exponential similarity kernel (default: `25`).
            kernel_fn (callable):   Similarity kernel taking distances and returning weights.
                                    If `None`, defaults to `sqrt(exp(-d^2 / kernel_width^2))`.

        Returns:
            A dictionary containing the following two keys (if `key` is specified) or one of the following:
            - `'query'`: a `numpy.ndarray` containing the LIME values for the query
            - `'context'`: a `numpy.ndarray` containing the LIME values for the contexts.
        """
        raise NotImplementedError('ExplainableAutoModelForGeneration objects must be instantiated using the `from_pretrained` method.')


#=======================================================================#
# Context Managers:                                                     #
#=======================================================================#

class Focus:
    def __init__(self, target:GeneratorExplanationBase, focus:Union[Tuple[int, int], str, None], *,
            token_processor:Optional[Callable[[str],str]]=None
        ) -> None:
        """Context manager to set the focus of the explanation.

        Args:
            target (GeneratorExplanationBase):               The target object to set the focus on.
            focus (Union[Tuple[int, int], str, None]):       The focus to set. Can be a tuple of two integers representing a slice,
                                                             a string representing a subsequence, or `None` to reset the focus.
            token_processor (Optional[Callable[[str],str]]): Optional function to process tokens before setting the focus.
                                                             If provided, it will be applied to each token in the focus string.
        """
        # set the target:
        self._target = target

        # if focus is a string, find it in the generated tokens:
        if isinstance(focus, str):
            # normalize focus string:
            focus = focus.lower().strip()

            # check that token_processor is provided if focus is a string:
            if token_processor is None:
                raise ValueError(f'`token_processor` must be specified in case `focus` is a string!')
            
            # find the subsequence in the generated tokens:
            tokens = [token_processor(t) for t in self._target.gen_tokens]
            num_tokens = len(tokens)

            self._focus = None
            for i in range(num_tokens):
                if self._focus is not None: break

                for j in range(i + 1, num_tokens):
                    # normalize candidate:
                    candidate = ''.join(tokens[i:j]).lower().strip()

                    # if candidate is not a prefix of focus, skip:
                    if not focus.startswith(candidate): break

                    if candidate == focus:
                        self._focus = (i, j)
                        break

            # raise an error if the focus string is not found:
            if self._focus is None:
                raise ValueError(f'Focus string "{focus}" not found in the generated tokens!')

        # othewise set the focus to the provided value:
        else: self._focus = focus

    def __enter__(self):
        """Sets the focus of the explanation."""
        self._target.focus = self._focus
        return self._target

    def __exit__(self, exc_type, exc_value, traceback):
        """Resets the focus of the explanation."""
        self._target.focus = None