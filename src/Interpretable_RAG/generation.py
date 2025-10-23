import os
import torch
import numpy as np
import transformers
import tqdm
import pickle

from scipy.special import comb
from sklearn.linear_model import LinearRegression

from resources.utils import decode_chat_template, get_model_type

from numpy.typing import NDArray
from typing import Union, List, Dict, Tuple, Optional, Literal, Iterable, Callable, Any

from abc import ABCMeta, abstractmethod

#=======================================================================#
# Declarations:                                                         #
#=======================================================================#

MAXINT = int(2**31 - 1) # Maximum value for signed int32

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

def create_rag_prompt(query:str, contexts:List[str], *, system:Optional[str]=None) -> List[Dict[str,str]]:
    """
    Creates chat-style messages for RAG using LLaMA-style chat template.

    Args:
        query (str):        The user's query.
        contexts (list):    A list of strings representing the retrieved documents.
        system (str):       An optional system prompt.

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

    # Format the context into a single message
    context_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(contexts)])

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{context_text}\n\nQuery: {query}"}
    ]

def generate_permutations_recursive(items:List[Any], func:Callable[[List[Any]], Any], perturbations:List[Any], index:int) -> Tuple[List[Tuple[int]], List[Tuple[int]]]:
    """Recursively generates index-tagged permutations of subsets of a given list,
    using bitmask logic to explore all combinations.

    This function is typically used internally by `generate_permutations`. It relies on 
    a bitmask (`index`) to determine which elements to include in each recursive step.
    The `func` is used to compute perturbations, which are memorized in the `perturbations` list.

    Args:
        items (list):         The list of items for which permutations are to be generated.
        func (callable):      A function applied to subsets of `items` to compute perturbations.
        perturbations (list): A list (with length 2^n) for memorization of perturbation results.
        index (int):          A bitmask representing the current subset of `items`.

    Returns:
        Tuple:
            - `permutations`:  A list of tuples, each representing a sequence of indices
                               of subsets in `perturbations` that define a permutation path.
            - `new_items`:     A list of tuples, each representing a sequence of indices
                               of items in `items` added to the previous set to define
                               a permutation path.
    """
    
    # if current perturbation undefined:
    if perturbations[index] is None:
        # create perturbation:
        perturbations[index] = func(items)

    # break on empty set:
    if index == 0: return [(index,)], [()]

    # calculate possible permutations:
    i, j, m = 0, 0, 1
    permutations, new_items = [], []
    while i < len(items):
        if index & m:
            child_permutations, child_new_items = generate_permutations_recursive(
                items=items[:i]+items[i+1:],
                func=func,
                perturbations=perturbations,
                index=index & ~m
            )
            permutations.extend([prm + (index,) for prm in child_permutations])
            new_items.extend([ni + (j,) for ni in child_new_items])

            i += 1

        m = m << 1
        j += 1

    return permutations, new_items

def generate_permutations(items:List[Any], func:Callable[[List[Any]], Any]) -> Tuple[NDArray[np.int_], NDArray[np.int_], List[Any]]:
    """Generates all possible bitmask-tagged permutations of subsets of a list,
    and computes a perturbation for each subset using a user-defined function.

    This function initializes the perturbation list and starts the recursive
    permutation generation using `generate_permutations_recursive`.

    Args:
        items (List[Any]):        The list of items to permute.
        func (List[Any]) -> Any): A function to compute a "perturbation" for any subset of items.

    Returns:
        Tuple:
            - `permutations`:  A matrix of shape (len(`permutations`), len(`items`) + 1) where
                               rows represent sequences of indices of subsets in
                               `perturbations` that define a permutation path.
            - `new_items`:     A matrix of shape (len(`permutations`), len(`items`)) where
                               rows represent sequences of indices of items in `items`
                               added to the previous set to define a permutation path.
            - `perturbations`: List of perturbation values for each subset.
    """
    # calculate number of possible perturbations:
    n = (2 ** len(items))

    # initialize perturbation list:
    perturbations = [None] * n

    # calculate all possible permuations:
    permutations, new_items = generate_permutations_recursive(
        items, func,
        perturbations,  # empty list to be filled
        n-1             # this creates a bitmap of n ones
    )

    # return tuple of permuations and perturbations: 
    return np.array(permutations), np.array(new_items), perturbations

def sample_perturbations(items:List[Any], func:Callable[[List[Any]], Any], num_samples:int, complementary:bool=False) -> Tuple[NDArray[np.bool_], List[Any]]:
    """Randomly samples a specified number of unique subsets of a given list,
    and returns their binary indicator features along with the result of applying a function
    to each sampled subset.

    The function constructs a random selection of bitmask-encoded subsets of the input list
    `items`, including the empty set and the full set. For each sampled subset, it applies
    the provided function `func` and stores the result. It also returns a binary feature
    matrix indicating which items are included in each sampled subset.

    Args:
        items (List[Any]):        The list of items to draw subsets from.
        func (List[Any]) -> Any): A function that computes a perturbation or
                                  feature from a subset of `items`.
        num_samples (int):        The number of unique subset samples to generate, including
                                  the empty set and full set.
        complementary (bool):     If `True`, makes sure that the sampled perturbations are
                                  pairs of complementary sets.

    Returns:
        Tuple:
            - `subsets`:       A boolean matrix of shape (len(`perturbations`), len(`items`)),
                               where each row indicates which items are included
                               in the corresponding subset.
            - `perturbations`: List of perturbation values for each subset.
    """
    
    # calculate number of possible perturbations:
    n = (2 ** len(items))

    # take sample of `num_samples` unique bitmasks (including empty and full):
    sample = np.empty(num_samples, dtype=int)
    sample[0]    = 0   # =0b000...0
    sample[-1]   = n-1 # =0b111...1
    if complementary:
        sample[1:-1:2] = np.random.choice(
            np.arange(1,(n//2)-1),
            size=((num_samples//2)-1),
            replace=False
        )
        sample[2:-1:2] = np.invert(sample[1:-1:2]) & (n-1)

    else:
        sample[1:-1] = np.random.choice(
            np.arange(1,n-1),
            size=(num_samples-2),
            replace=False
        )

    # generate perturbations:
    perturbations = [None] * num_samples
    subsets       = np.zeros((num_samples, len(items)), dtype=bool)
    for i, bitmask in enumerate(sample):
        # translate bitmask to set:
        current_items, m = [], 1
        for j in range(len(items)):
            if bitmask & m:
                current_items.append(items[j])
                subsets[i, j] = 1.
            m = m << 1

        # generate perturbation:
        perturbations[i] = func(current_items)

    return subsets, perturbations

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

    #===================================================================#
    # Methods:                                                          #
    #===================================================================#

    @abstractmethod
    def get_shapley_values(self,
            key:Union[Literal['query', 'context'], None],
            aggregation:Literal['token', 'sequence', 'bow', 'nucleus']='token',
            **kwargs
        ) -> Union[Dict[Literal['query', 'context'], NDArray[np.float64]], NDArray[np.float64]]:
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

    def save_values(self, path:Optional[str]=None, *,
            aggregations:Optional[List[Literal['token', 'sequence', 'bow', 'nucleus']]]=None,
        ) -> Union[str, None]:
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
            aggregations = ['token', 'sequence', 'bow', 'nucleus']

        for aggregation in aggregations:
            data_to_save['shapley_values_' + aggregation] = self.get_shapley_values(None, aggregation)

        if path is None: return data_to_save

        with open(path, 'wb') as f:
            pickle.dump(data_to_save, f)


class GeneratorExplanation(GeneratorExplanationBase):
    @classmethod
    def load(cls, saved_data:Union[str, Dict, Iterable], *,
        model_name_or_path:Optional[str]=None,
        tokenizer:Optional[transformers.PreTrainedTokenizer]=None
    ) -> Union['GeneratorExplanation', List['GeneratorExplanation']]:
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
                return cls.load([os.path.join(saved_data, file)
                    for file in os.listdir(saved_data)
                    if file.endswith('.pkl')])

            # load file:
            with open(saved_data, 'rb') as f:
                data = pickle.load(f)

        elif isinstance(saved_data, dict):
            data = saved_data

        elif hasattr(saved_data, '__iter__') or hasattr(saved_data, '__len__'):
            result = {}

            # load all entries separatelly:
            for item in saved_data:
                try:
                    # load next file:
                    result[item] = cls.load(
                            item,
                            model_name_or_path=model_name_or_path,
                            tokenizer=tokenizer
                        )
                    

                    # avoid multiple instances of the tokenizer:
                    if tokenizer is None: tokenizer = result[0].tokenizer

                except Exception as e:
                    print(f'WARNING: Could not load "{item}": {e}')

            return result

        else: raise ValueError("`saved_data` must be a path, an iterable, or a dictionary")

        result = cls()
        result._qry_tokens = data['qry_tokens']
        result._gen_tokens = data['gen_tokens']
        result._qry_precise = data['shap_qry_precise']
        result._ctx_precise = data['shap_ctx_precise']

        result._shapley_values = {}
        for aggregation in ['token', 'sequence', 'bow', 'nucleus']:
            result._shapley_values[aggregation] = data['shapley_values_' + aggregation]

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
        return self._gen_tokens

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

    def get_shapley_values(self,
            key:Union[Literal['query', 'context'], None],
            aggregation:Literal['token', 'sequence', 'bow', 'nucleus']='token',
            **kwargs
        ) -> Union[Dict[Literal['query', 'context'], NDArray[np.float64]], NDArray[np.float64]]:
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
        if key is None: return self._shapley_values[aggregation]
        else: return self._shapley_values[aggregation][key]


#=======================================================================#
# Generic Model Class:                                                  #
#=======================================================================#

class ExplainableAutoModelForGeneration:
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path:str, *args, **kwargs):
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
        class _ExplainableAutoModelForGeneration(T, GeneratorExplanationBase):
            def __init__(self, config, *inputs, **kwargs):
                super().__init__(config, *inputs, **kwargs)
                self._tokenizer:transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(config.name_or_path)
                self._tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self._explain:bool = False
                self._exp_logits:List[torch.Tensor] = []
                self._gen_logits:torch.Tensor = []
                self._gen_output = None
                self._shap_cache = None

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
                if len(self._gen_logits) == 0: return None

                # return the focus:
                if hasattr(self, '_focus'): return self._focus
                else: return slice(0, self._gen_output.shape[1])
            
            @focus.setter
            def focus(self, value:Union[Tuple[int, int], None]) -> None:
                """Sets the focus of the explanation."""
                # generate(...) needs to be called first:
                if (value is not None) and (len(self._gen_logits) == 0):
                    raise AttributeError('`generate(...)` needs to be called at least once before setting `focus`!')

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
                if len(self._gen_logits) == 0:
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
            def gen_token_probs(self) -> NDArray[np.float64]:
                """Probability of each token in the original generation."""
                # generate(...) needs to be called first:
                if len(self._gen_logits) == 0:
                    raise AttributeError('`generate(...)` needs to be called at least once before accessing `gen_token_probs`!')

                # get focus:
                focus = self.focus

                # get generated token ids:
                output_ids = self._gen_output[:, focus]

                # get logits:
                logits = self._gen_logits[:, focus, :]

                # return probability of each token in the original generation:
                return np.array([ 
                    [float(logits[i, j, id]) for j, id  in enumerate(seq)]
                    for i, seq in enumerate(output_ids)
                ])

            @property
            def cmp_token_probs(self) -> NDArray[np.float64]:
                """Probability of each token in the original generation given the compared input."""
                # compare(...) needs to be called first:
                if len(self._exp_logits) == 0:
                    raise AttributeError(
                        '`generate(...)` and `compare(...)` need to be called ' +
                        'at least once before accessing `cmp_token_probs`!'
                    )

                # get focus:
                focus = self.focus

                # get generated token ids:
                output_ids = self._gen_output[:, focus]

                # get logits:
                logits = [t[:, focus, :] for t in self._exp_logits]

                # return probability of each token in the original generation:
                return [np.array([ 
                    [float(t[i, j, id]) for j, id  in enumerate(seq)]
                    for i, seq in enumerate(output_ids)
                ]) for t in logits]


            @property
            def gen_sequence_prob(self) -> NDArray[np.float64]:
                """Total probability of generating the original sequence."""
                # generate(...) needs to be called first:
                if len(self._gen_logits) == 0:
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
            def cmp_sequence_probs(self) -> NDArray[np.float64]:
                """Total probability of generating the original sequence given the compared input."""
                # compare(...) needs to be called first:
                if len(self._exp_logits) == 0:
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
            def gen_bow_probs(self) -> NDArray[np.float64]:
                """Average probability of each token in the vocabualry of being generated given the original input."""
                # generate(...) needs to be called first:
                if len(self._gen_logits) == 0:
                    raise AttributeError('`generate(...)` needs to be called at least once before accessing `gen_bow_probs`!')

                # get focus:
                focus = self.focus

                # get probabilities:
                probs = logits2probs(self._gen_logits[:, focus, :], normalization='softmax')

                # return accumulated probability of each token in the vocabualry:
                return probs.mean(dim=1).float().numpy()
            
            @property
            def cmp_bow_probs(self) -> NDArray[np.float64]:
                """Average probability of each token in the vocabualry of being generated given the compared input."""
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

                # return accumulated probability of each token in the vocabualry:
                return [t.mean(dim=1).float().numpy() for t in probs]
            

            #@property
            def gen_nucleus_probs(self, p:float=0.9) -> NDArray[np.float64]:
                """Average probability of each token in the vocabualry of being generated given the original input."""
                # generate(...) needs to be called first:
                if len(self._gen_logits) == 0:
                    raise AttributeError('`generate(...)` needs to be called at least once before accessing `gen_nucleus_probs`!')

                # get focus:
                focus = self.focus

                # get probabilities:
                probs = logits2probs(self._gen_logits[:, focus, :], normalization='softmax')

                # return accumulated probability of each token in the vocabualry:
                return _nucleus_sampling(probs.float(),p=p).mean(dim=1).numpy()

            #@property
            def cmp_nucleus_probs(self, p:float=0.9) -> NDArray[np.float64]:
                """Average probability of each token in the vocabualry of being generated given the compared input."""
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

                # return accumulated probability of each token in the vocabualry:
                return [_nucleus_sampling(t.float(),p=p).mean(dim=1).numpy() for t in probs]

            #===============================================================#
            # Methods:                                                      #
            #===============================================================#

            def forward(self, *args, **kwargs):
                # get token probabilities:
                outputs = super().forward(*args, **kwargs)

                # save token probabilities:
                if self._explain: self._exp_logits[-1].append(outputs.logits[:,-1:,:].detach().cpu())
                else:             self._gen_logits.append(outputs.logits[:,-1:,:].detach().cpu())

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
                self._gen_logits  = []
                self._exp_logits  = []

                # generate:
                self._gen_output = super().generate(**inputs, **kwargs).sequences[:, inputs.input_ids.shape[-1]:]

                # finalize probabilities:
                self._gen_logits  = torch.concatenate(self._gen_logits, dim=1)

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
                inputs = self.tokenizer(inputs, truncation=True, padding=True, return_tensors='pt')

                # deactivate explanation mode:
                self._explain    = True

                # reset token probabilities:
                self._exp_logits.append([])

                # generate:
                output = super().generate(**inputs, **kwargs).sequences[:, inputs.input_ids.shape[-1]:]

                # finalize probabilities:
                self._exp_logits[-1] = torch.concatenate(self._exp_logits[-1], dim=1)

                # split batch in elements if multiple inputs:
                if len(inputs) > 1:
                    self._exp_logits.extend([t.unsqueeze(0) for t in self._exp_logits.pop(-1)])

                # return generated tokens:
                return output

            def __compare_conditional(self, inputs:List[str], outputs:Union[List[str], torch.LongTensor], batch_size:int=1, **kwargs) -> torch.LongTensor:
                # get batch size:
                single_input  = len(inputs) == 1
                single_output = len(outputs) == 1
                assert (len(inputs) == len(outputs)) or single_output
                if not single_input: batch_size = len(inputs)

                # activate explanation mode:
                self._explain = True

                # reset token probabilities:
                self._exp_logits.append([])

                # convert string to Iterable of tokens:
                if isinstance(outputs[0], str):
                    outputs = self.tokenizer(outputs, add_special_tokens=False, return_attention_mask=False, return_tensors='pt').input_ids
                outputs = outputs.to(self.device)

                # tokenize input:
                inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
                input_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)

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
                v0, v1 =  [int(i) for i in transformers.__version__.split('.')[:2]]
                if v0 >= 4 and v1 >= 52:
                    model_kwargs = self._get_initial_cache_position(
                        seq_length=input_ids.shape[1],
                        device=input_ids.device,
                        model_kwargs=model_kwargs,
                    )

                else:
                    model_kwargs = self._get_initial_cache_position(
                        input_ids=input_ids, 
                        model_kwargs=model_kwargs
                    )
                    
                with torch.no_grad():

                    # calculate p(t_0):
                    model_inputs = self.prepare_inputs_for_generation(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **model_kwargs
                    )
                    model_outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=True, **model_kwargs)
                    model_kwargs = self._update_model_kwargs_for_generation(model_outputs, model_kwargs, num_new_tokens=input_ids.shape[1])
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
                        model_inputs = self.prepare_inputs_for_generation(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
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
                    self._exp_logits[-1] = torch.concatenate(self._exp_logits[-1], dim=0).transpose(0,1)

                else: self._exp_logits[-1] = torch.concatenate(self._exp_logits[-1], dim=1)

                # split batch in elements if multiple inputs for the same output:
                if not single_input and single_output:
                    self._exp_logits.extend([t.unsqueeze(0) for t in self._exp_logits.pop(-1)])

                # return generated tokens:
                return torch.argmax(self._exp_logits[-1], dim=-1)


            def explain_generate(self, query:str, contexts:List[str], *, batch_size:int=32, max_samples:Union[int, Literal['inf', 'auto']]='auto', conditional:bool=True, system:Optional[str]=None, **kwargs):
                """Generates continuations of the passed input prompt(s) as well as perturbations for all retrieved documents.

                Args:
                    query (str):        The user's query.
                    contexts (list):    A list of strings representing the retrieved documents.
                    batch_size (int):   The batch size for generating perturbations (default: `32`).
                    max_samples (int):  Maximum number of samples used for computing SHAP atribution values.
                                        If `2**len(contexts) <= max_samples`, this automatically computes the precise SHAP values instead of kernel SHAP approximations.
                                        If `inf` is passed, always computes the precise SHAP values.
                                        If `auto` is passed, `max_samples` get's the same value as `batch_size` (default: `auto`).
                    conditional (bool): Whether to compute the compared values conditioned on the original generation (default: `True`).
                    system (str):       An optional system prompt.

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
                complete_rag_prompt = create_rag_prompt(query, contexts, system=system)
                output = self.generate(
                    [self.tokenizer.apply_chat_template(complete_rag_prompt, tokenize=False)],
                    **kwargs
                )

                # generate perturbed prompts:
                if   max_samples == 'auto': max_samples = batch_size
                elif max_samples == 'inf':  max_samples = MAXINT

                self._qry_tokens = query.split()
                self._qry_tokens[1:] = [' ' + t for t in self._qry_tokens[1:]]

                self._shap_cache, perturbed_prompts = {}, {}
                self._shap_cache['query'], perturbed_prompts['query'] = self.__generate_prompts(
                    self._qry_tokens,                                                        # permute the query
                    lambda items:create_rag_prompt(''.join(items), contexts, system=system), # build a prompt for each permutation
                    int(max_samples//2 + 1)
                )
                self._shap_cache['context'], perturbed_prompts['context'] = self.__generate_prompts(
                    contexts,                                                    # permute the contexts
                    lambda items:create_rag_prompt(query, items, system=system), # build a prompt for each permutation
                    int(max_samples//2 + 1)
                )

                # combine prompt lists comparison output:
                perturbed_prompts_combined = perturbed_prompts['query'][:-1] + perturbed_prompts['context'][1:]

                cache  = self._shap_cache['query']
                offset = len(perturbed_prompts['context']) - 2
                if cache['precise']: cache['indices'][:,-1] += offset
                else: cache['indices'][-1] += offset

                cache = self._shap_cache['context']
                offset = len(perturbed_prompts['query']) - 2
                if cache['precise']: cache['indices'][:,1:] += offset
                else: cache['indices'][1:] += offset

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
                        'last' if conditional else None,
                        **kwargs
                    )

                    # print empty line:
                    if num_batches > 1: print()

                return complete_rag_prompt + decode_chat_template(output[0], self.config.name_or_path) 

            def __generate_prompts(self, items, func, max_samples):
                # calculate number of samples needed for precise calculation:
                n = 2 ** len(items)

                # generate prompts:
                if max_samples >= n:
                    # generate prompts for perturbed inputs (precise SHAP values):
                    permutations, new_items, perturbed_prompts = generate_permutations(items, func)
                    cache = {'precise': True, 'indices': permutations, 'new_docs': new_items}

                elif max_samples < n:
                    # sample prompts for kernel SHAP:
                    perturbations, perturbed_prompts = sample_perturbations(items, func, max_samples)
                    cache = {'precise': False, 'indices': np.arange(perturbations.shape[0]), 'sets': perturbations}

                else: raise ValueError(f'Unknown value for parameter `max_samples`: {max_samples}')

                return cache, perturbed_prompts


            def get_shapley_values(self,
                key:Union[Literal['query', 'context'], None],
                aggregation:Literal['token', 'sequence', 'bow', 'nucleus']='token',
                num_samples:int=100,
                sample_size:int=10,
                **kwargs
            ) -> Union[Dict[Literal['query', 'context'], NDArray[np.float64]], NDArray[np.float64]]:
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
                    probs = [p.flatten() for p in eval(f'self.cmp_{aggregation}_probs')]

                    # Add the generated token probabilities as the final "player" in the SHAP context:
                    probs.append(eval(f'self.gen_{aggregation}_probs').flatten())


                else: raise ValueError(f'Unknown value for parameter `aggregation`: "{aggregation}"')

                # Call actual method:
                result = {'query': None, 'context': None} if key is None else {key: None}
                for k in result:
                    if self._shap_cache[k]['precise']: result[k] = self._get_shapley_values_precise(probs, **self._shap_cache[k])
                    else: result[k] = self._get_shapley_values_monte_carlo(probs, num_samples=num_samples, sample_size=sample_size, **self._shap_cache[k])

                return result if key is None else result[key]

            def _get_shapley_values_precise(self, probs:NDArray[np.float64], indices:NDArray[np.int_], new_docs:NDArray[np.int_], precise:bool) -> NDArray[np.float64]:
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

            def _get_shapley_values_monte_carlo(self, probs:NDArray[np.float64], indices:NDArray[np.int_], sets:NDArray[np.bool_], precise:bool, num_samples:int=100, sample_size:int=10) -> NDArray[np.float64]:
                assert precise is False, 'Monte Carlo SHAP values can only be calculated for approximate values!'

                # Generate Shapley values for `num_samples` samples of `sample_size` documents:
                attributions = []
                for _ in range(num_samples):
                    sample = np.random.choice(len(indices)-2, size=sample_size-2, replace=False) + 1 # skip the first and last indices

                    attributions.append(
                        self._get_shapley_values_kernel(
                            probs   = probs,
                            indices = np.concatenate([indices[:1], indices[sample], indices[-1:]]),
                            sets    = np.concatenate([sets[:1], sets[sample], sets[-1:]]),
                            precise = precise
                        )
                    )

                # Return the mean of the attributions across all samples:
                return np.mean(attributions, axis=0)
            
            def _get_shapley_values_complementary(self, probs:NDArray[np.float64], indices:NDArray[np.int_], sets:NDArray[np.bool_], precise:bool) -> NDArray[np.float64]:
                assert precise is False, 'Complementary SHAP values can only be calculated for approximate values!'

                # Initialize array to store marginal contributions for each permutation step
                p_marginal = np.empty((indices.shape[0] - 2, 2) + probs[0].shape, dtype=probs[0].dtype)

                # calculate complementary marginal contributions:
                p_marginal[:,0] = np.stack([probs[i] for i in indices[1:-1]]) - probs[0]
                p_marginal[:,1] = probs[-1] - p_marginal[:,0]

                # fit a linear regressor using the SHAP kernel:
                lr = LinearRegression()
                x  = np.concatenate((
                    #[sets[0], sets[-1]],
                    sets[1:-1],
                    ~sets[1:-1]
                ), axis=0, dtype=float)
                y  = np.concatenate((
                    #[probs[0], probs[-1]],
                    p_marginal[:,0],
                    p_marginal[:,1]
                ), axis=0, dtype=float)
                w  = [1. / (comb(len(z), sum(z)) * (len(z) - sum(z)))
                      for z in x]
                lr.fit(x, y, w)
                #lr.fit(x, y)

                # attributions are estimated SHAP values:
                attributions = lr.coef_.T

                # rescale attributions to fit prediction:
                return attributions
            
            def _get_shapley_values_kernel(self, probs:NDArray[np.float64], indices:NDArray[np.int_], sets:NDArray[np.bool_], precise:bool) -> NDArray[np.float64]:
                assert precise is False, 'Kernel SHAP values can only be calculated for approximate values!'

                # fit a linear regressor using the SHAP kernel:
                lr = LinearRegression()
                x  = sets[1:-1].astype(float)
                y  = np.stack([probs[i] for i in indices[1:-1]])
                w  = [(len(z)-1) / (comb(len(z), sum(z)) * sum(z) * -sum(z-1))
                      for z in x]
                lr.fit(x, y, w)

                # attributions are estimated SHAP values:
                attributions = lr.coef_.T

                # rescale attributions to fit prediction:
                return attributions
            
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