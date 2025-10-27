import os
import json
import torch
import numpy as np
from importlib import import_module
from typing import Optional, Callable, Dict, List, Union, Tuple

from numpy.typing import NDArray

__RESOURCE_DIR__ = os.path.dirname(os.path.abspath(__file__))

#====================================================================================================#
# General purpose helper functions:                                                                  #
#====================================================================================================#

def find_subseq(seq:List, subseq:List, start:int=0) -> int:
    """Finds subsequences in list-like sequences and returns the index of the first occurence.

    Args:
        seq (List):    The sequence which contains the subsequnece.
        subseq (List): The subsequence to look for.
        start (int):   The first index of `seq` to be checked.

    Returns:
        The index of the first occurence of `subseq` in `seq[start:]`.
    """
    seq_size    = len(seq)
    subseq_size = len(subseq)
    for i in range(start, seq_size-subseq_size+1):
        if seq[i:i+subseq_size] == subseq:
            return i
    raise ValueError(f'{subseq} not in {seq}.')

def nucleus_sample_tokens(scores:Union[NDArray, torch.Tensor], tokens:List[str], threshold:float=.9) -> Tuple[NDArray, List[str]]:
    """Nucleus samples the tokens for which the acumulated absolute scores are higher than `threshold`.

    Args:
        scores (NDArray | Tensor):  The scores.
        tokens (List):              The tokens.
        threshold (float):          The threshold that must be exceeded (default: `.9`).

    Returns:
        A tuple containing the following elements:
         - An `NDArray` with the sorted scores. The last element is the sum of the remaining non-sampled scores.
         - A list of token strings in the same order as the returned scores. The last element is a placeholder for the remaining elements.
    """
    if (threshold < 0) or (threshold > 1):
        raise ValueError (f'Parameter `threshold` must be in intervall `[0.,1.]`')

    # convert tensors to numpy:
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    # convert tokens to numpy:
    tokens = np.array(tokens, dtype=object)

    # combine identical tokens:
    for token in set(tokens):
        mask = tokens == token
        if sum(mask) > 1:
            # sum the contributions and store them
            # to the first occurence of the token:
            i = np.flatnonzero(mask)[0]
            scores[i] = sum(scores[mask])

            # Delete all but the first occurence
            # of the token:
            mask[i] = False
            scores = scores[~mask]
            tokens = tokens[~mask]

    # nucelus sample tokens with 90% absolute impact:
    s_abs = np.abs(scores)
    s_ind = np.argsort(s_abs)[::-1]
    s_cum = np.cumsum(s_abs[s_ind])
    s_sel = (s_cum / s_cum[-1]) <= threshold

    return (
        scores[s_ind[s_sel]].tolist() + [scores[s_ind[~s_sel]].sum()],
        [tokens[i] for i in s_ind[s_sel]] + [f'+{np.sum(~s_sel):d} other']
    )

def tokens2words(tokens:List[str], *, token_processor:Optional[Callable[[str],str]]=None, separator:str=' ', filter_tokens:List[str]=[]):
    """Combines tokens to words by splitting at `separator`.

    Args:
        tokens (List[str]):             The list of token strings.
        token_processor ((str) -> str): An optional function applied to each token.
        separator (str):                The separtaor to split on (default: `' '`).
        filter_tokens (List[str]):      A list of tokens to be ignored.
    """

    words = []
    for i, token in enumerate(tokens):
        # filter tokens:
        if token in filter_tokens: continue

        # apply token preprocessing:
        if token_processor is not None:
            token = token_processor(token)

        # start new word on first token or separator:
        if (len(words) == 0) or token.startswith(separator):
           words.append([i])

        # otherwise continue last word:
        else: words[-1].append(i)

    return words

#====================================================================================================#
# Messy transformers convenience functions:                                                          #
#====================================================================================================#

def get_model_type(model_name_or_path:str):
    # load dictionary of known types:
    try:
        with open(os.path.join(__RESOURCE_DIR__, 'model_types.json'), 'r') as file:
            model_types = json.load(file)
    except: model_types = {}

    # select matching type:
    try: model_type = model_types[model_name_or_path]
    except KeyError:
        # load tokenizer:
        from transformers import AutoModelForCausalLM
        model_type = type(AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map='auto',
            torch_dtype=torch.bfloat16  # save some space
        ))

        # save new template:
        model_types[model_name_or_path] = model_type.__name__
        with open(os.path.join(__RESOURCE_DIR__, 'model_types.json'), 'w') as file:
            json.dump(model_types, file)

    # instantiate type names:
    if isinstance(model_type, str):
        model_type = getattr(import_module('transformers'), model_type)

    return model_type

def get_chat_template(model_name_or_path:str):
    # load dictionary of known templates:
    try:
        with open(os.path.join(__RESOURCE_DIR__, 'chat_templates.json'), 'r') as file:
            templates = json.load(file)
    except: templates = {}

    # select matching template:
    try: template = templates[model_name_or_path]
    except KeyError:
        # load tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # extract chat pattern:
        pattern = tokenizer.apply_chat_template([{'role':'[ROLE]', 'content':'[CONTENT]'}], add_special_tokens=False, tokenize=False)
        sot, rest = pattern.split('[ROLE]')
        sep, eot  = rest.split('[CONTENT]')

        # try to cut system prompts:
        if sep in sot and eot in sot:
            sot = sot.split(eot)[-1]

        # tokenize pattern:
        sot_tokens = tokenizer(sot, add_special_tokens=False, return_tensors='np').input_ids
        sep_tokens = tokenizer(sep, add_special_tokens=False, return_tensors='np').input_ids
        eot_tokens = tokenizer(eot, add_special_tokens=False, return_tensors='np').input_ids

        # create new template:
        template = {
            'sot': {'tokens': tokenizer.convert_ids_to_tokens(sot_tokens[0]), 'ids': sot_tokens[0].tolist()},
            'sep': {'tokens': tokenizer.convert_ids_to_tokens(sep_tokens[0]), 'ids': sep_tokens[0].tolist()},
            'eot': {'tokens': tokenizer.convert_ids_to_tokens(eot_tokens[0]), 'ids': eot_tokens[0].tolist()}
        }

        # save new template:
        templates[model_name_or_path] = template
        with open(os.path.join(__RESOURCE_DIR__, 'chat_templates.json'), 'w') as file:
            json.dump(templates, file)

    return template

def decode_chat_template(inputs:Union[List[int], List[str], str], model_name_or_path:str, *, return_indices:bool=False) -> List[Dict[str, Union[List[int], List[str], str]]]:
    """Decodes a chat format string into a list of roles.

    Args:
        inputs (List[int] | str):
            The input content, either a raw string or a list of integer token IDs.

        model_name_or_path (str):
            Full model identifier, e.g., 'meta-llama/Llama-3.1-8B-Instruct',
            'mistralai/Mistral-7B-Instruct-v0.2', or 'google/gemma-7b-it'.

    Returns:
        A list of dictionaries with one entry per role and turn.
    """

    result = []
    
    # get chat template data:
    template = get_chat_template(model_name_or_path)

    # deal with string inputs:
    if isinstance(inputs, str):
        # load tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        sot = tokenizer.decode(template['sot']['ids'])
        sep = tokenizer.decode(template['sep']['ids'])
        eot = tokenizer.decode(template['eot']['ids'])
        del tokenizer

        i = 0
        try:
            while i < len(inputs):
                role_start = inputs.index(sot, i) + len(sot)
                role_end   = inputs.index(sep, role_start)
                role       = slice(role_start, role_end) if return_indices else inputs[role_start:role_end]

                content_start = role_end + len(sep)
                content_end   = inputs.index(eot, content_start)
                content       = slice(content_start, content_end) if return_indices else inputs[content_start:content_end]

                result.append({'role':role, 'content':content})
                i = content_end + len(eot)

        except ValueError: pass
        return result
    
    # deal with lists of integers or strings:
    elif hasattr(inputs, '__len__'):
        # make sure we have a list:
        inputs = list(inputs)

        use_str = isinstance(inputs[0], str)
        sot = template['sot']['tokens'] if use_str else template['sot']['ids']
        sep = template['sep']['tokens'] if use_str else template['sep']['ids']
        eot = template['eot']['tokens'] if use_str else template['eot']['ids']

        i = 0
        try:
            while i < len(inputs):
                role_start = find_subseq(inputs, sot, i) + len(sot)
                role_end   = find_subseq(inputs, sep, role_start)
                role       = slice(role_start, role_end) if return_indices else inputs[role_start:role_end]

                content_start = role_end + len(sep)
                content_end   = find_subseq(inputs, eot, content_start)
                content       = slice(content_start, content_end) if return_indices else inputs[content_start:content_end]

                result.append({'role':role, 'content':content})
                i = content_end + len(eot)

        except ValueError: pass
        return result

    else: raise TypeError()