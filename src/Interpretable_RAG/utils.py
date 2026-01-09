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

def flatten_token_attributions(attribution:List[float], tokens:List[str], *, token_processor:Optional[Callable[[str],str]]=None):
    """Flatten token-level attributions to a character-level attribution array.
    Each element of `attribution` is assumed to correspond to the token at the
    same position in `tokens`.

    Args:
        attribution (List[float]):
            A sequence of scalar attribution scores, one per token.
        tokens (List[str]):
            A sequence of token strings corresponding to `attribution`.
        token_processor ((str) -> str, optional):
            If provided, a function applied to each token string before counting its
            characters and appending to the output text (for example, to normalize white spaces). If None, the raw token strings are used.

    Returns:
        out (Tuple[np.ndarray, str]):
            A tuple `(attribution, text)`, where...
            - `attribution` is a 1-D numpy array of floats whose length equals the
            total number of characters in `text`. Each token's attribution value
            from `attribution` is repeated for each character of the corresponding
            processed token.
            - `text` is the concatenated string produced by joining all processed
            tokens in order.

    Example:
        >>> attribution = [0.1, -0.2]
        >>> tokens = ["Hello", " world"]
        >>> flatten_token_attributions(attribution, tokens)
        # returns (array([0.1,0.1,0.1,0.1,0.1, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2]),
        #          "Hello world")
    """

    attribution_out, txt_out = [], ''
    for a, s in zip(attribution, tokens, strict=True):
        if token_processor is not None:
            s = token_processor(s)
        
        attribution_out += [a]*len(s)
        txt_out += s

    return np.array(attribution_out), txt_out

def match_token_attributions(
        ret_attribution:List[float], ret_tokens:List[str],
        gen_attribution:List[float], gen_tokens:List[str],
        *,
        ret_token_processor:Optional[Callable[[str],str]]=None,
        gen_token_processor:Optional[Callable[[str],str]]=None
    ):
    """Align and match token-level attributions from retriever and generator.

    This function:
    - flattens token attributions to character-level attribution arrays using the
      provided token processors,
    - finds the overlapping substring between the flattened retriever and
      generator texts,
    - extracts the aligned character-level attribution vectors for the common
      substring,
    - groups adjacent characters that share identical retriever and generator
      attribution values into tokens,
    - normalizes the resulting retriever and generator attribution arrays so
      each sums to 1.

    Args:
        ret_attribution (List[float]):                Attribution values for retriever tokens.
        ret_tokens (List[str]):                       Retriever token strings corresponding to ret_attribution.
        gen_attribution (List[float]):                Attribution values for generator tokens.
        gen_tokens (List[str]):                       Generator token strings corresponding to gen_attribution.
        ret_token_processor ((str) -> str, optional): Optional function applied to each retriever token before flattening.
        gen_token_processor ((str) -> str, optional): Optional function applied to each generator token before flattening.

    Returns:
        Tuple[List[str], np.ndarray, np.ndarray]:
            A tuple `(out_tokens, out_ret_attr, out_gen_attr)`, where ...
            - `out_tokens` is a list of grouped character sequences (strings).
            - `out_ret_attr` is a 1-D numpy array of normalized retriever attributions per group (sums to 1).
            - `out_gen_attr`is a 1-D numpy array of normalized generator attributions per group (sums to 1).

    Raises:
        ValueError: if no overlapping substring can be found between the flattened
                    retriever and generator texts.

    Example:
        >>> ret_attr = [0.1, 0.2]
        >>> ret_toks = ["Hello", " world"]
        >>> gen_attr = [0.05, 0.25]
        >>> gen_toks = ["Hello", " world"]
        >>> match_token_attributions(ret_attr, ret_toks, gen_attr, gen_toks)
        (['Hello world'], array([0.33333333, 0.66666667]), array([0.16666667, 0.83333333]))
    """

    # flatten retreiver input:
    ret_attr, ret_txt = flatten_token_attributions(
        ret_attribution, ret_tokens,
        token_processor = ret_token_processor
    )

    # flatten generator input:
    gen_attr, gen_txt = flatten_token_attributions(
        gen_attribution, gen_tokens,
        token_processor = gen_token_processor
    )

    # find position of generator text in retriever text:
    offset = gen_txt.find(ret_txt)
    if offset >= 0:
        gen_txt  = gen_txt[offset:offset+len(ret_txt)]
        gen_attr = gen_attr[offset:offset+len(ret_txt)]

    else:
        # if not found: find position of retriever query in generator query:
        offset = ret_txt.find(gen_txt)
        if offset >= 0:
            ret_txt  = ret_txt[offset:offset+len(gen_txt)]
            ret_attr = ret_attr[offset:offset+len(gen_txt)]

        # if still not found raise ValueError:
        else: raise ValueError()

    assert gen_txt == ret_txt

    # combine:
    out_tokens   = [gen_txt[0]]
    out_gen_attr = [gen_attr[0]]
    out_ret_attr = [ret_attr[0]]
    for i in range(1, len(gen_txt)):
        if gen_attr[i] != out_gen_attr[-1] or ret_attr[i] != out_ret_attr[-1]:
            out_tokens.append('')
            out_gen_attr.append(gen_attr[i])
            out_ret_attr.append(ret_attr[i])

        out_tokens[-1] += gen_txt[i]

    # normalize:
    out_ret_attr = np.array(out_ret_attr)
    out_ret_attr /= out_ret_attr.sum()

    out_gen_attr = np.array(out_gen_attr)
    out_gen_attr /= out_gen_attr.sum()

    return out_tokens, out_ret_attr, out_gen_attr

def bootstrap_ci(data, num_samples=1000, confidence_level=0.95):
    """Compute confidence intervalls for the mean of `data`. The function performs
    `num_samples` resamples of `data` with replacement, computes the mean of each
    resample, and returns the lower and upper percentiles of those bootstrap means
    that correspond to the specified `confidence_level` (percentile method).

    Args:
        data (array-like):                  1-D sequence of numeric observations.
        num_samples (int, optional):        Number of bootstrap resamples to draw. Default is 1000.
        confidence_level (float, optional): Confidence level for the interval, between 0 and 1. Default is 0.95.

    Returns:
        tuple: (lower_bound, upper_bound) giving the percentile-based bootstrap confidence interval
               for the sample mean corresponding to the requested confidence level.
    """

    samples = np.random.choice(data, (num_samples, len(data)), replace=True)
    means = np.mean(samples, axis=1)
    lower_bound = np.percentile(means, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(means, (1 + confidence_level) / 2 * 100)
    return float(lower_bound), float(upper_bound)

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