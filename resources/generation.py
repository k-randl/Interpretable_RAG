import torch
import numpy as np
from scipy.special import comb
from sklearn.linear_model import LinearRegression
from transformers import PreTrainedModel, PreTrainedTokenizer , AutoTokenizer
from numpy.typing import NDArray
from typing import Union, List, Dict, Tuple, Optional, Literal
import tqdm
import pickle

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

def plot_shap_attributions(shap_values:NDArray[np.float_], tokens:List[str], *, normalize:bool=True, cmap:str='tab10') -> None:
    """
    Plot stacked SHAP attributions for multiple documents as positive and negative bar segments.

    Args:
        shap_values (NDArray[np.float_]): A 2D array of SHAP values with shape `(num_documents, num_tokens)`, where each row represents a document and each column corresponds to a token's attribution score.
        tokens (List[str]):               A list of token strings corresponding to the SHAP values. Length must match the number of columns in `shap_values`.
        normalize (bool):                 If `True`, normalizes the SHAP values across tokens so that the sum of absolute values equals 1 (default=`True`).
        cmap (str):                       The name of a matplotlib colormap used for highlighting.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Normalize so that values sum to 1 (if enabled):
    if normalize:
        shap_values = shap_values / np.abs(shap_values).sum(axis=0)

    # Get colormap:
    colors = cm.get_cmap(cmap)

    p_pos = 0.
    p_neg = 0.
    for i, p_doc in enumerate(shap_values):
        p_doc_pos = np.maximum(p_doc, 0)
        plt.bar(range(len(p_doc)), p_doc_pos, bottom=p_pos, color=colors(i), label=f'Document {i+1:d}')
        p_pos += p_doc_pos

        p_doc_neg = np.minimum(p_doc, 0)
        plt.bar(range(len(p_doc)), p_doc_neg, bottom=p_neg, color=colors(i))
        p_neg += p_doc_neg

    # Tick and axis settings:
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    if normalize:
        plt.ylim(-1, 1)

    # Clean frame (keep only left spine):
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add horizontal grid line at zero:
    plt.axhline(y=0, color='black', linewidth=0.5)

    # Add vertical grid lines at each tick:
    for tick_loc in range(len(tokens)):
        plt.axvline(x=tick_loc, color='lightgray', linewidth=0.5, zorder=0)

    plt.legend()
    plt.tight_layout()
    plt.show()

def highlight_dominant_passages(shap_values:NDArray[np.float_], tokens:List[str], *, threshold:float=.5, cmap:str='tab10') -> str:
    """
    Highlights tokens in a text sequence where a single document contributes
    at least `threshold` of the total positive SHAP value at that token.

    Args:
        shap_values (NDArray[np.float_]): A 2D array of SHAP values with shape `(num_documents, num_tokens)`,
                                          where each row represents a document and each column corresponds to a token's attribution score.
        tokens (List[str]):               A list of token strings corresponding to the SHAP values.
                                          Length must match the number of columns in `shap_values`.
        threshold (float):                Minimum attribution for highlighting in the intervall `[0., 1.]` (default: `0.5`).
        cmap (str):                       The name of a matplotlib colormap used for highlighting.

    Returns:
        An HTML-formatted string with spans highlighting dominant SHAP regions.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    #import matplotlib.colormaps as cm
    from IPython.display import display, HTML

    # extract only on positive contributions:
    shap_values_pos = np.maximum(shap_values, 0)
    num_docs, num_tokens = shap_values_pos.shape
    token_docs = -np.ones(num_tokens, dtype=int)

    for t in range(num_tokens):
        column = shap_values_pos[:, t]
        total = column.sum()
        if total > 0:
            dominant_doc = np.argmax(column)
            if column[dominant_doc] / total >= threshold:
                token_docs[t] = dominant_doc

    # Prepare color map
    cmap = cm.get_cmap(cmap)
    rgb_colors = [mcolors.to_hex(cmap(i)) for i in range(num_docs)]

    # Build highlighed HTML:
    html_tokens = []
    for tok, doc in zip(tokens, token_docs):
        if doc >= 0: html_tok = f'<span style="background-color:{rgb_colors[doc]}; padding:2px; border-radius:3px;">{tok}</span>'
        else:        html_tok = tok
        html_tokens.append(html_tok)

    html_response = ' '.join(html_tokens)

    # Build legend:
    html_legend   = ' '.join([f'<i style="background-color:{c}; padding:2px; border-radius:3px;">Document {i+1:d}</i>' for i, c in enumerate(rgb_colors)])
    
    # Build, display, and return final HTML:
    html_str = (
        '<table>\n' +
        '<tr style="border-top: 1px solid">\n' +
        '<td style="text-align:right; vertical-align:top"><b style="line-height:2">Response:</b></td>\n' +
        '<td style="text-align:left; vertical-align:top"><div style="line-height:2">' + html_response + '</div></td>\n' +
        '</tr>\n' +
        '<tr style="border-top: 1px solid">\n' +
        '<td style="text-align:right; vertical-align:top"><i style="line-height:2">Legend:</i></td>\n' +
        '<td style="text-align:left; vertical-align:top"><div style="line-height:2">' + html_legend + '</div></td>\n' +
        '</tr>\n' +
        '</table>\n'
    )
    display(HTML(html_str))
    return html_str

#=======================================================================#
# Generic Model Class:                                                  #
#=======================================================================#

def ExplainableAutoModelForGeneration(T:type):
    # make sure T is derived from PreTrainedModel:
    assert issubclass(T, PreTrainedModel)

    # generic class definition:
    class _ExplainableAutoModelForGeneration(T):
        def __init__(self, config, *inputs, **kwargs):
            super().__init__(config, *inputs, **kwargs)
            self.tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained(config.name_or_path)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self._explain:bool = False
            self._exp_probs:List[torch.Tensor] = []
            self._gen_probs:torch.Tensor = []
            self._gen_output = None
            self._shap_cache = None
            self._shap_precise = None
            self.all_top_scores = None
            self.all_top_tokens = None
            

        #===============================================================#
        # Properties:                                                   #
        #===============================================================#
        #>> Pairs of properties named `gen_[name]_probs` and
        #>> `cmp_[name]_probs` will be automatically used by the
        #>> `get_shapley_values(...)` method!

        @property
        def gen_token_probs(self):
            '''Probability of each token in the original generation.'''
            # generate(...) needs to be run first:
            if len(self._gen_probs) == 0: return None

            # return probability of each token in the original generation:
            return np.array([ 
                [float(self._gen_probs[i, j, id]) for j, id  in enumerate(seq)]
                for i, seq in enumerate(self._gen_output)
            ])

        @property
        def cmp_token_probs(self):
            '''Probability of each token in the original generation given the compared input.'''
            # compare(...) needs to be run first:
            if len(self._exp_probs) == 0: return None

            # return probability of each token in the original generation:
            return [np.array([ 
                [float(t[i, j, id]) for j, id  in enumerate(seq)]
                for i, seq in enumerate(self._gen_output)
            ]) for t in self._exp_probs]


        @property
        def gen_sequence_prob(self):
            '''Total probability of generating the original sequence.'''
            # generate(...) needs to be run first:
            if len(self._gen_probs) == 0: return None

            # return the multiplied probability of each token in the original generation:
            return np.prod([ 
                [float(self._gen_probs[i, j, id]) for j, id  in enumerate(seq)]
                for i, seq in enumerate(self._gen_output)
            ], axis=-1)

        @property
        def cmp_sequence_probs(self):
            '''Total probability of generating the original sequence given the compared input.'''
            # compare(...) needs to be run first:
            if len(self._exp_probs) == 0: return None

            # return the multiplied probability of each token in the original generation:
            return [np.prod([ 
                [float(t[i, j, id]) for j, id  in enumerate(seq)]
                for i, seq in enumerate(self._gen_output)
            ], axis=-1) for t in self._exp_probs]


        @property
        def gen_bow_probs(self):
            '''Accumulated probability of each token in the vocabualry of being generated given the original input.'''
            # generate(...) needs to be run first:
            if len(self._gen_probs) == 0: return None

            # return accumulated probability of each token in the vocabualry:
            return self._gen_probs.mean(dim=1).float().numpy()
        
        @property
        def cmp_bow_probs(self):
            '''Accumulated probability of each token in the vocabualry of being generated given the compared input.'''
            # compare(...) needs to be run first:
            if len(self._exp_probs) == 0: return None

            # return accumulated probability of each token in the vocabualry:
            return [t.mean(dim=1).float().numpy() for t in self._exp_probs]
        

        #@property
        def gen_nucleus_probs(self, p:float=0.9):
            '''Accumulated probability of each token in the vocabualry of being generated given the original input.'''
            # generate(...) needs to be run first:
            if len(self._gen_probs) == 0: return None

            # return accumulated probability of each token in the vocabualry:
            return _nucleus_sampling(self._gen_probs.float(),p=p).mean(dim=1).numpy()

        #@property
        def cmp_nucleus_probs(self, p:float=0.9):
            '''Accumulated probability of each token in the vocabualry of being generated given the compared input.'''
            # compare(...) needs to be run first:
            if len(self._exp_probs) == 0: return None

            # return accumulated probability of each token in the vocabualry:
            return [_nucleus_sampling(t.float(),p=p).mean(dim=1).numpy() for t in self._exp_probs]

        #===============================================================#
        # Methods:                                                      #
        #===============================================================#

        def forward(self, *args, **kwargs):
            # get token probabilities:
            outputs = super().forward(*args, **kwargs)

            # save token probabilities:
            #if self._explain: self._exp_probs[-1].append(softmax(outputs['logits'][:,-1:,:].detach().cpu(), dim=-1))
            #else:             self._gen_probs.append(softmax(outputs['logits'][:,-1:,:].detach().cpu(), dim=-1))
            if self._explain: self._exp_probs[-1].append(outputs.logits[:,-1:,:].detach().cpu())
            else:             self._gen_probs.append(outputs.logits[:,-1:,:].detach().cpu())

            # return token probabilities:
            return outputs

        def generate(self, inputs:List[str], **kwargs) -> List[str]:
            '''Generates continuations of the passed input prompt(s).

            Args:
                inputs:             The string(s) used as a prompt for the generation.
                generation_config:  The generation configuration to be used as base parametrization for the generation call.
                stopping_criteria:  Custom stopping criteria that complements the default stopping criteria built from arguments and ageneration config.
                kwargs:             Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs.

            Returns:
                List of generated strings.
            '''
            # tokenize inputs:
            inputs = self.tokenizer(inputs, truncation=True, return_tensors='pt')

            # deactivate explanation mode:
            self._explain    = False

            # reset token probabilities:
            self._gen_probs  = []
            self._exp_probs  = []

            # generate:
            self._gen_output = super().generate(**inputs, **kwargs).sequences[:, inputs.input_ids.shape[-1]:]

            # finalize probabilities:
            self._gen_probs  = torch.concatenate(self._gen_probs, dim=1)

            # return generated text:
            return self.tokenizer.batch_decode(self._gen_output)

        def compare(self, inputs:List[str], outputs:Optional[Union[List[str], torch.LongTensor]]=None, batch_size:int=1, **kwargs) -> torch.LongTensor:
            '''Calculates the probability `p(outputs) = p(t_0) * p(t_1|t_0) * ... * p(t_n|t_0...t_(n-1))` of
            a specific output `outputs = [t_0, t_1, ..., t_n]` to happen given an input prompt `inputs`.

            Args:
                inputs:     List of input propmts. If `outputs` is specified, `compare(...)` calculates the
                            probability `p(outputs) = p(t_0) * p(t_1|t_0) * ... * p(t_n|t_0...t_(n-1))` for each
                            token in `outputs = [t_0, t_1, ..., t_n]` given ``. Otherwise, it calculates the
                            unconditional probability (similar to `generate(...)`).
                outputs:    List of tokens `t_i` or (strings containing those) for which to compute the probability
                            (optional).
                batch_size: Batch size. Ignored if `len(inputs) > 1` or `outputs` not specified (optional).

            Returns:
                Tensor of generated token ids .
            '''

            if batch_size < 1:
                raise ValueError(f'Parameter batch_size must be a positive integer but got {batch_size:d}.')

            if outputs is None:
                if batch_size > 1:
                    print('WARNING: when outputs is not specified the parameter batch_size is ignored.')

                return self.__compare_unconditional(inputs=inputs, **kwargs)
            
            else: return self.__compare_conditional(inputs=inputs, outputs=outputs, batch_size=batch_size, **kwargs)

        def __compare_unconditional(self, inputs:List[str], **kwargs) -> torch.LongTensor:
            # tokenize inputs:
            inputs = self.tokenizer(inputs, truncation=True, return_tensors='pt')

            # deactivate explanation mode:
            self._explain    = True

            # reset token probabilities:
            self._exp_probs.append([])

            # generate:
            output = super().generate(**inputs, **kwargs).sequences[:, inputs.input_ids.shape[-1]:]

            # finalize probabilities:
            self._exp_probs[-1] = torch.concatenate(self._exp_probs[-1], dim=1)

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
            self._exp_probs.append([])

            # convert string to Iterable of tokens:
            if isinstance(outputs[0], str):
                outputs = self.tokenizer(outputs, add_special_tokens=False, return_attention_mask=False, return_tensors='pt').input_ids

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
            cur_len = input_ids.shape[1]
            #model_kwargs = self._get_initial_cache_position(input_ids, **model_kwargs)
            model_kwargs = self._get_initial_cache_position(
                            seq_length=input_ids.shape[1],          # length of the prompt
                            device=input_ids.device,               # usually cuda:0 / cpu
                            model_kwargs=model_kwargs,             # the whole dict, **not** unpacked
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
                self._exp_probs[-1] = torch.concatenate(self._exp_probs[-1], dim=0).transpose(0,1)

            else: self._exp_probs[-1] = torch.concatenate(self._exp_probs[-1], dim=1)

            # split batch in elements if multiple inputs for the same output.
            if not single_input and single_output:
                self._exp_probs.extend([t.unsqueeze(0) for t in self._exp_probs.pop(-1)])

            # return generated tokens:
            return torch.argmax(self._exp_probs[-1], dim=-1)


        def explain_generate(self, query:str, contexts:List[str], *, batch_size:int=32, max_samples:Union[int, Literal['inf', 'auto']]='auto', system:Optional[str]=None, **kwargs):
            """
            Generates continuations of the passed input prompt(s) as well as perturbations for all retrieved documents.

            Args:
                query (str):       The user's query.
                contexts (list):   A list of strings representing the retrieved documents.
                batch_size (int):  The batch size for generating perturbations (default: `32`).
                max_samples (int): Maximum number of samples used for computing SHAP atribution values.
                                   If `2**len(contexts) <= max_samples`, this automatically computes the precise SHAP values instead of kernel SHAP approximations.
                                   If `inf` is passed, always computes the precise SHAP values.
                                   If `auto` is passed, `max_samples` get's the same value as `batch_size` (default: `auto`).  
                system (str):      An optional system prompt.

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

            # calculate number of samples needed for precise calculation:
            n = 2 ** len(contexts)
            if   max_samples == 'auto': max_samples = batch_size
            elif max_samples == 'inf':  max_samples = n
            perturbed_rag_prompts = [None] * min(n, max_samples)

            # generate prompts:
            if max_samples >= n:
                # generate prompts for perturbed inputs (precise SHAP values):
                perturbed_rag_prompts[-1] = complete_rag_prompt
                self._shap_cache   = np.array(self.__generate_permutations(query, contexts, perturbed_rag_prompts, n-1, system=system))
                self._shap_precise = True

            elif max_samples < n:
                # sample prompts for kernel SHAP:
                self._shap_cache   = self.__generate_sample(query, contexts, perturbed_rag_prompts, max_samples, n-1, system=system)
                self._shap_precise = False

            else: raise ValueError(f'Unknown value for parameter `max_samples`: {max_samples}')

            # generate comparison output:
            num_batches = int(np.ceil(len(perturbed_rag_prompts[:-1]) / batch_size))
            for i in range(num_batches):
                # print batch number:
                if num_batches > 1: print(f'Batch {i+1:d} of {num_batches:d}:')

                # get prompts of this batch:
                prompts_batch = perturbed_rag_prompts[:-1][i * batch_size:(i+1) * batch_size]

                # generate probabilities:
                self.compare(
                    [self.tokenizer.apply_chat_template(prmpt, tokenize=False) for prmpt in prompts_batch],
                    output,
                    **kwargs
                )

                # print empty line:
                if num_batches > 1: print()

            return output

        def __generate_permutations(self, query:str, contexts:List[str], perturbed_rag_prompts:List[str], index:int, *, system:Optional[str]=None):
            # create perturbed prompt if necessary:
            if perturbed_rag_prompts[index] is None:
                perturbed_rag_prompts[index] = create_rag_prompt(query, contexts, system=system)

            # break on empty set:
            if index == 0: return [(index,)]

            # calculate possible permutations:
            i, m = 0, 1
            permutations = []
            while i < len(contexts):
                if index & m:
                    child_permutations = self.__generate_permutations(query, contexts[:i]+contexts[i+1:],
                        perturbed_rag_prompts=perturbed_rag_prompts,
                        index=index & ~m,
                        system=system
                    )
                    permutations.extend([prm + (index,) for prm in child_permutations])
                    i += 1

                m = m << 1

            return permutations

        def __generate_sample(self, query:str, contexts:List[str], perturbed_rag_prompts:List[str], num_samples:int, num_perturbations:int, *, system:Optional[str]=None):
            # take sample of `num_samples` unique sets of documents (including empty and full):
            sample = np.empty(num_samples, dtype=int)
            sample[0]    = 0
            sample[-1]   = num_perturbations-1
            sample[1:-1] = np.random.choice(np.arange(1,num_perturbations-1), size=(num_samples-2), replace=False)

            # generate corresponding prompts:
            features = np.zeros((num_samples, len(contexts)), dtype=float)
            for i, index in enumerate(sample[:-1]):
                # translate index to set:
                j, m = 0, 1
                current_contexts = []
                while j < len(contexts):
                    if index & m:
                        current_contexts.append(contexts[j])
                        features[i, j] = 1.
                    j += 1
                    m = m << 1

                # generate prompt:
                perturbed_rag_prompts[i] = create_rag_prompt(query, current_contexts, system=system)

            return features
            

        def get_shapley_values(self, aggregation:Literal['token', 'sequence', 'bow', 'nucleus']='token', **kwargs) -> NDArray[np.float_]:
            """
            Generates Shapley feature attribution values for the chose aggregation method.

            Args:
                aggregation (str):  Aggregation method for probabilities (default: `'token'`).

            Returns:
                A `numpy.ndarray` containing the Shapley values.
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
            if self._shap_precise: return self._get_shapley_values_precise(probs)
            else: return self._get_shapley_values_kernel(probs)

        def _get_shapley_values_precise(self, probs:NDArray[np.float_]) -> NDArray[np.float_]:
            # Get the shape of the permutations matrix: (num_permutations, num_documents)
            num_permutations, num_docs = self._shap_cache.shape

            # Initialize array to store marginal contributions for each permutation step
            p_marginal = np.empty(self._shap_cache.shape + probs[0].shape, dtype=probs[0].dtype)

            # For each permutation, calculate the marginal contributions
            for i in range(num_permutations):
                # First document's contribution is its raw probability
                p_marginal[i, 0] = probs[self._shap_cache[i, 0]]

                for j in range(1, num_docs):
                    # Difference in output probability when adding the j-th document
                    prev = probs[self._shap_cache[i, j - 1]]
                    curr = probs[self._shap_cache[i, j]]
                    p_marginal[i, j] = curr - prev

            # Encode the permutation transitions using bitwise operations
            new_doc = np.empty(self._shap_cache.shape, dtype=np.int16)

            # First position is zeroed (i.e., no document yet included)
            new_doc[:, 0] = 0

            for j in range(1, num_docs):
                # Bitwise difference to capture which bit changed, then take log2
                prev = self._shap_cache[:, j - 1]
                curr = self._shap_cache[:, j]
                new_doc[:, j] = np.log2(curr & ~prev) + 1

            # Initialize SHAP value container: one entry per document
            p_shap = np.empty((num_docs,) + probs[0].shape, dtype=probs[0].dtype)

            # For each document, aggregate all matching marginal contributions
            for j in range(num_docs):
                # Mean over all marginal contributions that map to document j
                p_shap[j] = p_marginal[new_doc == j].mean(0)

            # Return SHAP values for all but the baseline (first one)
            return p_shap[1:]

        def _get_shapley_values_kernel(self, probs:NDArray[np.float_]) -> NDArray[np.float_]:
            # fit a linear regressor using the SHAP kernel:
            lr = LinearRegression()
            x = self._shap_cache[1:-1]
            y = np.stack(probs[1:-1])
            w = [(len(z)-1) / (comb(len(z), sum(z)) * sum(z) * -sum(z-1))
                for z in x]
            lr.fit(x, y, w)

            # attributions are estimated SHAP values:
            attributions = lr.coef_.T

            # rescale attributions to fit prediction:
            return attributions / np.abs(attributions.sum(axis=0)) * (probs[-1] - probs[0])
        
        def _extract_top_exp_prob(self, top_k = 200):
            """
            Extracts the top-k probabilities and their corresponding tokens from the generated tensors.

            Args:
                top_k (int): The number of top probabilities to extract (default: 50).

            Returns:
                A tuple containing two tensors:
                - `permutation_top_vals`: A tensor of shape (n_perm, top_k) containing the top-k scores.
                - `permutation_top_tokens`: A tensor of shape (n_perm, top_k, 2) containing the step and token indices.
            """
            # ensure that we have at least one tensor to process:
            if len(self._exp_probs) == 0: return None, None

            # get the list of tensors:
            tensor_list = self._exp_probs
            all_top_scores = []           # list of (max_new_tokens, top_k)
            all_top_tokens = []           # list of (max_new_tokens, top_k)

            for scores in tensor_list:    # scores: (max_new_tokens, vocab_size)
                # torch.topk along the *token* dimension (dim=-1)
                vals, idx = torch.topk(scores, k=top_k, dim=-1)   # both (max_new_tokens, top_k)

                all_top_scores.append(vals)   # keep the scores
                all_top_tokens.append(idx)    # keep the token IDs

            # (optional) stack into big tensors: (n_perm, max_new_tokens, top_k)
            self.all_top_scores = torch.stack(all_top_scores).squeeze(1)
            self.all_top_tokens = torch.stack(all_top_tokens).squeeze(1)

        def save_values(self,path: str):
            """
            Saves the SHAP values and exp_probs and gen_probs to a file.

            Args:
                path (str): The path where the SHAP values should be saved.
            """
            #if self.all_top_scores is None or self.all_top_tokens is None:
            #    self._extract_top_exp_prob()
            #top_exp_probs, top_exp_tokens = self.all_top_scores, self.all_top_tokens
            
            data_to_save = {
                'generated_output': self._gen_output,
                'shap_precise': self._shap_precise,
                'shapley_values_tokens': self.get_shapley_values('token'),
                'shapley_values_passages' : self.get_shapley_values('sequence'),
                'shapley_values_bow': self.get_shapley_values('bow'),
                'shapley_values_nucleus': self.get_shapley_values('nucleus'),
            }
            with open(path, 'wb') as f:
                pickle.dump(data_to_save, f)        
            return data_to_save
    return _ExplainableAutoModelForGeneration