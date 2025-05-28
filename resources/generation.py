import torch
import numpy as np
from torch.nn.functional import softmax
from transformers import PreTrainedModel, AutoTokenizer
from typing import Union, List, Tuple, Optional

#=======================================================================#
# Helper Functions:                                                     #
#=======================================================================#

def _to_batch(input_ids:torch.Tensor,
              attention_mask:torch.Tensor,
              output_ids:Union[List[int],torch.Tensor],
              pad_token_id:int,
              batch_size:int) -> Tuple[torch.Tensor, torch.Tensor]:
    
    input_size = input_ids.shape[1]

    batched_inputs_ids = torch.full(
        (batch_size, input_size + batch_size),
        pad_token_id,
        device=input_ids.device,
        dtype=input_ids.dtype
    )
    batched_attention_mask = torch.zeros_like(batched_inputs_ids)

    for i in range(batch_size):
        batched_inputs_ids[-input_size-i-1:-i-1] = input_ids
        batched_inputs_ids[-i-1:] = output_ids[:i]

        batched_attention_mask[-input_size-i-1:-i-1] = attention_mask
        batched_attention_mask[-i-1:] = 1

    return batched_inputs_ids, batched_attention_mask

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
            self.tokenizer   = AutoTokenizer.from_pretrained(config.name_or_path)
            self._explain    = False
            self._exp_probs  = []
            self._gen_probs  = []
            self._gen_output = None

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
            return np.array([ 
                [float(self._exp_probs[i, j, id]) for j, id  in enumerate(seq)]
                for i, seq in enumerate(self._gen_output)
            ])

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
        def cmp_sequence_prob(self):
            '''Total probability of generating the original sequence given the compared input.'''
            # compare(...) needs to be run first:
            if len(self._exp_probs) == 0: return None

            # return the multiplied probability of each token in the original generation:
            return np.prod([ 
                [float(self._exp_probs[i, j, id]) for j, id  in enumerate(seq)]
                for i, seq in enumerate(self._gen_output)
            ], axis=-1)

        @property
        def gen_bow_probs(self):
            '''Accumulated probability of each token in the vocabualry of being generated given the original input.'''
            # generate(...) needs to be run first:
            if len(self._gen_probs) == 0: return None

            # return accumulated probability of each token in the vocabualry:
            return self._gen_probs.mean(dim=1).numpy()

        @property
        def cmp_bow_probs(self):
            '''Accumulated probability of each token in the vocabualry of being generated given the compared input.'''
            # compare(...) needs to be run first:
            if len(self._exp_probs) == 0: return None

            # return accumulated probability of each token in the vocabualry:
            return self._exp_probs.mean(dim=1).numpy()

        def forward(self, *args, **kwargs):
            # get token probabilities:
            p = super().forward(*args, **kwargs)

            # save token probabilities:
            if self._explain: self._exp_probs.append(softmax(p.logits[:,-1:,:].detach().cpu(), dim=-1))
            else:             self._gen_probs.append(softmax(p.logits[:,-1:,:].detach().cpu(), dim=-1))

            # return token probabilities:
            return p
        
        def generate(self, inputs:Union[List[str], str], **kwargs) -> List[str]:
            '''Generates continuatiations of the passed input prompt(s).

            Args:
                inputs:             The string(s) used as a prompt for the generation.
                generation_config:  The generation configuration to be used as base parametrization for the generation call.
                stopping_criteria:  Custom stopping criteria that complements the default stopping criteria built from arguments and ageneration config.
                kwargs:             Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs.

            Returns:                List of generated strings.
            '''
            # tokenize inputs:
            inputs = self.tokenizer(inputs, truncation=True, return_tensors='pt')

            # deactivate explanation mode:
            self._explain    = False

            # reset token probabilities:
            self._gen_probs  = []

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

            Returns:        Tensor of generated token ids .'''

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
            self._exp_probs  = []

            # generate:
            output = super().generate(**inputs, **kwargs).sequences[:, inputs.input_ids.shape[-1]:]

            # finalize probabilities:
            self._exp_probs  = torch.concatenate(self._exp_probs, dim=1)

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
            self._exp_probs = []

            # convert string to Iterable of tokens:
            if isinstance(outputs[0], str):
                outputs = self.tokenizer(outputs, add_special_tokens=False, return_attention_mask=False,return_tensors='pt').input_ids

            # tokenize input:
            inputs = self.tokenizer(inputs, truncation=True, return_tensors='pt')
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            # batch processing in case of single input:
            if single_input: input_ids, attention_mask = _to_batch(
                input_ids, attention_mask, outputs, self.tokenizer.pad_token_id, batch_size
            )

            with torch.no_grad():

                # calculate p(t_0):
                self.forward(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

                # update inputs:
                if single_input:    nxt = torch.unsqueeze(outputs[0,:batch_size], dim=1).to(input_ids)
                elif single_output: nxt = torch.full((batch_size, 1), outputs[0,0], device=input_ids.device, dtype=input_ids.dtype)
                else:               nxt = torch.unsqueeze(outputs[:,0], dim=1).to(input_ids)

                input_ids = torch.concatenate((input_ids, nxt), dim=-1)
                attention_mask = torch.concatenate((attention_mask, torch.full((batch_size, 1), 1, device=input_ids.device, dtype=input_ids.dtype)), dim=-1)

                # p(outputs) = p(t_0) * p(t_1|t_0) * ... * p(t_1|t_0...t_(j-1)):
                for i in range(1, outputs.shape[1]):
                    self.forward(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

                    # update inputs:
                    if single_input:    nxt = torch.unsqueeze(outputs[0,(i*batch_size):((i+1)*batch_size)], dim=1).to(input_ids)
                    elif single_output: nxt = torch.full((batch_size, 1), outputs[0,i], device=input_ids.device, dtype=input_ids.dtype)
                    else:               nxt = torch.unsqueeze(outputs[:,i], dim=1).to(input_ids)

                    input_ids = torch.concatenate((input_ids, nxt), dim=-1)
                    attention_mask = torch.concatenate((attention_mask, torch.full((batch_size, 1), 1, device=input_ids.device, dtype=input_ids.dtype)), dim=-1)

            # finalize probabilities:
            self._exp_probs = torch.concatenate(self._exp_probs, dim=0 if single_input else 1)

            # return generated tokens:
            return torch.argmax(self._exp_probs, dim=-1)

    return _ExplainableAutoModelForGeneration