import torch
from torch.nn.functional import softmax
from transformers import PreTrainedModel, AutoTokenizer
from typing import Union, List, Tuple

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
            self._exp_output = None
            self._gen_output = None

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
            self._gen_output = super().generate(**inputs,**kwargs).sequences[:, inputs.input_ids.shape[-1]:]

            # finalize probabilities:
            self._gen_probs  = torch.concatenate(self._gen_probs, dim=1)

            # return generated text:
            return self.tokenizer.batch_decode(self._gen_output)

        def compare_unconditional(self, inputs:List[str]) -> List[str]:
            '''Generates continuatiations of the passed input prompts.

            Args:
                inputs:     List of input propmts.

            Returns:        List of generated strings.
            '''
            # tokenize inputs:
            inputs = self.tokenizer(inputs, truncation=True, return_tensors='pt')

            # deactivate explanation mode:
            self._explain    = True

            # reset token probabilities:
            self._exp_probs  = []

            # generate:
            self._exp_output = super().generate(**inputs).sequences[:, inputs.input_ids.shape[-1]:]

            # finalize probabilities:
            self._exp_probs  = torch.concatenate(self._exp_probs, dim=1)

            # return generated text:
            return self.tokenizer.batch_decode(self._exp_output)

        def compare_conditional(self, inputs:List[str], outputs:Union[List[str], torch.LongTensor], batch_size:int=1, **kwargs) -> List[str]:
            '''Calculates the probability `p(outputs) = p(t_0) * p(t_1|t_0) * ... * p(t_n|t_0...t_(n-1))` of
            a specific output `outputs = [t_0, t_1, ..., t_n]` to happen given an input prompt `inputs`.

            Args:
                inputs:     List of input propmts.
                outputs:    List of tokens `t_i` or (strings containing those) for which to compute the probability.
                batch_size: Batch size. Ignored if `len(inputs) > 1`.

            Returns:        List of generated strings .'''

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
            self._exp_probs = torch.concatenate(self._exp_probs, dim=1)

            # return generated tokens:
            self._exp_output = torch.argmax(self._exp_probs, dim=-1)
            return self.tokenizer.batch_decode(self._exp_output)

    return _ExplainableAutoModelForGeneration