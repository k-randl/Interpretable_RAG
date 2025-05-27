import torch
from torch.nn.functional import softmax
from transformers import PreTrainedModel, AutoTokenizer
from typing import Union, List

#=======================================================================#
# Generic Model Class:                                                  #
#=======================================================================#

def ExplainableGenerator(T:type):
    # make sure T is derived from PreTrainedModel:
    assert issubclass(T, PreTrainedModel)

    # generic class definition:
    class _ExplainableGenerator(T):
        def __init__(self, config, *inputs, **kwargs):
            super().__init__(config, *inputs, **kwargs)
            self.tokenizer   = AutoTokenizer.from_pretrained(config.name_or_path)
            self._explain    = False
            self._exp_probs  = []
            self._gen_probs  = []

        def forward(self, *args, **kwargs):
            # get token probabilities:
            p = super().forward(*args, **kwargs)

            # save token probabilities:
            if self._explain: self._exp_probs.append(softmax(p, dim=-1))
            else:             self._gen_probs.append(softmax(p, dim=-1))

            # return token probabilities:
            return p
        
        def generate(self, *args, **kwargs):
            # deactivate explanation mode:
            self._explain    = False

            # reset token probabilities:
            self._gen_probs  = []

            # generate:
            return super().generate(*args, **kwargs)

        def compare(self, inputs:List[str], outputs:Union[List[str], torch.LongTensor], **kwargs):
            '''Calculates the probability `p(outputs) = p(t_0) * p(t_1|t_0) * ... * p(t_n|t_0...t_(n-1))` of
            a specific output `outputs = [t_0, t_1, ..., t_n]` to happen given an input prompt `inputs`.

            Args:
                inputs:     Iterable of tokens `t_i` in the input propmt.
                outputs:    Iterable of tokens `t_i` for which to compute the probability.

            Returns:        Token probabilities of the last next-token-prediction step with shape = (bs,)'''

            # get batch size:
            assert (len(inputs) == len(outputs)) or (len(outputs) == 1)
            single_output = len(outputs) == 1
            bs = len(inputs)

            # activate explanation mode:
            self._explain = True

            # reset token probabilities:
            self._exp_probs = []

            # convert string to Iterable of tokens:
            if isinstance(outputs, str):
                outputs = self.tokenizer(outputs, add_special_tokens=False, return_attention_mask=False).input_ids

            # tokenize input:
            inputs = self.tokenizer(inputs, truncation=True, return_tensors='pt')
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            # calculate p(t_0):
            self.forward(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            # update inputs:
            if single_output: input_ids = torch.concatenate((input_ids, torch.full((bs, 1), outputs[0], device=input_ids.device, dtype=input_ids.dtype)), dim=-1)
            else:             input_ids = torch.concatenate((input_ids, outputs[:,0].to(input_ids)), dim=-1)
            attention_mask = torch.concatenate((attention_mask, torch.full((bs, 1), 1, device=self._x.device, dtype=self._x.dtype)), dim=-1)

            # p(outputs) = p(t_0) * p(t_1|t_0) * ... * p(t_1|t_0...t_(j-1)):
            for t in outputs[1:]:
                self.forward(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

                # update inputs:
                if single_output: input_ids = torch.concatenate((input_ids, torch.full((bs, 1), outputs[0], device=input_ids.device, dtype=input_ids.dtype)), dim=-1)
                else:             input_ids = torch.concatenate((input_ids, outputs[:,0].to(input_ids)), dim=-1)
                attention_mask = torch.concatenate((attention_mask, torch.full((bs, 1), 1, device=self._x.device, dtype=self._x.dtype)), dim=-1)

            return self._exp_probs

    return _ExplainableGenerator