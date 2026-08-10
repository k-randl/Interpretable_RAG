import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Optional
from numpy.typing import NDArray
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_TEMPLATE = (
    "Use the following retrieved documents, ranked from highest "
    "to lowest relevance, to answer the user's query. "
    "Be thorough and accurate, and cite documents when useful. "
    "Keep the answer under 200 words."
    "\n\n{context}"
    "\n\nQuery: {query}"
)

def join_contexts(contexts:List[str], *, max_document_size:Optional[int]=None):
    # Apply size limit (build a new list instead of mutating the caller's `contexts`)
    if max_document_size is not None:
        contexts = [
            doc[:max_document_size - 3] + '...' if len(doc) > max_document_size else doc
            for doc in contexts
        ]

    # Format the context into a single message
    return "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(contexts)])


class ExternalAutoModel(ABC):
    '''Shared base class for black-box "external" (non-SHAP) generation-explanation
    methods (ContextCite, MIRAGE, ...) used as explainers with `AIPCForGeneration`.
    Provides prompt construction, plain generation, and teacher-forced probability
    scoring; subclasses only need to implement `.explain(...)`.
    '''

    def __init__(self, pretrained_model_name_or_path, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.generator = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)

    def _prompt_ids(self, query:str, contexts:List[str], *, max_document_size:Optional[int]=None) -> List[int]:
        messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(query=query, context=join_contexts(contexts, max_document_size=max_document_size))}]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer.encode(chat_prompt, add_special_tokens=False)

    def __call__(self, query:str, contexts:List[str], *, max_document_size:Optional[int]=None,
            max_new_tokens:int=256, **kwargs
        ) -> str:
        # default max_new_tokens instead of leaving it unset -- an unset value falls back to
        # the model's own (often very large) max_length and triggers HF's "max_new_tokens not
        # set" warning; 256 matches the length used elsewhere (e.g. MirageAutoModel.explain()'s
        # own default, and the AIPC ablation scripts' generation settings):
        prompt_ids = self._prompt_ids(query, contexts, max_document_size=max_document_size)
        input_ids  = torch.tensor([prompt_ids], device=self.generator.device)
        # a single, unpadded sequence -- the whole row is real content, so attention_mask is
        # trivially all-ones. Passing it explicitly (rather than letting HF try to infer it from
        # input_ids != pad_token_id) also avoids HF's "attention mask not set" warning and sidesteps
        # the hazard where pad_token_id == eos_token_id (e.g. Llama) makes that inference unsafe for
        # prompts containing legitimate mid-sequence eos tokens (e.g. chat-template turn separators):
        attention_mask = torch.ones_like(input_ids)

        output_ids = self.generator.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, **kwargs)[0, len(prompt_ids):]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def _token_probs(self, prompt_ids:List[int], output_ids:List[int], *, fast:bool=True) -> torch.Tensor:
        '''Per-token P(output_i | prompt + output[:i]).

        If `fast` is `True` (default), computed via a single forward pass over the whole
        concatenated (prompt + output) sequence. If `False`, computed iteratively -- one
        token at a time, reusing a KV cache -- matching the step-by-step decoding approach
        used by `ExplainableAutoModelForGeneration.gen_token_probs`/`cmp_token_probs`
        (src/Interpretable_RAG/generation.py) exactly, at the cost of `len(output_ids)+1`
        sequential forward passes instead of one. Empirically the two give nearly identical
        results on real prompts (within ~0.001), so `fast=True` is a safe default; `fast=False`
        is available in case they ever diverge meaningfully for some model/prompt.
        '''
        device = self.generator.device

        if fast:
            input_ids = torch.tensor([prompt_ids + output_ids], device=device)
            with torch.no_grad():
                logits = self.generator(input_ids).logits

            output_logits = logits[0, len(prompt_ids)-1:len(prompt_ids)+len(output_ids)-1]
            probs         = torch.softmax(output_logits.float(), dim=-1)
            ids           = torch.tensor(output_ids, device=probs.device)
            return probs.gather(-1, ids[:,None])[:,0]

        with torch.no_grad():
            # prefill: process the full prompt once, building the KV cache:
            out             = self.generator(torch.tensor([prompt_ids], device=device), use_cache=True)
            past_key_values = out.past_key_values
            next_logits     = out.logits[0, -1]

            probs = []
            for token_id in output_ids:
                probs.append(torch.softmax(next_logits.float(), dim=-1)[token_id])

                # feed just the single new token, reusing the cache:
                out             = self.generator(torch.tensor([[token_id]], device=device), past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                next_logits     = out.logits[0, -1]

        return torch.stack(probs)

    def compare(self, query:str, contexts:List[str], output:str, *, max_document_size:Optional[int]=None, fast:bool=True) -> float:
        '''Score the probability that `output` would be generated for the given
        query/contexts, via teacher-forced decoding. Used by AIPCForGeneration
        to measure how the probability of the original generation changes as
        context documents are perturbed.

        Args:
            query (str):                        The query.
            contexts (List[str]):               The (possibly perturbed) context documents.
            output (str):                       The fixed continuation to score.
            max_document_size (int, optional):  An optional size limit of context documents in characters.
            fast (bool, optional):              Decoding strategy passed through to `_token_probs`
                                                (default `True`: single forward pass; `False`: iterative
                                                with a KV cache, matching `ExplainableAutoModelForGeneration`
                                                exactly at the cost of `len(output_ids)+1` forward passes).

        Returns:
            float: Arithmetic mean of the per-token probabilities of `output`, matching
                the per-token-softmax + arithmetic-mean aggregation used by
                `gen_token_probs`/`cmp_token_probs` in `ExplainableAutoModelForGeneration`
                (src/Interpretable_RAG/generation.py), so scores from AIPCForGeneration
                and AIPCForGenerationFast are computed the same way.
        '''
        prompt_ids = self._prompt_ids(query, contexts, max_document_size=max_document_size)
        output_ids = self.tokenizer.encode(output, add_special_tokens=False)
        if len(output_ids) == 0: return 1.

        return self._token_probs(prompt_ids, output_ids, fast=fast).mean().item()

    @abstractmethod
    def explain(self, query:str, contexts:List[str], *, max_document_size:Optional[int]=None, **kwargs) -> NDArray[np.float32]:
        '''Return one relevancy score per document in `contexts`.'''
        raise NotImplementedError
