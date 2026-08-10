import numpy as np
import torch
from typing import List, Optional
from numpy.typing import NDArray

from .external import ExternalAutoModel, PROMPT_TEMPLATE, join_contexts


def _document_char_spans(contexts:List[str], *, max_document_size:Optional[int]=None) -> List[tuple]:
    '''Character spans (start, end) of each document's CONTENT (excluding the
    "Document N:\\n" header) within `join_contexts(contexts, max_document_size=...)`.
    '''
    if max_document_size is not None:
        contexts = [
            doc[:max_document_size - 3] + '...' if len(doc) > max_document_size else doc
            for doc in contexts
        ]

    spans = []
    pos = 0
    for i, doc in enumerate(contexts):
        if i > 0: pos += 2  # the "\n\n" separator inserted between documents by join_contexts
        pos += len(f"Document {i+1}:\n")
        spans.append((pos, pos + len(doc)))
        pos += len(doc)
    return spans


class MirageAutoModel(ExternalAutoModel):
    '''Self-contained reimplementation of MIRAGE (https://github.com/Betswish/MIRAGE)'s
    context-attribution method, without depending on the `inseq`/`captum` libraries used
    by the original implementation. Reuses `ExternalAutoModel`'s prompt construction,
    `generate` and `compare` (both are architecture-agnostic and identical in spirit for
    any model wrapper), and only adds the MIRAGE-specific explanation method below.

    MIRAGE explains a generation in two stages:
      1. CTI (Context-sensitive Token Identification): find output tokens whose probability
         is substantially higher with the context included than without it (a contrastive
         probability-difference test), i.e. tokens the context actually influenced.
      2. CCI (Context Cue Identification): for each such token, compute a gradient-based
         saliency attribution of its probability with respect to the input embeddings, keep
         only the most salient input positions, and accumulate them across all CTI-selected
         tokens.

    The resulting per-input-token scores are aggregated into one score per context document
    using each document's known character span within the prompt.
    '''

    def __init__(self, pretrained_model_name_or_path, **kwargs):
        super().__init__(pretrained_model_name_or_path, **kwargs)
        # we only ever need gradients w.r.t. an input-embedding leaf tensor, never w.r.t.
        # the (billions of) model parameters, so freeze them to save memory/compute:
        self.generator.requires_grad_(False)

    def explain(self, query:str, contexts:List[str], *, max_document_size:Optional[int]=None,
            max_new_tokens:int=256,
            cti_std:float=0.0,
            cci_top_percent:float=5.0,
            max_cti_tokens:Optional[int]=None,
            **kwargs
        ) -> NDArray[np.float32]:
        '''Compute MIRAGE-style per-document relevancy scores for a query/context pair.

        Args:
            query, contexts, max_document_size:  As elsewhere.
            max_new_tokens (int):    Max length of the (greedily generated) reference completion
                                     that attributions are computed for.
            cti_std (float, optional):  CTI threshold, in standard deviations above the mean
                                        contrastive (with- vs. without-context) log-probability
                                        difference across output tokens, applied to its ABSOLUTE
                                        value -- an output token counts as context-sensitive whether
                                        context helps (raises its probability) or hurts (lowers it),
                                        matching the original implementation's `filter_rank_tokens`
                                        (`abs(score) >= threshold`), not just the "context helps" case.
                                        Matches `context_sensitivity_std_threshold` (default `0`) --
                                        not the separate, later `--CTI` CLI flag (default `1`), which
                                        only re-filters already-computed scores during citation
                                        formatting and has no equivalent in our single-pass design.
            cci_top_percent (float, optional):  For each CTI-selected token, prompt positions are
                                        kept if their saliency is within the top `cci_top_percent`%
                                        of the (min, max) RANGE of saliency values -- not the top
                                        `cci_top_percent`% by COUNT/rank -- and selection only
                                        competes among document/context token positions (positions
                                        in the instruction preamble, chat template, or query can
                                        never be attributed to a document, so they're excluded from
                                        competing for this budget). Matches `--CCI` (default `-5`,
                                        i.e. top 5% of the range) and the original implementation's
                                        `input_context_scores` scoping (context tokens only).
            max_cti_tokens (int, optional):  Upper bound on the number of CTI-selected tokens to run
                                        the (expensive) gradient-based CCI step for. If more tokens
                                        pass the CTI threshold, only the `max_cti_tokens` with the
                                        largest contrastive magnitude are used. `None` (default) runs
                                        every CTI-selected token, matching the original MIRAGE method
                                        exactly; a reference-implementation comparison found that a
                                        cap of 20 discarded over half the CTI-selected tokens on real
                                        samples, flattening the resulting per-document scores and
                                        likely explaining a low AIPC score in a prior ablation run.

        Returns:
            NDArray[np.float32]: One relevancy score per document in `contexts`.
        '''
        prompt_ids      = self._prompt_ids(query, contexts, max_document_size=max_document_size)
        contextless_ids = self._prompt_ids(query, [], max_document_size=max_document_size)

        # greedily generate the reference completion (with the full context). A single, unpadded
        # sequence has an all-ones attention_mask by construction; passing it explicitly avoids
        # HF's "attention mask not set" warning and the hazard of HF inferring it from
        # input_ids != pad_token_id, which is unsafe when pad_token_id == eos_token_id (e.g. Llama):
        input_ids      = torch.tensor([prompt_ids], device=self.generator.device)
        attention_mask = torch.ones_like(input_ids)
        gen_ids    = self.generator.generate(input_ids, attention_mask=attention_mask, do_sample=False, max_new_tokens=max_new_tokens)
        output_ids = gen_ids[0, len(prompt_ids):].tolist()
        if len(output_ids) == 0:
            return np.zeros(len(contexts), dtype=np.float32)

        # === CTI: contrastive (with- vs. without-context) log-probability of each output token ===
        logp_with    = self._teacher_forced_logprobs(prompt_ids, output_ids)
        logp_without = self._teacher_forced_logprobs(contextless_ids, output_ids)
        contrast     = (logp_with - logp_without).cpu().numpy()

        threshold = contrast.mean() + cti_std * contrast.std()
        abs_contrast = np.abs(contrast)
        cti_idx   = np.nonzero(abs_contrast >= threshold)[0]
        if len(cti_idx) == 0:
            cti_idx = np.array([int(np.argmax(abs_contrast))])
        if max_cti_tokens is not None and len(cti_idx) > max_cti_tokens:
            cti_idx = cti_idx[np.argsort(abs_contrast[cti_idx])[-max_cti_tokens:]]

        # === CCI: gradient-based saliency of each CTI token's probability w.r.t. input embeddings ===
        saliencies = self._cci_saliency(prompt_ids, output_ids, cti_idx)

        # map each prompt token to the document (if any) it belongs to, so the top-`cci_top_percent`
        # selection below only ever competes among document/context token positions:
        doc_idx_per_token  = self._document_token_map(query, contexts, prompt_ids, max_document_size=max_document_size)
        context_positions  = np.nonzero(doc_idx_per_token >= 0)[0]

        p            = cci_top_percent / 100.
        n_prompt     = len(prompt_ids)
        saliency_sum = np.zeros(n_prompt, dtype=np.float64)
        for saliency in saliencies.values():
            # keep positions whose saliency is within the top `cci_top_percent`% of the range
            # spanned by document/context positions only (matches the original's range-interpolated
            # threshold `(1-p)*max + p*min`, applied only to `input_context_scores`):
            candidate = saliency[context_positions]
            cci_threshold = (1 - p) * candidate.max() + p * candidate.min()
            keep = context_positions[candidate >= cci_threshold]
            saliency_sum[keep] += saliency[keep]

        # === aggregate per-token scores into per-document scores ===
        return self._aggregate_by_document(doc_idx_per_token, contexts, saliency_sum)

    def _teacher_forced_logprobs(self, prompt_ids:List[int], output_ids:List[int]) -> torch.Tensor:
        '''Per-token log P(output_i | prompt + output[:i]), reusing the shared
        `ExternalAutoModel._token_probs` forward pass.'''
        return self._token_probs(prompt_ids, output_ids).log()

    def _cci_saliency(self, prompt_ids:List[int], output_ids:List[int], cti_idx:NDArray[np.int_]) -> dict:
        '''L2-norm gradient-based saliency of log P(output[i] | prompt + output[:i]) with respect
        to each prompt-token embedding, for every token index `i` in `cti_idx`. Uses a single
        forward pass (with gradients enabled) and one `backward(retain_graph=True)` call per
        CTI token, reusing the same stored activations rather than re-running the forward pass.
        '''
        n_prompt = len(prompt_ids)
        full_ids = torch.tensor([prompt_ids + output_ids], device=self.generator.device)

        embed_layer = self.generator.get_input_embeddings()
        with torch.no_grad():
            embeds = embed_layer(full_ids).float()
        embeds.requires_grad_(True)

        saliencies = {}
        with torch.enable_grad():
            logits    = self.generator(inputs_embeds=embeds.to(self.generator.dtype)).logits
            log_probs = torch.log_softmax(logits[0, n_prompt-1:n_prompt+len(output_ids)-1].float(), dim=-1)

            for i in cti_idx:
                i = int(i)
                target_id = output_ids[i]
                score     = log_probs[i, target_id]

                if embeds.grad is not None: embeds.grad.zero_()
                score.backward(retain_graph=True)
                saliencies[i] = embeds.grad[0, :n_prompt].norm(dim=-1).detach().cpu().numpy().copy()

        return saliencies

    def _document_token_map(self, query:str, contexts:List[str], prompt_ids:List[int], *,
            max_document_size:Optional[int]=None
        ) -> NDArray[np.int64]:
        '''For each prompt token, the index (into `contexts`) of the document whose character
        span contains it, or `-1` if the token falls outside every document (e.g. it's part of
        the instruction preamble, chat template, or query text).
        '''
        context_str  = join_contexts(contexts, max_document_size=max_document_size)
        content      = PROMPT_TEMPLATE.format(query=query, context=context_str)
        messages     = [{"role": "user", "content": content}]
        chat_prompt  = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # _document_char_spans returns spans relative to `context_str` alone, so we need
        # BOTH offsets: where `context_str` sits within `content` (after PROMPT_TEMPLATE's
        # instruction preamble) and where `content` sits within the chat-templated prompt --
        # using only the latter put every span ~200 chars too early, into the preamble text
        # instead of the actual document content:
        offset = chat_prompt.index(content) + content.index(context_str)
        doc_spans = [(s + offset, e + offset) for s, e in _document_char_spans(contexts, max_document_size=max_document_size)]

        encoded = self.tokenizer(chat_prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded['offset_mapping'][:len(prompt_ids)]

        doc_idx_per_token = np.full(len(prompt_ids), -1, dtype=np.int64)
        for tok_idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= tok_start: continue  # special/empty-span tokens
            for doc_idx, (doc_start, doc_end) in enumerate(doc_spans):
                if doc_start <= tok_start < doc_end:
                    doc_idx_per_token[tok_idx] = doc_idx
                    break

        return doc_idx_per_token

    def _aggregate_by_document(self, doc_idx_per_token:NDArray[np.int64], contexts:List[str],
            token_scores:NDArray[np.float64]
        ) -> NDArray[np.float32]:
        '''Sum per-token `token_scores` (aligned with `doc_idx_per_token`) within each document.'''
        scores = np.zeros(len(contexts), dtype=np.float32)
        mask   = doc_idx_per_token >= 0
        np.add.at(scores, doc_idx_per_token[mask], token_scores[mask])
        return scores
