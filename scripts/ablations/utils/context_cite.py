import re
from context_cite import ContextCiter
from context_cite.context_partitioner import SimpleContextPartitioner

from typing import List, Optional
from numpy.typing import NDArray

from .external import ExternalAutoModel, PROMPT_TEMPLATE, join_contexts


class DocumentContextPartitioner(SimpleContextPartitioner):
    def __init__(self, context:str) -> None:
        super().__init__(context)
        # "Document N:" is followed by a newline, but "Query:" is followed by a space
        # (matching PROMPT_TEMPLATE / src/Interpretable_RAG/generation.py's create_rag_prompt).
        self._split_expression = re.compile(r'\n\n(?:(?:Document)\s\d+:\n|(?:Query):\s)')
        self._cache = {}

    def split_context(self) -> None:
        parts      = self._split_expression.split(self.context)
        separators = self._split_expression.findall(self.context)
        self._cache["instruction"] = parts[0] + separators[0]
        self._cache["parts"]       = parts[1:-1]
        self._cache["separators"]  = separators[:-1]
        self._cache["query"]       = separators[-1] + parts[-1]

    @property
    def instruction(self) -> str:
        if self._cache.get("instruction") is None:
            self.split_context()
        return self._cache["instruction"]

    @property
    def query(self) -> str:
        if self._cache.get("query") is None:
            self.split_context()
        return self._cache["query"]

    def get_context(self, mask:Optional[NDArray]=None) -> str:
        return self.instruction + super().get_context(mask) + self.query


class ContextCiteAutoModel(ExternalAutoModel):
    def explain(self, query:str, contexts:List[str], *, max_document_size:Optional[int]=None, **kwargs):
        # build the fully-formatted prompt text up front, so "Document 1:" gets a leading
        # separator too (otherwise DocumentContextPartitioner would fold it into `instruction`)
        # and pass a passthrough template to ContextCiter, since `context_text` already
        # contains the preamble and "Query: ..." suffix that PROMPT_TEMPLATE would otherwise add again.
        context_text = PROMPT_TEMPLATE.format(query=query, context=join_contexts(contexts, max_document_size=max_document_size))

        cp = DocumentContextPartitioner(context_text)
        cc = ContextCiter(self.generator, self.tokenizer, context_text, query,
                          partitioner=cp, prompt_template="{context}")
        return cc.get_attributions()