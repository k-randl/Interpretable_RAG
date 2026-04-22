"""
Type definitions for Interpretable RAG.

This module provides type aliases and protocols used throughout the library.
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


# ============================================================================
# Basic Type Aliases
# ============================================================================

# Numeric types
Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# Generic numpy array
NDArrayFloat = Union[Float32Array, Float64Array]
NDArray = npt.NDArray[Any]

# Tensor types
TensorOrArray = Union[Tensor, NDArray]
Device = Union[str, torch.device]

# Token types
TokenList = List[str]
TokenizedInput = Dict[str, Tensor]

# ============================================================================
# Shapley Value Types
# ============================================================================

# Coalition representation (set of player indices)
Coalition = frozenset[int]

# Value function: Coalition -> float
ValueFunction = Callable[[Coalition], float]

# Shapley values result: array of shape [num_players]
ShapleyValues = Float64Array

# Attribution types
AttributionMethod = Literal["token", "sequence", "bow", "nucleus"]
ShapleyResult = Dict[str, NDArrayFloat]

# ============================================================================
# Retrieval Types
# ============================================================================

# Distance/similarity metrics
MetricType = Literal["IP", "L2", "inner_product", "euclidean"]

# Query/Document representations
Embedding = Float32Array  # Shape: [dim]
EmbeddingBatch = Float32Array  # Shape: [batch, dim]

# Search results
SearchDistances = Float32Array  # Shape: [num_queries, top_k]
SearchIndices = IntArray  # Shape: [num_queries, top_k]
SearchResult = Tuple[SearchDistances, SearchIndices]

# Gradient-based attribution methods
GradientMethod = Literal["grad", "gradIn", "intGrad", "aGrad"]
BaselineType = Literal["zero", "unk", "mask", "random"]

# ============================================================================
# Generation Types
# ============================================================================

# Message format for chat models
ChatMessage = Dict[str, str]  # {"role": str, "content": str}
ChatHistory = List[ChatMessage]

# Generation parameters
GenerationParams = Dict[str, Any]

# ============================================================================
# IR Evaluation Types
# ============================================================================

# Query-document relevance
Qrels = Dict[str, Dict[str, int]]  # qid -> {docid -> relevance}

# Retrieval results
RetrievalResults = Dict[str, Dict[str, float]]  # qid -> {docid -> score}

# ============================================================================
# Protocols (Structural Subtyping)
# ============================================================================

class Tokenizer(Protocol):
    """Protocol for tokenizer-like objects."""
    
    def __call__(
        self, 
        text: Union[str, List[str]], 
        **kwargs
    ) -> Dict[str, Any]:
        ...
    
    def decode(self, token_ids: Union[List[int], Tensor]) -> str:
        ...
    
    def encode(self, text: str, **kwargs) -> List[int]:
        ...


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""
    
    def encode(
        self, 
        sentences: Union[str, List[str]], 
        **kwargs
    ) -> NDArrayFloat:
        ...


class LanguageModel(Protocol):
    """Protocol for language models."""
    
    def generate(
        self, 
        input_ids: Tensor, 
        **kwargs
    ) -> Tensor:
        ...
    
    def forward(
        self, 
        input_ids: Tensor, 
        **kwargs
    ) -> Any:
        ...


# ============================================================================
# Type Variables for Generic Functions
# ============================================================================

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=LanguageModel)
ArrayT = TypeVar("ArrayT", bound=NDArray)
