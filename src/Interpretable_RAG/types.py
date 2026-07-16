"""
Type definitions for Interpretable RAG.

This module provides type aliases and protocols used throughout the library.
"""

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Union,
)

import numpy as np
import numpy.typing as npt
from torch import Tensor, FloatTensor


# ============================================================================
# Basic Type Aliases
# ============================================================================

# Numeric types
FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int_]
BoolArray = npt.NDArray[np.bool_]

# Tensor types
TensorOrArray = Union[Tensor, npt.NDArray[Any]]
FloatTensorOrArray = Union[FloatTensor, FloatArray]

# Token types
TokenList = List[str]
TokenizedInput = Dict[str, Tensor]

# ============================================================================
# Retrieval Types
# ============================================================================

# Query/Document representations
Embedding = FloatArray  # Shape: [dim]
EmbeddingBatch = FloatArray  # Shape: [batch, dim]

# ============================================================================
# Generation Types
# ============================================================================

# Message format for chat models
ChatMessage = Dict[str, str]  # {"role": str, "content": str}
ChatHistory = List[ChatMessage]