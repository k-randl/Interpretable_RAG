"""
Interpretable RAG - Explainable Retrieval-Augmented Generation

This library provides tools for building interpretable RAG systems using
Shapley values for generation attribution and gradient-based methods for
retrieval explanation.

Main Components:
    - ExplainableAutoModelForGeneration: Explainable text generation with Shapley values
    - ExplainableAutoModelForRAG: End-to-end explainable RAG pipeline
    - Retrieval explanation models: Gradient-based attribution for retrievers

Quick Start:
    >>> from Interpretable_RAG import ExplainableAutoModelForGeneration
    >>> model = ExplainableAutoModelForGeneration.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    >>> result = model.explain_generate(
    ...     query="What is the capital of France?",
    ...     contexts=["Paris is the capital of France.", "France is in Europe."]
    ... )

Configuration:
    >>> from Interpretable_RAG.config import config
    >>> config.generation.MAX_GEN_LEN = 500
"""

__version__ = "0.1.0"
__author__ = "Korbinian Randl & Guido Rocchietti"

# Core classes
from .generation import ExplainableAutoModelForGeneration
from .rag import ExplainableAutoModelForRAG

# Retrieval models
from .retrieval import RetrieverExplanation, RetrieverExplanationBase
from .retrieval_online import ExplainableAutoModelForRetrieval

# Utilities
from .utils import (
    create_faiss_index_flat,
    load_faiss_index,
    save_faiss_index,
)

# Methods
from .methods import get_shapley_values

# Type definitions
from .types import (
    ShapleyValues,
    Embedding,
    EmbeddingBatch,
)
from .retrieval import RetrieverMethods_t
from .generation import GeneratorMethods_t, GeneratorAggregations_t

__all__ = [
    # Version
    "__version__",
    # Core
    "ExplainableAutoModelForGeneration",
    "ExplainableAutoModelForRAG",
    # Retrieval
    "RetrieverExplanation",
    "RetrieverExplanationBase",
    "ExplainableAutoModelForRetrieval",
    # Tools
    "create_faiss_index_flat",
    "load_faiss_index",
    "save_faiss_index",
    # Methods
    "get_shapley_values",
    # Types
    "ShapleyValues",
    "RetrieverMethods_t",
    "GeneratorMethods_t",
    "GeneratorAggregations_t",
    "Embedding",
    "EmbeddingBatch",
]
