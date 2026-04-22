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
__author__ = "Guido Rocchietti"

# Core classes
from .generation import ExplainableAutoModelForGeneration
from .rag import ExplainableAutoModelForRAG

# Retrieval models
from .retrieval import RetrieverExplanation, RetrieverExplanationBase
from .retrieval_offline import ExplainableAutoModelForRetrieval
# from .retrieval_online import ExplainableRetriever

# Utilities
from .tools import (
    create_faiss_index_flat,
    create_flat_index,
    load_trained_index,
    save_index,
    embed_passages,
)
from .search_tools import (
    search,
    load_faiss_index,
    generate_query_embeddings,
    calculate_results,
)

# Methods
from .methods import get_shapley_values

# Type definitions
from .types import (
    ShapleyValues,
    AttributionMethod,
    GradientMethod,
    Embedding,
    EmbeddingBatch,
)

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
    "create_flat_index",
    "load_trained_index",
    "save_index",
    "embed_passages",
    "search",
    "load_faiss_index",
    "generate_query_embeddings",
    "calculate_results",
    # Methods
    "get_shapley_values",
    # Types
    "ShapleyValues",
    "AttributionMethod",
    "GradientMethod",
    "Embedding",
    "EmbeddingBatch",
]
