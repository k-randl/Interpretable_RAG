import numpy as np
import random
from typing import List, Dict, Any

class PromptPerturbationModule:
    """
    Module for generating document ablations and perturbations for the generator prompt.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def generate_setup_a(self, ranked_docs: List[str], k: int = 5) -> Dict[str, List[str]]:
        """
        Setup A (Top vs Low): Retrieve Top-k and Low-k documents from the ranked list.
        Condition A1: Prompt order [Top-k, Low-k]
        Condition A2: Prompt order [Low-k, Top-k]
        Condition A3: Random shuffle of these 2k documents
        """
        top_docs = ranked_docs[:k]
        # Low docs are the bottom k documents in the ranked list
        low_docs = ranked_docs[-k:] if len(ranked_docs) >= 2 * k else ranked_docs[k:]
        
        # If there are not enough documents, just duplicate or use what's available
        # But normally we have enough docs in the ranked list (e.g. 100)
        
        a1 = top_docs + low_docs
        a2 = low_docs + top_docs
        
        a3 = a1.copy()
        random.shuffle(a3)
        
        return {
            'A1': a1,
            'A2': a2,
            'A3': a3
        }

    def generate_setup_b(self, top_docs: List[str], corpus_docs: List[str], k: int = 5) -> Dict[str, List[str]]:
        """
        Setup B (Top vs Random): Retrieve Top-k documents and select k completely Random documents from the corpus.
        Condition B1: Prompt order [Top-k, Random]
        Condition B2: Prompt order [Random, Top-k]
        Condition B3: Random shuffle of these 2k documents
        """
        # Ensure we do not select documents that are already in top_docs
        available_corpus = [doc for doc in corpus_docs if doc not in top_docs]
        
        if len(available_corpus) >= k:
            random_docs = random.sample(available_corpus, k)
        else:
            # Fallback if corpus is small
            random_docs = available_corpus + [random.choice(corpus_docs) for _ in range(k - len(available_corpus))]
            
        b1 = top_docs + random_docs
        b2 = random_docs + top_docs
        
        b3 = b1.copy()
        random.shuffle(b3)
        
        return {
            'B1': b1,
            'B2': b2,
            'B3': b3
        }

    def generate_intertwined_setup(self, ranked_docs: List[str], k: int = 5) -> Dict[str, List[str]]:
        """
        Creates C1, C2, C3 variations from a ranked list of documents.
        C1: [Top-k, Low-k]
        C2: [Low-k, Top-k]
        C3: Shuffled [Top-k, Low-k]
        """
        top_docs = ranked_docs[:k]
        low_docs = ranked_docs[k:2*k] if len(ranked_docs) >= 2*k else ranked_docs[k:]
        
        c1 = top_docs + low_docs
        c2 = low_docs + top_docs
        
        c3 = c1.copy()
        random.shuffle(c3)
        
        return {
            'C1': c1,
            'C2': c2,
            'C3': c3
        }
