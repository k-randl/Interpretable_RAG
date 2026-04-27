import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the parent directory to the path so we can import src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.Interpretable_RAG.eval_datasets import DatasetLoader
from src.Interpretable_RAG.perturbations import PromptPerturbationModule
from src.Interpretable_RAG.metrics import calculate_metrics
from analysis.correlation_analysis import calculate_warg, calculate_failure_modes

def run_tests():
    print("--- 1. Testing DatasetLoader (MuSiQue) ---")
    try:
        # Load a tiny fraction for testing (e.g., train split with streaming or just a few examples if possible, 
        # but validation is usually small enough. To avoid long downloads we can mock or just load 2 examples).
        loader = DatasetLoader('musique', split='validation[:2]')
        data = loader.load()
        print(f"Loaded {len(data)} items from MuSiQue.")
        print(f"Sample: Query='{data[0]['query']}', Answers={data[0]['answers']}")
    except Exception as e:
        print(f"Warning: Could not load MuSiQue (maybe offline or missing datasets): {e}")
        data = [{'query_id': 'test_0', 'query': 'Who is the author?', 'answers': ['John Doe']}]

    print("\n--- 2. Testing PromptPerturbationModule ---")
    perturbator = PromptPerturbationModule()
    ranked_docs = [f"Doc_{i}" for i in range(10)]
    corpus_docs = ranked_docs + [f"Random_{i}" for i in range(5)]
    
    a_vars = perturbator.generate_setup_a(ranked_docs, k=2)
    b_vars = perturbator.generate_setup_b(ranked_docs[:2], corpus_docs, k=2)
    
    print(f"Setup A1 (Top/Low): {a_vars['A1']}")
    print(f"Setup B1 (Top/Random): {b_vars['B1']}")

    print("\n--- 3. Testing Metrics (including BERTScore) ---")
    prediction = "The author of the book is John Doe."
    ground_truths = data[0]['answers']
    metrics = calculate_metrics(prediction, ground_truths)
    print(f"Metrics for Prediction: '{prediction}' | GT: {ground_truths}")
    print(f"EM: {metrics['exact_match']:.2f}, F1: {metrics['f1_score']:.2f}, ROUGE-L: {metrics['rougeL']:.2f}, BERTScore: {metrics['bert_score']:.2f}")

    print("\n--- 4. Testing Correlation & Failure Modes ---")
    ret_ranking = [0, 1, 2, 3, 4]
    gen_ranking = [0, 2, 1, 4, 3] # mild permutation
    
    warg = calculate_warg(ret_ranking, gen_ranking)
    print(f"WARG Score: {warg:.4f}")
    
    print("Failure Modes for k=2:")
    # For Wasted Retrieval: ret_top (0) in gen is rank 0. <= 2. So 0.
    # For Noise Distraction: gen_top (0) in ret is rank 0. <= 2. So 0.
    fm_2 = calculate_failure_modes(ret_ranking, gen_ranking, k=2)
    print(fm_2)
    
    bad_gen_ranking = [4, 3, 2, 1, 0] # complete reversal
    fm_2_bad = calculate_failure_modes(ret_ranking, bad_gen_ranking, k=2)
    print(f"Failure Modes for bad gen ranking {bad_gen_ranking}:")
    print(fm_2_bad)
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    # Ensure CUDA devices are set as requested by user
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    run_tests()
