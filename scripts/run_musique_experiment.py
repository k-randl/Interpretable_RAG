import argparse
import os
import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

# Add the parent directory to the path so we can import src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.Interpretable_RAG.eval_datasets import DatasetLoader
from src.Interpretable_RAG.retrieval_online import ExplainableAutoModelForRetrieval
from src.Interpretable_RAG.generation import ExplainableAutoModelForGeneration
from src.Interpretable_RAG.perturbations import PromptPerturbationModule
from analysis.correlation_analysis import calculate_warg, calculate_failure_modes

def setup_models(retriever_id: str, generator_id: str):
    print(f"INFO: Loading Generator {generator_id}...")
    generator = ExplainableAutoModelForGeneration.from_pretrained(
        generator_id,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    
    print(f"INFO: Loading Retriever {retriever_id}...")
    if 'dragon' in retriever_id.lower():
        retriever = ExplainableAutoModelForRetrieval.from_pretrained(
            query_encoder_name_or_path='facebook/dragon-plus-query-encoder',
            context_encoder_name_or_path='facebook/dragon-plus-context-encoder'
        ).to('cuda:3' if torch.cuda.device_count() > 3 else 'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        retriever = ExplainableAutoModelForRetrieval.from_pretrained(
            query_encoder_name_or_path=retriever_id,
            context_encoder_name_or_path=retriever_id
        ).to('cuda:3' if torch.cuda.device_count() > 3 else 'cuda' if torch.cuda.is_available() else 'cpu')
        
    return retriever, generator

def main():
    parser = argparse.ArgumentParser(description="Runs MuSiQue online RAG experiment.")
    parser.add_argument("--retriever_id", type=str, required=True, help="HF model ID or 'dragon'")
    parser.add_argument("--generator_id", type=str, required=True, help="HF generator ID")
    parser.add_argument("--output_path", type=str, required=True, help="Base folder where to save all the results.")
    parser.add_argument("--num_docs_context", type=int, default=10, help="Number of documents to retrieve initially for Setup A/B.")
    parser.add_argument("--max_gen_len", type=int, default=300, help="Maximum length of the generated response.")
    args = parser.parse_args()

    retriever, generator = setup_models(args.retriever_id, args.generator_id)
    perturbator = PromptPerturbationModule()

    # Load data
    loader = DatasetLoader('musique', split='validation')

    from datasets import load_dataset
    raw_dataset = load_dataset('bdsaglam/musique', split='validation')

    normalized_data = loader.load()
    
    # Ground truths logging
    gt_log = []
    
    gen_model_name = args.generator_id.split('/')[-1]
    ret_model_name = args.retriever_id.split('/')[-1] if 'dragon' not in args.retriever_id else 'dragon'

    base_out_dir = Path(args.output_path) / f"{ret_model_name}_{gen_model_name}"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(normalized_data)), desc="Processing MuSiQue"):
        item = normalized_data[i]
        raw_item = raw_dataset[i]
        
        query_id = item['query_id']
        query = item['query']
        answers = item['answers']
        
        gt_log.append({'query_id': query_id, 'query': query, 'answers': answers})
        
        # Get paragraphs
        paragraphs = [p['paragraph_text'] for p in raw_item['paragraphs']]
        
        if len(paragraphs) == 0:
            continue
            
        # 1. Retrieve
        retrieved_ids, retrieved_sim = retriever(
            query=query,
            contexts=paragraphs,
            k=args.num_docs_context,
            reorder=True
        )
        
        retrieved_docs = [paragraphs[idx] for idx in retrieved_ids]
        
        # We need top-k and random. 
        # For setup A, we need top-5 and low-5 from retrieved. (If we retrieved 10, low 5 is the bottom 5).
        k = 5 if len(retrieved_docs) >= 10 else len(retrieved_docs) // 2
        if k == 0:
            k = 1
            
        a_vars = perturbator.generate_setup_a(retrieved_docs, k=k)
        
        # For setup B, we need top-k and random from corpus. Corpus is all paragraphs.
        b_vars = perturbator.generate_setup_b(retrieved_docs[:k], paragraphs, k=k)
        
        variations = {'original': retrieved_docs[:k*2]}
        variations.update(a_vars)
        variations.update(b_vars)
        
        for var_name, var_docs in variations.items():
            var_dir = base_out_dir / var_name
            var_dir.mkdir(parents=True, exist_ok=True)
            
            save_file = var_dir / f"{gen_model_name}_qid_{query_id}.pkl"
            if save_file.exists():
                continue # Resume support
                
            generator.explain_generate(
                query,
                var_docs,
                max_new_tokens=args.max_gen_len,
                batch_size=2,
                max_samples_query=32,
                max_samples_context=32,
                do_sample=False
            )
            
            generator.save_values(str(save_file))
            
    # Save ground truths
    pd.DataFrame(gt_log).to_csv(base_out_dir / "ground_truths.csv", index=False)
    
    print(f"Finished processing {ret_model_name} + {gen_model_name}")

if __name__ == "__main__":
    main()
