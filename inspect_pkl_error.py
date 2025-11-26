import pickle
import numpy as np
import sys

file_path = "results/generation/trec19_19_11_2025/randomized/Llama-3.1-8B-Instruct_qid_32_10.pkl"

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print("Keys in data:", data.keys())
    
    if 'shapley_values_token' in data:
        sv = data['shapley_values_token']
        if 'context' in sv:
            ctx = sv['context']
            print(f"Context shape: {ctx.shape}")
        else:
            print("No 'context' in shapley_values_token")
            
        if 'query' in sv:
            qry = sv['query']
            print(f"Query shape: {qry.shape}")
        else:
            print("No 'query' in shapley_values_token")

        if 'gen_tokens' in data:
            print(f"gen_tokens length: {len(data['gen_tokens'])}")
            
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"Error: {e}")
