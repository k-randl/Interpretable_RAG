import pickle
import numpy as np
import sys
import torch

def check_shapes(filepath):
    print(f"--- Checking shapes for {filepath} ---")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if 'shapley_values_token' in data:
        print("Found 'shapley_values_token'")
        sv = data['shapley_values_token']
        if isinstance(sv, dict):
            for k, v in sv.items():
                if isinstance(v, np.ndarray):
                    print(f"  Key '{k}': shape {v.shape}")
                elif isinstance(v, list):
                     print(f"  Key '{k}': list of length {len(v)}")
    
    if 'qry_tokens' in data:
        print(f"qry_tokens length: {len(data['qry_tokens'])}")
    if 'gen_tokens' in data:
        print(f"gen_tokens length: {len(data['gen_tokens'])}")
        
    if 'intGrad' in data:
        print("Found 'intGrad'")
        ig = data['intGrad']
        if isinstance(ig, dict):
            for k, v in ig.items():
                if isinstance(v, torch.Tensor):
                    print(f"  Key '{k}': shape {v.shape}")
                elif isinstance(v, np.ndarray):
                    print(f"  Key '{k}': shape {v.shape}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_shapes(sys.argv[1])
