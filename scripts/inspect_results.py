import pickle
import os
import sys

def inspect_pickle(filepath):
    print(f"--- Inspecting {filepath} ---")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            for k, v in data.items():
                print(f"Key: {k}, Type: {type(v)}")
                if hasattr(v, '__len__'):
                    print(f"  Length: {len(v)}")
                # Print a small sample if possible
                try:
                    print(f"  Sample: {str(v)[:200]}...")
                except:
                    pass
        elif isinstance(data, list):
            print(f"Length: {len(data)}")
            if len(data) > 0:
                print(f"Sample item type: {type(data[0])}")
                print(f"Sample item: {str(data[0])[:200]}...")
        else:
            print(f"Value: {str(data)[:200]}...")
            
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    print("\n")

if __name__ == "__main__":
    files_to_inspect = [
        "results/retrieval/trec19/query_31_1.pkl",
        "results/generation/trec19_19_11_2025/original/Llama-3.1-8B-Instruct_qid_31_1.pkl",
        "results/retrieval/efra_06_11/query_0.pkl",
        # Add more as discovered
    ]
    
    # Add one from efra generation if found (will be added dynamically if I knew the name, but for now I'll just use what I know or pass args)
    
    if len(sys.argv) > 1:
        files_to_inspect = sys.argv[1:]

    base_dir = "/Users/anonymized_user_2/Documents/Lavoro/Interpretable_RAG"
    for f in files_to_inspect:
        full_path = os.path.join(base_dir, f) if not f.startswith("/") else f
        if os.path.exists(full_path):
            inspect_pickle(full_path)
        else:
            print(f"File not found: {full_path}")
