import pickle
import sys
from pathlib import Path

def inspect_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Type of data: {type(data)}")
        if isinstance(data, dict):
            print("Keys:", data.keys())
            for k, v in data.items():
                print(f"Key: {k}, Type: {type(v)}")
                if hasattr(v, 'shape'):
                    print(f"Shape: {v.shape}")
                elif isinstance(v, list):
                    print(f"Length: {len(v)}")
                    if len(v) > 0:
                        print(f"First element type: {type(v[0])}")
        else:
            print(data)
    except Exception as e:
        print(f"Error loading pickle: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_pickle(sys.argv[1])
    else:
        print("Please provide a file path.")
