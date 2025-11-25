import pickle
import sys

def inspect_gen_shapley(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        sv = data.get('shapley_values_token', {})
        print(f"Shapley Token Keys Sample: {list(sv.keys())[:5]}")
        first_val = list(sv.values())[0]
        print(f"Shapley Token Value Type: {type(first_val)}")
        print(f"Shapley Token Value Sample: {first_val}")
        
        # Also check token lists
        print(f"Query Tokens Sample: {data.get('qry_tokens', [])[:5]}")
        print(f"Gen Tokens Sample: {data.get('gen_tokens', [])[:5]}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_gen_shapley(sys.argv[1])
