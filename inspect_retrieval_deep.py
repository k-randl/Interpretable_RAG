import pickle
import sys

def inspect_retrieval_deep(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print("--- Input ---")
        inp = data.get('input', {})
        if isinstance(inp, dict):
            print("Keys:", inp.keys())
            if 'query' in inp:
                print("Input Query Type:", type(inp['query']))
                print("Input Query Sample:", inp['query'][:50] if isinstance(inp['query'], str) else inp['query'][:5])
            if 'context' in inp:
                print("Input Context Type:", type(inp['context']))
                # Context might be a list of contexts
                if isinstance(inp['context'], list):
                     print("Input Context Len:", len(inp['context']))
                     print("Input Context Sample[0]:", inp['context'][0][:50] if isinstance(inp['context'][0], str) else inp['context'][0][:5])

        
        print("\n--- IntGrad ---")
        ig = data.get('intGrad', {})
        if isinstance(ig, dict):
            print("Keys:", ig.keys())
            if 'query' in ig:
                print("IG Query Type:", type(ig['query']))
                print("IG Query Sample:", ig['query'][:5] if isinstance(ig['query'], list) else "Not a list")
            if 'context' in ig:
                print("IG Context Type:", type(ig['context']))
                if isinstance(ig['context'], list):
                     print("IG Context Len:", len(ig['context']))
                     print("IG Context Sample[0]:", ig['context'][0][:5] if isinstance(ig['context'][0], list) else "Not a list")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_retrieval_deep(sys.argv[1])

