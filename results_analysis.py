#%%
import pickle


pickle_path = 'perturbed_outputs.pkl'
with open(pickle_path, 'rb') as f:
    perturbed_outputs = pickle.load(f)
    
#%%
import pandas as pd
ranked_chunks = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/code/efra_retrieval/results/ir_results_chunks.csv')
topics = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/Evaluation Dataset/topics.tsv', sep='\t')
evaluation_dataset = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/code/efra_retrieval/validation_Dataset_with_chunks_ids.csv')    
#%%
import nltk
from nltk.corpus import stopwords
def clean_tokens(tokens):
    stop_words = list(stopwords.words('english'))
    stop_words += ['', '\n', '.', ',', '(', ')', '[', ']', '{', '}', ':', ';', '?', '!', '"', "'", '-', '_', '/', '\\', '*', '&', '^', '%', '$', '#', '@', '~']
    tokens_filtered = [token.strip() for token in tokens if token.strip(' \n') not in stop_words + ['']]
    return tokens_filtered

def clean_tokens_mask(tokens):
    """
    Returns a boolean mask for filtering out stopwords and punctuation.

    Args:
        tokens (list of str): The list of tokens.

    Returns:
        List[bool]: Mask with True for tokens to keep.
    """
    stop_words = set(stopwords.words('english'))
    stop_symbols = {'', '\n', '.', ',', '(', ')', '[', ']', '{', '}', ':', ';',
                    '?', '!', '"', "'", '-', '_', '/', '\\', '*', '&', '^', '%',
                    '$', '#', '@', '~'}
    combined_stops = stop_words.union(stop_symbols)

    mask = [(token.strip() not in combined_stops) for token in tokens]
    return mask
#%%
from transformers import AutoTokenizer
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
# Collect all token sets and map probabilities
token_sets = {}
token_probs = {}
topk= 100
top_tokens = []


for desc, data in perturbed_outputs.items():
    if 'complete' in  desc:
        probs = data['gen_probs'].mean(dim=1)  # Average probabilities across all samples
    else:
        probs  = data['exp_probs'].mean(dim=1)  # Average probabilities across all samples
    topk_indices = torch.topk(probs, topk+len(list(stopwords.words('english')))).indices.tolist()[0]
    topk_proabilities = torch.topk(probs, topk+len(list(stopwords.words('english')))).values.tolist()[0]
    topk_tokens = [tokenizer.decode([idx]) for idx in topk_indices]
    mask = clean_tokens_mask(topk_tokens)
    filtered_tokens = [t.strip() for t, keep in zip(topk_tokens, mask) if keep][:topk]
    filtered_probs = [p for p, keep in zip(topk_proabilities, mask) if keep][:topk]
    token_sets[desc] = filtered_tokens
    token_probs[desc] = {token: prob for token, prob in zip(filtered_tokens, filtered_probs)}

#%%
### Find intersection of tokens across all descriptions
intersection = set.intersection(*[set(tokens) for tokens in token_sets.values()])

#%%

def plot_token_probabilities(shared_tokens, token_probs, title="Token Probabilities Across Variants"):
    for token in shared_tokens:
        values = []
        labels = []
        for desc in token_probs:
            prob = token_probs[desc].get(token, 0.0)
            values.append(prob)
            labels.append(desc)
        plt.plot(labels, values, marker='o', label=f"Token: {token}")

    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.ylabel("Probability")
    plt.xlabel("Prompt Variant")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Only plot a subset to avoid clutter
plot_token_probabilities(list(intersection)[:5], {key: token_probs[key] for key in [key for key in token_probs if 'constrained' in key or 'complete' in key]}, title="Token Probabilities for Constrained Prompt Variants")
plot_token_probabilities(list(intersection)[:5], {key: token_probs[key] for key in [key for key in token_probs if 'constrained' not in key]}, title="Token Probabilities for Prompt Variants")

#%%
for i, desc1 in enumerate(all_descriptions):
    for j, desc2 in enumerate(all_descriptions):
        if i < j:
            diff = token_sets[desc1] ^ token_sets[desc2]
            print(f"Token difference between '{desc1}' and '{desc2}': {diff}")
            
#%%

import seaborn as sns
import pandas as pd

def build_token_prob_df(shared_tokens, token_probs):
    data = {desc: [token_probs[desc].get(token, 0.0) for token in shared_tokens]
            for desc in token_probs}
    df = pd.DataFrame(data, index=shared_tokens)
    return df

df = build_token_prob_df(list(intersection)[:10], token_probs)

plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap='Blues')
plt.title("Probabilities of Shared Tokens Across Prompt Variants")
plt.ylabel("Token")
plt.xlabel("Variant")
plt.tight_layout()
plt.show()