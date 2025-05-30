#%%
import pickle


pickle_path = 'perturbed_outputs.pkl'
with open(pickle_path, 'rb') as f:
    perturbed_outputs = pickle.load(f)


#%%
from collections import defaultdict
import matplotlib.pyplot as plt

# Collect all token sets and map probabilities
token_sets = {}
token_probs = {}

for desc, data in perturbed_outputs.items():
    tokens = data['gen_tokens']
    probs = data['gen_probs']
    token_sets[desc] = set(tokens)
    token_probs[desc] = dict(zip(tokens, probs))

# Intersect tokens
all_descriptions = list(token_sets.keys())
intersection = set.intersection(*token_sets.values())
print(f"Shared tokens across all variants: {intersection}")
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
plot_token_probabilities(list(intersection)[:5], token_probs)

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