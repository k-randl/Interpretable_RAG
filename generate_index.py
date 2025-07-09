#%%
import torch
from transformers import AutoTokenizer
from resources.retrieval_offline import ExplainableAutoModelForContextEncoding, ExplainableAutoModelForRetrieval
tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
query_encoder = ExplainableAutoModelForContextEncoding.from_pretrained('facebook/dragon-plus-query-encoder')
context_encoder = ExplainableAutoModelForRetrieval.from_pretrained('facebook/dragon-plus-context-encoder')

# We use msmarco query and passages as an example
query =  "Where was Marie Curie born?"
contexts = [
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
]
# Apply tokenizer
query_input = tokenizer(query, return_tensors='pt')
ctx_input = tokenizer(contexts, padding=True, truncation=True, return_tensors='pt')
# Compute embeddings: take the last-layer hidden state of the [CLS] token
query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]
ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]
#%%
import matplotlib.pyplot as plt

def decode_tokens(input_ids):
    return [[tokenizer.decode(token) for token in text] for text in input_ids]

#%% attention rollout:
fig, axs = plt.subplots(3,1)

tokens = decode_tokens(query_input.input_ids)[0]
axs[0].imshow(query_encoder.attentionRollout()[:,0])
axs[0].set_xticks(range(len(tokens)), labels=tokens)
axs[0].set_yticks([])
axs[0].set_title('Query:')

for i in range(ctx_emb.shape[0]):
    tokens = decode_tokens(ctx_input.input_ids)[i]
    axs[i+1].imshow(context_encoder.attentionRollout()[i,0].reshape(1,-1))
    axs[i+1].set_xticks(range(len(tokens)), labels=tokens, rotation=90)
    axs[i+1].set_yticks([])
    axs[i+1].set_title(f'Context {i+1:d}:')

plt.show()

#%% aGrad:
fig, axs = plt.subplots(3,1)

tokens = decode_tokens(query_input.input_ids)[0]
axs[0].imshow(query_encoder.aGrad().mean(axis=1)[:,0])
axs[0].set_xticks(range(len(tokens)), labels=tokens)
axs[0].set_yticks([])
axs[0].set_title('Query:')

for i in range(ctx_emb.shape[0]):
    tokens = decode_tokens(ctx_input.input_ids)[i]
    axs[i+1].imshow(context_encoder.aGrad().mean(axis=1)[i,0].reshape(1,-1))
    axs[i+1].set_xticks(range(len(tokens)), labels=tokens, rotation=90)
    axs[i+1].set_yticks([])
    axs[i+1].set_title(f'Context {i+1:d}:')

plt.show()

#%% Grad x In:
fig, axs = plt.subplots(3,1)

tokens = decode_tokens(query_input.input_ids)[0]
axs[0].imshow(query_encoder.gradIn())
axs[0].set_xticks(range(len(tokens)), labels=tokens)
axs[0].set_yticks([])

for i in range(ctx_emb.shape[0]):
    tokens = decode_tokens(ctx_input.input_ids)[i]
    axs[i+1].imshow(context_encoder.gradIn()[i].reshape(1,-1))
    axs[i+1].set_xticks(range(len(tokens)), labels=tokens, rotation=90)
    axs[i+1].set_yticks([])
    axs[i+1].set_title(f'Context {i+1:d}:')

plt.show()