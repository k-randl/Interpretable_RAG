#%%
import torch
import matplotlib.pyplot as plt
from resources.modelling import ExplainableAutoModelForRAG

# We use msmarco query and passages as an example
query =  "Where was Marie Curie born?"
contexts = [
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
]

rag = ExplainableAutoModelForRAG.from_pretrained(
    'facebook/dragon-plus-query-encoder',
    'facebook/dragon-plus-context-encoder'
)

# Create RAG model:
rag(query, contexts, output_attentions=True, output_hidden_states=True)

#%% aGrad:
fig, axs = plt.subplots(3,1)

tokens = rag.in_tokens['query'][0]
axs[0].imshow(rag.aGrad()['query'].mean(axis=1).abs())
axs[0].set_xticks(range(len(tokens)), labels=tokens)
axs[0].set_yticks([])
axs[0].set_title('Query:')

for i in range(len(contexts)):
    tokens = rag.in_tokens['context'][i]
    axs[i+1].imshow(rag.aGrad()['context'].mean(axis=1).abs()[i].reshape(1,-1))
    axs[i+1].set_xticks(range(len(tokens)), labels=tokens, rotation=90)
    axs[i+1].set_yticks([])
    axs[i+1].set_title(f'Context {i+1:d}:')

plt.show()

#%% Grad x In:
fig, axs = plt.subplots(3,1)

tokens = rag.in_tokens['query'][0]
axs[0].imshow(rag.gradIn()['query'].abs())
axs[0].set_xticks(range(len(tokens)), labels=tokens)
axs[0].set_yticks([])
axs[0].set_title('Query:')

for i in range(len(contexts)):
    tokens = rag.in_tokens['context'][i]
    axs[i+1].imshow(rag.gradIn()['context'][i].abs().reshape(1,-1))
    axs[i+1].set_xticks(range(len(tokens)), labels=tokens, rotation=90)
    axs[i+1].set_yticks([])
    axs[i+1].set_title(f'Context {i+1:d}:')

plt.show()

#%% Grad:
fig, axs = plt.subplots(3,1)

tokens = rag.in_tokens['query'][0]
axs[0].imshow(rag.grad()['query'].mean(axis=-1).abs())
axs[0].set_xticks(range(len(tokens)), labels=tokens)
axs[0].set_yticks([])
axs[0].set_title('Query:')

for i in range(len(contexts)):
    tokens = rag.in_tokens['context'][i]
    axs[i+1].imshow(rag.grad()['context'][i].mean(axis=-1).abs().reshape(1,-1))
    axs[i+1].set_xticks(range(len(tokens)), labels=tokens, rotation=90)
    axs[i+1].set_yticks([])
    axs[i+1].set_title(f'Context {i+1:d}:')

plt.show()