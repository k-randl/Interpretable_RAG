#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from resources.modelling_online import ExplainableAutoModelForRAG

# We use msmarco query and passages as an example
query =  "Where was Marie Curie born?"
contexts = [
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
]

rag = ExplainableAutoModelForRAG.from_pretrained(
    'facebook/dragon-plus-query-encoder',
    'facebook/dragon-plus-context-encoder'
).to('cuda' if torch.cuda.is_available() else 'cpu')

# Create RAG model:
rag(query, contexts, output_attentions=True, output_hidden_states=True)

#%%
def plot_importance(ax, scores, tokens, title):
    assert len(scores) == len(tokens)
    y = np.arange(len(scores))[::-1]
    ax.barh(y, scores)
    ax.set_yticks(y, labels=tokens)
    ax.set_title(title)

#%% aGrad:
fig, axs = plt.subplots(1, 3)

plot_importance(axs[0], 
    scores = rag.aGrad()['query'][0].mean(axis=0), 
    tokens = rag.in_tokens['query'][0],
    title  = 'Query:'
)

for i in range(len(contexts)):
    plot_importance(axs[1+i], 
        scores = rag.aGrad()['context'][i].mean(axis=0), 
        tokens = rag.in_tokens['context'][i],
        title  = f'Context {i+1:d}:'
    )

plt.tight_layout()
plt.show()

#%% Grad x In:
fig, axs = plt.subplots(1, 3)

plot_importance(axs[0], 
    scores = rag.gradIn()['query'][0], 
    tokens = rag.in_tokens['query'][0],
    title  = 'Query:'
)

for i in range(len(contexts)):
    plot_importance(axs[1+i], 
        scores = rag.gradIn()['context'][i], 
        tokens = rag.in_tokens['context'][i],
        title  = f'Context {i+1:d}:'
    )

plt.tight_layout()
plt.show()

#%% Grad:
fig, axs = plt.subplots(1, 3)

plot_importance(axs[0], 
    scores = rag.grad()['query'][0].mean(axis=-1), 
    tokens = rag.in_tokens['query'][0],
    title  = 'Query:'
)

for i in range(len(contexts)):
    plot_importance(axs[1+i], 
        scores = rag.grad()['context'][i].mean(axis=-1), 
        tokens = rag.in_tokens['context'][i],
        title  = f'Context {i+1:d}:'
    )

plt.tight_layout()
plt.show()

#%% Shap:
import shap

SEPARATOR = rag.tokenizer.sep_token

# create text masker:
class InvariantTextMasker(shap.maskers.Text):
    def __call__(self, mask, s, **kwargs):
        mask = np.bitwise_or(mask, self.invariants(s)[0])
        return super().__call__(mask, s, **kwargs)

def predict_shap(inputs):
    return [rag(*s.split(SEPARATOR))[0][0].detach().numpy() for s in inputs]

def create_data_shap(query:str, contexts:list):
    return [query + SEPARATOR + context for context in contexts]

masker = InvariantTextMasker(rag.tokenizer, collapse_mask_token=True)
explainer = shap.Explainer(predict_shap, masker)
shap_values = explainer(create_data_shap(query, contexts))

#%%
fig, axs = plt.subplots(1,2)

for i in range(len(shap_values.values)):
    plot_importance(axs[i], 
        scores = shap_values.values[i], 
        tokens = shap_values.data[i],
        title  = f'Context {i+1:d}:'
    )

plt.tight_layout()
plt.show()

#%%
shap.plots.text(shap_values)