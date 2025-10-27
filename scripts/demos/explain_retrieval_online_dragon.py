#%%
import sys
sys.path.insert(0, "../..")

import torch
from src.Interpretable_RAG.plotting import plot_importance_retriever, higlight_importance_retriever
from src.Interpretable_RAG.retrieval_online import ExplainableAutoModelForRetrieval

# We use msmarco query and passages as an example
query =  "Where was Marie Curie born?"
contexts = [
    "Maria Skłodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace.",
    "Maria Skłodowska was born in Warsaw, in Congress Poland in the Russian Empire, as the fifth and youngest child of well-known teachers Bronisława, née Boguska, and Władysław Skłodowski.",
    "While a French citizen, Marie Skłodowska Curie, who used both surnames, never lost her sense of Polish identity. She taught her daughters the Polish language and took them on visits to Poland.",
    "Marie Curie founded the Curie Institute in Paris in 1920, and the Curie Institute in Warsaw in 1932.",
]

retriever = ExplainableAutoModelForRetrieval.from_pretrained(
    'facebook/dragon-plus-query-encoder',
    'facebook/dragon-plus-context-encoder'
).to('cuda' if torch.cuda.is_available() else 'cpu')

# Create RAG model:
retriever(query, contexts, output_attentions=True, output_hidden_states=True)

#%% aGrad:
plot_importance_retriever(retriever, method='aGrad')
higlight_importance_retriever(retriever, method='aGrad', token_processor=lambda s: s[2:] if s.startswith('##') else ' ' + s)

#%% Grad:
plot_importance_retriever(retriever, method='grad')
higlight_importance_retriever(retriever, method='grad', token_processor=lambda s: s[2:] if s.startswith('##') else ' ' + s)

#%% Grad x In:
plot_importance_retriever(retriever, method='gradIn')
higlight_importance_retriever(retriever, method='gradIn', token_processor=lambda s: s[2:] if s.startswith('##') else ' ' + s)

#%% Integrated Gradients:
plot_importance_retriever(retriever, method='intGrad')
higlight_importance_retriever(retriever, method='intGrad', token_processor=lambda s: s[2:] if s.startswith('##') else ' ' + s)