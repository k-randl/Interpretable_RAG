#%%
import os, pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['TRANSFORMERS_CACHE'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
os.environ['HF_TOKEN'] = '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'

import torch
import pandas as pd
from transformers import AutoTokenizer
from methods import *
from resources.generation import ExplainableAutoModelForGeneration, plot_shap_attributions, highlight_dominant_passages

MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token_id = tokenizer.eos_token_id
#%%
results_files = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/results_with_shap/'
files = os.listdir(results_files)

file = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/results_with_shap/Llama-3.1-8B-Instruct_top6_input_5.pkl'
#%%
with open(os.path.join(results_files, file), 'rb') as f:
    results = pickle.load(f)

# %%
## read a xlsx with multiple sheets
ranked_chunks = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/code/efra_retrieval/results/ir_results_chunks.csv')
topics = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/Evaluation Dataset/topics.tsv', sep='\t')
#topics = pd.read_csv('/home/francomaria.nardini/raid/guidoroc
#%%
validation = pd.read_excel('/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/evaluation_dataset_v2.xlsx')
### join validation with ranked_chunks
validation = validation.merge(topics, on='query',how='left')
### merge the ranked_chunks 'score' column with validation on retrieved_text and query_id
validation = validation.merge(ranked_chunks[['retrieved_text', 'query_id', 'score']],
                              left_on=['retrieved_text', 'query_id'],
                              right_on=['retrieved_text', 'query_id'],
                              how='inner')
#%%
#calculate  rouge score between validation['retrieved_text'] and ranked_chunks['retrieved_text'
# using the rouge_score package
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
validation['rouge_score'] = validation.apply(lambda x: scorer.score(x['retrieved_text'], x['retrieved_text'])['rouge1'].fmeasure, axis=1)

#%%

tokens= [x.lstrip("Ġ").lstrip('Ċ') for x in tokenizer.convert_ids_to_tokens(results['generated_output'][0])]
#%%

# Imposta la dimensione della figura prima di fare il plot
plt.figure(figsize=(20, 8))

# Plot con rotazione dei token e maggiore spazio
plot_shap_attributions(results['shapley_values_tokens'], tokens, normalize=True)

# Regola il layout per evitare il taglio delle etichette
plt.tight_layout()

#%%
plt.figure(figsize=(20, 8))

# Plot con rotazione dei token e maggiore spazio
plot_shap_attributions(results['shapley_values_tokens'], tokens, normalize=False)

# Regola il layout per evitare il taglio delle etichette
plt.tight_layout()
# %%
highlight_dominant_passages(
    results['shapley_values_tokens'],
   tokens
)

# %%
