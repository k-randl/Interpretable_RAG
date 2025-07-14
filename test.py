# %%
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import torch
import numpy as np
import pandas as pd
from transformers import LlamaForCausalLM
from tqdm import tqdm
from resources.generation import ExplainableAutoModelForGeneration, plot_shap_attributions, highlight_dominant_passages

import nltk
from nltk.corpus import stopwords

           
#%%
MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'
MAX_SEQ_LEN = 1024
MAX_GEN_LEN = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# %% Load Pipeline:
model = ExplainableAutoModelForGeneration(LlamaForCausalLM).from_pretrained(
    MODEL_ID,
    device_map='auto',
    torch_dtype=torch.bfloat16
)
# %%
ranked_chunks = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/code/efra_retrieval/results/ir_results_chunks.csv')
evaluation_dataset = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/code/efra_retrieval/validation_Dataset_with_chunks_ids.csv')    
topics = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/Evaluation Dataset/topics.tsv', sep='\t')
do_evalualuation = True

#%%
input_dict = {}
queries = topics['query'].tolist()
contexts = ranked_chunks['retrieved_text'].groupby(ranked_chunks['query_id']).apply(list).to_dict()
contexts_topfive = {k: v[:5] for k, v in contexts.items()}  # Limit to top 5 contexts
#%%
#query = queries[0:4]  # Example query
#context = [contexts[i][:5] for i in range(4)]  # Example contexts for the first query
query = queries[0]  # Example query
context = contexts_topfive[0]  # Example contexts for the first query
# %%
import time
start_time = time.time()
output = model.explain_generate(
    query,
    context,
    max_new_tokens=MAX_GEN_LEN,
    do_sample=False,
    top_p=1,
    temperature=0.7,
    num_beams=1,
    max_samples=32
)
end_time = time.time()
print(f"Time taken for generation: {end_time - start_time} seconds")
#%%
start_time = time.time()
save_path = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/results_with_shap/'
results = model.save_values(save_path+'prova.pkl')
end_time = time.time()
print(f"Time taken for saving results: {end_time - start_time} seconds")
#%%
