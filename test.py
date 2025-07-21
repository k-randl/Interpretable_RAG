# %%
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
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
MAX_GEN_LEN = 300
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/results_with_shap/'
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
num_docs_context = 6
contexts = {k: v[:num_docs_context] for k, v in contexts.items()}  # Limit to num_docs contexts


#%%
for i in tqdm(range(0, len(queries))):
    query = queries[i]
    context = contexts[i]  # Example contexts for the i-th query
    output = model.explain_generate(
        query,
        context,
        max_new_tokens=MAX_GEN_LEN,
        batch_size=2**num_docs_context,
        do_sample=False,
        top_p=1,
        temperature=0.7,
        num_beams=1,
        max_samples=2**num_docs_context
    )
    model.save_values(SAVE_PATH + f'{MODEL_ID.split('/')[-1]}_top{num_docs_context}_input_{i}.pkl')

# %%
