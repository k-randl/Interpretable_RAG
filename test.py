# %%
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5,6,7'
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from resources.generation import ExplainableAutoModelForGeneration

import nltk
from nltk.corpus import stopwords

           
#%%
MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'
MAX_SEQ_LEN = 1024
MAX_GEN_LEN = 300
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/results/new_exp/'
os.makedirs(SAVE_PATH, exist_ok=True)
# %% Load Pipeline:
model = ExplainableAutoModelForGeneration.from_pretrained(
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
random_context = {k: np.random.choice(v, num_docs_context, replace=False).tolist() for k, v in contexts.items()}
no_duplicate_contexts = {k: list(set(v))[:num_docs_context] for k, v in contexts.items()}  # Remove duplicates 
#%%

#%%
do_original = False
n_steps = len(queries)  # Assuming you want to iterate over all queries
n_steps = 37
if do_original:
    print('Generating original inputs...')
    NEW_PATH = os.path.join(SAVE_PATH , 'original')
    os.makedirs(NEW_PATH, exist_ok=True)
    input_dict = {'query_id':[],'queries': [], 'contexts': []}
    for i in tqdm(range(0, n_steps)):
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
        model.save_values( os.path.join(NEW_PATH,f'{MODEL_ID.split('/')[-1]}_top{num_docs_context}_input_{i}.pkl'))
        input_dict['queries'].append(query)
        input_dict['contexts'].append(context)
        input_dict['query_id'].append(i)
        input_df = pd.DataFrame(input_dict)
        input_df.to_csv(os.path.join(NEW_PATH,f'{MODEL_ID.split("/")[-1]}_top{num_docs_context}_input.csv'), index=False)
# %%
do_random = False

if do_random:
    print('Generating randomized inputs...')
    NEW_PATH = os.path.join(SAVE_PATH , 'randomized')
    os.makedirs(NEW_PATH, exist_ok=True)
    input_dict = {'query_id':[],'queries': [], 'contexts': []}
    for i in tqdm(range(0, n_steps)):
        query = queries[i]
        context = random_context[i]  # Example contexts for the i-th query
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
        model.save_values(os.path.join(NEW_PATH, f'{MODEL_ID.split('/')[-1]}_top{num_docs_context}_input_{i}_randomized.pkl'))
        input_dict['queries'].append(query)
        input_dict['contexts'].append(context)
        input_dict['query_id'].append(i)
        input_df = pd.DataFrame(input_dict)
        input_df.to_csv(os.path.join(NEW_PATH, f'{MODEL_ID.split("/")[-1]}_top{num_docs_context}_input_randomized.csv'), index=False)

# %%
do_no_duplicates = True

if do_no_duplicates:
    print('Generating no duplicates inputs...')
    NEW_PATH = os.path.join(SAVE_PATH , 'no_duplicates')
    os.makedirs(NEW_PATH, exist_ok=True)
    #n_steps = len(queries)  # Assuming you want to iterate over all queries
    input_dict = {'query_id':[],'queries': [], 'contexts': []}
    for i in tqdm(range(0, n_steps)):
        query = queries[i]
        context = no_duplicate_contexts[i]  # Example contexts for the i-th query
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
        model.save_values(os.path.join(NEW_PATH,f'{MODEL_ID.split('/')[-1]}_top{num_docs_context}_input_{i}_no_duplicates.pkl'))
        input_dict['queries'].append(query)
        input_dict['contexts'].append(context)
        input_dict['query_id'].append(i)
        input_df = pd.DataFrame(input_dict)
        input_df.to_csv(os.path.join(NEW_PATH,f'{MODEL_ID.split("/")[-1]}_top{num_docs_context}_input_no_duplicates.csv'), index=False)
    
    
    
#%%
