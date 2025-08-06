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
SAVE_PATH = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/results/treccast_exp/'
os.makedirs(SAVE_PATH, exist_ok=True)

# %% Load Pipeline:
model = ExplainableAutoModelForGeneration.from_pretrained(
    MODEL_ID,
    device_map='auto',
    torch_dtype=torch.bfloat16
)

#%% Load TRECCAST data
RANKED_LIST_PATH = '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2019/data/results/results_snowflake_flat.csv'
TOPICS_PATH = '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2019/data/topics/topics.tsv'
PASSAGES_PATH = '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2019/data/CAST2019collection.tsv'


if os.path.exists('/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/cast19_retrieval_results.csv'):
    ranked_passages = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/cast19_retrieval_results.csv', index_col=0)
    ranked_passages.columns = ['query_id', 'docno', 'rank',	'score','id','retrieved_text']
else:
    collection_df = pd.read_csv(PASSAGES_PATH, sep='\t', names=['id', 'text'])
    # Assumendo che il file delle ranked list sia in formato CSV con colonne: query_id, retrieved_text, rank, score
    ranked_passages = pd.read_csv(RANKED_LIST_PATH)  # Sostituisci con il percorso corretto
    ranked_passages_with_text = ranked_passages.merge(collection_df, left_on='docno', right_on='id', how='left')
    ranked_passages.columns = ['query_id', 'docno', 'rank',	'score','id','retrieved_text']

    # Assumendo che il file delle query sia in formato CSV/TSV con colonne: query_id, query
queries_df = pd.read_csv(TOPICS_PATH, names=['query_id', 'query'])  # Sostituisci con il percorso corretto
queries_df = queries_df[queries_df['query_id'].isin(ranked_passages['query_id'].unique())]  # Fil
   
#%%
# Preparazione dei dati
queries = queries_df['query'].tolist()
contexts = ranked_passages['retrieved_text'].groupby(ranked_passages['query_id']).apply(list).to_dict()

num_docs_context = 6  # Puoi modificare questo numero in base alle tue esigenze
contexts = {k: v[:num_docs_context] for k, v in contexts.items()}  # Limita a num_docs contesti

# Crea versioni alternative dei contesti
random_context = {k: np.random.choice(v, num_docs_context, replace=False).tolist() for k, v in contexts.items()}
# Crea versioni alternative dei contesti senza duplicati, mantieni l'ordine originale, quindi non usare set
no_duplicate_contexts = {k: list(dict.fromkeys(v))[:num_docs_context] for k, v in contexts.items()}  # Rimuove i duplicati

#%% Esperimento con contesti originali
do_original = True
n_steps = len(queries)  # Numero totale di query da processare

if do_original:
    print('Generazione con input originali...')
    NEW_PATH = os.path.join(SAVE_PATH, 'original')
    os.makedirs(NEW_PATH, exist_ok=True)
    input_dict = {'query_id': [], 'queries': [], 'contexts': []}
    
    for i in tqdm(range(0, n_steps)):
        query = queries[i]
        query_id = queries_df.iloc[i]['query_id']  # Assicurati che questo corrisponda all'ID corretto
        context = contexts[query_id]  # Usa query_id per ottenere i contesti corretti
        
        output = model.explain_generate(
            query,
            context,
            max_new_tokens=MAX_GEN_LEN,
            batch_size=2**num_docs_context,
            do_sample=False,
            top_p=1,
            temperature=0.7,
            num_beams=1,
            max_samples=2**num_docs_context*2
        )
        
        model.save_values(os.path.join(NEW_PATH, f'{MODEL_ID.split('/')[-1]}_top{num_docs_context}_input_{query_id}.pkl'))
        input_dict['queries'].append(query)
        input_dict['contexts'].append(context)
        input_dict['query_id'].append(query_id)
        
        # Salva i risultati parziali dopo ogni query
        input_df = pd.DataFrame(input_dict)
        input_df.to_csv(os.path.join(NEW_PATH, f'{MODEL_ID.split("/")[-1]}_top{num_docs_context}_input.csv'), index=False)

#%% Esperimento con contesti randomizzati
do_random = True

if do_random:
    print('Generazione con input randomizzati...')
    NEW_PATH = os.path.join(SAVE_PATH, 'randomized')
    os.makedirs(NEW_PATH, exist_ok=True)
    input_dict = {'query_id': [], 'queries': [], 'contexts': []}
    
    for i in tqdm(range(0, n_steps)):
        query = queries[i]
        query_id = queries_df.iloc[i]['query_id']
        context = random_context[query_id]
        
        output = model.explain_generate(
            query,
            context,
            max_new_tokens=MAX_GEN_LEN,
            batch_size=2**num_docs_context,
            do_sample=False,
            top_p=1,
            temperature=0.7,
            num_beams=1,
            max_samples=2**num_docs_context*2
        )
        
        model.save_values(os.path.join(NEW_PATH, f'{MODEL_ID.split('/')[-1]}_top{num_docs_context}_input_{query_id}_randomized.pkl'))
        input_dict['queries'].append(query)
        input_dict['contexts'].append(context)
        input_dict['query_id'].append(query_id)
        
        input_df = pd.DataFrame(input_dict)
        input_df.to_csv(os.path.join(NEW_PATH, f'{MODEL_ID.split("/")[-1]}_top{num_docs_context}_input_randomized.csv'), index=False)

#%% Esperimento senza duplicati
do_no_duplicates = False

if do_no_duplicates:
    print('Generazione con input senza duplicati...')
    NEW_PATH = os.path.join(SAVE_PATH, 'no_duplicates')
    os.makedirs(NEW_PATH, exist_ok=True)
    input_dict = {'query_id': [], 'queries': [], 'contexts': []}
    
    for i in tqdm(range(0, n_steps)):
        query = queries[i]
        query_id = queries_df.iloc[i]['query_id']
        context = no_duplicate_contexts[query_id]
        
        output = model.explain_generate(
            query,
            context,
            max_new_tokens=MAX_GEN_LEN,
            batch_size=2**num_docs_context,
            do_sample=False,
            top_p=1,
            temperature=0.7,
            num_beams=1,
            max_samples=2**num_docs_context*2
        )
        
        model.save_values(os.path.join(NEW_PATH, f'{MODEL_ID.split('/')[-1]}_top{num_docs_context}_input_{query_id}_no_duplicates.pkl'))
        input_dict['queries'].append(query)
        input_dict['contexts'].append(context)
        input_dict['query_id'].append(query_id)
        
        input_df = pd.DataFrame(input_dict)
        input_df.to_csv(os.path.join(NEW_PATH, f'{MODEL_ID.split("/")[-1]}_top{num_docs_context}_input_no_duplicates.csv'), index=False)


#%%
from resources.generation import GeneratorExplanation
# Load the saved values from the mode
files =  GeneratorExplanation.load('/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/results/treccast_exp/original/')
# %%
from resources.plotting import plot_attribution_generator, higlight_attribution_generator

plot_attribution_generator(files[key])

higlight_attribution_generator(files[key],  token_processor=lambda s: s.replace('Ġ', ' ').strip('Ċ'))
# %%
