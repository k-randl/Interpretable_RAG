# %%
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3,4,5'
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
SAVE_PATH = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/results/treccast_exp_15_09_25/'
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
    # Assuming the ranked list file is in CSV format with columns: query_id, retrieved_text, rank, score
    ranked_passages = pd.read_csv(RANKED_LIST_PATH)  # Replace with the correct path
    ranked_passages_with_text = ranked_passages.merge(collection_df, left_on='docno', right_on='id', how='left')
    ranked_passages.columns = ['query_id', 'docno', 'rank',	'score','id','retrieved_text']

    # Assuming the query file is in CSV/TSV format with columns: query_id, query
queries_df = pd.read_csv(TOPICS_PATH, names=['query_id', 'query'])  # Replace with the correct path
queries_df = queries_df[queries_df['query_id'].isin(ranked_passages['query_id'].unique())]  # Fil
   
#%%
# Data preparation
queries = queries_df['query'].tolist()
contexts = ranked_passages['retrieved_text'].groupby(ranked_passages['query_id']).apply(list).to_dict()

num_docs_context = 6  # You can change this number based on your needs
contexts = {k: v[:num_docs_context] for k, v in contexts.items()}  # Limit to num_docs contexts

# Create alternative versions of the contexts
random_context = {k: np.random.choice(v, num_docs_context, replace=False).tolist() for k, v in contexts.items()}
# Create alternative versions of the contexts without duplicates, maintaining the original order, so don't use set
no_duplicate_contexts = {k: list(dict.fromkeys(v))[:num_docs_context] for k, v in contexts.items()}  # Removes duplicates

#%% Function to run experiments
def run_experiment(exp_type: str, context_dict: dict, queries_df: pd.DataFrame, model, save_path: str, num_docs_context: int):
    """
    Runs a generation experiment for a given context type.
    
    Args:
        exp_type (str): Experiment type ('original', 'randomized', or 'no_duplicates')
        context_dict (dict): Dictionary with the contexts to use
        queries_df (pd.DataFrame): DataFrame with the queries
        model: Model to use for generation
        save_path (str): Base path to save the results
        num_docs_context (int): Number of context documents to use
    """
    print(f'Generation with {exp_type} input...')
    NEW_PATH = os.path.join(save_path, exp_type)
    os.makedirs(NEW_PATH, exist_ok=True)
    
    input_dict = {'query_id': [], 'queries': [], 'contexts': []}
    queries = queries_df['query'].tolist()
    
    for i in tqdm(range(len(queries))):
        query = queries[i]
        query_id = queries_df.iloc[i]['query_id']
        context = context_dict[query_id]
        
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
        
        # Build the filename based on the experiment type
        suffix = '' if exp_type == 'original' else f'_{exp_type}'
        save_file = f'{MODEL_ID.split('/')[-1]}_top{num_docs_context}_input_{query_id}{suffix}.pkl'
        
        model.save_values(os.path.join(NEW_PATH, save_file))
        input_dict['queries'].append(query)
        input_dict['contexts'].append(context)
        input_dict['query_id'].append(query_id)
        
        # Save partial results after each query
        input_df = pd.DataFrame(input_dict)
        input_df.to_csv(os.path.join(NEW_PATH, f'{MODEL_ID.split("/")[-1]}_top{num_docs_context}_input{suffix}.csv'), index=False)

#%% Configuration of the experiments to run
experiments_to_run = {
    'original': True,       # Experiment with original contexts
    'randomized': True,     # Experiment with randomized contexts
    'no_duplicates': False  # Experiment without duplicates
}

#%% Run the experiments
# Map of contexts for each experiment type
context_maps = {
    'original': contexts,
    'randomized': random_context,
    'no_duplicates': no_duplicate_contexts
}

# Run the configured experiments
for exp_type, should_run in experiments_to_run.items():
    if should_run:
        print(f"\nStarting experiment: {exp_type}")
        run_experiment(exp_type, context_maps[exp_type], queries_df, model, SAVE_PATH, num_docs_context)


#%%
#from resources.generation import GeneratorExplanation
# Load the saved values from the mode


saved_data= '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/results/treccast_exp_15_09_25/original/'
files = os.listdir(saved_data)
from pickle import load
data = load(open(os.path.join(saved_data, files[0]), 'rb'))
# %%
data
# %%
