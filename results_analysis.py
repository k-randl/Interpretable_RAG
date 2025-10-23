#%%
import os, pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['TRANSFORMERS_CACHE'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
os.environ['HF_TOKEN'] = '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'

import torch
import pandas as pd
from transformers import AutoTokenizer
from methods import *
from resources.generation import GeneratorExplanation
from code.Interpretable_RAG.resources.plotting_16_10_2025 import *

MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token_id = tokenizer.eos_token_id
#%%
ranked_chunks = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/code/efra_retrieval/results/ir_results_chunks.csv')
topics = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/Evaluation Dataset/topics.tsv', sep='\t')
aligned_validation = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/aligned_validation_data.csv')

#%%
RESULTS_FOLDER = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/results/new_exp/'

if os.path.exists(os.path.join(RESULTS_FOLDER,'original')):
    original_path = os.path.join(RESULTS_FOLDER,'original')
    original_files = os.listdir(original_path)
    ### order files by name, beware that the name contains the number _0_ _1_ _2_ etc
    original_files.sort(key=lambda x: int(x.replace('.pkl','').split('_')[3]) if x.endswith('.pkl') else 0)
    csvs = [f for f in original_files if f.endswith('.csv')]
    original_files = [f for f in original_files if f.endswith('.pkl')]
    if csvs:
        original_df = pd.read_csv(os.path.join(original_path, original_files[0]))
        #original_files = [f for f in original_files if f.endswith('.pkl')]
if os.path.exists(os.path.join(RESULTS_FOLDER,'randomized')):
    randomized_path = os.path.join(RESULTS_FOLDER,'randomized')
    randomized_files = os.listdir(randomized_path)
    ### order files by name, beware that the name contains the number _0_ _1_ _2_ etc
    randomized_files.sort(key=lambda x: int(x.replace('.pkl','').split('_')[3]) if x.endswith('.pkl') else 0)
    csvs = [f for f in randomized_files if f.endswith('.csv')]
    randomized_files = [f for f in randomized_files if f.endswith('.pkl')]
    if csvs:
        # Read the CSV and convert the contexts column from string to list
        randomized_df = pd.read_csv(os.path.join(randomized_path,csvs[0]))
        randomized_df['contexts'] = randomized_df['contexts'].apply(eval)  # Convert string to list
        ### put every value in the contexts lists as a single row
        randomized_df = randomized_df.explode('contexts').reset_index(drop=True)
        #randomized_files = [f for f in randomized_files if f.endswith('.pkl')]

if os.path.exists(os.path.join(RESULTS_FOLDER,'no_duplicates')):
    no_duplicates_path = os.path.join(RESULTS_FOLDER,'no_duplicates')
    no_duplicates_files = os.listdir(no_duplicates_path)
    ### order files by name, beware that the name contains the number _0_ _1_ _2_ etc
    no_duplicates_files.sort(key=lambda x: int(x.replace('.pkl','').split('_')[3]) if x.endswith('.pkl') else 0)
    csvs = [f for f in no_duplicates_files if f.endswith('.csv')]
    no_duplicates_files = [f for f in no_duplicates_files if f.endswith('.pkl')]
    if csvs:
        # Read the CSV and convert the contexts column from string to list
        no_duplicates_df = pd.read_csv(os.path.join(no_duplicates_path, csvs[0]))
        no_duplicates_df['contexts'] = no_duplicates_df['contexts'].apply(eval)  # Convert string to list
        no_duplicates_df = no_duplicates_df.explode('contexts').reset_index(drop=True)
        #no_duplicates_files = [f for f in no_duplicates_files if f.endswith('.pkl')]
# Function to calculate the similarity between two texts

# Function to calculate the Jaccard similarity between two texts
def calculate_similarity(text1, text2):
    # Convert texts to sets of words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate intersection and union
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    # Calculate Jaccard ratio
    if union == 0:
        return 0
    return intersection / union

#%%
# Alignment of dataframes with aligned_validation
if 'randomized_df' in locals():
    # Add context_id column to randomized_df
    randomized_df['context_id'] = None
    similarity_threshold = 0.8  # Same threshold used in align_data.py
    
    # For each query_id
    for query_id in aligned_validation['query_id'].unique():
        # Get the contexts and their IDs from the validation
        query_contexts = aligned_validation[aligned_validation['query_id'] == query_id]
        
        # For each row in the randomized dataset
        mask = (randomized_df['query_id'] == query_id)
        for idx in randomized_df[mask].index:
            context = randomized_df.loc[idx, 'contexts']
            
            # Find the best match based on similarity
            best_similarity = 0
            best_match = None
            
            for _, validation_row in query_contexts.iterrows():
                similarity = calculate_similarity(context, validation_row['context'])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = validation_row
            
            # If the similarity is above the threshold, use this match
            if best_similarity >= similarity_threshold:
                randomized_df.loc[idx, 'context_id'] = best_match['context_id']
                randomized_df.loc[idx, 'similarity_score'] = best_similarity

#%%
if 'no_duplicates_df' in locals():
    # Add context_id column to no_duplicates_df
    no_duplicates_df['context_id'] = None
    similarity_threshold = 0.8  # Same threshold used in align_data.py
    
    # For each query_id
    for query_id in aligned_validation['query_id'].unique():
        # Get the contexts and their IDs from the validation
        query_contexts = aligned_validation[aligned_validation['query_id'] == query_id]
        
        # For each row in the no_duplicates dataset
        mask = (no_duplicates_df['query_id'] == query_id)
        for idx in no_duplicates_df[mask].index:
            context = no_duplicates_df.loc[idx, 'contexts']
            
            # Find the best match based on similarity
            best_similarity = 0
            best_match = None
            
            for _, validation_row in query_contexts.iterrows():
                similarity = calculate_similarity(context, validation_row['context'])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = validation_row
            
            # If the similarity is above the threshold, use this match
            if best_similarity >= similarity_threshold:
                no_duplicates_df.loc[idx, 'context_id'] = best_match['context_id']
                no_duplicates_df.loc[idx, 'similarity_score'] = best_similarity


# Verify the results
print("\nNumber of aligned context_ids:")
print(f"Randomized: {randomized_df['context_id'].notna().sum()}")

# Save the updated DataFrames
randomized_df.to_csv(os.path.join(randomized_path, 'randomized_with_context_ids.csv'), index=False)
        
#%%
###


#%%
query_id = 3

randomized_results    = GeneratorExplanation.load(os.path.join(randomized_path, randomized_files[query_id]), model_name_or_path='MODEL_ID', tokenizer=tokenizer)
original_results      = GeneratorExplanation.load(os.path.join(original_path, original_files[query_id]), model_name_or_path='MODEL_ID', tokenizer=tokenizer)
no_duplicates_results = GeneratorExplanation.load(os.path.join(no_duplicates_path, no_duplicates_files[query_id]), model_name_or_path='MODEL_ID', tokenizer=tokenizer)

#%%
# Plot with token rotation and more space
plot_attribution_generator(original_results, aggregation='token', normalize=True, figsize=(20, 8))

### print the contexts
print(f"Contexts for query_id {query_id}:")
for i, row in enumerate(aligned_validation[aligned_validation.query_id == query_id].iloc[:6].iterrows()):
    print(f'DOCUMENT {row[1]['context_id']}' + row[1]['context'][:100] + '...')  # Print the first 100 characters of each context
#%%
# Plot with token rotation and more space
plot_attribution_generator(randomized_results, aggregation='token', normalize=True, figsize=(20, 8))

### print the contexts
print(f"Contexts for query_id {query_id} Randomized:")
for i, row in enumerate(randomized_df[randomized_df.query_id == query_id].iloc[:6].iterrows()):
    print(f'DOCUMENT {row[1]['context_id']}' + row[1]['contexts'][:100] + '...')  # Print the first 100 characters of each context
    
#%%
# Plot with token rotation and more space
plot_attribution_generator(no_duplicates_results, aggregation='token', normalize=True, figsize=(20, 8))

### print the contexts
print(f"Contexts for query_id {query_id} No duplicates:")
for i, row in enumerate(no_duplicates_df[no_duplicates_df.query_id == query_id].iloc[:6].iterrows()):
    print(f'DOCUMENT {i} : {row[1]['context_id']}' + row[1]['contexts'][:100] + '...')  # Print the first 100 characters of each context
# %%
higlight_attribution_generator(original_results, token_processor=lambda s: s.replace('Ġ', ' ').strip('Ċ'))

# %%
higlight_attribution_generator(randomized_results, token_processor=lambda s: s.replace('Ġ', ' ').strip('Ċ'))   

#%%
higlight_attribution_generator(no_duplicates_results['shapley_values_tokens'], token_processor=lambda s: s.replace('Ġ', ' ').strip('Ċ'))

# %%
aligned_validation[aligned_validation.query_id == query_id].iloc[:6]
# %%
