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
        # Leggi il CSV e converti la colonna contexts da stringa a lista
        randomized_df = pd.read_csv(os.path.join(randomized_path,csvs[0]))
        randomized_df['contexts'] = randomized_df['contexts'].apply(eval)  # Converte la stringa in lista
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
        # Leggi il CSV e converti la colonna contexts da stringa a lista
        no_duplicates_df = pd.read_csv(os.path.join(no_duplicates_path, csvs[0]))
        no_duplicates_df['contexts'] = no_duplicates_df['contexts'].apply(eval)  # Converte la stringa in lista
        no_duplicates_df = no_duplicates_df.explode('contexts').reset_index(drop=True)
        #no_duplicates_files = [f for f in no_duplicates_files if f.endswith('.pkl')]
# Funzione per calcolare la similarità tra due testi

# Funzione per calcolare la similarità di Jaccard tra due testi
def calculate_similarity(text1, text2):
    # Converti i testi in set di parole
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calcola l'intersezione e l'unione
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    # Calcola il rapporto di Jaccard
    if union == 0:
        return 0
    return intersection / union

#%%
# Allineamento dei dataframe con aligned_validation
if 'randomized_df' in locals():
    # Aggiungi colonna context_id a randomized_df
    randomized_df['context_id'] = None
    similarity_threshold = 0.8  # Stessa soglia usata in align_data.py
    
    # Per ogni query_id
    for query_id in aligned_validation['query_id'].unique():
        # Prendi i contesti e i loro ID dalla validation
        query_contexts = aligned_validation[aligned_validation['query_id'] == query_id]
        
        # Per ogni riga nel randomized dataset
        mask = (randomized_df['query_id'] == query_id)
        for idx in randomized_df[mask].index:
            context = randomized_df.loc[idx, 'contexts']
            
            # Trova il miglior match in base alla similarità
            best_similarity = 0
            best_match = None
            
            for _, validation_row in query_contexts.iterrows():
                similarity = calculate_similarity(context, validation_row['context'])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = validation_row
            
            # Se la similarità è sopra la soglia, usa questo match
            if best_similarity >= similarity_threshold:
                randomized_df.loc[idx, 'context_id'] = best_match['context_id']
                randomized_df.loc[idx, 'similarity_score'] = best_similarity

#%%
if 'no_duplicates_df' in locals():
    # Aggiungi colonna context_id a no_duplicates_df
    no_duplicates_df['context_id'] = None
    similarity_threshold = 0.8  # Stessa soglia usata in align_data.py
    
    # Per ogni query_id
    for query_id in aligned_validation['query_id'].unique():
        # Prendi i contesti e i loro ID dalla validation
        query_contexts = aligned_validation[aligned_validation['query_id'] == query_id]
        
        # Per ogni riga nel no_duplicates dataset
        mask = (no_duplicates_df['query_id'] == query_id)
        for idx in no_duplicates_df[mask].index:
            context = no_duplicates_df.loc[idx, 'contexts']
            
            # Trova il miglior match in base alla similarità
            best_similarity = 0
            best_match = None
            
            for _, validation_row in query_contexts.iterrows():
                similarity = calculate_similarity(context, validation_row['context'])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = validation_row
            
            # Se la similarità è sopra la soglia, usa questo match
            if best_similarity >= similarity_threshold:
                no_duplicates_df.loc[idx, 'context_id'] = best_match['context_id']
                no_duplicates_df.loc[idx, 'similarity_score'] = best_similarity


# Verifica i risultati
print("\nNumero di context_id allineati:")
print(f"Randomized: {randomized_df['context_id'].notna().sum()}")

# Salva i DataFrame aggiornati
randomized_df.to_csv(os.path.join(randomized_path, 'randomized_with_context_ids.csv'), index=False)
        
#%%
###


#%%
query_id = 3

with open(os.path.join(randomized_path, randomized_files[query_id]), 'rb') as f:
    randomized_results = pickle.load(f)
with open(os.path.join(original_path, original_files[query_id]), 'rb') as f:
    original_results = pickle.load(f)
with open(os.path.join(no_duplicates_path, no_duplicates_files[query_id]), 'rb') as f:
    no_duplicates_results = pickle.load(f)
    
tokens_randomized= [x.lstrip("Ġ").lstrip('Ċ') for x in tokenizer.convert_ids_to_tokens(original_results['generated_output'][0])]
tokens_no_duplicates = [x.lstrip("Ġ").lstrip('Ċ') for x in tokenizer.convert_ids_to_tokens(no_duplicates_results['generated_output'][0])]
token_original = [x.lstrip("Ġ").lstrip('Ċ') for x in tokenizer.convert_ids_to_tokens(randomized_results['generated_output'][0])]
#%%
# Imposta la dimensione della figura prima di fare il plot
plt.figure(figsize=(20, 8))

# Plot con rotazione dei token e maggiore spazio
plot_shap_attributions(original_results['shapley_values_tokens'], token_original, normalize=True)

# Regola il layout per evitare il taglio delle etichette
plt.tight_layout()
### print the contexts
print(f"Contexts for query_id {query_id}:")
for i, row in enumerate(aligned_validation[aligned_validation.query_id == query_id].iloc[:6].iterrows()):
    print(f'DOCUMENT {row[1]['context_id']}' + row[1]['context'][:100] + '...')  # Stampa i primi 100 caratteri di ogni contesto
#%%
# Imposta la dimensione della figura prima di fare il plot
plt.figure(figsize=(20, 8))

# Plot con rotazione dei token e maggiore spazio
plot_shap_attributions(randomized_results['shapley_values_tokens'], tokens_randomized, normalize=True)

# Regola il layout per evitare il taglio delle etichette
plt.tight_layout()

### print the contexts
print(f"Contexts for query_id {query_id} Randomized:")
for i, row in enumerate(randomized_df[randomized_df.query_id == query_id].iloc[:6].iterrows()):
    print(f'DOCUMENT {row[1]['context_id']}' + row[1]['contexts'][:100] + '...')  # Stampa i primi 100 caratteri di ogni contesto
    
#%%
plt.figure(figsize=(20, 8))

# Plot con rotazione dei token e maggiore spazio
plot_shap_attributions(no_duplicates_results['shapley_values_tokens'], tokens_no_duplicates, normalize=True)

# Regola il layout per evitare il taglio delle etichette
plt.tight_layout()

### print the contexts
print(f"Contexts for query_id {query_id} No duplicates:")
for i, row in enumerate(no_duplicates_df[no_duplicates_df.query_id == query_id].iloc[:6].iterrows()):
    print(f'DOCUMENT {i} : {row[1]['context_id']}' + row[1]['contexts'][:100] + '...')  # Stampa i primi 100 caratteri di ogni contesto
# %%
highlight_dominant_passages(
    original_results['shapley_values_tokens'],
   token_original
)

# %%
highlight_dominant_passages(
    randomized_results['shapley_values_tokens'],
   tokens_randomized
)   

#%%
highlight_dominant_passages(
    no_duplicates_results['shapley_values_tokens'],
   tokens_no_duplicates
)
# %%
aligned_validation[aligned_validation.query_id == query_id].iloc[:6]
# %%
