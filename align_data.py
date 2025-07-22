#%%
import pandas as pd

evaluation_dataset = pd.read_excel('/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/evaluation_dataset_v2.xlsx')
ranked_chunks = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/code/efra_retrieval/results/ir_results_chunks.csv')
topics = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/Evaluation Dataset/topics.tsv', sep='\t')
# %%
# Primo merge per ottenere i query_id
validation = evaluation_dataset.merge(topics, on='query', how='left')
sorted_validation = validation.sort_values(by=['query_id'])

# Prepara i dati ranked_chunks
ranked_chunks_grouped = ranked_chunks.groupby('query_id')

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

# Funzione per trovare la corrispondenza migliore nel gruppo di chunk
def find_matching_chunk(row, chunks_group, similarity_threshold=0.8):
    validation_text = row['retrieved_text']
    
    # Prima prova la corrispondenza esatta
    matching_chunks = chunks_group[chunks_group['retrieved_text'] == validation_text]
    if not matching_chunks.empty:
        return matching_chunks.iloc[0]
    
    # Se non trova corrispondenza esatta, cerca per similarità
    best_similarity = 0
    best_match = None
    
    for idx, chunk in chunks_group.iterrows():
        similarity = calculate_similarity(validation_text, chunk['retrieved_text'])
        if similarity > best_similarity and similarity >= similarity_threshold:
            best_similarity = similarity
            best_match = chunk
    
    if best_match is not None:
        print(f"Trovata corrispondenza con similarità {best_similarity:.2f} per query_id {row['query_id']}")
    
    return best_match

# Lista per i risultati allineati
aligned_data = []

# Itera su ogni riga della validazione

for idx, validation_row in sorted_validation.iterrows():
    
    query_id = validation_row['query_id']

    # Prendi i chunk per questa query
    if query_id in ranked_chunks_grouped.groups:
        chunks_for_query = ranked_chunks_grouped.get_group(query_id)
        
        # Trova il chunk corrispondente
        matching_chunk = find_matching_chunk(validation_row, chunks_for_query)
        
        if matching_chunk is not None:
            # Aggiungi i dati allineati
            aligned_row = {
                'query_id': query_id,
                'query': validation_row['query'],
                'doc_id': matching_chunk['doc_id'] if 'doc_id' in matching_chunk else None,
                'context': validation_row['retrieved_text'],
                
                'relevancy': validation_row['relevancy'] if 'relevancy' in validation_row else None,  # Aggiungi la colonna 'relevance' se esiste
                'relevancy_numeric': validation_row['relevancy_numeric'] if 'relevancy_numeric' in validation_row else None,  # Aggiungi la colonna 'relevance_numeric' se esiste
                'score': matching_chunk['score'],
                'rank': matching_chunk.name,  # Questo sarà l'indice originale nel ranked_chunks
               # 'relevance': validation_row['relevance']
            }
            aligned_data.append(aligned_row)

        else:
            print(f"WARNING: No matching chunk found for query_id {query_id} and context:\n{validation_row['retrieved_text'][:100]}...")

# Crea il DataFrame finale allineato
aligned_df = pd.DataFrame(aligned_data)

# Ordina per query_id e score
aligned_df = aligned_df.sort_values(by=['query_id', 'score'], ascending=[True, False])

# Aggiungi una colonna per context_id strutturata come f"'{query_id}_{rank_position}' meaning that the rank position reset for each query_id
# (e.g., '1_0', '1_1', '2_0', '2_1', etc.)
context_id = []
old_query_id = None
for query_id in aligned_df['query_id'].tolist():
    if query_id != old_query_id:
        # Reset della posizione di rango per un nuovo query_id
        rank_position = 0
    context_id.append(f"{query_id}_{rank_position}")
    old_query_id = query_id
    rank_position += 1

aligned_df['context_id'] = context_id


# Salva il risultato
aligned_df.to_csv('aligned_validation_data.csv', index=False)

# Stampa alcune statistiche
print(f"\nStatistiche di allineamento:")
print(f"Righe nel dataset di validazione originale: {len(sorted_validation)}")
print(f"Righe nel dataset allineato: {len(aligned_df)}")
print(f"Righe non allineate: {len(sorted_validation) - len(aligned_df)}")

# Controlla i query_id con problemi
problematic_queries = set(sorted_validation['query_id']) - set(aligned_df['query_id'])
if problematic_queries:
    print(f"\nQuery ID con problemi di allineamento: {sorted(problematic_queries)}")

#%%andas as pd

#%%

# %%
