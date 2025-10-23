# run_pipeline.py
import argparse
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

# Aggiungi la cartella 'resources' al path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.Interpretable_RAG.generation import ExplainableAutoModelForGeneration

def setup(model_id: str, device: str):
    """Carica il modello generativo e il tokenizer."""
    print(f"INFO: Caricamento del modello {model_id}...")
    model = ExplainableAutoModelForGeneration.from_pretrained(
        model_id,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    print("INFO: Modello caricato con successo.")
    return model

def load_data(topics_path: Path, ranked_list_path: Path, collection_path: Path, top_k: int):
    """Carica le query e i documenti recuperati."""
    print("INFO: Caricamento dei dati di input...")
    queries_df = pd.read_csv(topics_path, names=['query_id', 'query'], sep='\t')
    ranked_passages_df = pd.read_csv(ranked_list_path)
    
    # Se il testo dei passaggi non è già presente, uniscilo dalla collezione
    if 'retrieved_text' not in ranked_passages_df.columns:
        collection_df = pd.read_csv(collection_path, sep='\t', names=['id', 'text'])
        ranked_passages_df = ranked_passages_df.merge(collection_df, left_on='docno', right_on='id', how='left')
        ranked_passages_df = ranked_passages_df.rename(columns={'text': 'retrieved_text'})

    # Raggruppa i contesti per query e prendi i primi 'top_k'
    contexts = ranked_passages_df.groupby('query_id')['retrieved_text'].apply(lambda x: x.head(top_k).tolist()).to_dict()
    
    # Filtra le query per avere solo quelle con contesti
    valid_qids = contexts.keys()
    queries_df = queries_df[queries_df['query_id'].isin(valid_qids)].reset_index(drop=True)
    
    print(f"INFO: Dati caricati per {len(queries_df)} query.")
    return queries_df, contexts

def create_context_variations(contexts: dict, num_docs: int) -> dict:
    """Crea le variazioni dei contesti: originale, randomizzato e senza duplicati."""
    print("INFO: Creazione delle variazioni di contesto...")
    variations = {
        'original': {k: v[:num_docs] for k, v in contexts.items()},
        'randomized': {k: np.random.choice(v, min(num_docs, len(v)), replace=False).tolist() for k, v in contexts.items()},
        'no_duplicates': {k: list(dict.fromkeys(v))[:num_docs] for k, v in contexts.items()}
    }
    print("INFO: Variazioni create.")
    return variations

def run_single_experiment(exp_type: str, context_dict: dict, queries_df: pd.DataFrame, model, args: argparse.Namespace):
    """Esegue un set di esperimenti di generazione e salva i risultati."""
    print(f"--- Avvio Esperimento: {exp_type.upper()} ---")
    save_dir = args.output_path / exp_type
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_log = []

    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc=f"Generazione per '{exp_type}'"):
        query_id = row['query_id']
        query = row['query']
        
        # Salta se non ci sono contesti per questa query
        if query_id not in context_dict or not context_dict[query_id]:
            continue
            
        context = context_dict[query_id]
        
        # Esegui la generazione e la spiegabilità
        model.explain_generate(
            query,
            context,
            max_new_tokens=args.max_gen_len,
            batch_size=2**len(context),
            do_sample=False,
            max_samples=2**len(context) * 2 # Per il calcolo preciso di SHAP
        )
        
        # Salva i dati di spiegabilità
        save_file = save_dir / f"{args.model_id.split('/')[-1]}_qid_{query_id}.pkl"
        model.save_values(str(save_file))
        
        results_log.append({'query_id': query_id, 'query': query, 'contexts': context})

    # Salva un log CSV con gli input utilizzati
    log_df = pd.DataFrame(results_log)
    log_df.to_csv(save_dir / "_inputs_log.csv", index=False)
    print(f"--- Esperimento {exp_type.upper()} completato. Risultati in: {save_dir} ---")

def main():
    parser = argparse.ArgumentParser(description="Esegue un pipeline RAG spiegabile end-to-end.")
    
    # Argomenti per i percorsi
    parser.add_argument("--topics_path", type=Path, required=True, help="Percorso del file con le query (topics).")
    parser.add_argument("--ranked_list_path", type=Path, required=True, help="Percorso della ranked list dal retrieval.")
    parser.add_argument("--collection_path", type=Path, required=True, help="Percorso della collezione di passaggi.")
    parser.add_argument("--output_path", type=Path, required=True, help="Cartella base dove salvare tutti i risultati.")

    # Argomenti per i modelli
    parser.add_argument("--model_id", type=str, default='meta-llama/Llama-3.1-8B-Instruct', help="ID del modello generativo da Hugging Face.")
    
    # Parametri dell'esperimento
    parser.add_argument("--num_docs_context", type=int, default=6, help="Numero di documenti da usare come contesto.")
    parser.add_argument("--max_gen_len", type=int, default=300, help="Lunghezza massima della risposta generata.")
    
    # Flag per controllare quali esperimenti eseguire
    parser.add_argument("--run_original", action='store_true', help="Esegui l'esperimento con i contesti originali.")
    parser.add_argument("--run_randomized", action='store_true', help="Esegui con contesti in ordine casuale.")
    parser.add_argument("--run_no_duplicates", action='store_true', help="Esegui con contesti senza duplicati.")

    args = parser.parse_args()

    # 1. Setup del modello
    model = setup(args.model_id, 'cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Caricamento e preparazione dei dati
    queries_df, contexts = load_data(args.topics_path, args.ranked_list_path, args.collection_path, args.num_docs_context * 2) # Carichiamo un po' più di documenti per la randomizzazione
    context_variations = create_context_variations(contexts, args.num_docs_context)

    # 3. Esecuzione degli esperimenti selezionati
    if args.run_original:
        run_single_experiment('original', context_variations['original'], queries_df, model, args)
    
    if args.run_randomized:
        run_single_experiment('randomized', context_variations['randomized'], queries_df, model, args)
        
    if args.run_no_duplicates:
        run_single_experiment('no_duplicates', context_variations['no_duplicates'], queries_df, model, args)
        
    print("\nPipeline completata.")

if __name__ == "__main__":
    main()
    
    '''python run_pipeline.py \
    --topics_path /percorso/dati/topics.tsv \
    --ranked_list_path /percorso/dati/results_snowflake_flat.csv \
    --collection_path /percorso/dati/CAST2019collection.tsv \
    --output_path /percorso/nuovi_risultati/ \
    --num_docs_context 6 \
    --run_original \
    --run_randomized'''