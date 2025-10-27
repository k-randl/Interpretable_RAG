# generate_index.py
import argparse
import numpy as np
import sys
from pathlib import Path

# Assicuriamoci che lo script possa trovare il modulo 'resources'
# Aggiunge la cartella genitore al percorso di ricerca di Python
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Importiamo solo le funzioni necessarie dai nostri moduli di utilità
from src.Interpretable_RAG.index_res import create_flat_index, save_index

def build_faiss_index(embeddings_path: Path, save_path: Path, metric: str = 'IP'):
    """
    Carica gli embedding, costruisce un indice FAISS e lo salva su disco.

    Args:
        embeddings_path (Path): Percorso del file .npy contenente gli embedding.
        save_path (Path): Percorso dove salvare l'indice FAISS generato.
        metric (str): La metrica di distanza da usare ('IP' per Inner Product o 'L2').
    """
    print(f"INFO: Caricamento degli embedding da: {embeddings_path}")
    if not embeddings_path.exists():
        print(f"ERRORE: File di embedding non trovato in {embeddings_path}")
        return

    embeddings = np.load(embeddings_path).astype(np.float32)
    print(f"INFO: Caricati {embeddings.shape[0]} vettori di dimensione {embeddings.shape[1]}.")

    # Crea l'indice FAISS
    dimension = embeddings.shape[1]
    index = create_flat_index(dimension, measure=metric)

    # Popola l'indice
    print("INFO: Aggiunta dei vettori all'indice...")
    index.add(embeddings)
    print(f"INFO: L'indice ora contiene {index.ntotal} vettori.")

    # Assicura che la cartella di output esista
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Salva l'indice
    save_index(index, str(save_path))

def main():
    """
    Funzione principale per eseguire lo script da riga di comando.
    """
    parser = argparse.ArgumentParser(
        description="Costruisce e salva un indice FAISS da un file di embedding .npy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--embeddings_path",
        type=Path,
        required=True,
        help="Percorso del file .npy contenente gli embedding dei passaggi."
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        required=True,
        help="Percorso del file dove salvare l'indice FAISS finale."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="IP",
        choices=["IP", "L2"],
        help="Metrica di distanza per l'indice FAISS (Inner Product o L2)."
    )
    
    args = parser.parse_args()
    
    build_faiss_index(args.embeddings_path, args.save_path, args.metric)

if __name__ == "__main__":
    main()