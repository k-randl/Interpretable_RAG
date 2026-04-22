#%%
import sys
from pathlib import Path
import argparse
import torch
import pandas as pd
import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.Interpretable_RAG.retrieval_online import ExplainableAutoModelForRetrieval

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

import csv
#%%
def load_dataframe(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline()
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(first_line)
        separator = dialect.delimiter
        f.seek(0)
        try:
            has_header = sniffer.has_header(f.read(2048))
        except Exception: 
            try:
                if " " not in first_line.replace(separator,''):
                    has_header = True
                else:
                    has_header = False
            except Exception:
                print("Warning: Could not determine if the file has a header. Assuming no header.")
                has_header = False
        f.seek(0)

    if has_header:
        return pd.read_csv(file_path, sep=separator)
    else:
        return pd.read_csv(file_path, sep=separator)

def find_column_name(columns, aliases):
    for alias in aliases:
        if alias in columns:
            return alias
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topics_path', type=str, required=True)
    parser.add_argument('--retrieval_results_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='Snowflake/snowflake-arctic-embed-l-v2.0')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    topics = load_dataframe(args.topics_path)
    retrieval_results = load_dataframe(args.retrieval_results_path)
    

    query_id_column = find_column_name(retrieval_results.columns, ['qid', 'query_id', 'topic_id'])
    retrieved_text_column = find_column_name(retrieval_results.columns, ['text', 'retrieved_text', 'document_text', 'passage'])
    query_column = find_column_name(topics.columns, ['query', 'query_text', 'topic'])
    qid_column = find_column_name(topics.columns, ['qid', 'query_id', 'topic_id'])
    print(query_id_column, retrieved_text_column, query_column, qid_column)

    topics = topics[topics[qid_column].isin(retrieval_results[query_id_column].unique())]

    if not query_id_column or not retrieved_text_column or not query_column:
        print("Error: Could not automatically detect column names. Please ensure your files have one of the following columns:")
        print("Query ID columns: qid, query_id, topic_id")
        print("Retrieved text columns: text, retrieved_text, document_text, passage")
        print("Query columns: query, query_text, topic")
        sys.exit(1)
    else:
        print(topics.head())
        print(retrieval_results.head())

    add_pooling_layer = 'snowflake' not in args.model_name.lower()

    rag = ExplainableAutoModelForRetrieval.from_pretrained(
        args.model_name,
        add_pooling_layer=add_pooling_layer
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    queries = topics[query_column].tolist()
    qids = topics[qid_column].tolist()
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"query_id_column: {query_id_column}, retrieved_text_column: {retrieved_text_column}, query_column: {query_column}")
    with open(output_dir / 'config.txt', 'w') as f:
        f.write(f"topics_path: {args.topics_path}\n")
        f.write(f"retrieval_results_path: {args.retrieval_results_path}\n")
        f.write(f"model_name: {args.model_name}\n")
        f.write(f"batch_size: {args.batch_size}\n")
    for qid in tqdm.tqdm(qids, total=len(queries), desc='Processing queries'):
        #print(f'Processing query id: {qid}')
        query = topics[topics[qid_column] == qid][query_column].values[0]
        #print(f'Query: {query}')
        contexts = retrieval_results[retrieval_results[query_id_column] == qid][retrieved_text_column].tolist()
        #ßßprint(contexts)
        if not contexts:
            contexts = []
        rag('query: ' + query, contexts=contexts, output_attentions=True, output_hidden_states=True)
        rag.save_values(str(output_dir / f'query_{qid}.pkl'), batch_size=16)

if __name__ == '__main__':
    main()