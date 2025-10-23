import os, json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from tqdm.auto import tqdm



"""
Generates a qrels file in TREC format with binary relevance (1 relevant document per query).

Parameters:
- df (pd.DataFrame): DataFrame containing query-document relevance information.
- qid_col (str): Column name for query IDs.
- doc_id_col (str): Column name for relevant document IDs.
- output_file (str): path of the output qrels file.

Returns:
- None: Writes a qrels file to disk.
"""


def create_qrels_file(df, qid_col="qid", doc_id_col="doc_id", sep= ',', output_file="qrels.qrel"):

    # Ensure the DataFrame contains only unique qid-doc_id pairs
    df = df[[qid_col, doc_id_col]].drop_duplicates()
    df.columns = ['qid','doc_id']
    # Add a fixed column (required for TREC format) and relevance score of 1
    df["dummy_col"] = 0
    df["relevance"] = 1
    df =df[['qid', "dummy_col", 'doc_id', "relevance"]]
    # Save to file
    df.to_csv(output_file, sep=",", index=False, header=False)
    
    print(f"Qrels file saved as {output_file}")
    return  df


#### GENERATE TOPICS FILE ####
"""
Generates a topics file in TREC format with a query per line.

Parameters:
- df (pd.DataFrame): DataFrame containing query information.
- qid_col (str): Column name for query IDs.
- query_col (str): Column name for queries.
- output_file (str): Name of the output topics file.

Returns:
- None: Writes a topics file to disk.
"""
def create_topics_file(df, qid_col="qid", query_col="query", output_file="topics.csv", sep = ","):
    
    # Save to file
    df[[qid_col, query_col]].to_csv(output_file, sep=sep, index=False, header=False)
    
    print(f"Topics file saved as {output_file}")
    return df[[qid_col, query_col]]



### READ JSON FILES ###
'''
### Define function to get texts and ids from parsed files (json) ###
### Every file contains a dictionary with keys as the query ids and values as the texts ###
### The key is the file name of the original HTML from where the passages where extracted ###
'''

def get_texts(files_path):
    texts = []
    ids = []
    parsed_files = os.listdir(files_path)
    for file in tqdm(parsed_files):
        with open(files_path + file, 'r') as f:
            html = json.load(f)
            for key in html.keys():
                texts += html[key]['texts']
                ids += ([key]*len(html[key]['texts']))
    return texts, ids





  
'''
### Define function to embed a list of passages using a model and a tokenizer ###
'''

def embed_passages(passages, model, tokenizer, device="cuda", max_length=512):
    inputs = tokenizer(passages, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings =outputs.last_hidden_state[:, 0, :]  # Mean pooling
    return embeddings.cpu().numpy()

def embed_passages_snowflake(queries, model,tokenizer, query=True, max_length=512):
    if query:
        query_prefix = 'query: '
        queries_with_prefix = ["{}{}".format(query_prefix, i) for i in queries]
    else:
        queries_with_prefix = queries
    query_tokens = tokenizer(queries_with_prefix, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
    query_tokens = {k: v.to('cuda') for k, v in query_tokens.items()}
    with torch.no_grad():
        query_embeddings = model(**query_tokens)[0][:, 0]
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    return query_embeddings.cpu().numpy()



'''
### Define function to generate the context embeddings used to later build the index ###
### The embeddings are saved in a folder as numpy arrays ###
### The function takes a list of texts, the model name and the output folder as input ###
### The function uses the last hidden state of the model to generate the embeddings ###
### The function uses a step parameter to avoid memory issues when generating the embeddings ###
### The function uses the first token of the last hidden state as the embedding for the context ###
'''

def generate_context_embeddings(texts,model_name = 'facebook/dragon-plus-query-encoder',output_folder = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/data/passages_tensors/', step = 256):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    context_encoder = AutoModel.from_pretrained(model_name, device_map='auto')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    context_encoder.to(device)
    for i in tqdm(range(0, len(texts), step)):
        tkns = tokenizer(texts[i:i+step], padding='max_length', truncation=True, return_tensors='pt', max_length=512)
        tkns = {k: v.to(device) for k, v in tkns.items()}
        embeddings = context_encoder(**tkns).last_hidden_state[:, 0, :]
        ## Save embeddings
        embeddings = embeddings.detach().cpu().numpy()
        with open(f'{output_folder}{i}.npy', 'wb') as f:
            np.save(f, embeddings)


'''
### Define function to load the embeddings from a folder ###
### The function loads all the numpy arrays in the folder and stacks them ###
'''
def load_embeddings_from_folder(path):
    embeddings = None
    files = os.listdir(path)
    for file in tqdm(files[:]):
        with open(path + file, 'rb') as f:
            if embeddings is None:
                embeddings = np.load(f)
            else:
                embeddings = np.vstack((embeddings, np.load(f)))
    return embeddings

'''
### Define function to create a FAISS index from a list of embeddings ###
### The function takes a numpy array of embeddings and an optional save_folder parameter ###
### The function creates a FAISS index and adds the embeddings to it ###
### The function saves the index to disk if a save_folder is provided ###
'''

def create_faiss_index_flat(embeddings, save_folder = None,type_index = 'IP'):
    assert type_index in ['IP','L2'], "Invalid index type. Choose from ['IP', 'L2'], default is IP"
    # Create a FAISS index
    num_vectors = len(embeddings)
    dim = len(embeddings[0])
    if type_index == 'IP':
        faiss_index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    else:
        from sklearn.preprocessing import normalize
        faiss_index = faiss.IndexFlatL2(dim)  # Euclidean distance
        embeddings = normalize(embeddings)
    print(f'Index created with {num_vectors} vectors of dimension {dim}')
    # Add vectors to the FAISS index
    faiss_index.add(np.array(embeddings, dtype=np.float32))
    if save_folder is not None:
        print(f'Saving index to {save_folder}')
        faiss.write_index(faiss_index, save_folder)
    return faiss_index



"""
Receives a list of strings for a 'bad' line.
We expect 2 columns: [id, text].
- If more than 2 columns, merge them into one.
- If fewer than 2, fill them with defaults.
- If we return None, that line is skipped.
"""
def fix_bad_line(bad_line: list[str]) -> list[str] | None:

    if len(bad_line) > 2:
        # E.g. if we have an extra column, merge columns 1..end into one
        return [bad_line[0], " ".join(bad_line[1:])]
    elif len(bad_line) == 1:
        # Only one column: fill the second with a default string
        return [bad_line[0], "MISSING_TEXT"]
    elif len(bad_line) == 0:
        # Completely empty line: fill both columns
        return ["MISSING_ID", "MISSING_TEXT"]
    else:
        # Exactly 2 columns => it's actually "bad" for some reason, 
        # but let's just keep them as-is
        return bad_line


"""
Apply the reduction described in "Speeding Up the Xbox Recommender System Using a
Euclidean Transformation for Inner-Product Spaces" to the documents.
- Divides the embeddings by the maximum norm.
- Adds a column to the embeddings with the last_d value for each document.
"""

def apply_l2_reduction_documents(passage_embeddings):
    all_norms = np.linalg.norm(passage_embeddings, axis=1)
    # Find the maximum norm
    max_norm = np.max(all_norms)
    # Convert passage_embeddings to a NumPy array (if it's not already)
    passage_embeddings = np.array(passage_embeddings)
    # Compute norms squared for each document
    n2_docs_sq = np.sum(passage_embeddings ** 2, axis=1)  # Sum of squares along the last axis
    # Compute last_d values
    last_d = np.sqrt(1 - (n2_docs_sq / (max_norm ** 2)))
    # Normalize embeddings
    normalized_embeddings = passage_embeddings / max_norm
    # Stack normalized embeddings with last_d column
    doc_data = np.column_stack((normalized_embeddings, last_d))
    doc_data = doc_data.astype(np.float32)
    return doc_data



"""
Apply the reduction described in "Speeding Up the Xbox Recommender System Using a
Euclidean Transformation for Inner-Product Spaces" to the query.
- Adds a column of zeros to the query embeddings.
- Normalizes the embeddings (L2).
"""
def apply_l2_reduction_query(query_embeddings):
    zero_column = np.zeros((query_embeddings.shape[0], 1))
    # Append the zero column to the embeddings
    extended_query_embeddings = np.hstack((query_embeddings, zero_column))
    # Normalize each row (L2 norm)
    data = extended_query_embeddings / np.linalg.norm(extended_query_embeddings, axis=1, keepdims=True)
    data = np.array(data).astype(np.float32)
    return data
