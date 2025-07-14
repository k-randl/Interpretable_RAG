#%%
from resources.index_res import *
import numpy as np

EMBEDDING_PATH = '/home/francomaria.nardini/raid/guidorocchietti/data/conversational/CAST2019/passage_embeddings/snowflake-arctic-embed-l-v2.0/passage_embeddings.npy'
METRIC = 'inner_product'  # or 'l2', 'euclidean', etc.
SAVE_DIR = '/home/francomaria.nardini/raid/guidorocchietti/data/RAG_Explainability/indexes/index_flat_ip_snowflake.faiss'


# %%
embeddings = np.load(EMBEDDING_PATH)
index = create_flat_index(embeddings.shape[1], METRIC)
index.verbose = True
index.add(embeddings)  # Add your data vectors to the index
print(f"Number of vectors in the index: {index.ntotal}")
# Save the populated index
save_index(index, SAVE_DIR)
# %%
