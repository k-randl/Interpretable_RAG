import os
import sys
sys.path.insert(0, "../..")
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4"

import torch
import pandas as pd
from src.Interpretable_RAG.retrieval_online import ExplainableAutoModelForRetrieval

retriever = ExplainableAutoModelForRetrieval.from_pretrained(
    'Snowflake/snowflake-arctic-embed-l-v2.0',
    add_pooling_layer = False
).to('cuda' if torch.cuda.is_available() else 'cpu')

texts = pd.read_csv('../../data/popqa/passages.csv')['text'].tolist()
retriever.compute_index(texts, batch_size=32, save_folder='../../data/popqa/snowflake-arctic-embed-l-v2.0/')