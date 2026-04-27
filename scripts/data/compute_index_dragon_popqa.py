import os
import sys
sys.path.insert(0, "../..")
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5,7"

import torch
import pandas as pd
from src.Interpretable_RAG.retrieval_online import ExplainableAutoModelForRetrieval

retriever = ExplainableAutoModelForRetrieval.from_pretrained(
    'facebook/dragon-plus-query-encoder',
    'facebook/dragon-plus-context-encoder'
).to('cuda' if torch.cuda.is_available() else 'cpu')

texts = pd.read_csv('../../data/popqa/passages.csv')['text'].tolist()
retriever.compute_index(texts, batch_size=64, max_length=512,
                        save_folder='../../data/popqa/facebook-dragon-plus/')