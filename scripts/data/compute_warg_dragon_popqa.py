import os
import sys
sys.path.insert(0, "../..")
#os.environ["CUDA_VISIBLE_DEVICES"] = "3,5,7"

import torch
import pandas as pd
from src.Interpretable_RAG.rag import WARGScorer

scorer = WARGScorer('../../data/popqa/facebook-dragon-plus/',
    query_encoder_name_or_path='facebook/dragon-plus-query-encoder',
    context_encoder_name_or_path='facebook/dragon-plus-context-encoder',
    retriever_query_format='query: {query}',
    retriever_kwargs={'add_pooling_layer':False},

    generator_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
    generator_kwargs={'device_map':'auto', 'dtype':torch.bfloat16}
)

warg = pd.read_csv('../../data/popqa/topics.tsv', sep='\t', index_col=None)
for p in range(1,10):
    warg[f'p={p*.1:0.2f}'] = scorer(warg['query'].tolist(), k=10, p=p*.1, batch_size=24,
                                    retriever_kwargs={'max_length':512},
                                    generator_kwargs={'max_document_size':2000},
                                    checkpoint_path='../../results/facebook-dragon-plus/scorer_popqa.json')
warg['passages'] = [scorer._queries[q][2] for q in warg['query'].tolist()]
warg.to_csv('../../results/facebook-dragon-plus/warg_popqa.csv', index=False)