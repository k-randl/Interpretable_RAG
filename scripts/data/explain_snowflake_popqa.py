#%%
import os
import sys
sys.path.insert(0, "../..")
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4,5"

import torch
import pandas as pd
from src.Interpretable_RAG.rag import ExplainableAutoModelForRAG

#%%
model = ExplainableAutoModelForRAG(
    query_encoder_name_or_path='Snowflake/snowflake-arctic-embed-l-v2.0',
#    dir='../../data/popqa/snowflake-arctic-embed-l-v2.0/',

    retriever_query_format='query: {query}',
    retriever_token_processor=lambda s: s.replace('▁', ' '),
    retriever_kwargs={'add_pooling_layer':False},

    generator_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
    generator_token_processor=lambda s: s.replace('Ġ', ' ').strip('Ċ'),
    generator_kwargs={'device_map':'auto', 'dtype':torch.bfloat16}
)

#%%
if os.path.exists('../../results/snowflake-arctic-embed-l-v2.0/results_popqa.csv'):
    results = pd.read_csv('../../results/snowflake-arctic-embed-l-v2.0/results_popqa.csv', index_col=None)

else:
    results = pd.read_csv('../../results/snowflake-arctic-embed-l-v2.0/warg_popqa.csv', index_col=None)
    results['generation'] = [None for _ in range(len(results))]

#%%
for i, (qry, ctx) in enumerate(results[['query', 'passages']].values):
    if not results['generation'].isnull()[i]:
        print(f'Skipping query {i:d}.')
        continue

    ctx = eval(ctx)
    results.loc[i,'generation'] = model(
        query=qry,
        contexts=ctx,
        generator_kwargs={
            'max_new_tokens':256,
            'do_sample':False,
            'top_p':1,
            'num_beams':1,
            'batch_size':32,
            'max_samples_query':32,
            'max_samples_context':32,
            'conditional':True
        }
    )[-1]['content']
    model.save_values('../../results/snowflake-arctic-embed-l-v2.0/explanations_popqa/', f'{i:d}.pkl',
                      ret_methods=['intGrad'], gen_aggregations=['token'], batch_size=16)
    results.to_csv('../../results/snowflake-arctic-embed-l-v2.0/results_popqa.csv', index=False)