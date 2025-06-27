#%%
import torch
import numpy as np
from resources.retrieval_online import ExplainableAutoModelForRetrieval
import pandas as pd
import pickle
import tqdm
#%%
# We use msmarco query and passages as an example
topics = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/Evaluation Dataset/topics.tsv', sep='\t')
retrieval_results = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/code/efra_retrieval/results/ir_results_chunks.csv')

rag = ExplainableAutoModelForRetrieval.from_pretrained(
    'Snowflake/snowflake-arctic-embed-l-v2.0',
    add_pooling_layer = False
).to('cuda' if torch.cuda.is_available() else 'cpu')
queries = topics['query'].tolist()

#%%
# Create RAG model:
list_of_importance_scores = []
for qid, query in tqdm.tqdm(enumerate(queries), total=len(queries), desc='Processing queries'):
    contexts = retrieval_results[retrieval_results['query_id'] == qid]['retrieved_text'].tolist()
    rag('query: ' + query, contexts, output_attentions=True, output_hidden_states=True)
    importance_score = rag.aGrad()
    in_tokens = rag.in_tokens
    importance_score['query_in_tokens'] = rag.in_tokens['query']
    importance_score['context_in_tokens'] = rag.in_tokens['context']
    pickle.dump(importance_score, open(f'/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/index_snowflake/importance_scores/importance_scores_{qid}.pkl', 'wb'))
    ### Save the importance scores for each query and context
    ### the importance scores are dictionaries with keys 'query' and 'context' 
    ### where 'query' is a list of scores for the query and 'context' is a list of scores for each context
    
    
if True: exit()

#%%
def plot_importance(ax, scores, tokens, title):
    assert len(scores) == len(tokens)
    y = np.arange(len(scores))[::-1]
    ax.barh(y, scores)
    ax.set_yticks(y, labels=tokens)
    ax.set_title(title)
