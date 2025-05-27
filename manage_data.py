#%%
import pandas as pd

chunks_with_id_path = '/home/francomaria.nardini/raid/guidorocchietti/code/efra_paper/chunks_df_relevance.csv'
chunks_with_id_df = pd.read_csv(chunks_with_id_path, index_col=0)
# %%
subset_with_rel = chunks_with_id_df[chunks_with_id_df['relevance'] != '0']
subset_without_rel = chunks_with_id_df[chunks_with_id_df['relevance'] == '0'].sample(n=len(subset_with_rel), random_state=42)
#%%
subset_for_explainability = pd.concat([subset_with_rel, subset_without_rel])


# %%
subset_for_explainability.to_csv('subset_for_explainability.csv', index =  False)

#%%
import pickle
path = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/index_snowflake/importance_scores/importance_scores_0.pkl'
with open(path, 'rb') as f:
    importance_scores = pickle.load(f)
# %%
