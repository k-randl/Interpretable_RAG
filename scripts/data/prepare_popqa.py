#%%
import sys
sys.path.insert(0, "../..")

import wikipedia
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from nltk.tokenize import word_tokenize

from src.Interpretable_RAG.data import load_html

from typing import List, Callable

wikipedia.set_lang("en")
def load_data_wiki(topics:List[str], window:int, *, tokenize:Callable[[str], List[str]]=word_tokenize, output_tokens:bool=False, explored_urls:List[str]=[]):
    for topic in tqdm(topics):
        try: pages = [wikipedia.page(topic)]

        except wikipedia.PageError as e:
            topic = wikipedia.search(topic)[0]
            try: pages = [wikipedia.page(topic)]
            except: continue

        except wikipedia.DisambiguationError as e:
            pages = []
            for t in str(e).split('\n')[1:]:
                try: pages.append(wikipedia.page(t))
                except: continue

        # process document:
        for page in pages:
            if page.url not in explored_urls:
                explored_urls.append(page.url)

                html = page.html()
                if not html:
                    html = page.content
                    print('Warning: html could not be loaded. Falling back to content.')
                for text, i, j in load_html(html=html, window=window-len(tokenize(topic)), tokenize=tokenize, output_tokens=output_tokens, handle_wiki_tags=True):
                    text = f'**{topic}:**\n' + text.strip()

                    if len(text) > 10*window:
                        yield topic, text[:(10*window)-3]+'...', page.url, i, j

                    elif len(text) > 0:
                        yield topic, text, page.url, i, j

#%% Load the PopQA dataset from Hugging Face
dataset = load_dataset("akariasai/PopQA")['test'].shuffle(seed=42)
topics  = dataset.select_columns(['id', 'question', 'possible_answers'])[:200]
dataset = dataset.select_columns(['id', 's_wiki_title', 'o_wiki_title'])[:200]
assert dataset['id'] == topics['id']
topics = pd.DataFrame(topics).rename(columns={'id':'query_id', 'question':'query'})
topics.to_csv('../../data/popqa/topics.tsv', sep='\t', index=False)
dataset

#%% Get relevant contexts from Wikipedia
chunks_subj = pd.DataFrame(
    [chunk for chunk in load_data_wiki(dataset['s_wiki_title'], 50)],
    columns=['topic', 'text', 'url', 'paragraph', 'part']
)
chunks_obj = pd.DataFrame(
    [chunk for chunk in load_data_wiki(dataset['o_wiki_title'], 50, explored_urls=chunks_subj['url'].unique().tolist())],
    columns=['topic', 'text', 'url', 'paragraph', 'part']
)
chunks = pd.concat([chunks_subj, chunks_obj]).reset_index(drop=True)

#%% Annotate relevant queries
ids = [[] for _ in range(len(chunks))]
for id, s, o in tqdm(zip(dataset['id'], dataset['s_wiki_title'], dataset['o_wiki_title'])):
    for index in (chunks[chunks['topic']==s].index.tolist() +
                  chunks[chunks['topic']==o].index.tolist()):
        ids[index].append(id)
chunks['ids'] = ids
chunks.to_csv('../../data/popqa/passages.csv')