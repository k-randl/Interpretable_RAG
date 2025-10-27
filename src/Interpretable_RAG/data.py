import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from io import StringIO
from html.parser import HTMLParser

from typing import Callable, Optional, Union, List
from transformers import PreTrainedTokenizer

#====================================================================================================#
# Data access and analytics:                                                                         #
#====================================================================================================#

from google.cloud import storage
from google.oauth2 import service_account

DATABASE    = 'c-labs1.efra.summaries_migrated', 'efra.summaries_v2_migrated', 'c-labs1.efra.summaries_v3'
CREDENTIALS = service_account.Credentials.from_service_account_file('data/service-account-external-efra.json')

def query_data(sql:str): return pd.read_gbq(sql, credentials=CREDENTIALS)

def get_unique(columns:Union[str, List[str]], db:int=-1):
    # make list if necessary:
    if isinstance(columns, str):
        columns = [columns,]

    return query_data(f'SELECT DISTINCT {", ".join(columns)} FROM `{DATABASE[db]}`')

def count_values(column:str, values:Union[str, List[str]], db:int=-1):
    # make list if necessary:
    if isinstance(values, str):
        values = [values,]

    return {value:query_data(f'SELECT COUNT({column}) FROM `{DATABASE[db]}` WHERE {column} = "{value}"').values.flatten()[0] for value in values}


#====================================================================================================#
# Data processing:                                                                                   #
#====================================================================================================#

from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from google.cloud.storage.bucket import Bucket

SENT_TOKENIZER = PunktSentenceTokenizer()

from html.parser import HTMLParser

class HTMLSplitter(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = ''

    @property
    def ends_with_space(self):
        if len(self.text) == 0: return True
        else: return self.text[-1] in (' ', '\n', '\r', '\t', '>')

    @property
    def ends_with_newline(self):
        if self.text.endswith('\n  - '): return True
        elif len(self.text) == 0: return True
        else: return self.text[-1] in ('\n', '\r', '>')
    
    def handle_starttag(self, tag, attrs):
        if tag in ('table', 'tr', 'th', 'td'):
            self.text += f'<{tag}>' if self.ends_with_newline else f'\n<{tag}>'
        
        elif tag == 'b' or tag == 'strong':
            self.text += '**' if self.ends_with_space else ' **'

        elif tag == 'i':
            if not self.ends_with_space:
                self.text += '*' if self.ends_with_space else ' *'
        
        elif tag == 'li':
            self.text += '  - ' if self.ends_with_newline else '\n  - '
        
        elif tag == 'p':
            if not self.ends_with_newline:
                self.text += '\n'
        
        elif tag.startswith('h'):
            self.text += '\n**' if self.ends_with_newline else '\n\n**'

        elif not self.ends_with_space:
            self.text += ' '

    def handle_endtag(self, tag):
        if tag in ('table', 'tr', 'th', 'td'):
            self.text += f'</{tag}>'
        
        if tag == 'b' or tag == 'strong':
            self.text += '** '

        elif tag == 'i':
            self.text += '* '
        
        elif tag == 'p':
            if not self.ends_with_space:
                self.text += '\n'
        
        elif tag.startswith('h'):
            self.text += '**\n'

        elif not self.ends_with_space:
            self.text += ' '
    
    def handle_data(self, data):
        self.text += data.replace('\n', '').replace('\r', '').strip()

    def get_data(self):
        return self.text.strip()

def retrieve_url(url:str, window:int, tokenize:Callable[[str], List]=word_tokenize, bucket:Optional[Bucket]=None):
    parts, paragraphs, tokens, texts = [], [], [], []

    # create bucket:
    if bucket is None:
        client = storage.Client(credentials=CREDENTIALS)
        bucket = client.bucket('c-labs1-efra')

    # retrieve urls:
    html = str(bucket.blob(url[18:]).download_as_string())

    # parse html:
    parser = HTMLSplitter()
    parser.feed(html)
    html = parser.get_data()

    # split text in paragraphs:
    part = 0
    cursor = 0
    remaining_tokens = -1
    for paragraph, txt in enumerate(html.split('\n\n')):
        for i,j in SENT_TOKENIZER.span_tokenize(txt):
            sentence = tokenize(txt[i:j+1])
            remaining_tokens -= len(sentence) - 1

            if remaining_tokens <= 0:
                parts.append(str(part))
                paragraphs.append(str(paragraph))
                tokens.append(sentence[:window])
                texts.append(txt[i:j+1])
                remaining_tokens = window - len(sentence) + 1
                    
                part += 1
                cursor = i

            else: 
                tokens[-1].extend(sentence[:window])
                texts[-1] = txt[cursor:j+1]

        remaining_tokens = -1

    return {'part':parts, 'paragraph':paragraphs, 'tokens':tokens, 'texts':texts}


def load_data(columns:Union[str, List[str]], window:int, tokenize:Callable[[List[str]], List]=word_tokenize, limit:Optional[int]=None, where:Optional[str]=None, db:int=-1):
    # make list if necessary:
    if isinstance(columns, str):
        columns = [columns,]

    # build query:
    query = f'SELECT {", ".join(columns)} FROM `{DATABASE[db]}`'
    if where is not None: query += f' WHERE {where}'
    if limit is not None: query += f' LIMIT {limit:d}'
    print(f'SQL query:      "{query};"')

    # download data:
    data = query_data(query).dropna()
    print(f'Retrieved data: {len(data):d} posts')

    # create bucket:
    client = storage.Client(credentials=CREDENTIALS)
    bucket = client.bucket('c-labs1-efra')

    for _, entry in tqdm(data.iterrows()):
        result = {}
        for column in columns:
            result[column] = entry[column]

            if column.endswith('_url'):
                result[column[:-4]] = retrieve_url(url=entry[column], window=window, tokenize=tokenize, bucket=bucket)
            
        yield result

def load_prompts(prompt:str, tokenizer:PreTrainedTokenizer, max_tokens:int, **kwargs):
    prefix = tokenizer.encode('**Task:**\n\n' + prompt)
    suffix = tokenizer.encode('\n\n**Summary:**\n\n')[1:]

    for entry in load_data(
        columns=['post_id', 'english_content_url', 'english_summary'],
        window=max_tokens - len(prefix) - len(suffix),
        tokenize=lambda s: tokenizer.encode(s)[1:],
        **kwargs):

        # add pre and suffix:
        entry['english_content']['tokens'] = [prefix + t + suffix for t in entry['english_content']['tokens']]

        yield entry

def pad(input_ids:torch.Tensor, tokenizer:PreTrainedTokenizer):
    batch_size = len(input_ids)
    seq_length = max([len(ids) for ids in input_ids])

    output = np.ones((batch_size, seq_length), dtype=int) * tokenizer.pad_token_id

    for i, ids in enumerate(input_ids):
        output[i, -len(ids):] = ids

    return torch.tensor(output, device="cuda")

def save_summary(dir:str, df:pd.DataFrame, output_text:List[str], tokenizer:PreTrainedTokenizer, init:bool=False):
    #print('Saving...')
    save_df = df.copy()
    save_df['prompt'] = [tokenizer.decode(m) for m in save_df['prompt']]
    save_df['summary'] = output_text
    save_df.to_csv(os.path.join(dir,'summaries.csv'), mode='w' if init else 'a', index=False, header=init)

    for i in save_df.ID.drop_duplicates():
        txt = ''
    
        for s in save_df[save_df.ID == i]['summary'].values:
            s = s.split('**Summary:**')[-1]
            s = s.split('**Sentence:**')[-1]
            s = s.split('**Answer:**')[-1]
    
            txt += s
    
        with open(os.path.join(dir,f'summary_{i}.md'), 'w') as file:
            file.write(txt)

    del save_df
    gc.collect()