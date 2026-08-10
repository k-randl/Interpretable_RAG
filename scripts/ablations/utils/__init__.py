import os
from getpass import getpass
def huggingface_login(token_path:str):
    if os.path.exists(token_path):
        with open(token_path, 'r') as file:
            token = file.read().strip()
    else:
        token = getpass(prompt='Huggingface login  token: ')

    try:
        from huggingface_hub import login
        login(token=token)
    except ImportError:
        # some huggingface_hub versions ship a broken (circular-import) `login()`;
        # fall back to writing the token directly to the standard HF cache location,
        # which `from_pretrained(...)` etc. pick up automatically.
        os.environ['HF_TOKEN'] = token
        cache_path = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'token')
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as file:
            file.write(token)


import json
from datasets import load_dataset
def load_ms_marco(num_samples:int=200, use_long_queries:bool=False):
    '''Loads the MS MARCO dataset from huggingface'''

    # Load the MS MARCO dataset version 2.1
    dataset = load_dataset("ms_marco", "v2.1", split="train")

    # Get a random sample of 200 documents
    sample = dataset.shuffle(seed=42).select(range(num_samples))

    # Set dataset format:
    if use_long_queries:
        with open(os.path.join(os.path.dirname(__file__), 'queries.json'), 'r') as file:
            queries = json.load(file) 

        sample = sample.map(lambda item: {'query':queries[item['query']], 'context':item['passages']['passage_text']})

    else: sample = sample.map(lambda item: {'query':item['query'], 'context':item['passages']['passage_text']})

    return sample