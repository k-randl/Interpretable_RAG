# %%
import os
import sys
sys.path.insert(0, "../..")

import torch
from src.Interpretable_RAG.plotting import visualize_attribution_generator
from src.Interpretable_RAG.generation import ExplainableAutoModelForGeneration

from huggingface_hub import login
from getpass import getpass

# %%
TOKEN_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '.huggingface.token')
if os.path.exists(TOKEN_PATH):
    with open(TOKEN_PATH, 'r') as file:
        login(token=file.read())

else: login(token=getpass(prompt='Huggingface login  token: '))

#%%
MAX_SEQ_LEN = 1024
MAX_GEN_LEN = 256
# %% Load Pipeline:
generator = ExplainableAutoModelForGeneration.from_pretrained(
    pretrained_model_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
    device_map='auto',
    torch_dtype=torch.bfloat16
)

# We use msmarco query and passages as an example
query =  "Where was Marie Curie born?"
contexts = [
    "Maria Skłodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace.",
    "Maria Skłodowska was born in Warsaw, in Congress Poland in the Russian Empire, as the fifth and youngest child of well-known teachers Bronisława, née Boguska, and Władysław Skłodowski.",
    "While a French citizen, Marie Skłodowska Curie, who used both surnames, never lost her sense of Polish identity. She taught her daughters the Polish language and took them on visits to Poland.",
    "Marie Curie founded the Curie Institute in Paris in 1920, and the Curie Institute in Warsaw in 1932.",
]

# %%
output = generator.explain_generate(
    query=query,
    contexts=contexts,
    max_new_tokens=MAX_GEN_LEN,
    do_sample=False,
    top_p=1,
    num_beams=1,
    batch_size=64,
    max_samples_query=24,
    max_samples_context=24,
    conditional=True
)
output

# %%
visualize_attribution_generator(generator, aggregation='token', token_processor=lambda s: s.replace('Ġ', ' ').strip('Ċ'))

#%%
visualize_attribution_generator(generator, aggregation='bow')

#%%
visualize_attribution_generator(generator, aggregation='nucleus')

#%%
visualize_attribution_generator(generator, aggregation='sequence')