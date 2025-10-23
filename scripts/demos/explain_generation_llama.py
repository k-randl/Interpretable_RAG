# %%
import os
import sys
sys.path.insert(0, "..")

import torch
from resources.generation import ExplainableAutoModelForGeneration
from code.Interpretable_RAG.resources.plotting_16_10_2025 import visualize_attribution_generator

from huggingface_hub import login
from getpass import getpass

# %%
if os.path.exists('.huggingface.token'):
    with open('.huggingface.token', 'r') as file:
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
    max_samples=64,
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