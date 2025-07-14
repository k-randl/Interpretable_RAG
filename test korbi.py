# %%
import os
#os.environ['TRANSFORMERS_CACHE'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import torch
import pandas as pd
from transformers import LlamaForCausalLM

from resources.generation import ExplainableAutoModelForGeneration, plot_shap_attributions, highlight_dominant_passages

from huggingface_hub import login
from getpass import getpass

torch.manual_seed(42)

if os.path.exists('.huggingface.token'):
    with open('.huggingface.token', 'r') as file:
        login(token=file.read())

else: login(token=getpass(prompt='Huggingface login  token: '))

#%%
MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'
MAX_SEQ_LEN = 1024
MAX_GEN_LEN = 256
# %% Load Pipeline:
model = ExplainableAutoModelForGeneration(LlamaForCausalLM).from_pretrained(
    MODEL_ID,
    device_map='auto',
    torch_dtype=torch.bfloat16
)

# We use msmarco query and passages as an example
query =  "Where was Marie Curie born?"
contexts = [
    "Maria Skłodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace.",
    "Maria Skłodowska was born in Warsaw, in Congress Poland in the Russian Empire, as the fifth and youngest child of well-known teachers Bronisława, née Boguska, and Władysław Skłodowski.",
    "While a French citizen, Marie Skłodowska Curie, who used both surnames,[5][6] never lost her sense of Polish identity. She taught her daughters the Polish language and took them on visits to Poland.",
    "Marie Curie founded the Curie Institute in Paris in 1920, and the Curie Institute in Warsaw in 1932.",
]

# %%
output = model.explain_generate(
    query,
    contexts,
    max_new_tokens=MAX_GEN_LEN,
    do_sample=False,
    top_p=1,
    temperature=0.7,
    num_beams=1,
    max_samples=31
)


# %%
plot_shap_attributions(
    model.get_shapley_values('token'),
    model.tokenizer.batch_decode(model.tokenizer(output, return_tensors='np', add_special_tokens=False).input_ids.T),
    normalize=True
)
#%%
highlight_dominant_passages(
    model.get_shapley_values('token'),
    model.tokenizer.batch_decode(model.tokenizer(output, return_tensors='np', add_special_tokens=False).input_ids.T),
)


#%%

p_shap = p_pos + p_neg
p_gen = model.gen_token_probs.flatten()
p_bas = model.cmp_token_probs[0].flatten()

plt.bar(range(len(p_gen)), p_gen-p_bas-p_shap)

# %%
model.cmp_token_probs