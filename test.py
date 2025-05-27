# %%
import os
import torch
from transformers import LlamaForCausalLM

from resources.generation import ExplainableGenerator

# %%
from huggingface_hub import login
from getpass import getpass

torch.manual_seed(42)
if os.path.exists('.huggingface.token'):
    with open('.huggingface.token', 'r') as file:
        login(token=file.read())

else: login(token=getpass(prompt='Huggingface login  token: '))

# %%
MAX_SEQ_LEN = 1024
MAX_GEN_LEN = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE

# %% Load Pipeline:
model = ExplainableGenerator(LlamaForCausalLM).from_pretrained(
    '',
    max_seq_len = MAX_SEQ_LEN,
    max_gen_len = MAX_GEN_LEN,
    device_map='auto',
    torch_dtype=torch.bfloat16
).to()
# %%
