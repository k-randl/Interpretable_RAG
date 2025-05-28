# %%
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import torch
import pandas as pd
from transformers import LlamaForCausalLM

from resources.generation import ExplainableAutoModelForGeneration, _to_batch
'''
from huggingface_hub import login
from getpass import getpass

torch.manual_seed(42)

if os.path.exists('.huggingface.token'):
    with open('.huggingface.token', 'r') as file:
        login(token=file.read())

else: login(token=getpass(prompt='Huggingface login  token: '))
'''
#%%
MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'
MAX_SEQ_LEN = 1024
MAX_GEN_LEN = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE
# %% Load Pipeline:
model = ExplainableAutoModelForGeneration(LlamaForCausalLM).from_pretrained(
    MODEL_ID,
    #max_seq_len = MAX_SEQ_LEN,
    #max_gen_len = MAX_GEN_LEN,
    device_map='auto',
    torch_dtype=torch.bfloat16
)
# %%
ranked_chunks = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/code/efra_retrieval/results/ir_results_chunks.csv')
topics = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/Evaluation Dataset/topics.tsv', sep='\t')
# %%
def create_rag_prompt(query, contexts):
    """
    Creates a RAG prompt for the Llama model.

    Args:
        query (str): The user's query.
        contexts (list): A list of strings representing the retrieved documents.

    Returns:
        str: The formatted RAG prompt.
    """
    prompt = f"You are an expert of Food and Risk related legislation. Assist me in answering the User demand. Use the following list of documents, ranked from high to low relevancy, to answer the user's query.\n\n"

    for i, context in enumerate(contexts[:10]):  # Use top 10 contexts
        prompt += f"Document {i+1}:\n{context}\n\n"

    prompt += f"Query: {query}\n\nAnswer:"
    return prompt

# Example usage:
query = topics.iloc[0]['query']  # Assuming the first query for example
contexts = ranked_chunks[ranked_chunks['query_id'] == 0]['retrieved_text'].tolist() # Assuming query_id 0 for example
#%%
rag_prompt = create_rag_prompt(query, contexts)
#   print(rag_prompt)

# You would then tokenize this prompt and pass it to the model for generation
# input_ids = model.tokenizer(rag_prompt, return_tensors="pt").input_ids.to(DEVICE)
# generated_output = model.generate(input_ids)
# print(model.tokenizer.decode(generated_output[0], skip_special_tokens=True))

# %%
output = model.generate(
    rag_prompt,
    max_new_tokens=MAX_GEN_LEN,
    do_sample=False,
    top_p=1,
    temperature=0.7,
    num_beams=1,
    return_dict_in_generate=True,
    output_scores=True
)

# %%
first_doc_removed_prompt = create_rag_prompt(query, contexts[1:])
output_first_doc_removed = model.compare(
    [first_doc_removed_prompt],output if type(output) is list else [output],batch_size = 4
)
# %%
exp_probs = model._exp_probs
gen_probs = model._gen_probs