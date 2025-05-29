# %%
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import torch
import numpy as np
import pandas as pd
from transformers import LlamaForCausalLM
from tqdm import tqdm
from resources.generation import ExplainableAutoModelForGeneration
import nltk
from nltk.corpus import stopwords
#%%%
#DEFINE METHODS

def clean_tokens(tokens):
    stop_words = list(stopwords.words('english'))
    stop_words += ['', '\n', '.', ',', '(', ')', '[', ']', '{', '}', ':', ';', '?', '!', '"', "'", '-', '_', '/', '\\', '*', '&', '^', '%', '$', '#', '@', '~']
    tokens_filtered = [token.strip() for token in tokens if token.strip(' \n') not in stop_words + ['']]
    return tokens_filtered

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

def generate_outputs(rag_prompt,output, model,batch_size, MAX_GEN_LEN=256 ):
    if type(rag_prompt) is str:
        rag_prompt = [rag_prompt]
    if output is not None:
        output = output if type(output) is list else [output]
        perturbed_output = model.compare( rag_prompt,output,batch_size = batch_size, max_new_tokens=MAX_GEN_LEN)
    else:
        perturbed_output = model.compare(rag_prompt,batch_size = batch_size, max_new_tokens=MAX_GEN_LEN,do_sample= False,    return_dict_in_generate=True,output_scores=True )
    gen_probs = model._gen_probs
    exp_probs = model._exp_probs
    meaned_gen_nucleus =  model.gen_nucleus_probs()
    idx_meaned_gen_nucleus = torch.argsort(meaned_gen_nucleus, dim=-1, descending=True)[0,:]
    gen_tokens =  model.tokenizer.batch_decode(idx_meaned_gen_nucleus)
    meaned_exp_nucleus = model.cmp_nucleus_probs()
    idx_meaned_exp_nucleus = torch.argsort(meaned_exp_nucleus, dim=-1, descending=True)[0,:]
    exp_tokens = model.tokenizer.batch_decode(idx_meaned_exp_nucleus)
    return perturbed_output, gen_tokens, exp_tokens, gen_probs, exp_probs
           
#%%
MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'
MAX_SEQ_LEN = 1024
MAX_GEN_LEN = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# %% Load Pipeline:
model = ExplainableAutoModelForGeneration(LlamaForCausalLM).from_pretrained(
    MODEL_ID,

    device_map='auto',
    torch_dtype=torch.bfloat16
)
# %%
ranked_chunks = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/code/efra_retrieval/results/ir_results_chunks.csv')
topics = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/Evaluation Dataset/topics.tsv', sep='\t')
query = topics.iloc[0]['query']  # Assuming the first query for example
contexts = ranked_chunks[ranked_chunks['query_id'] == 0]['retrieved_text'].tolist() # Assuming query_id 0 for example
complete_rag_prompt = create_rag_prompt(query, contexts)
perturbed_prompts = []
for i in range(len(contexts)):
    perturbed_prompts.append((create_rag_prompt(query, contexts[:i]+contexts[i+1:]),'masking document '+str(i+1)))

# %%
output = model.generate(
    [complete_rag_prompt],
    max_new_tokens=MAX_GEN_LEN,
    do_sample=False,
    top_p=1,
    temperature=0.7,
    num_beams=1,
    return_dict_in_generate=True,
    output_scores=True
    )
meaned_gen_nucleus = model.gen_nucleus_probs()
idx_meaned_gen_nucleus = torch.argsort(meaned_gen_nucleus, dim=-1, descending=True)[0,:]
gen_tokens =  model.tokenizer.batch_decode(idx_meaned_gen_nucleus)
#%%
BATCH_SIZE = 16
perturbed_outputs = {}
i= 0
for prompt, description in (perturbed_prompts):
    i+=1
    print(f'Processing {str(i)}...')
    perturbed_output, gen_tokens, exp_tokens, gen_probs, exp_probs = generate_outputs(prompt, None, model, batch_size=BATCH_SIZE, MAX_GEN_LEN=MAX_GEN_LEN)
    perturbed_outputs[description] = {
        'perturbed_output': perturbed_output,
        'gen_tokens': gen_tokens,
        'exp_tokens': exp_tokens,
        'gen_probs': gen_probs,
        'exp_probs': exp_probs
    }
    perturbed_output, gen_tokens, exp_tokens, gen_probs, exp_probs = generate_outputs(prompt, output, model, batch_size=BATCH_SIZE, MAX_GEN_LEN=MAX_GEN_LEN)
    perturbed_outputs[description+'_constrained'] = {
        'perturbed_output': perturbed_output,
        'gen_tokens': gen_tokens,
        'exp_tokens': exp_tokens,
        'gen_probs': gen_probs,
        'exp_probs': exp_probs
    }


# %%
perturbed_outputs['complete'] = {
    'perturbed_output': output,
    'gen_tokens': gen_tokens,
    'exp_tokens': '',
    'gen_probs': model._gen_probs,
    'exp_probs': ''
}

#%%
### Save the perturbed outputs to a file
import pickle
with open('perturbed_outputs.pkl', 'wb') as f:
    pickle.dump(perturbed_outputs, f)