# %%
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
import torch
import numpy as np
import pandas as pd
from transformers import LlamaForCausalLM

from resources.generation import ExplainableAutoModelForGeneration
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
gen_probs = model._gen_probs
# %%
first_doc_removed_prompt = create_rag_prompt(query, contexts[1:])
output_first_doc_removed = model.compare(
    [first_doc_removed_prompt],output if type(output) is list else [output],batch_size = 4
)
exp_probs = model._exp_probs
# %%
output_not_constrained = model.compare(
    [first_doc_removed_prompt],    max_new_tokens=MAX_GEN_LEN,
    do_sample=False,
    top_p=1,
    temperature=0.7,
    num_beams=1,
    return_dict_in_generate=True,
    output_scores=True
)
not_cont_probs = model._exp_probs

# %%
idx = np.argsort(model.gen_bow_probs)[0,::-1]
model.tokenizer.batch_decode(idx[:10])

# %%
idx = np.argsort(model.cmp_bow_probs)[0,::-1]
model.tokenizer.batch_decode(idx[:10])


#%%


#%%
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
# %%
exp_nucleus = nucleus_sampling_preserve_order(exp_probs, p=0.9)
meaned_exp_nucleus = exp_nucleus.mean(dim=1)
idx_meaned_exp_nucleus = torch.argsort(meaned_exp_nucleus, dim=-1, descending=True)[0,:100]
exp_tokens = model.tokenizer.batch_decode(idx_meaned_exp_nucleus)
# %%
gen_nucleus = nucleus_sampling_preserve_order(gen_probs, p=0.9)
meaned_gen_nucleus = gen_nucleus.mean(dim=1)
idx_meaned_gen_nucleus = torch.argsort(meaned_gen_nucleus, dim=-1, descending=True)[0,:100]
gen_tokens =  model.tokenizer.batch_decode(idx_meaned_gen_nucleus)
#%%
not_cont_nucleus = nucleus_sampling_preserve_order(not_cont_probs, p=0.9)
meaned_not_cont_nucleus = not_cont_nucleus.mean(dim=1)
idx_meaned_not_cont_nucleus = torch.argsort(meaned_not_cont_nucleus, dim=-1, descending=True)[0,:100]
not_cont_tokens =  model.tokenizer.batch_decode(idx_meaned_not_cont_nucleus)
# %%
def clean_tokens(tokens):
    stop_words = list(stopwords.words('english'))
    stop_words += ['', '\n', '.', ',', '(', ')', '[', ']', '{', '}', ':', ';', '?', '!', '"', "'", '-', '_', '/', '\\', '*', '&', '^', '%', '$', '#', '@', '~']
    tokens_filtered = [token.strip() for token in tokens if token.strip(' \n') not in stop_words + ['']]
    return tokens_filtered

#%%
gen_tokens_filtered = [token.strip() for token in gen_tokens if token.strip(' \n') not in stop_words + ['']]
exp_tokens_filtered = [token.strip() for token in exp_tokens if token.strip(' \n') not in stop_words + ['']]
not_cont_tokens_filtered = [token.strip() for token in not_cont_tokens if token.strip(' \n') not in stop_words + ['']]
# %%
shared =[token for token in gen_tokens_filtered if token in exp_tokens_filtered]
shared_not_cont = [token for token in gen_tokens_filtered if token in not_cont_tokens_filtered]
shared_exp_not_cont = [token for token in exp_tokens_filtered if token in not_cont_tokens_filtered]

#%%
#prompts = generate_ordered_prompt_variations(query, contexts, min_docs=2, max_docs=)
prompts = []
for i in range(len(contexts)):
    prompts.append((create_rag_prompt(query, contexts[:i]+contexts[i+1:]),'masking document '+str(i+1)))

# %%
def generate_outputs(rag_prompt,output, model,batch_size, MAX_GEN_LEN=MAX_GEN_LEN ):
    if output is None:
        output = output if type(output) is list else [output]
        perturbed_output = model.compare( [rag_prompt],output,batch_size = batch_size)
        gen_probs = model._gen_probs
        exp_probs = model._exp_probs
        meaned_gen_nucleus =  model.gen_nucleus_probs()
        idx_meaned_gen_nucleus = torch.argsort(meaned_gen_nucleus, dim=-1, descending=True)[0,:]
        gen_tokens =  model.tokenizer.batch_decode(idx_meaned_gen_nucleus)
        meaned_exp_nucleus = model.exp_nucleus_probs()
        idx_meaned_exp_nucleus = torch.argsort(meaned_exp_nucleus, dim=-1, descending=True)[0,:]
        exp_tokens = model.tokenizer.batch_decode(idx_meaned_exp_nucleus)
        return perturbed_output, gen_tokens, exp_tokens
    else:
        perturbed_output = model.compare( [rag_prompt],batch_size = batch_size)
        gen_probs = model._gen_probs
        exp_probs = model._exp_probs
        meaned_gen_nucleus =  model.gen_nucleus_probs()
        idx_meaned_gen_nucleus = torch.argsort(meaned_gen_nucleus, dim=-1, descending=True)[0,:]
        gen_tokens =  model.tokenizer.batch_decode(idx_meaned_gen_nucleus)
        meaned_exp_nucleus = model.exp_nucleus_probs()
        idx_meaned_exp_nucleus = torch.argsort(meaned_exp_nucleus, dim=-1, descending=True)[0,:]
        exp_tokens = model.tokenizer.batch_decode(idx_meaned_exp_nucleus)
        return perturbed_output, gen_tokens, exp_tokens
           

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
meaned_gen_nucleus = model.gen_nucleus_probs()
idx_meaned_gen_nucleus = torch.argsort(meaned_gen_nucleus, dim=-1, descending=True)[0,:]
gen_tokens =  model.tokenizer.batch_decode(idx_meaned_gen_nucleus)
#%%
perturbed_outputs = {}
for prompt, description in tqdm(prompts):
    perturbed_output, gen_tokens, exp_tokens = generate_outputs(prompt, output, model, batch_size=64)
    perturbed_outputs[description] = {
        'perturbed_output': perturbed_output,
        'gen_tokens': gen_tokens,
        'exp_tokens': exp_tokens
    }
    #print(f"Generated tokens for {description}: {gen_tokens[:10]}")  # Print first 10 tokens for brevity
