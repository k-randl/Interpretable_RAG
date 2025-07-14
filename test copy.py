# %%
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import torch
import numpy as np
import pandas as pd
from transformers import LlamaForCausalLM
from tqdm import tqdm
from resources.generation import ExplainableAutoModelForGeneration, plot_shap_attributions, highlight_dominant_passages

import nltk
from nltk.corpus import stopwords

           
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
evaluation_dataset = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/code/efra_retrieval/validation_Dataset_with_chunks_ids.csv')    
topics = pd.read_csv('/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/Evaluation Dataset/topics.tsv', sep='\t')
do_evalualuation = True

#%%
input_dict = {}
queries = topics['query'].tolist()
contexts = ranked_chunks['retrieved_text'].groupby(ranked_chunks['query_id']).apply(list).to_dict()
contexts_topfive = {k: v[:5] for k, v in contexts.items()}  # Limit to top 5 contexts
#%%
#query = queries[0:4]  # Example query
#context = [contexts[i][:5] for i in range(4)]  # Example contexts for the first query
query = queries[0]  # Example query
context = contexts_topfive[0]  # Example contexts for the first query
# %%
import time
start_time = time.time()
output = model.explain_generate(
    query,
    context,
    max_new_tokens=MAX_GEN_LEN,
    do_sample=False,
    top_p=1,
    temperature=0.7,
    num_beams=1,
    max_samples=32
)
end_time = time.time()
print(f"Time taken for generation: {end_time - start_time} seconds")
#%%
start_time = time.time()
save_path = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/results_with_shap/'
reults = model.save_values(save_path+'prova.pkl')
end_time = time.time()
print(f"Time taken for saving results: {end_time - start_time} seconds")
#%%
'''
for qid in tqdm(range(len(topics)), desc='Processing queries',total=len(topics)):
    query = topics.iloc[qid]['query']
    if do_evalualuation: 
        contexts = evaluation_dataset[evaluation_dataset['query'] == query]['text'].tolist()
    else:
        contexts = ranked_chunks[ranked_chunks['query_id'] == qid]['retrieved_text'].tolist()
    complete_rag_prompt = create_rag_messages(query, contexts)
    complete_rag_prompt = model.tokenizer.apply_chat_template(complete_rag_prompt, tokenize=False, add_generation_prompt=True)
    perturbed_prompts = []
    for j in range(len(contexts)):
        perturbed_prompts.append((create_rag_messages(query, contexts[:j]+contexts[j+1:]),'masking document '+str(j+1)))
    chat_templates = [(model.tokenizer.apply_chat_template(x[0], tokenize=False, add_generation_prompt=True), x[1]) for x in perturbed_prompts]
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
    BATCH_SIZE = 8
    perturbed_outputs = {}
    i= 0
    for prompt, description in (chat_templates):
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

    perturbed_outputs['complete'] = {
        'perturbed_output': output,
        'gen_tokens': gen_tokens,
        'exp_tokens': '',
        'gen_probs': model._gen_probs,
        'exp_probs': ''
    }

    ### Save the perturbed outputs to a file
    import pickle
    import datetime
    output_folder = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/outputs_evaluation/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(os.path.join(output_folder,f'perturbed_outputs_query_{qid}.pkl')):
        with open(os.path.join(output_folder,f'perturbed_outputs_query_{qid}.pkl'), 'wb') as f:
            pickle.dump(perturbed_outputs, f)
    else:
        with open(os.path.join(output_folder,str(datetime.datetime.now().strftime('%Y%m%d_%H%M'))+'perturbed_outputs.pkl'), 'wb') as f:
            pickle.dump(perturbed_outputs, f)
# %%

query = topics.iloc[1]['query']  # Assuming the first query for example
query_id = topics.iloc[1]['query_id']  # Assuming the first query_id for example
contexts = ranked_chunks[ranked_chunks['query_id'] == query_id]['retrieved_text'].tolist() # Assuming query_id 0 for example
complete_rag_prompt = create_rag_messages(query, contexts)
complete_rag_prompt = model.tokenizer.apply_chat_template(complete_rag_prompt, tokenize=False, add_generation_prompt=True)
perturbed_prompts = []
for i in range(len(contexts)):
    perturbed_prompts.append((create_rag_messages(query, contexts[:i]+contexts[i+1:]),'masking document '+str(i+1)))
chat_templates = [(model.tokenizer.apply_chat_template(x[0], tokenize=False, add_generation_prompt=True),x[1]) for x in perturbed_prompts]
# %%

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
BATCH_SIZE = 8
perturbed_outputs = {}
i= 0
for prompt, description in (chat_templates):
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
import datetime
if not os.path.exists('perturbed_outputs.pkl'):
    with open('perturbed_outputs.pkl', 'wb') as f:
        pickle.dump(perturbed_outputs, f)
else:
    with open(str(datetime.datetime.now().strftime('%Y%m%d_%H%M'))+'perturbed_outputs.pkl', 'wb') as f:
        pickle.dump(perturbed_outputs, f)
'''