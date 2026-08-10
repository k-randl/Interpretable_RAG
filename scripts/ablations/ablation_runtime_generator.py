# %%% ===============================================================================================#
# Setup:                                                                                             #
#====================================================================================================#

import os
import sys
sys.path.insert(0, "../..")

# Paths:
TOKEN_PATH   = os.path.join(os.path.dirname(__file__), '..', '..', '.huggingface.token')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results', 'runtime_generator')
COMPLEMENTARY = True
os.makedirs(RESULTS_PATH, exist_ok=True)

# Parameters:
STEP_SIZE    = 1
NUM_DOCS     = 5

# %% Load data sample:
from utils import huggingface_login, load_ms_marco
huggingface_login(TOKEN_PATH)
sample = load_ms_marco(num_samples=10)
sample = list(zip(sample['query'], sample['context'], strict=True))

# %%% ===============================================================================================#
# Load RAG-E Pipeline:                                                                               #
#====================================================================================================#
import time
import torch
from tqdm.autonotebook import tqdm
from src.Interpretable_RAG.generation import ExplainableAutoModelForGeneration

generator = ExplainableAutoModelForGeneration.from_pretrained(
    pretrained_model_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
    device_map='auto',
    dtype=torch.bfloat16
)

# calculate explanations:
lime = []
for i, (qry, ctx) in enumerate(tqdm(sample, desc='Testing lime')):
    item = {}
    item['query']       = qry
    item['contexts']    = ctx

    t0 = time.time_ns()
    item['answer']      = generator.explain_generate(qry, ctx)
    item['attribution'] = generator.lime('context')
    item['dt']          = (time.time_ns() - t0) / 1e9

    lime.append(item)

print(lime)

# calculate explanations:
kshap = []
for i, (qry, ctx) in enumerate(tqdm(sample, desc='Testing kSHAP')):
    item = {}
    item['query']       = qry
    item['contexts']    = ctx

    t0 = time.time_ns()
    item['answer']      = generator.explain_generate(qry, ctx)
    item['attribution'] = generator.shap('context', num_samples=1, sample_size=1000000)
    item['dt']          = (time.time_ns() - t0) / 1e9

    kshap.append(item)

print(kshap)

# calculate explanations:
pmcshap = []
for i, (qry, ctx) in enumerate(tqdm(sample, desc='Testing pmcSHAP')):
    item = {}
    item['query']       = qry
    item['contexts']    = ctx

    t0 = time.time_ns()
    item['answer']      = generator.explain_generate(qry, ctx)
    item['attribution'] = generator.shap('context')
    item['dt']          = (time.time_ns() - t0) / 1e9

    pmcshap.append(item)

print(pmcshap)

# %%% ===============================================================================================#
# Load ContextCite Pipeline:                                                                         #
#====================================================================================================#

import time
import torch
from tqdm.autonotebook import tqdm
from utils.context_cite import ContextCiteAutoModel

generator = ContextCiteAutoModel('meta-llama/Llama-3.1-8B-Instruct',
    device_map='auto',
    dtype=torch.bfloat16
)

# calculate explanations:
cc = []
for qry, ctx in tqdm(sample, desc='Testing ContextCite'):
    item = {}
    item['query']       = qry
    item['contexts']    = ctx

    t0 = time.time_ns()
    item['answer']      = generator(qry, ctx)
    item['attribution'] = generator.explain(qry, ctx)
    item['dt']          = (time.time_ns() - t0) / 1e9

    cc.append(item)

print(cc)

# %%% ===============================================================================================#
# Load MIRAGE Pipeline:                                                                              #
#====================================================================================================#

import time
import torch
from tqdm.autonotebook import tqdm
from utils.mirage import MirageAutoModel

generator = MirageAutoModel('meta-llama/Llama-3.1-8B-Instruct',
    device_map='auto',
    dtype=torch.bfloat16
)

# calculate explanations:
mirage = []
for qry, ctx in tqdm(sample, desc='Testing MIRAGE'):
    item = {}
    item['query']       = qry
    item['contexts']    = ctx

    t0 = time.time_ns()
    item['answer']      = generator(qry, ctx)
    item['attribution'] = generator.explain(qry, ctx)
    item['dt']          = (time.time_ns() - t0) / 1e9

    mirage.append(item)

print(mirage)

# %%
import pandas as pd

file = os.path.join(RESULTS_PATH, 'runtime.json')
if os.path.exists(file):
    results = pd.read_json(file)
else: results = pd.DataFrame()

if 'lime' in locals():
    print('Found new LIME')
    results['lime'] = [item['dt'] for item in lime]

if 'kshap' in locals():
    print('Found new kSHAP')
    results['kshap'] = [item['dt'] for item in kshap]

if 'pmcshap' in locals():
    print('Found new pmcSHAP')
    results['pmcshap'] = [item['dt'] for item in pmcshap]

if 'context_cite' in locals():
    print('Found new ContextCite')
    results['context_cite'] = [item['dt'] for item in cc]

if 'mirage' in locals():
    print('Found new MIRAGE')
    results['mirage'] = [item['dt'] for item in mirage]

results.to_json(os.path.join(RESULTS_PATH, 'runtime.json'))

print(results.describe().T.to_latex(float_format="{:.1f}".format))