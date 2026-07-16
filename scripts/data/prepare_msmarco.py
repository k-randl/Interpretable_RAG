# %%% ===============================================================================================#
# Setup:                                                                                             #
#====================================================================================================#

import os
import sys
sys.path.insert(0, "../..")

import pandas as pd
from datasets import load_dataset

os.makedirs('../../data/msmarco/', exist_ok=True)

# Parameters:
NUM_DOCS     = 5

# %% Load data sample:
# Load the MS MARCO dataset version 2.1
dataset = load_dataset("ms_marco", "v2.1", split="train")

# Get a random sample of 200 documents
sample = dataset.shuffle(seed=42).select(range(200))

# Set dataset format:
topics = sample.map(lambda item: {
    'passages':item['passages']['passage_text'][:NUM_DOCS],
    'possible_answers':item['answers']
}).select_columns(['query_id', 'query', 'passages', 'possible_answers'])
topics = pd.DataFrame(topics)
topics.to_csv('../../data/msmarco/topics.tsv', sep='\t', index=False)