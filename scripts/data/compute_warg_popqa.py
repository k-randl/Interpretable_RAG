import os
import sys
sys.path.insert(0, "../..")

import argparse
import torch
import pandas as pd
from src.Interpretable_RAG.rag import WARGScorer

#================================================================================#
# Initialization arguments for supported modules:                                #
#================================================================================#
RET_INIT_ARGS = {
    'dragon': {
        'query_encoder_name_or_path': 'facebook/dragon-plus-query-encoder',
        'context_encoder_name_or_path': 'facebook/dragon-plus-context-encoder',
    },
    'snowflake': {
        'query_encoder_name_or_path': 'Snowflake/snowflake-arctic-embed-l-v2.0',
        'retriever_query_format': 'query: {query}',
        'retriever_kwargs': {'add_pooling_layer': False},
    },
}

GEN_INIT_ARGS = {
    'llama8B': {
        'generator_name_or_path': 'meta-llama/Llama-3.1-8B-Instruct',
        'generator_kwargs': {'device_map': 'auto', 'dtype': torch.bfloat16},
    },
    'qwen7B': {
        'generator_name_or_path': 'Qwen/Qwen2.5-7B-Instruct',
        'generator_kwargs': {'device_map': 'auto', 'dtype': torch.bfloat16},
    },
    'gemma3_12B': {
        'generator_name_or_path': 'google/gemma-3-12b-it',
        'generator_kwargs': {'device_map': 'auto', 'dtype': torch.bfloat16},
    },
}

#================================================================================#
# Call-time arguments for supported modules:                                     #
#================================================================================#
RET_CALL_ARGS = {
    'dragon': {'max_length': 512},
    'snowflake': {},
}

GEN_CALL_ARGS = {
    'llama8B': {'max_document_size': 2000},
    'qwen7B': {'max_document_size': 2000},
    'gemma3_12B': {'max_document_size': 2000},
}

#================================================================================#
# Folder names for supported modules:                                            #
#================================================================================#
RET_FOLDERS = {
    'dragon': 'facebook-dragon-plus',
    'snowflake': 'snowflake-arctic-embed-l-v2.0',
}

GEN_FOLDERS = {
    'llama8B': 'llama-3.1-8b-instruct',
    'qwen7B': 'qwen2.5-7b-instruct',
    'gemma3_12B': 'gemma-3-12b-it',
}

P_VALUES = [round(p * 0.1, 1) for p in range(1, 10)]

#================================================================================#
# Main function:                                                                 #
#================================================================================#
def main(generator: str, retriever: str, devices: str | None = None):
    index_path = f'../../data/popqa/{RET_FOLDERS[retriever]}/'
    scorer_path = f'../../results/{RET_FOLDERS[retriever]}/{GEN_FOLDERS[generator]}/scorer_popqa.json'
    results_path = f'../../results/{RET_FOLDERS[retriever]}/{GEN_FOLDERS[generator]}/warg_popqa.csv'

    if devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices

    scorer = WARGScorer(index_path, **RET_INIT_ARGS[retriever], **GEN_INIT_ARGS[generator])

    warg = pd.read_csv('../../data/popqa/topics.tsv', sep='\t', index_col=None)

    for p in P_VALUES:
        warg[f'p={p:.2f}'] = scorer(
            warg['query'].tolist(),
            k=10,
            p=p,
            batch_size=24,
            retriever_kwargs=RET_CALL_ARGS[retriever],
            generator_kwargs=GEN_CALL_ARGS[generator],
            checkpoint_path=scorer_path,
        )

    warg['passages'] = [scorer._queries[q][2] for q in warg['query'].tolist()]
    warg.to_csv(results_path, index=False)

#================================================================================#
# Command line interface:                                                        #
#================================================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute WARG scores on the PopQA dataset.'
    )
    parser.add_argument(
        '--generator',
        choices=list(GEN_INIT_ARGS.keys()),
        required=True,
        help='Generator model to use.',
    )
    parser.add_argument(
        '--retriever',
        choices=list(RET_INIT_ARGS.keys()),
        required=True,
        help='Retriever model to use.',
    )
    parser.add_argument(
        '--devices',
        default=None,
        help='Comma-separated CUDA device indices (e.g. "0,1"). Defaults to all visible devices.',
    )
    args = parser.parse_args()
    main(generator=args.generator, retriever=args.retriever, devices=args.devices)
