import os
import sys
sys.path.insert(0, "../..")

import argparse
import torch
import pandas as pd
from src.Interpretable_RAG.retrieval_online import ExplainableAutoModelForRetrieval

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
        'add_pooling_layer': False,
    },
}

#================================================================================#
# Call-time arguments for supported modules:                                     #
#================================================================================#
RET_CALL_ARGS = {
    'dragon': {'batch_size': 64, 'max_length': 512},
    'snowflake': {'batch_size': 32},
}

#================================================================================#
# Folder names for supported modules:                                            #
#================================================================================#
RET_FOLDERS = {
    'dragon': 'facebook-dragon-plus',
    'snowflake': 'snowflake-arctic-embed-l-v2.0',
}

#================================================================================#
# Main function:                                                                 #
#================================================================================#
def main(retriever: str, devices: str | None = None):
    save_folder = f'../../data/popqa/{RET_FOLDERS[retriever]}/'

    if devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices

    model = ExplainableAutoModelForRetrieval.from_pretrained(
        **RET_INIT_ARGS[retriever]
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    texts = pd.read_csv('../../data/popqa/passages.csv')['text'].tolist()
    model.compute_index(texts, save_folder=save_folder, **RET_CALL_ARGS[retriever])

#================================================================================#
# Command line interface:                                                        #
#================================================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute retrieval index over PopQA passages.'
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
    main(retriever=args.retriever, devices=args.devices)