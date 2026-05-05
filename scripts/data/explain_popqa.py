import os
import re
import ast
import sys
sys.path.insert(0, "../..")

import argparse
import nltk
import torch
import pandas as pd
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from src.Interpretable_RAG.rag import ExplainableAutoModelForRAG

nltk.download('punkt_tab', quiet=True)


def _ret_token_processor(s: str) -> str:
    return s.replace('▁', ' ')

def _gen_token_processor(s: str) -> str:
    return s.replace('Ġ', ' ').strip('Ċ')


_stemmer = PorterStemmer()

def _first_sentence(text: str) -> str:
    sentences = sent_tokenize(text.strip())
    return sentences[0] if sentences else text

def _normalize_text(text: str) -> list[str]:
    text = text.lower()
    return [_stemmer.stem(w) for w in word_tokenize(text) if w.isalnum()]

def _token_f1(prediction: str, ground_truths: list[str]) -> float:
    pred_tokens = _normalize_text(prediction)
    best = 0.0
    for gt in ground_truths:
        gt_tokens = _normalize_text(gt)
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


#================================================================================#
# Initialization arguments for supported modules:                                #
#================================================================================#
RET_INIT_ARGS = {
    'dragon': {
        'query_encoder_name_or_path': 'facebook/dragon-plus-query-encoder',
        'context_encoder_name_or_path': 'facebook/dragon-plus-context-encoder',
        'retriever_token_processor': _ret_token_processor,
    },
    'snowflake': {
        'query_encoder_name_or_path': 'Snowflake/snowflake-arctic-embed-l-v2.0',
        'retriever_query_format': 'query: {query}',
        'retriever_token_processor': _ret_token_processor,
        'retriever_kwargs': {'add_pooling_layer': False},
    },
}

GEN_INIT_ARGS = {
    'llama8B': {
        'generator_name_or_path': 'meta-llama/Llama-3.1-8B-Instruct',
        'generator_token_processor': _gen_token_processor,
        'generator_kwargs': {'device_map': 'auto', 'dtype': torch.bfloat16},
    },
    'qwen7B': {
        'generator_name_or_path': 'Qwen/Qwen2.5-7B-Instruct',
        'generator_token_processor': _gen_token_processor,  # BPE tokenizer, same Ġ prefix
        'generator_kwargs': {'device_map': 'auto', 'dtype': torch.bfloat16},
    },
    'gemma3_12B': {
        'generator_name_or_path': 'google/gemma-3-12b-it',
        'generator_token_processor': _ret_token_processor,  # SentencePiece tokenizer, ▁ prefix
        'generator_kwargs': {'device_map': 'auto', 'dtype': torch.bfloat16},
    },
}

#================================================================================#
# Call-time arguments for supported modules:                                     #
#================================================================================#
RET_CALL_ARGS = {
    'dragon': {'retriever_kwargs': {'max_length': 512}},
    'snowflake': {},
}

GEN_CALL_ARGS = {
    'llama8B': {
        'generator_kwargs': {
            'max_new_tokens': 256,
            'do_sample': False,
            'top_p': 1,
            'num_beams': 1,
            'batch_size': 32,
            'max_samples_query': 32,
            'max_samples_context': 32,
            'conditional': True,
        },
    },
    'qwen7B': {
        'generator_kwargs': {
            'max_new_tokens': 256,
            'do_sample': False,
            'top_p': 1,
            'num_beams': 1,
            'batch_size': 32,
            'max_samples_query': 32,
            'max_samples_context': 32,
            'conditional': True,
        },
    },
    'gemma3_12B': {
        'generator_kwargs': {
            'max_new_tokens': 256,
            'do_sample': False,
            'top_p': 1,
            'num_beams': 1,
            'batch_size': 32,
            'max_samples_query': 32,
            'max_samples_context': 32,
            'conditional': True,
        },
    },
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

#================================================================================#
# Main function:                                                                 #
#================================================================================#
def main(generator: str, retriever: str, devices: str | None = None):
    results_dir = f'../../results/{RET_FOLDERS[retriever]}/{GEN_FOLDERS[generator]}'
    results_path = f'{results_dir}/results_popqa.csv'
    warg_path = f'{results_dir}/warg_popqa.csv'
    explanations_dir = f'{results_dir}/explanations_popqa/'

    if devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices

    model = ExplainableAutoModelForRAG(**RET_INIT_ARGS[retriever], **GEN_INIT_ARGS[generator])

    if os.path.exists(results_path):
        results = pd.read_csv(results_path, index_col=None)
    else:
        results = pd.read_csv(warg_path, index_col=None)
        results['generation'] = [None for _ in range(len(results))]

    for i, (qry, ctx) in enumerate(results[['query', 'passages']].values):
        if not results['generation'].isnull()[i]:
            print(f'Skipping query {i:d}.')
            continue

        ctx = eval(ctx)
        results.loc[i, 'generation'] = model(
            query=qry,
            contexts=ctx,
            **RET_CALL_ARGS[retriever],
            **GEN_CALL_ARGS[generator],
        )[-1]['content']
        model.save_values(
            explanations_dir, f'{i:d}.pkl',
            ret_methods=['intGrad'], gen_aggregations=['token'], batch_size=16,
        )
        results.to_csv(results_path, index=False)

    results['f1'] = results.apply(
        lambda row: _token_f1(
            _first_sentence(str(row['generation'])),
            ast.literal_eval(row['possible_answers']),
        ) if pd.notna(row['generation']) else float('nan'),
        axis=1,
    )
    results.to_csv(results_path, index=False)

#================================================================================#
# Command line interface:                                                        #
#================================================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate explanations on the PopQA dataset.'
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