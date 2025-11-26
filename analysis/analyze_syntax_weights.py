import sys
import os
import pickle
import numpy as np
import nltk
from nltk.tokenize import TreebankWordTokenizer
import torch
import glob
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from Interpretable_RAG.utils import tokens2words

# Initialize tokenizer
tokenizer = TreebankWordTokenizer()

def map_nltk_to_llm_weights(text, llm_tokens, llm_weights):
    """
    Maps LLM token weights to NLTK tokens.
    """
    # Get NLTK spans and tokens
    # span_tokenize gives (start, end) indices
    try:
        spans = list(tokenizer.span_tokenize(text))
    except:
        # Fallback if text is empty or problematic
        return [], []
        
    nltk_tokens = [text[s:e] for s, e in spans]
    
    # Get POS tags
    # pos_tag expects a list of words
    try:
        tagged = nltk.pos_tag(nltk_tokens)
    except IndexError:
        return [], []
    
    # Create character map to LLM token index
    token_spans = [] 
    cursor = 0
    for i, token_str in enumerate(llm_tokens):
        t_len = len(token_str)
        token_spans.append({
            'start': cursor,
            'end': cursor + t_len,
            'weight': llm_weights[i],
            'token': token_str
        })
        cursor += t_len

    # Map NLTK tokens to these spans
    results = []
    for idx, (token_text, pos) in enumerate(tagged):
        s_start, s_end = spans[idx]
        
        token_weight = 0.0
        
        for span in token_spans:
            overlap_start = max(s_start, span['start'])
            overlap_end = min(s_end, span['end'])
            
            if overlap_start < overlap_end:
                overlap_len = overlap_end - overlap_start
                span_len = span['end'] - span['start']
                
                if span_len > 0:
                    fraction = overlap_len / span_len
                    token_weight += span['weight'] * fraction
                
        results.append({
            'text': token_text,
            'pos': pos,
            'weight': token_weight,
            'i': idx
        })
        
    return results

def analyze_retrieval(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    raw_tokens = data['input']['query']
    if isinstance(raw_tokens[0], list):
        raw_tokens = raw_tokens[0] 
        
    weights = data['intGrad']['query']
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    
    if len(weights.shape) > 1:
        weights = weights.flatten()
        
    full_text = "".join(raw_tokens)
    
    mapped_tokens = map_nltk_to_llm_weights(full_text, raw_tokens, weights)
    
    return mapped_tokens

def analyze_generation(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    raw_tokens = data['qry_tokens']
    
    sv = data['shapley_values_token']['query']
    if isinstance(sv, torch.Tensor):
        sv = sv.detach().cpu().numpy()
        
    # Mean Absolute Value
    token_weights = np.mean(np.abs(sv), axis=0) 
    
    full_text = "".join(raw_tokens)
    
    mapped_tokens = map_nltk_to_llm_weights(full_text, raw_tokens, token_weights)
    
    return mapped_tokens

def find_patterns(mapped_tokens):
    """
    Finds Preposition + Substantive patterns using NLTK tags.
    IN = Preposition
    NN, NNS, NNP, NNPS = Nouns
    """
    patterns = []
    
    for i, token_data in enumerate(mapped_tokens):
        # Pattern: IN (Preposition) + Noun
        if token_data['pos'] == 'IN':
            if i + 1 < len(mapped_tokens):
                next_token = mapped_tokens[i+1]
                if next_token['pos'].startswith('NN'):
                    patterns.append({
                        'pattern': 'ADP+NOUN',
                        'term1': token_data['text'],
                        'term2': next_token['text'],
                        'weight1': token_data['weight'],
                        'weight2': next_token['weight'],
                        'combined_weight': token_data['weight'] + next_token['weight']
                    })
        
        # Pattern: DT (Determiner) + Noun
        if token_data['pos'] == 'DT':
            if i + 1 < len(mapped_tokens):
                next_token = mapped_tokens[i+1]
                if next_token['pos'].startswith('NN'):
                    patterns.append({
                        'pattern': 'DET+NOUN',
                        'term1': token_data['text'],
                        'term2': next_token['text'],
                        'weight1': token_data['weight'],
                        'weight2': next_token['weight'],
                        'combined_weight': token_data['weight'] + next_token['weight']
                    })
                    
        # Pattern: Verb + Noun (Object?)
        if token_data['pos'].startswith('VB'):
             if i + 1 < len(mapped_tokens):
                next_token = mapped_tokens[i+1]
                if next_token['pos'].startswith('NN'):
                    patterns.append({
                        'pattern': 'VERB+NOUN',
                        'term1': token_data['text'],
                        'term2': next_token['text'],
                        'weight1': token_data['weight'],
                        'weight2': next_token['weight'],
                        'combined_weight': token_data['weight'] + next_token['weight']
                    })

        # Adjective + Noun (JJ, JJR, JJS)
        if token_data['pos'].startswith('JJ'):
             if i + 1 < len(mapped_tokens):
                next_token = mapped_tokens[i+1]
                if next_token['pos'].startswith('NN'):
                    patterns.append({
                        'pattern': 'ADJ+NOUN',
                        'term1': token_data['text'],
                        'term2': next_token['text'],
                        'weight1': token_data['weight'],
                        'weight2': next_token['weight'],
                        'combined_weight': token_data['weight'] + next_token['weight']
                    })

    return patterns

def main():
    print("Starting analysis with NLTK...")
    
    retrieval_files = glob.glob("results/retrieval/**/*.pkl", recursive=True)
    generation_files = glob.glob("results/generation/**/*.pkl", recursive=True)
    
    print(f"Found {len(retrieval_files)} retrieval files and {len(generation_files)} generation files.")
    
    stats = []
    
    # Process Retrieval
    print("Processing Retrieval...")
    for f in tqdm(retrieval_files):
        try:
            tokens = analyze_retrieval(f)
            patterns = find_patterns(tokens)
            
            all_weights = [t['weight'] for t in tokens]
            avg_weight = np.mean(np.abs(all_weights)) if all_weights else 1.0
            
            for p in patterns:
                p['type'] = 'retrieval'
                p['file'] = os.path.basename(f)
                p['global_avg_weight'] = avg_weight
                p['normalized_combined'] = p['combined_weight'] / avg_weight if avg_weight != 0 else 0
                stats.append(p)
        except Exception as e:
            # print(f"Skipping {f}: {e}")
            pass

    # Process Generation
    print("Processing Generation...")
    for f in tqdm(generation_files):
        try:
            tokens = analyze_generation(f)
            patterns = find_patterns(tokens)
            
            all_weights = [t['weight'] for t in tokens]
            avg_weight = np.mean(np.abs(all_weights)) if all_weights else 1.0

            for p in patterns:
                p['type'] = 'generation'
                p['file'] = os.path.basename(f)
                p['global_avg_weight'] = avg_weight
                p['normalized_combined'] = p['combined_weight'] / avg_weight if avg_weight != 0 else 0
                stats.append(p)
        except Exception as e:
            # print(f"Skipping {f}: {e}")
            pass

    if not stats:
        print("No patterns found.")
        return

    df = pd.DataFrame(stats)
    
    print("\n--- Analysis Results ---")
    
    # Group by Type and Pattern
    summary = df.groupby(['type', 'pattern'])['normalized_combined'].agg(['mean', 'count', 'std'])
    print(summary)
    
    print("\n--- Top Examples (Retrieval - ADP+NOUN) ---")
    ret_ex = df[(df['type']=='retrieval') & (df['pattern']=='ADP+NOUN')].sort_values('normalized_combined', ascending=False).head(5)
    print(ret_ex[['term1', 'term2', 'normalized_combined']])
    
    print("\n--- Top Examples (Generation - DET+NOUN) ---")
    gen_ex = df[(df['type']=='generation') & (df['pattern']=='DET+NOUN')].sort_values('normalized_combined', ascending=False).head(5)
    print(gen_ex[['term1', 'term2', 'normalized_combined']])

    # Save
    df.to_csv("analysis/syntax_weight_analysis.csv", index=False)
    print("\nSaved detailed stats to analysis/syntax_weight_analysis.csv")

if __name__ == "__main__":
    main()
