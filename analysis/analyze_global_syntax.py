import sys
import os
import pickle
import numpy as np
import glob
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import nltk
from nltk.tokenize import TreebankWordTokenizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# Try to import Spacy, handle failure
HAS_SPACY = False
try:
    import spacy
    # Try loading to see if it crashes
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
    print("Spacy loaded successfully.")
except Exception as e:
    print(f"Spacy load failed ({e}). Falling back to NLTK heuristics.")

# Initialize NLTK
tokenizer = TreebankWordTokenizer()

def map_weights_to_tokens(text, llm_tokens, llm_weights):
    """
    Maps LLM token weights to word-level tokens (NLTK/Spacy).
    """
    # 1. Tokenize text into words
    if HAS_SPACY:
        doc = nlp(text)
        word_tokens = [t.text for t in doc]
        # Spacy spans
        spans = [(t.idx, t.idx + len(t.text)) for t in doc]
    else:
        # NLTK spans
        try:
            spans = list(tokenizer.span_tokenize(text))
            word_tokens = [text[s:e] for s, e in spans]
        except:
            return [], []

    # 2. Align LLM tokens to character spans
    llm_spans = []
    cursor = 0
    for i, token_str in enumerate(llm_tokens):
        t_len = len(token_str)
        llm_spans.append({
            'start': cursor,
            'end': cursor + t_len,
            'weight': llm_weights[i]
        })
        cursor += t_len
        
    # 3. Map weights to word tokens
    mapped_tokens = []
    
    # If using Spacy, we can fill in POS/Dep here
    if HAS_SPACY:
        for idx, token in enumerate(doc):
            s_start, s_end = spans[idx]
            w_weight = 0.0
            
            for l_span in llm_spans:
                overlap_start = max(s_start, l_span['start'])
                overlap_end = min(s_end, l_span['end'])
                if overlap_start < overlap_end:
                    overlap_len = overlap_end - overlap_start
                    l_len = l_span['end'] - l_span['start']
                    if l_len > 0:
                        w_weight += l_span['weight'] * (overlap_len / l_len)
            
            mapped_tokens.append({
                'text': token.text,
                'pos': token.pos_,  # Spacy simplified POS
                'tag': token.tag_,  # Spacy detailed POS
                'dep': token.dep_,  # Dependency
                'head': token.head.text,
                'weight': w_weight
            })
            
    else:
        # NLTK Fallback
        # Get POS tags
        try:
            tagged = nltk.pos_tag(word_tokens)
        except:
            tagged = [(w, 'NN') for w in word_tokens] # Fallback
            
        for idx, (token_text, pos) in enumerate(tagged):
            s_start, s_end = spans[idx]
            w_weight = 0.0
            
            for l_span in llm_spans:
                overlap_start = max(s_start, l_span['start'])
                overlap_end = min(s_end, l_span['end'])
                if overlap_start < overlap_end:
                    overlap_len = overlap_end - overlap_start
                    l_len = l_span['end'] - l_span['start']
                    if l_len > 0:
                        w_weight += l_span['weight'] * (overlap_len / l_len)
            
            mapped_tokens.append({
                'text': token_text,
                'pos': map_nltk_to_simplified(pos), # Map NLTK tag to simplified (NOUN, VERB)
                'tag': pos,
                'dep': None,
                'weight': w_weight
            })
            
        # Apply NLTK Heuristics for Dependency/Function
        mapped_tokens = apply_nltk_heuristics(mapped_tokens)

    return mapped_tokens

def map_nltk_to_simplified(tag):
    if tag.startswith('NN'): return 'NOUN'
    if tag.startswith('VB'): return 'VERB'
    if tag.startswith('JJ'): return 'ADJ'
    if tag.startswith('RB'): return 'ADV'
    if tag.startswith('IN'): return 'ADP'
    if tag.startswith('DT'): return 'DET'
    if tag.startswith('PRP'): return 'PRON'
    return 'OTHER'

def apply_nltk_heuristics(tokens):
    """
    Attempts to label Subject, Object, etc. using NLTK chunking or heuristics.
    """
    # Simple Heuristic:
    # 1. Find main VERB (first verb?)
    # 2. Noun Phrase before Verb -> nsubj
    # 3. Noun Phrase after Verb -> dobj
    # 4. Noun Phrase after ADP -> pobj
    
    # We can use NLTK RegexpParser for chunking
    grammar = r"""
      NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}
      VP: {<VB.*>+}
      PP: {<IN><NP>}
    """
    cp = nltk.RegexpParser(grammar)
    
    # Prepare for chunker
    tagged = [(t['text'], t['tag']) for t in tokens]
    try:
        tree = cp.parse(tagged)
    except:
        return tokens

    # Traverse tree to assign labels
    # This is tricky because tree structure doesn't map 1:1 to flat list indices easily without tracking.
    # Let's just do a flat pass with state.
    
    state = 'BEFORE_VERB' # BEFORE_VERB, AFTER_VERB
    
    current_idx = 0
    
    for child in tree:
        if isinstance(child, nltk.Tree):
            chunk_len = len(child)
            label = child.label()
            
            role = 'OTHER'
            
            if label == 'VP':
                state = 'AFTER_VERB'
                role = 'VERB_PHRASE'
            
            elif label == 'NP':
                if state == 'BEFORE_VERB':
                    role = 'SUBJECT' # Likely subject
                else:
                    role = 'OBJECT' # Likely object
            
            elif label == 'PP':
                role = 'PREP_PHRASE'
                
            # Assign to tokens
            for i in range(chunk_len):
                if current_idx + i < len(tokens):
                    tokens[current_idx + i]['dep'] = role
            
            current_idx += chunk_len
            
        else:
            # Single token (outside chunk)
            current_idx += 1
            
    return tokens

def analyze_ngrams(tokens, n=2):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        window = tokens[i:i+n]
        
        # Create key: "DET + NOUN"
        key = " + ".join([t['pos'] for t in window])
        
        # Calculate combined weight
        # Sum or Mean? Sum reflects total attention to the phrase.
        weight = sum([t['weight'] for t in window])
        
        ngrams.append({
            'pattern': key,
            'weight': weight,
            'text': " ".join([t['text'] for t in window])
        })
    return ngrams

def analyze_functional_roles(tokens):
    roles = []
    for t in tokens:
        if t['dep']:
            roles.append({
                'role': t['dep'],
                'weight': t['weight'],
                'text': t['text'],
                'pos': t['pos']
            })
    return roles

def process_file(file_path, file_type):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        if file_type == 'retrieval':
            raw_tokens = data['input']['query']
            if isinstance(raw_tokens[0], list): raw_tokens = raw_tokens[0]
            
            weights = data['intGrad']['query']
            if isinstance(weights, pd.DataFrame): weights = weights.values
            if hasattr(weights, 'detach'): weights = weights.detach().cpu().numpy()
            if len(weights.shape) > 1: weights = weights.flatten()
            
        else: # generation
            raw_tokens = data['qry_tokens']
            sv = data['shapley_values_token']['query']
            if hasattr(sv, 'detach'): sv = sv.detach().cpu().numpy()
            # Mean absolute weight across generated tokens
            weights = np.mean(np.abs(sv), axis=0)

        full_text = "".join(raw_tokens)
        
        # Map
        tokens = map_weights_to_tokens(full_text, raw_tokens, weights)
        
        # Calculate global average for normalization
        all_w = [t['weight'] for t in tokens]
        avg_w = np.mean(np.abs(all_w)) if all_w else 1.0
        
        # Analyze
        bi_grams = analyze_ngrams(tokens, n=2)
        tri_grams = analyze_ngrams(tokens, n=3)
        roles = analyze_functional_roles(tokens)
        
        results = []
        
        # N-grams
        for bg in bi_grams:
            results.append({
                'category': 'POS_BIGRAM',
                'key': bg['pattern'],
                'weight': bg['weight'],
                'norm_weight': bg['weight'] / avg_w if avg_w else 0,
                'text': bg['text']
            })
            
        for tg in tri_grams:
            results.append({
                'category': 'POS_TRIGRAM',
                'key': tg['pattern'],
                'weight': tg['weight'],
                'norm_weight': tg['weight'] / avg_w if avg_w else 0,
                'text': tg['text']
            })
            
        # Functional
        for r in roles:
            # We care about the role itself (e.g. SUBJECT)
            # But maybe filter only NOUNs in those roles to avoid noise?
            if r['pos'] in ['NOUN', 'PROPN']:
                results.append({
                    'category': 'FUNCTIONAL_ROLE',
                    'key': r['role'], # nsubj, dobj, SUBJECT, OBJECT
                    'weight': r['weight'],
                    'norm_weight': r['weight'] / avg_w if avg_w else 0,
                    'text': r['text']
                })
                
        return results
        
    except Exception as e:
        # print(f"Error processing {file_path}: {e}")
        return []

def main():
    print("Starting Global Syntax & Logic Analysis...")
    
    retrieval_files = glob.glob("results/retrieval/**/*.pkl", recursive=True)
    generation_files = glob.glob("results/generation/**/*.pkl", recursive=True)
    
    all_results = []
    
    print("Processing Retrieval...")
    for f in tqdm(retrieval_files):
        res = process_file(f, 'retrieval')
        for r in res:
            r['type'] = 'retrieval'
            all_results.append(r)
            
    print("Processing Generation...")
    for f in tqdm(generation_files):
        res = process_file(f, 'generation')
        for r in res:
            r['type'] = 'generation'
            all_results.append(r)
            
    if not all_results:
        print("No results generated.")
        return
        
    df = pd.DataFrame(all_results)
    
    # Aggregation
    # We want: Most weighted combinations (Norm Weight) and Frequency
    
    print("\n=== TOP POS BIGRAMS (Weighted) ===")
    bigrams = df[df['category'] == 'POS_BIGRAM']
    grp_bi = bigrams.groupby(['type', 'key'])['norm_weight'].agg(['mean', 'count', 'std'])
    # Filter rare ones
    grp_bi = grp_bi[grp_bi['count'] >= 3]
    print(grp_bi.sort_values('mean', ascending=False).head(10))

    print("\n=== TOP POS TRIGRAMS (Weighted) ===")
    trigrams = df[df['category'] == 'POS_TRIGRAM']
    grp_tri = trigrams.groupby(['type', 'key'])['norm_weight'].agg(['mean', 'count'])
    grp_tri = grp_tri[grp_tri['count'] >= 3]
    print(grp_tri.sort_values('mean', ascending=False).head(10))
    
    print("\n=== FUNCTIONAL ROLES (Subject/Object/etc) ===")
    roles = df[df['category'] == 'FUNCTIONAL_ROLE']
    if not roles.empty:
        grp_role = roles.groupby(['type', 'key'])['norm_weight'].agg(['mean', 'count', 'std'])
        print(grp_role.sort_values('mean', ascending=False))
    else:
        print("No functional roles identified (Parsing issue?).")
        
    # Save
    output_path = "analysis/global_syntax_logic_analysis.csv"
    df.to_csv(output_path, index=False)
    print(f"\nFull data saved to {output_path}")

if __name__ == "__main__":
    main()
