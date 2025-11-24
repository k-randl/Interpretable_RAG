import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import nltk
from nltk import pos_tag

# Import existing analysis tools
from analyze_results import load_pickle, analyze_retrieval_results, analyze_generation_results
from generate_plots import (
    plot_shapley_heatmap, 
    plot_mean_doc_importance, 
    plot_pos_importance,
    clean_token
)
from analyze_retrieval import (
    plot_query_heatmap, 
    plot_context_heatmap, 
    plot_token_comparison, 
    plot_document_relevance,
    plot_weighted_token_overlap
)

# Ensure NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('taggers/universal_tagset')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('universal_tagset', quiet=True)

# --- HTML Report Styles ---
REPORT_CSS = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; color: #333; background-color: #f9f9f9; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
h2 { color: #2980b9; margin-top: 30px; border-left: 5px solid #3498db; padding-left: 10px; }
h3 { color: #7f8c8d; }
.experiment-block { border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; margin-bottom: 30px; background-color: #fff; }
.plot-container { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 15px; justify-content: center; }
.plot-card { border: 1px solid #eee; padding: 15px; border-radius: 8px; background: #fff; box-shadow: 0 2px 5px rgba(0,0,0,0.05); width: 45%; min-width: 400px; text-align: center; }
.plot-card img { max-width: 100%; height: auto; border-radius: 4px; }
.description { font-style: italic; color: #666; margin-top: 10px; font-size: 0.9em; background: #f0f8ff; padding: 10px; border-radius: 4px; text-align: left; }
.stats-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }
.stats-table th, .stats-table td { padding: 8px; border: 1px solid #ddd; text-align: left; }
.stats-table th { background-color: #f2f2f2; }
.badge { display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; color: #fff; font-weight: bold; }
.badge-gen { background-color: #e67e22; }
.badge-ret { background-color: #27ae60; }
"""

EXPLANATIONS = {
    "position_bias": "<b>Position Bias Analysis:</b> Comparison of document importance based on its position in the prompt. <br>If 'Original' and 'Randomized' both show a downward trend, the model has a <b>Primacy Bias</b> (favors the first thing it reads). <br>If 'Randomized' is flat or different, the model might be attending to content rather than position.",
    "pos_distribution": "<b>Part-of-Speech Impact:</b> Which grammatical roles drive the model's generation? High importance on Nouns/Verbs indicates content focus; high importance on Determiners/Prepositions might indicate noise.",
    "top_tokens": "<b>Top Influential Tokens:</b> The specific words (excluding stop words) that accumulated the most importance across all queries. These are the 'drivers' of your model's outputs.",
    "sparsity": "<b>Attention Sparsity (Gini):</b> Measures how 'focused' the model's attention is. <br>High Gini = Model focuses on a few specific words. <br>Low Gini = Model attends to everything equally (diffuse attention).",
    "doc_relevance": "<b>Retrieval Confidence:</b> Average importance score of documents by their retrieved rank."
}

# --- Aggregation Classes ---

class ConditionStats:
    def __init__(self, name):
        self.name = name
        self.gen_doc_scores = [] # List of arrays
        self.gen_pos_scores = defaultdict(list) # {pos: [scores]}
        self.token_counts = Counter() # {token_str: total_importance}
        self.sparsity_scores = [] # Gini coefficients per query
        self.ret_doc_scores = []

    def add_generation(self, shapley_context, shapley_query, gen_tokens, qry_tokens):
        # 1. Doc Position Scores
        doc_imps = np.sum(np.abs(shapley_context), axis=1)
        self.gen_doc_scores.append(doc_imps)
        
        # 2. Token Statistics (Top Tokens & Sparsity)
        # Aggregate importance per input token (query + context)
        # shapley_query: [num_gen, num_qry] -> sum(axis=0) = total impact OF each query token
        qry_imp = np.sum(np.abs(shapley_query), axis=0)
        # shapley_context: [num_docs, num_gen] -> We need token-level context?
        # Actually shapley_context usually comes aggregated by document in the plotting utils.
        # If we want top *words* in context, we need the full token-level matrix if available.
        # For now, let's focus on Query Tokens + Generated Tokens self-importance (if available) or just Query.
        
        # Let's track Query Token Importance for now as it's cleaner
        for tok, score in zip(qry_tokens, qry_imp):
            if len(tok) > 2: # Skip tiny tokens
                self.token_counts[tok] += score
                
        # 3. Sparsity (Gini of Query Importance)
        # How concentrated is the reliance on specific query tokens?
        if len(qry_imp) > 0:
            self.sparsity_scores.append(gini(qry_imp))

        # 4. POS Analysis (on Generated Tokens)
        # Importance per generated token
        gen_imp = np.sum(np.abs(shapley_query), axis=1) 
        # We need context contribution too. Assuming shapley_context [num_docs, num_gen]
        if shapley_context.shape[1] == len(gen_imp):
             gen_imp += np.sum(np.abs(shapley_context), axis=0)
        
        tags = pos_tag(gen_tokens[:len(gen_imp)], tagset='universal')
        for (token, tag), score in zip(tags, gen_imp):
            self.gen_pos_scores[tag].append(score)

    def add_retrieval(self, analysis):
         if 'intGrad' in analysis and 'context_score' in analysis['intGrad']:
            scores = analysis['intGrad']['context_score']
            doc_imp = np.sum(np.abs(scores), axis=1)
            self.ret_doc_scores.append(doc_imp)

def gini(array):
    if np.amin(array) < 0:
        array -= np.amin(array) # Values cannot be negative
    array = np.add(array, 1e-9) # Values cannot be 0
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

class ExperimentAccumulator:
    def __init__(self, name):
        self.name = name
        self.conditions = {} # 'original': ConditionStats, 'randomized': ConditionStats

    def get_condition(self, cond_name):
        if cond_name not in self.conditions:
            self.conditions[cond_name] = ConditionStats(cond_name)
        return self.conditions[cond_name]

    def plot_doc_position_comparison(self, output_path):
        """Plots Document Importance by Position side-by-side for all conditions."""
        plt.figure(figsize=(12, 6))
        
        has_data = False
        conditions = list(self.conditions.keys())
        # Define distinct colors for known conditions, fallback for others
        color_map = {'original': '#3498db', 'randomized': '#e74c3c', 'default': 'gray'}
        bar_width = 0.8 / len(conditions) if conditions else 0.8
        
        all_means = []
        max_len = 0
        
        # Pre-calculate means to find dimensions
        for cond_name in conditions:
            stats = self.conditions[cond_name]
            if not stats.gen_doc_scores: 
                all_means.append(None)
                continue
            
            max_docs = max(len(x) for x in stats.gen_doc_scores)
            max_len = max(max_len, max_docs)
            padded = np.array([np.pad(x, (0, max_docs - len(x)), constant_values=np.nan) for x in stats.gen_doc_scores])
            means = np.nanmean(padded, axis=0)
            errs = np.nanstd(padded, axis=0) / np.sqrt(len(padded))
            all_means.append((means, errs))
            has_data = True
            
        if not has_data:
            plt.close()
            return False
            
        # Plot bars
        indices = np.arange(max_len)
        for i, cond_name in enumerate(conditions):
            if all_means[i] is None: continue
            means, errs = all_means[i]
            
            # Handle length mismatch if one condition has fewer docs
            current_means = np.pad(means, (0, max_len - len(means)), constant_values=0)
            current_errs = np.pad(errs, (0, max_len - len(errs)), constant_values=0)
            
            pos = indices + (i * bar_width)
            c = color_map.get(cond_name, plt.cm.tab10(i))
            
            plt.bar(pos, current_means, width=bar_width, label=cond_name.capitalize(), 
                   yerr=current_errs, capsize=5, color=c, alpha=0.8)
            
        plt.title(f"Position Bias: Document Importance ({self.name})")
        plt.xlabel("Document Position in Prompt")
        plt.ylabel("Mean Importance (Shapley)")
        plt.xticks(indices + bar_width * (len(conditions) - 1) / 2, [str(i+1) for i in indices])
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return True

    def plot_top_tokens(self, output_path):
        # Aggregate across conditions or just pick 'original'? Let's aggregate all.
        total_counts = Counter()
        for stats in self.conditions.values():
            total_counts.update(stats.token_counts)
            
        if not total_counts: return False
        
        top_tokens = total_counts.most_common(15)
        tokens, scores = zip(*top_tokens)
        
        plt.figure(figsize=(10, 6))
        plt.barh(tokens, scores, color='#8e44ad')
        plt.gca().invert_yaxis()
        plt.title(f"Top Influential Query Tokens ({self.name})")
        plt.xlabel("Total Accumulated Importance")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return True
        
    def plot_pos_distribution(self, output_path):
        # Aggregate
        pos_agg = defaultdict(list)
        for stats in self.conditions.values():
            for tag, vals in stats.gen_pos_scores.items():
                pos_agg[tag].extend(vals)
        
        if not pos_agg: return False
        
        avg_scores = {tag: np.mean(vals) for tag, vals in pos_agg.items()}
        sorted_tags = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        tags, values = zip(*sorted_tags)
        
        plt.figure(figsize=(8, 5))
        plt.bar(tags, values, color='#2ecc71')
        plt.title(f"Part-of-Speech Importance ({self.name})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return True

def generate_local_index(file_dir, file_name, file_type):
    """Generates a simple index.html for a specific query folder."""
    
    plots = []
    if file_type == 'generation':
        plots = [
            ("Document Importance", "doc_imp.png", "doc_importance"),
            ("POS Importance", "pos_imp.png", "pos_distribution"),
            ("Query Heatmap", "qry_map.png", "shapley_heatmap")
        ]
    else:
        plots = [
            ("Document Relevance", "ret_relevance.png", "doc_relevance"),
            ("Weighted Overlap", "token_overlap.png", "token_overlap"),
            ("Token Comparison", "token_comp.png", "top_tokens")
        ]
        
    html = [f"""<!DOCTYPE html>
<html>
<head>
    <title>Detail: {file_name}</title>
    <style>{REPORT_CSS}</style>
</head>
<body>
    <div class="container">
        <a href="../../../report.html" style="display:inline-block; margin-bottom:20px; text-decoration:none; color:#3498db;">&larr; Back to Global Report</a>
        <h1>Analysis: {file_name}</h1>
        <div class="experiment-block">
            <div class="plot-container">"""]
        
    for title, fname, key in plots:
        if os.path.exists(os.path.join(file_dir, fname)):
             html.append(f"""
            <div class="plot-card">
                <h3>{title}</h3>
                <img src="{fname}" alt="{title}">
            </div>""")
            
    html.append("""
            </div>
        </div>
    </div>
</body>
</html>""")
    
    with open(os.path.join(file_dir, "index.html"), "w") as f:
        f.write("\n".join(html))

# --- Main Analysis Logic ---

def infer_metadata(file_path):
    parts = Path(file_path).parts
    # Assumptions based on path structure: results/generation/<EXPERIMENT>/<CONDITION>/<file>
    # e.g. results/generation/trec19_19_11_2025/original/file.pkl
    
    try:
        if 'generation' in parts:
            idx = parts.index('generation')
            experiment = parts[idx+1]
            condition = parts[idx+2] if idx+2 < len(parts)-1 else 'default'
            return experiment, condition
        elif 'retrieval' in parts:
            idx = parts.index('retrieval')
            experiment = parts[idx+1]
            return experiment, 'retrieval'
    except:
        pass
    return 'unknown_experiment', 'default'

def main():
    parser = argparse.ArgumentParser(description="Master Analysis Script")
    parser.add_argument("input_path", help="Root folder to scan")
    parser.add_argument("--output_dir", default="analysis_report_v2", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Scan Files
    files = []
    if os.path.isdir(args.input_path):
        files = list(Path(args.input_path).glob("**/*.pkl"))
    else:
        files = [Path(args.input_path)]
        
    experiments = {} # name -> ExperimentAccumulator
    processed_files_log = []
    
    print(f"Scanning {len(files)} files...")
    
    # 2. Process Files
    for file_path in files:
        try:
            exp_name, condition = infer_metadata(file_path)
            
            if exp_name not in experiments:
                experiments[exp_name] = ExperimentAccumulator(exp_name)
                
            # Setup local output
            file_clean_name = file_path.stem
            local_out_dir = os.path.join(args.output_dir, exp_name, condition, file_clean_name)
            os.makedirs(local_out_dir, exist_ok=True)
            
            # Load
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            ftype = 'unknown'
            
            # Analyze
            if 'gen_tokens' in data: # Generation
                ftype = 'generation'
                cond_stats = experiments[exp_name].get_condition(condition)
                
                qry = [clean_token(t) for t in data['qry_tokens']]
                gen = [clean_token(t) for t in data['gen_tokens']]
                s_ctx = data['shapley_values_token']['context']
                s_qry = data['shapley_values_token']['query']
                
                # Accumulate
                cond_stats.add_generation(s_ctx, s_qry, gen, qry)
                
                # Local Plots
                plot_mean_doc_importance(s_ctx, os.path.join(local_out_dir, "doc_imp.png"))
                plot_pos_importance(gen, s_qry, s_ctx, os.path.join(local_out_dir, "pos_imp.png"))
                plot_shapley_heatmap(s_qry, gen, qry, "Query Heatmap", os.path.join(local_out_dir, "qry_map.png"))
                
            elif 'input' in data: # Retrieval
                ftype = 'retrieval'
                # For retrieval, condition might be just 'default' or folder name
                cond_stats = experiments[exp_name].get_condition('retrieval_agg') 
                
                analysis = analyze_retrieval_results(data)
                cond_stats.add_retrieval(analysis)
                
                # Local Plots
                plot_document_relevance(analysis, os.path.join(local_out_dir, "ret_relevance.png"))
                plot_weighted_token_overlap(analysis, os.path.join(local_out_dir, "token_overlap.png"))
                plot_token_comparison(analysis, os.path.join(local_out_dir, "token_comp.png"))
                
            # Log for report
            processed_files_log.append({
                'exp': exp_name, 'cond': condition, 'name': file_clean_name,
                'type': ftype, 'path': local_out_dir
            })
            
            # Generate Index Page
            generate_local_index(local_out_dir, file_clean_name, ftype)
            
            print(f"Processed: {exp_name} / {condition} / {file_clean_name}")
            
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    # 3. Generate Global Plots per Experiment
    print("\nGenerating Global Plots...")
    exp_plots_map = defaultdict(list)
    
    for exp_name, accumulator in experiments.items():
        exp_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # A. Comparison: Doc Position
        if accumulator.plot_doc_position_comparison(os.path.join(exp_dir, "compare_position.png")):
            exp_plots_map[exp_name].append(("compare_position.png", "position_bias"))
            
        # B. Top Tokens
        if accumulator.plot_top_tokens(os.path.join(exp_dir, "top_tokens.png")):
            exp_plots_map[exp_name].append(("top_tokens.png", "top_tokens"))
            
        # C. POS Distribution
        if accumulator.plot_pos_distribution(os.path.join(exp_dir, "pos_dist.png")):
             exp_plots_map[exp_name].append(("pos_dist.png", "pos_distribution"))

    # 4. Generate HTML Report
    generate_full_report(args.output_dir, experiments, exp_plots_map, processed_files_log)
    print(f"Report ready at {args.output_dir}/report.html")

def generate_full_report(out_dir, experiments, exp_plots_map, file_log):
    html = [f"""<!DOCTYPE html>
<html>
<head>
    <title>RAG Comparative Analysis</title>
    <style>{REPORT_CSS}</style>
</head>
<body>
    <div class="container">
        <h1>RAG Pipeline Comparative Analysis</h1>
        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
    """]
    
    # --- Experiment Sections ---
    for exp_name in experiments.keys():
        html.append(f"""
        <div class="experiment-block">
            <h2>Experiment: {exp_name}</h2>
            <div class="plot-container">""")
            
        # Add Global Plots
        if exp_name in exp_plots_map:
            for img, desc_key in exp_plots_map[exp_name]:
                desc = EXPLANATIONS.get(desc_key, "")
                html.append(f"""
                <div class="plot-card">
                    <img src="{exp_name}/{img}" alt="{desc_key}">
                    <div class="description">{desc}</div>
                </div>""")
                
        html.append("</div>")
        
        # Add File List
        html.append(f"""
            <h3>Detailed Logs ({exp_name})</h3>
            <table class="stats-table">
                <thead><tr><th>Condition</th><th>File</th><th>Type</th><th>Links</th></tr></thead>
                <tbody>""")
        
        # Filter logs for this experiment
        exp_logs = [f for f in file_log if f['exp'] == exp_name]
        exp_logs.sort(key=lambda x: (x['cond'], x['name']))
        
        for log in exp_logs:
            rel_path = os.path.relpath(log['path'], out_dir)
            badge_class = "badge-gen" if log['type'] == 'generation' else "badge-ret"
            
            # Determine main image for quick preview link
            preview_img = "doc_imp.png" if log['type'] == 'generation' else "token_overlap.png"
            
            html.append(f"""
            <tr>
                <td>{log['cond']}</td>
                <td>{log['name']}</td>
                <td><span class="badge {badge_class}">{log['type']}</span></td>
                <td>
                    <a href="{rel_path}/index.html" target="_blank">Detailed View</a> | 
                    <a href="{rel_path}/" target="_blank">Folder</a>
                </td>
            </tr>""")
            
        html.append("</tbody></table></div>")
        
    html.append("</div></body></html>")
    
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write("\n".join(html))

if __name__ == "__main__":
    main()
