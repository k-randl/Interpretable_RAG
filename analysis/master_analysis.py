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
import concurrent.futures
import traceback

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
    print("tqdm not found. Install with 'pip install tqdm' for progress bars.")

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

def infer_metadata(file_path):
    parts = Path(file_path).parts
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

def gini(array):
    if np.amin(array) < 0:
        array -= np.amin(array) # Values cannot be negative
    array = np.add(array, 1e-9) # Values cannot be 0
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

# --- Helper functions for HTML generation inside workers ---
def generate_generation_stats_html(output_dir, gen_tokens, qry_tokens, shapley_context, shapley_query):
    q_imp = np.sum(np.abs(shapley_query), axis=1)
    c_imp = np.sum(np.abs(shapley_context), axis=0)
    
    min_len = min(len(q_imp), len(c_imp))
    total_gen_importance = np.sum(np.abs(shapley_query), axis=0) + np.sum(np.abs(shapley_context), axis=0)
    
    top_gen_indices = np.argsort(total_gen_importance)[-10:][::-1]
    
    stats_html = [f"""<!DOCTYPE html>
<html>
<head>
    <title>Generation Stats</title>
    <style>{REPORT_CSS}</style>
</head>
<body>
    <div class=\"container\">
        <h1>Generation Statistics</h1>
        <p><b>Query:</b> {" ".join(qry_tokens)}</p>
        <p><b>Generated Text:</b> {" ".join(gen_tokens)}</p>
        
        <div class=\"experiment-block\">
            <h3>Top 10 Most 'Expensive' Generated Tokens</h3>
            <p>Tokens that required the most attribution (Context + Query).</p>
            <table class=\"stats-table\">
                <thead><tr><th>Rank</th><th>Token</th><th>Total Importance</th></tr></thead>
                <tbody>""" ]
                
    for i, idx in enumerate(top_gen_indices):
        token = gen_tokens[idx] if idx < len(gen_tokens) else "N/A"
        score = total_gen_importance[idx] if idx < len(total_gen_importance) else 0.0
        stats_html.append(f"<tr><td>{i+1}</td><td>{token}</td><td>{score:.4f}</td></tr>")
        
    stats_html.append("</tbody></table></div></div></body></html>")

    with open(os.path.join(output_dir, "gen_stats.html"), "w") as f:
        f.write("\n".join(stats_html))

def generate_local_index(file_dir, file_name, file_type, query_text="", documents=[]):
    plots = []
    extra_links = ""
    
    if file_type == 'generation':
        plots = [
            ("Document Importance", "doc_imp.png", "doc_importance"),
            ("POS Importance", "pos_imp.png", "pos_distribution"),
            ("Query Heatmap", "qry_map.png", "shapley_heatmap")
        ]
        extra_links = f'<a href="gen_stats.html" class="badge badge-gen" style="font-size:1em; text-decoration:none; padding:10px;">View Full Generation Stats</a>'
    else:
        plots = [
            ("Document Relevance", "ret_relevance.png", "doc_relevance"),
            ("Weighted Overlap", "token_overlap.png", "token_overlap"),
            ("Token Comparison", "token_comp.png", "top_tokens")
        ]
    
    doc_html = ""
    if documents:
        doc_html = "<div class='experiment-block'><h3>Documents</h3>"
        for i, doc in enumerate(documents):
            doc_html += f"""
            <details style=\"margin-bottom:10px; border:1px solid #ddd; padding:5px; border-radius:5px;\">
                <summary style=\"cursor:pointer; font-weight:bold;\">Document {i+1} (Click to Expand)</summary>
                <div style=\"padding:10px; background:#f9f9f9; font-family:monospace; white-space:pre-wrap;\">{doc}</div>
            </details>
            """
        doc_html += "</div>"
        
    html = [f"""<!DOCTYPE html>
<html>
<head>
    <title>Detail: {file_name}</title>
    <style>{REPORT_CSS}</style>
</head>
<body>
    <div class=\"container\">
        <a href=\"../../../report.html\" style=\"display:inline-block; margin-bottom:20px; text-decoration:none; color:#3498db;\" >&larr; Back to Global Report</a>
        <h1>Analysis: {file_name}</h1>
        <div style=\"background:#f0f8ff; padding:15px; border-radius:5px; margin-bottom:20px;\">
            <strong>Query:</strong> {query_text}
        </div>
        {extra_links}
        <div class=\"experiment-block\">
            <div class=\"plot-container\">""" ]
        
    for title, fname, key in plots:
        if os.path.exists(os.path.join(file_dir, fname)):
             html.append(f"""
            <div class=\"plot-card\">
                <h3>{title}</h3>
                <img src=\"{fname}\" alt=\"{title}\">
            </div>""")
            
    html.append(f"""
            </div>
        </div>
        {doc_html}
    </div>
</body>
</html>""")
    
    with open(os.path.join(file_dir, "index.html"), "w") as f:
        f.write("\n".join(html))

# --- Worker Function ---
def process_file_worker(args):
    file_path, output_dir = args
    result_payload = {
        'status': 'failed',
        'file_path': str(file_path),
        'error': None
    }
    
    try:
        exp_name, condition = infer_metadata(file_path)
        file_clean_name = file_path.stem
        local_out_dir = os.path.join(output_dir, exp_name, condition, file_clean_name)
        os.makedirs(local_out_dir, exist_ok=True)
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        stats_summary = {}
        ftype = 'unknown'
        query_text = "N/A"
        documents = []

        if 'gen_tokens' in data: # Generation
            ftype = 'generation'
            
            qry = [clean_token(t) for t in data['qry_tokens']]
            gen = [clean_token(t) for t in data['gen_tokens']]
            s_ctx = data['shapley_values_token']['context']
            s_qry = data['shapley_values_token']['query']
            
            if len(qry) > 0:
                query_text = " ".join(qry).replace('Ġ', ' ')
            
            # Local Plots
            plot_mean_doc_importance(s_ctx, os.path.join(local_out_dir, "doc_imp.png"))
            plot_pos_importance(gen, s_qry, s_ctx, os.path.join(local_out_dir, "pos_imp.png"))
            plot_shapley_heatmap(s_qry, gen, qry, "Query Heatmap", os.path.join(local_out_dir, "qry_map.png"))
            generate_generation_stats_html(local_out_dir, gen, qry, s_ctx, s_qry)
            
            # Compute Stats for Aggregation
            # 1. Doc scores
            doc_imps = np.sum(np.abs(s_ctx), axis=1)
            
            # 2. Query Impact
            qry_imp = np.sum(np.abs(s_qry), axis=1)
            
            # 3. Gen Impact (POS)
            gen_imp = np.sum(np.abs(s_qry), axis=0)
            if s_ctx.shape[1] == len(gen_imp):
                gen_imp += np.sum(np.abs(s_ctx), axis=0)
            
            gen_tags = pos_tag(gen[:len(gen_imp)], tagset='universal')
            pos_scores = defaultdict(float)
            for (token, tag), score in zip(gen_tags, gen_imp):
                 pos_scores[tag] += score # Pre-aggregate sum per tag for this file
            
            # 4. Token Counts (Query)
            token_counts = defaultdict(float)
            for tok, score in zip(qry, qry_imp):
                if len(tok) > 2:
                    token_counts[tok] += score
            
            stats_summary = {
                'type': 'generation',
                'doc_imps': doc_imps,
                'qry_imp_gini': gini(qry_imp) if len(qry_imp) > 0 else None,
                'pos_scores': dict(pos_scores), # {tag: total_score}
                'pos_counts': dict(Counter([t for _, t in gen_tags])), # Count occurrences for averaging
                'token_counts': dict(token_counts)
            }

        elif 'input' in data: # Retrieval
            ftype = 'retrieval'
            analysis = analyze_retrieval_results(data)
            
            if 'query_tokens' in analysis:
                query_text = " ".join(clean_token(t) for t in analysis['query_tokens']).replace('Ġ', ' ')
            
            if 'context_tokens_list' in analysis:
                for doc_toks in analysis['context_tokens_list']:
                    doc_text = " ".join(clean_token(t) for t in doc_toks).replace('Ġ', ' ')
                    documents.append(doc_text)
            
            # Local Plots
            plot_document_relevance(analysis, os.path.join(local_out_dir, "ret_relevance.png"))
            plot_weighted_token_overlap(analysis, os.path.join(local_out_dir, "token_overlap.png"))
            plot_token_comparison(analysis, os.path.join(local_out_dir, "token_comp.png"))
            
            # Compute Stats
            ret_doc_scores = None
            if 'intGrad' in analysis and 'context_score' in analysis['intGrad']:
                scores = analysis['intGrad']['context_score']
                ret_doc_scores = np.sum(np.abs(scores), axis=1)
            
            stats_summary = {
                'type': 'retrieval',
                'doc_imps': ret_doc_scores
            }

        generate_local_index(local_out_dir, file_clean_name, ftype, query_text, documents)
        
        result_payload = {
            'status': 'success',
            'exp_name': exp_name,
            'condition': condition,
            'name': file_clean_name,
            'ftype': ftype,
            'path': local_out_dir,
            'query': query_text,
            'stats': stats_summary
        }
        
    except Exception as e:
        result_payload['error'] = f"{str(e)} | {traceback.format_exc()}"

    return result_payload


# --- Aggregation Classes (Revised to use pre-computed stats) ---
class ConditionStats:
    def __init__(self, name):
        self.name = name
        self.gen_doc_scores = [] 
        self.gen_pos_scores = defaultdict(list) # {pos: [avg_scores_per_file]} or accumulated total?
        # Let's store total mass and total count to average later
        self.pos_total_score = defaultdict(float)
        self.pos_total_count = defaultdict(int)
        
        self.token_counts = Counter() 
        self.sparsity_scores = [] 
        self.ret_doc_scores = []

    def ingest_generation_stats(self, stats):
        if stats.get('doc_imps') is not None:
            self.gen_doc_scores.append(stats['doc_imps'])
        
        if stats.get('qry_imp_gini') is not None:
            self.sparsity_scores.append(stats['qry_imp_gini'])
            
        # Token Counts
        for tok, score in stats.get('token_counts', {}).items():
            self.token_counts[tok] += score
            
        # POS
        # We have total score per tag and count per tag for this file
        # We want global importance per tag. 
        # Simple way: append raw scores? No, too much data.
        # Just accumulate totals.
        for tag, score in stats.get('pos_scores', {}).items():
            self.pos_total_score[tag] += score
            
        for tag, count in stats.get('pos_counts', {}).items():
            self.pos_total_count[tag] += count

    def ingest_retrieval_stats(self, stats):
        if stats.get('doc_imps') is not None:
            self.ret_doc_scores.append(stats['doc_imps'])

class ExperimentAccumulator:
    def __init__(self, name):
        self.name = name
        self.conditions = {} 

    def get_condition(self, cond_name):
        if cond_name not in self.conditions:
            self.conditions[cond_name] = ConditionStats(cond_name)
        return self.conditions[cond_name]

    def plot_doc_position_comparison(self, output_path):
        plt.figure(figsize=(12, 6))
        has_data = False
        conditions = list(self.conditions.keys())
        color_map = {'original': '#3498db', 'randomized': '#e74c3c', 'default': 'gray'}
        bar_width = 0.8 / len(conditions) if conditions else 0.8
        
        all_means = []
        max_len = 0
        
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
            
        indices = np.arange(max_len)
        for i, cond_name in enumerate(conditions):
            if all_means[i] is None: continue
            means, errs = all_means[i]
            
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
        # Calculate average score per POS tag across all files
        pos_avgs = defaultdict(float)
        
        # Aggregate across conditions
        grand_total_score = defaultdict(float)
        grand_total_count = defaultdict(int)
        
        for stats in self.conditions.values():
            for tag, val in stats.pos_total_score.items():
                grand_total_score[tag] += val
            for tag, count in stats.pos_total_count.items():
                grand_total_count[tag] += count
                
        if not grand_total_score: return False
        
        for tag in grand_total_score:
            if grand_total_count[tag] > 0:
                pos_avgs[tag] = grand_total_score[tag] / grand_total_count[tag]
        
        sorted_tags = sorted(pos_avgs.items(), key=lambda x: x[1], reverse=True)
        tags, values = zip(*sorted_tags)
        
        plt.figure(figsize=(8, 5))
        plt.bar(tags, values, color='#2ecc71')
        plt.title(f"Part-of-Speech Importance ({self.name})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return True

def generate_global_syntax_section():
    html = ["<div class='experiment-block'><h2>Global Syntax & Logic Analysis</h2>"]
    syntax_path = "analysis/global_syntax_logic_analysis.csv"
    if os.path.exists(syntax_path):
        try:
            df = pd.read_csv(syntax_path)
            html.append("<h3>Top POS Bigrams (Weighted)</h3>")
            bigrams = df[df['category'] == 'POS_BIGRAM']
            grp_bi = bigrams.groupby(['type', 'key'])['norm_weight'].agg(['mean', 'count']).reset_index()
            grp_bi = grp_bi[grp_bi['count'] >= 3].sort_values('mean', ascending=False).head(10)
            html.append("<table class='stats-table'><thead><tr><th>Type</th><th>Pattern</th><th>Mean Norm. Weight</th><th>Count</th></tr></thead><tbody>")
            for _, row in grp_bi.iterrows():
                html.append(f"<tr><td>{row['type']}</td><td>{row['key']}</td><td>{row['mean']:.4f}</td><td>{row['count']}</td></tr>")
            html.append("</tbody></table>")
        except Exception as e:
            html.append(f"<p>Error loading syntax analysis: {e}</p>")
            
    weights_path = "analysis/syntax_weight_analysis.csv"
    if os.path.exists(weights_path):
        try:
            df = pd.read_csv(weights_path)
            html.append("<h3>Pattern Weight Summary</h3>")
            summary = df.groupby(['type', 'pattern'])['normalized_combined'].agg(['mean', 'count']).reset_index()
            html.append("<table class='stats-table'><thead><tr><th>Type</th><th>Pattern</th><th>Mean Norm. Weight</th><th>Count</th></tr></thead><tbody>")
            for _, row in summary.iterrows():
                html.append(f"<tr><td>{row['type']}</td><td>{row['pattern']}</td><td>{row['mean']:.4f}</td><td>{row['count']}</td></tr>")
            html.append("</tbody></table>")
        except Exception as e:
            html.append(f"<p>Error loading syntax weights: {e}</p>")
    html.append("</div>")
    return "\n".join(html)

def generate_full_report(out_dir, experiments, exp_plots_map, file_log, global_syntax_html=""):
    html = [f"""<!DOCTYPE html>
<html>
<head>
    <title>RAG Comparative Analysis</title>
    <style>{REPORT_CSS}</style>
</head>
<body>
    <div class=\"container\">
        <h1>RAG Pipeline Comparative Analysis</h1>
        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
        {global_syntax_html}
    """ ]
    
    for exp_name in experiments.keys():
        html.append(f"""
        <div class=\"experiment-block\">
            <h2>Experiment: {exp_name}</h2>
            <div class=\"plot-container\">""")
            
        if exp_name in exp_plots_map:
            for img, desc_key in exp_plots_map[exp_name]:
                desc = EXPLANATIONS.get(desc_key, "")
                html.append(f"""
                <div class=\"plot-card\">
                    <img src=\"{exp_name}/{img}\" alt=\"{desc_key}\">
                    <div class=\"description\">{desc}</div>
                </div>""")
        html.append("</div>")
        
        html.append(f"            <h3>Detailed Logs ({exp_name})</h3>")
        html.append("            <table class='stats-table'>")
        html.append("                <thead><tr><th>Condition</th><th>File</th><th>Query Text</th><th>Type</th><th>Links</th></tr></thead>")
        html.append("                <tbody>")
        
        # Filter logs for this experiment
        exp_logs = [f for f in file_log if f['exp_name'] == exp_name]
        exp_logs.sort(key=lambda x: (x['condition'], x['name']))
        
        for log in exp_logs:
            rel_path = os.path.relpath(log['path'], out_dir)
            badge_class = "badge-gen" if log['ftype'] == 'generation' else "badge-ret"
            query_text = log.get('query', 'N/A')
            
            html.append("            <tr>")
            html.append(f"                <td>{log['condition']}</td>")
            html.append(f"                <td>{log['name']}</td>")
            html.append(f"                <td style='max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;' title='{query_text}'>{query_text}</td>")
            html.append(f"                <td><span class='badge {badge_class}'>{log['ftype']}</span></td>")
            html.append(f"                <td>")
            html.append(f"                    <a href='{rel_path}/index.html' target='_blank'>Detailed View</a> | ")
            html.append(f"                    <a href='{rel_path}/' target='_blank'>Folder</a>")
            html.append("                </td>")
            html.append("            </tr>")
            
        html.append("</tbody></table></div>")
    html.append("</div></body></html>")
    
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write("\n".join(html))

def main():
    parser = argparse.ArgumentParser(description="Master Analysis Script (Optimized)")
    parser.add_argument("input_path", help="Root folder to scan")
    parser.add_argument("--output_dir", default="analysis_report_v2", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Scan Files
    files = []
    if os.path.isdir(args.input_path):
        files = list(Path(args.input_path).glob("**/*.pkl"))
    else:
        files = [Path(args.input_path)]
        
    experiments = {} 
    processed_files_log = []
    
    print(f"Scanning {len(files)} files using {args.workers} workers...")
    
    # 2. Parallel Processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Prepare args for worker (file_path, output_dir)
        work_items = [(f, args.output_dir) for f in files]
        
        # Submit all
        futures = [executor.submit(process_file_worker, item) for item in work_items]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc="Processing"):
            res = future.result()
            
            if res['status'] == 'failed':
                print(f"Error processing {res['file_path']}: {res['error']}")
                continue
                
            # Integrate Results
            exp_name = res['exp_name']
            condition = res['condition']
            
            if exp_name not in experiments:
                experiments[exp_name] = ExperimentAccumulator(exp_name)
                
            stats = res['stats']
            if stats:
                cond_stats = experiments[exp_name].get_condition(condition if res['ftype']=='generation' else 'retrieval_agg')
                
                if res['ftype'] == 'generation':
                    cond_stats.ingest_generation_stats(stats)
                elif res['ftype'] == 'retrieval':
                    cond_stats.ingest_retrieval_stats(stats)
                    
            processed_files_log.append(res)

    # 3. Generate Global Plots
    print("\nGenerating Global Plots...")
    exp_plots_map = defaultdict(list)
    
    for exp_name, accumulator in experiments.items():
        exp_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        if accumulator.plot_doc_position_comparison(os.path.join(exp_dir, "compare_position.png")):
            exp_plots_map[exp_name].append(("compare_position.png", "position_bias"))
        if accumulator.plot_top_tokens(os.path.join(exp_dir, "top_tokens.png")):
            exp_plots_map[exp_name].append(("top_tokens.png", "top_tokens"))
        if accumulator.plot_pos_distribution(os.path.join(exp_dir, "pos_dist.png")):
             exp_plots_map[exp_name].append(("pos_dist.png", "pos_distribution"))

    # 4. Generate Final Report
    print("Generating Report...")
    global_syntax_html = generate_global_syntax_section()
    generate_full_report(args.output_dir, experiments, exp_plots_map, processed_files_log, global_syntax_html)
    print(f"Report ready at {args.output_dir}/report.html")

if __name__ == "__main__":
    main()