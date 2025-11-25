# Analysis Toolkit Documentation

This directory contains a suite of scripts to interpret, visualize, and analyze the results of your RAG (Retrieval-Augmented Generation) pipeline.

## 1. Core Analysis Scripts

### `analyze_global_syntax.py` (The "Grammar" Analyzer)
**What it does:**  
This script performs a deep linguistic analysis of your model's attention. It answers questions like: *"Does the model focus more on Verbs or Nouns?"* or *"Does it pay attention to the Subject or the Object of the sentence?"*

**How it works:**
1.  **Reconstruction:** It takes the raw tokens (e.g., `["un", "believ", "able"]`) and stitches them into full words (`"unbelievable"`).
2.  **Alignment:** It maps the model's attribution scores (gradients/Shapley values) from tokens to these full words.
3.  **Parsing:** It uses NLTK to tag every word with its Part-of-Speech (POS) and grammatical role (Subject, Object, etc.).
4.  **Aggregation:** It calculates the average importance of specific patterns (e.g., "Adjective + Noun") relative to the global average.

**Output:** `analysis/global_syntax_logic_analysis.csv`

---

### `master_analysis.py` (The Report Generator)
**What it does:**  
This is your "one-stop-shop" script. It scans your entire `results/` directory, generates visualizations for *every* experiment file, and compiles them into a browsable HTML report.

**Features:**
- **Position Bias Plots:** Checks if your model ignores documents at the bottom of the list.
- **Heatmaps:** Visualizes exactly which words in the query or context triggered the response.
- **HTML Report:** Creates a `report.html` that organizes all your results by experiment and condition.

**Usage:**
```bash
python3 analysis/master_analysis.py results/ --output_dir analysis_report
```

---

### `analyze_syntax_weights.py` (The Pattern Hunter)
**What it does:**  
Similar to the global syntax analyzer but more targeted. It specifically looks for and quantifies "Preposition + Substantive" and "Determiner + Noun" patterns to test specific hypotheses about syntactic glue.

**Output:** `analysis/syntax_weight_analysis.csv`

---

## 2. Visualization & Inspection Tools

### `visualize_results.py`
**What it does:**  
Generates quick heatmaps and plots for a **single** result file (`.pkl`). Use this when you want to debug or inspect one specific query.

**Usage:**
```bash
python3 analysis/visualize_results.py results/generation/path/to/file.pkl
```

### `explain_pickle.py`
**What it does:**  
A simple utility that peeks inside a `.pkl` file and tells you what it contains (Retrieval vs. Generation data, available metrics, tensor shapes, etc.).

**Usage:**
```bash
python3 analysis/explain_pickle.py results/path/to/file.pkl
```

## 3. Library Scripts (Helpers)

- **`analyze_results.py`**: The core library containing functions to load data and calculate statistics. Used by other scripts.
- **`analyze_retrieval.py`**: Contains specialized plotting logic for retrieval tasks (e.g., weighted token overlap).
- **`generate_plots.py`**: Contains plotting logic for generation tasks (e.g., Shapley heatmaps).

## Quick Start Guide

**1. To generate a full HTML report of all experiments:**
```bash
python3 analysis/master_analysis.py results/
```

**2. To analyze the linguistic/syntactic logic of the model:**
```bash
python3 analysis/analyze_global_syntax.py
```

**3. To inspect a single file:**
```bash
python3 analysis/visualize_results.py <path_to_pkl_file>
```
