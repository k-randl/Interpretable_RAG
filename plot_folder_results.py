#%%
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

# Add the project's root directory to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Interpretable_RAG.plotting import (
    plot_attribution_generator,
    higlight_attribution_generator,
    plot_attribution_summary_generator,
    plot_shap_by_pos,
    tokens_to_pos,
    shap_by_pos,
    process_global_importance,
    plot_global_importance_distribution,
    safe_load_pickle
)
from src.Interpretable_RAG.generation import GeneratorExplanation

def main():
    parser = argparse.ArgumentParser(description="Load and visualize explanation data from a folder of pickle files.")
    parser.add_argument("pickle_folder", type=str, help="The absolute path to the folder containing the pickle files.")
    args = parser.parse_args()

    print(f"Loading data from folder: {args.pickle_folder}")

    # Load all pickle files in the directory
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    explanations = GeneratorExplanation.load(args.pickle_folder, tokenizer=tokenizer)

    print(f"Loaded {len(explanations)} explanation files. Generating plots for each...")

    # Create an output directory for the plots
    output_dir = "plots_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved in the '{output_dir}' directory.")

    all_pickle_files = [os.path.join(args.pickle_folder, f) for f in os.listdir(args.pickle_folder) if f.endswith('.pkl')]

    for file_path, explanation in explanations.items():
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\\nProcessing: {base_filename}")

        # 1. Plot token-level attribution
        fig1 = plot_attribution_generator(explanation, aggregation='token', normalize=True, show=False)
        fig1.savefig(os.path.join(output_dir, f"{base_filename}_token_attribution.png"))
        plt.close(fig1)

        # 2. Highlight dominant passages in the generated text
        html_output = higlight_attribution_generator(explanation, show=False)
        with open(os.path.join(output_dir, f"{base_filename}_highlighted_text.html"), "w") as f:
            f.write(html_output)

        # 3. Plot document-level attribution summary
        fig2 = plot_attribution_summary_generator(explanation, aggregation='sequence', absolute=False, normalize=True, show=False)
        fig2.savefig(os.path.join(output_dir, f"{base_filename}_document_summary.png"))
        plt.close(fig2)

        # 4. POS Analysis
        print(f"Performing POS analysis for {base_filename}...")
        shap_values = explanation.get_shapley_values('context', 'token')
        tokens = explanation.gen_tokens
        pos_tags, _, _, _ = tokens_to_pos(tokens)
        df_pos = shap_by_pos(shap_values, pos_tags)
        fig3 = plot_shap_by_pos(df_pos, show=False)
        if fig3:
            fig3.savefig(os.path.join(output_dir, f"{base_filename}_pos_analysis.png"))
            plt.close(fig3)

    # 5. Global Document Importance Analysis (after processing all files)
    print("\\nPerforming global document importance analysis...")
    df_global_importance = process_global_importance(all_pickle_files, safe_load_pickle)
    if not df_global_importance.empty:
        fig4 = plot_global_importance_distribution(df_global_importance)
        if fig4:
            fig4.savefig(os.path.join(output_dir, "global_document_importance.png"))
            plt.close(fig4)
            print("Saved global_document_importance.png")

    print(f"\\nAll plots have been generated and saved in '{output_dir}'.")

if __name__ == "__main__":
    main()
else:
    path = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/results/treccast_exp_15_09_25/original/'
    ### set the arguments for testing in jupyter notebook
    class Args:
        pickle_folder = path
    args = Args()
    print(f"Loading data from folder: {args.pickle_folder}")
    # Load all pickle files in the directory
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    explanations = GeneratorExplanation.load(args.pickle_folder, tokenizer=tokenizer)
    print(f"Loaded {len(explanations)} explanation files. Generating plots for each...")
    # Create an output directory for the plots
    output_dir = "plots_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved in the '{output_dir}' directory.")
    all_pickle_files = [os.path.join(args.pickle_folder, f) for f in os.listdir(args.pickle_folder) if f.endswith('.pkl')]
    for file_path, explanation in explanations.items():
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\\nProcessing: {base_filename}")
        # 1. Plot token-level attribution
        fig1 = plot_attribution_generator(explanation, aggregation='token', normalize=True, show=False)
        fig1.savefig(os.path.join(output_dir, f"{base_filename}_token_attribution.png"))
        plt.close(fig1)
        # 2. Highlight dominant passages in the generated text
        html_output = higlight_attribution_generator(explanation, show=False)
        with open(os.path.join(output_dir, f"{base_filename}_highlighted_text.html"), "w") as f:
            f.write(html_output)
        # 3. Plot document-level attribution summary
        fig2 = plot_attribution_summary_generator(explanation, aggregation='sequence', absolute=False, normalize=True, show=False)
        fig2.savefig(os.path.join(output_dir, f"{base_filename}_document_summary.png"))
        plt.close(fig2)
        # 4. POS Analysis
        print(f"Performing POS analysis for {base_filename}...")
        shap_values = explanation.get_shapley_values('context', 'token')
        tokens = explanation.gen_tokens
        pos_tags, _, _, _ = tokens_to_pos(tokens)
        df_pos = shap_by_pos(shap_values, pos_tags)
        fig3 = plot_shap_by_pos(df_pos, show=False)
        if fig3:
            fig3.savefig(os.path.join(output_dir, f"{base_filename}_pos_analysis.png"))
            plt.close(fig3)
    # 5. Global Document Importance Analysis (after processing all files)
    print("\\nPerforming global document importance analysis...")
    df_global_importance = process_global_importance(all_pickle_files, safe_load_pickle)
    if not df_global_importance.empty:
        fig4 = plot_global_importance_distribution(df_global_importance)
        if fig4:
            fig4.savefig(os.path.join(output_dir, "global_document_importance.png"))
            plt.close(fig4)
            print("Saved global_document_importance.png")
    print(f"\\nAll plots have been generated and saved in '{output_dir}'.")
#====================================================================#
 #                           IMPORTS                                 #
 #====================================================================# 
 
# %%
