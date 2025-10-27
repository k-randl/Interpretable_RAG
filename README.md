# Interpretable RAG

This project provides a framework for building and evaluating interpretable Retrieval-Augmented Generation (RAG) pipelines. The core of the project is the ability to explain the contribution of retrieved documents and query parts to the final generated output, using Shapley values.

## Features

*   **Interpretable RAG Pipeline:** A RAG pipeline that can explain the importance of retrieved documents and query components.
*   **Shapley Value-based Explanations:** Uses Shapley values to quantify the contribution of each input feature (e.g., retrieved documents, query tokens) to the generated output.
*   **Multiple Explanation Aggregations:** Supports different aggregation methods for Shapley values, such as token-level, sequence-level, and bag-of-words.
*   **Online and Offline Retrieval:** Supports both online and offline retrieval methods.
*   **Extensible and Modular:** The code is organized in a modular way, making it easy to extend and adapt to different models and datasets.

## Installation

After cloning or downloading this repository, first run the Linux shell script [`./setup.sh`](https://github.com/k-randl/self-explaining_llms/blob/main/setup.sh).
It will initialize the workspace by performing the following steps:

1. It will install the required Python modules by running `pip install -r "./requirements.txt"`
2. It will download the necessary Python code to compute the [BARTScore](https://github.com/neulab/BARTScore) by Yuan et al. (2021) to "./resources/bart_score.py".
3. It will download the necessary Python code to compute the [LongDocFACTScore](https://github.com/jbshp/LongDocFACTScore) by Bishop et al. (2024) to "./resources/ldfacts.py".

After running the script, copy your credential file to "./data/service-account-external-efra.json"

## Usage

The main entry point for running experiments is the `scripts/run_pipeline.py` script. Here is an example of how to run it:

```bash
python scripts/run_pipeline.py \
    --topics_path /path/to/your/topics.tsv \
    --ranked_list_path /path/to/your/ranked_list.csv \
    --collection_path /path/to/your/collection.tsv \
    --output_path /path/to/your/output_directory/ \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --num_docs_context 6 \
    --max_gen_len 300 \
    --run_original \
    --run_randomized
```

### Arguments

*   `--topics_path`: Path to the file with the queries.
*   `--ranked_list_path`: Path to the ranked list from the retrieval.
*   `--collection_path`: Path to the collection of passages.
*   `--output_path`: Base directory to save all the results.
*   `--model_id`: ID of the generative model from Hugging Face.
*   `--num_docs_context`: Number of documents to use as context.
*   `--max_gen_len`: Maximum length of the generated response.
*   `--run_original`: Run the experiment with the original contexts.
*   `--run_randomized`: Run with contexts in random order.
*   `--run_no_duplicates`: Run with contexts without duplicates.

## Project Structure

*   `src/Interpretable_RAG`: Contains the core source code for the interpretable RAG pipeline.
*   `scripts`: Contains scripts for running experiments, building indexes, and performing analysis.
*   `data`: Contains the data used for the experiments.
*   `resources`: Contains additional resources, such as evaluation scripts.
*   `outputs_evaluation`: Contains the evaluation outputs.
*   `outputs_retrieved`: Contains the retrieved outputs.
*   `results`: Contains the results of the experiments.
*   `plots`: Contains plots for the analysis of the results.

## How it Works

The interpretability of the RAG pipeline is achieved by using Shapley values. The contribution of each retrieved document and each query token is computed by analyzing the change in the model's output probability when that feature is included or excluded. The project implements both precise Shapley value calculation and approximations like KernelSHAP.

The `ExplainableAutoModelForGeneration` class wraps a Hugging Face `transformers` model and provides the functionality to compute the Shapley values. The `ExplainableAutoModelForRAG` class combines this explainable generator with a retriever to create the full interpretable RAG pipeline.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.