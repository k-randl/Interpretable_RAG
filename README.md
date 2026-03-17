# RAG-E

[![Deploy PyPI](https://github.com/k-randl/Interpretable_RAG/actions/workflows/python-publish.yml/badge.svg?branch=main)](https://github.com/k-randl/Interpretable_RAG/actions/workflows/python-publish.yml)

This project provides a framework for building and evaluating interpretable Retrieval-Augmented Generation (RAG) pipelines. The core of the project is the ability to explain the contribution of retrieved documents and query parts to the final generated output, using Shapley values.

## Features

*   **Interpretable RAG Pipeline:** A RAG pipeline that can explain the importance of retrieved documents and query components.
*   **Shapley Value-based Explanations:** Uses Shapley values to quantify the contribution of each input feature (e.g., retrieved documents, query tokens) to the generated output.
*   **Multiple Explanation Aggregations:** Supports different aggregation methods for Shapley values, such as token-level, sequence-level, and bag-of-words.
*   **Online and Offline Retrieval:** Supports both online and offline retrieval methods.
*   **Extensible and Modular:** The code is organized in a modular way, making it easy to extend and adapt to different models and datasets.

## Installation

### Install source code fom GitHub
This option is mainly useful for development or replication our experiments. First clone this repository.
After cloning or downloading the repository, run the Linux shell script `./setup.sh`.
It will initialize the workspace by performing the following steps:

1. It will install the required Python modules by running `pip install -r "./requirements.txt"`
2. It will download the necessary Python code to compute the [BARTScore](https://github.com/neulab/BARTScore) by Yuan et al. (2021) to "./resources/bart_score.py".
3. It will download the necessary Python code to compute the [LongDocFACTScore](https://github.com/jbshp/LongDocFACTScore) by Bishop et al. (2024) to "./resources/ldfacts.py".

## Usage

### How to use RAG-E:

The classes `ExplainableAutoModelForRetrieval` and `ExplainableAutoModelForGeneration` wrap Hugging Face [`transformers`](https://github.com/huggingface/transformers) models and provide the functionality to compute attribution scores.
For **retrieval**, RAG-E supports the following token attribution methods:

- `.grad(...)`: Raw gradients towards the inputs of the last batch 
- `.aGrad(...)`: [AGrad (`da ⊙ a`)](https://doi.org/10.1109/BigData52589.2021.9671639) by Liu et al. (2021) of the last batch.
- `.gradIn(...)`: Gradient times input (`dx ⊙ x`) scores of the last batch.
- `.intGrad(...)`: [Integrated Gradients](https://doi.org/10.48550/arXiv.1703.01365) by Sundararajan et al. (2017) of the last batch.
  
For **generation** RAG-E supports precise and Kernel-SHAP approximated Shapley Values for document attribution.

The `ExplainableAutoModelForRAG` class combines these models to create the full interpretable RAG pipeline.
Here is an Example with Llama 3.1 8B and Snowflake Arctic Embed v2 (*more examples in `scripts/demos`*):
```python
from src.Interpretable_RAG.rag import ExplainableAutoModelForRAG

# Load Pipeline:
model = ExplainableAutoModelForRAG(
    # Retriever info:
    query_encoder_name_or_path='Snowflake/snowflake-arctic-embed-l-v2.0',
    retriever_query_format='query: {query}',
    retriever_token_processor=lambda s: s.replace('▁', ' '),
    retriever_kwargs={'add_pooling_layer':False},

    # Generator info:
    generator_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
    generator_token_processor=lambda s: s.replace('Ġ', ' ').strip('Ċ'),
    generator_kwargs={'device_map':'auto', 'torch_dtype':torch.bfloat16}
)

# MSMarco query and passage as an example:
query =  "Where was Marie Curie born?"
contexts = [
    "Maria Skłodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace.",
    "Maria Skłodowska was born in Warsaw, in Congress Poland in the Russian Empire, as the fifth and youngest child of well-known teachers Bronisława, née Boguska, and Władysław Skłodowski.",
    "While a French citizen, Marie Skłodowska Curie, who used both surnames, never lost her sense of Polish identity. She taught her daughters the Polish language and took them on visits to Poland.",
    "Marie Curie founded the Curie Institute in Paris in 1920, and the Curie Institute in Warsaw in 1932.",
]

# Generate answer:
output = model(
    query=query,
    contexts=contexts,
    k=5,
    generator_kwargs={
        'max_new_tokens':256,
        'do_sample':False,
        'top_p':1,
        'num_beams':1,
        'batch_size':64,
        'max_samples_query':32,
        'max_samples_context':32,
        'conditional':True
    }
)

# Explain retriever:
ret_attributions = model.retriever.intGrad()

# Explain generator:
gen_attributions = model.generator.get_shapley_values()

```

To visualize these explanations, RAG-E includes easy-to-use plotting functions:
```python
from src.Interpretable_RAG.plotting import visualize_importance_retriever, visualize_importance_generator, plot_document_importance_rag, higlight_importance_rag

# Functions to normalize tokens:
retriever_token_processor=lambda s: s.replace('▁', ' '),
generator_token_processor=lambda s: s.replace('Ġ', ' ').strip('Ċ')

# Generate highlighted tokens for the retriever:
visualize_importance_retriever(model.retriever, method='intGrad', token_processor=retriever_token_processor, show:bool=True)

# Generate highlighted tokens for the generator:
visualize_attribution_generator(model.generator, aggregation='token', token_processor=generator_token_processor, show:bool=True)

# Generate highlighted tokens for the rag pipeline:
higlight_importance_rag(model, retriever_method='intGrad', show:bool=True,
        retriever_token_processor=retriever_token_processor,
        generator_token_processor=generator_token_processor
)

# Plot document importance for the rag pipeline:
plot_document_importance_rag(model, show:bool=True)
```

### Replicating our experiments:
Replication our experiments requires installation **Option 2**. The main entry point for running experiments is the `scripts/run_pipeline.py` script. Here is an example of how to run it:

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

**Arguments:**

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

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the **GPL v3 License**. See the `LICENSE` file for details.
