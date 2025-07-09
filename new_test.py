# %%
import os
import json
import pickle
import datetime
import logging
import torch
import numpy as np
import pandas as pd
from transformers import LlamaForCausalLM
from tqdm import tqdm
from resources.generation import ExplainableAutoModelForGeneration
import nltk
from nltk.corpus import stopwords
from pathlib import Path
import gc
from typing import List, Dict, Tuple, Optional, Any
import traceback
import time

# %%
# CONFIGURATION
class Config:
    def __init__(self):
        # Environment setup
        self.transformers_cache = '/home/francomaria.nardini/raid/guidorocchietti/.cache/huggingface'
        self.cuda_devices = '3,4,5,6'
        
        # Model configuration
        self.model_id = 'meta-llama/Llama-3.1-8B-Instruct'
        self.max_seq_len = 1024
        self.max_gen_len = 256
        self.batch_size = 8
        
        # Paths
        self.ranked_chunks_path = '/home/francomaria.nardini/raid/guidorocchietti/code/efra_retrieval/results/ir_results_chunks.csv'
        self.evaluation_dataset_path = '/home/francomaria.nardini/raid/guidorocchietti/code/efra_retrieval/validation_Dataset_with_chunks_ids.csv'
        self.topics_path = '/home/francomaria.nardini/raid/guidorocchietti/data/EFRA/Evaluation Dataset/topics.tsv'
        self.output_folder = '/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG/outputs_evaluation/'
        
        # Processing options
        self.do_evaluation = True
        self.save_frequency = 1  # Save after every query
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        # Resume configuration
        self.checkpoint_file = 'processing_checkpoint.json'
        self.resume_from_checkpoint = True

config = Config()

# Set environment variables
os.environ['TRANSFORMERS_CACHE'] = config.transformers_cache
os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_devices
# %%
# LOGGING SETUP
def setup_logging():
    """Setup logging configuration"""
    log_dir = Path(config.output_folder) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f'processing_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# %%
# CHECKPOINT MANAGEMENT
class CheckpointManager:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        
    def save_checkpoint(self, data: Dict[str, Any]):
        """Save checkpoint data"""
        try:
            with open(self.checkpoint_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Checkpoint saved: {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint data"""
        if not self.checkpoint_path.exists():
            return None
            
        try:
            with open(self.checkpoint_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Checkpoint loaded: {self.checkpoint_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
            
    def clear_checkpoint(self):
        """Remove checkpoint file"""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info("Checkpoint cleared")

checkpoint_manager = CheckpointManager(config.checkpoint_file)

# %%
# UTILITY FUNCTIONS
def clean_tokens(tokens: List[str]) -> List[str]:
    """Clean and filter tokens"""
    stop_words = list(stopwords.words('english'))
    stop_words += ['', '\n', '.', ',', '(', ')', '[', ']', '{', '}', ':', ';', '?', '!', '"', "'", '-', '_', '/', '\\', '*', '&', '^', '%', '$', '#', '@', '~']
    tokens_filtered = [token.strip() for token in tokens if token.strip(' \n') not in stop_words + ['']]
    return tokens_filtered

def create_rag_messages(query: str, contexts: List[str]) -> List[Dict[str, str]]:
    """Creates chat-style messages for RAG using LLaMA-style chat template."""
    system_prompt = (
        "You are an expert on Food and Risk-related legislation. Use the following retrieved documents, "
        "ranked from highest to lowest relevance, to answer the user's query. "
        "Be thorough and accurate, and cite documents when useful."
    )
    
    context_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(contexts)])
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{context_text}\n\nQuery: {query}"},
    ]
    return messages

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def save_results(data: Dict[str, Any], query_id: int, output_folder: str):
    """Save results with proper error handling"""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f'perturbed_outputs_query_{query_id}.pkl'
    filepath = output_path / filename
    
    # If file exists, create timestamped version
    if filepath.exists():
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'perturbed_outputs_query_{query_id}_{timestamp}.pkl'
        filepath = output_path / filename
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Results saved: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save results for query {query_id}: {e}")
        raise

# %%
# IMPROVED GENERATION FUNCTIONS
def generate_outputs_with_retry(
    rag_prompt: str, 
    output: Optional[Any], 
    model: Any, 
    batch_size: int, 
    max_gen_len: int = 256,
    max_retries: int = 3,
    retry_delay: int = 5
) -> Tuple[Any, List[str], List[str], Any, Any]:
    """Generate outputs with retry logic for error handling"""
    
    for attempt in range(max_retries):
        try:
            clear_gpu_memory()
            
            if isinstance(rag_prompt, str):
                rag_prompt = [rag_prompt]
                
            if output is not None:
                output = output if isinstance(output, list) else [output]
                perturbed_output = model.compare(
                    rag_prompt, output, 
                    batch_size=batch_size, 
                    max_new_tokens=max_gen_len
                )
            else:
                perturbed_output = model.compare(
                    rag_prompt, 
                    batch_size=batch_size, 
                    max_new_tokens=max_gen_len,
                    do_sample=False, 
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            gen_probs = model._gen_probs
            exp_probs = model._exp_probs
            meaned_gen_nucleus = model.gen_nucleus_probs()
            idx_meaned_gen_nucleus = torch.argsort(meaned_gen_nucleus, dim=-1, descending=True)[0, :]
            gen_tokens = model.tokenizer.batch_decode(idx_meaned_gen_nucleus)
            meaned_exp_nucleus = model.cmp_nucleus_probs()
            idx_meaned_exp_nucleus = torch.argsort(meaned_exp_nucleus, dim=-1, descending=True)[0, :]
            exp_tokens = model.tokenizer.batch_decode(idx_meaned_exp_nucleus)
            
            return perturbed_output, gen_tokens, exp_tokens, gen_probs, exp_probs
            
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"CUDA OOM on attempt {attempt + 1}/{max_retries}: {e}")
            clear_gpu_memory()
            
            if attempt < max_retries - 1:
                # Reduce batch size on retry
                batch_size = max(1, batch_size // 2)
                logger.info(f"Reducing batch size to {batch_size} and retrying...")
                time.sleep(retry_delay)
            else:
                logger.error("Max retries reached for CUDA OOM")
                raise
                
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

def process_single_query(
    query_id: int, 
    query: str, 
    contexts: List[str], 
    model: Any, 
    config: Config
) -> Dict[str, Any]:
    """Process a single query with comprehensive error handling"""
    
    logger.info(f"Processing query {query_id}: {query[:100]}...")
    
    try:
        # Create complete RAG prompt
        complete_rag_prompt = create_rag_messages(query, contexts)
        complete_rag_prompt = model.tokenizer.apply_chat_template(
            complete_rag_prompt, tokenize=False, add_generation_prompt=True
        )
        
        # Create perturbed prompts
        perturbed_prompts = []
        for j in range(len(contexts)):
            perturbed_contexts = contexts[:j] + contexts[j+1:]
            perturbed_msg = create_rag_messages(query, perturbed_contexts)
            perturbed_prompts.append((perturbed_msg, f'masking document {j+1}'))
        
        chat_templates = [
            (model.tokenizer.apply_chat_template(x[0], tokenize=False, add_generation_prompt=True), x[1]) 
            for x in perturbed_prompts
        ]
        
        # Generate complete output
        output = model.generate(
            [complete_rag_prompt],
            max_new_tokens=config.max_gen_len,
            do_sample=False,
            top_p=1,
            temperature=0.7,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        meaned_gen_nucleus = model.gen_nucleus_probs()
        idx_meaned_gen_nucleus = torch.argsort(meaned_gen_nucleus, dim=-1, descending=True)[0, :]
        gen_tokens = model.tokenizer.batch_decode(idx_meaned_gen_nucleus)
        
        # Process perturbed outputs
        perturbed_outputs = {}
        
        for i, (prompt, description) in enumerate(chat_templates):
            logger.info(f'Processing perturbation {i+1}/{len(chat_templates)}: {description}')
            
            try:
                # Unconstrained generation
                perturbed_output, gen_tokens_pert, exp_tokens, gen_probs, exp_probs = generate_outputs_with_retry(
                    prompt, None, model, config.batch_size, config.max_gen_len, config.max_retries, config.retry_delay
                )
                
                perturbed_outputs[description] = {
                    'perturbed_output': perturbed_output,
                    'gen_tokens': gen_tokens_pert,
                    'exp_tokens': exp_tokens,
                    'gen_probs': gen_probs,
                    'exp_probs': exp_probs
                }
                
                # Constrained generation
                perturbed_output_const, gen_tokens_const, exp_tokens_const, gen_probs_const, exp_probs_const = generate_outputs_with_retry(
                    prompt, output, model, config.batch_size, config.max_gen_len, config.max_retries, config.retry_delay
                )
                
                perturbed_outputs[description + '_constrained'] = {
                    'perturbed_output': perturbed_output_const,
                    'gen_tokens': gen_tokens_const,
                    'exp_tokens': exp_tokens_const,
                    'gen_probs': gen_probs_const,
                    'exp_probs': exp_probs_const
                }
                
            except Exception as e:
                logger.error(f"Failed to process perturbation {description}: {e}")
                # Continue with other perturbations instead of failing completely
                continue
        
        # Add complete output
        perturbed_outputs['complete'] = {
            'perturbed_output': output,
            'gen_tokens': gen_tokens,
            'exp_tokens': '',
            'gen_probs': model._gen_probs,
            'exp_probs': ''
        }
        
        return perturbed_outputs
        
    except Exception as e:
        logger.error(f"Failed to process query {query_id}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# %%
# MAIN PROCESSING FUNCTION
def main():
    """Main processing function with checkpoint support"""
    
    logger.info("Starting RAG processing with interpretability analysis")
    
    try:
        # Load model
        logger.info(f"Loading model: {config.model_id}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = ExplainableAutoModelForGeneration(LlamaForCausalLM).from_pretrained(
            config.model_id,
            device_map='auto',
            torch_dtype=torch.bfloat16
        )
        
        # Load datasets
        logger.info("Loading datasets...")
        ranked_chunks = pd.read_csv(config.ranked_chunks_path)
        evaluation_dataset = pd.read_csv(config.evaluation_dataset_path)
        topics = pd.read_csv(config.topics_path, sep='\t')
        
        # Check for existing checkpoint
        checkpoint_data = None
        start_idx = 0
        
        if config.resume_from_checkpoint:
            checkpoint_data = checkpoint_manager.load_checkpoint()
            if checkpoint_data:
                start_idx = checkpoint_data.get('last_processed_idx', 0) + 1
                logger.info(f"Resuming from query index {start_idx}")
        
        # Process queries
        total_queries = len(topics)
        failed_queries = []
        
        for qid in tqdm(range(start_idx, total_queries), desc='Processing queries', initial=start_idx, total=total_queries):
            try:
                query = topics.iloc[qid]['query']
                
                # Get contexts
                if config.do_evaluation:
                    contexts = evaluation_dataset[evaluation_dataset['query'] == query]['text'].unique().tolist()
                else:
                    contexts = ranked_chunks[ranked_chunks['query_id'] == qid]['retrieved_text'].unique().tolist()
                
                if not contexts:
                    logger.warning(f"No contexts found for query {qid}")
                    continue
                
                # Process query
                perturbed_outputs = process_single_query(qid, query, contexts, model, config)
                
                # Save results
                save_results(perturbed_outputs, qid, config.output_folder)
                
                # Update checkpoint
                checkpoint_data = {
                    'last_processed_idx': qid,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'total_queries': total_queries,
                    'failed_queries': failed_queries
                }
                checkpoint_manager.save_checkpoint(checkpoint_data)
                
                # Clear memory periodically
                if qid % 5 == 0:
                    clear_gpu_memory()
                
            except Exception as e:
                logger.error(f"Failed to process query {qid}: {e}")
                failed_queries.append({'query_id': qid, 'error': str(e)})
                
                # Save failed queries info
                checkpoint_data = {
                    'last_processed_idx': qid,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'total_queries': total_queries,
                    'failed_queries': failed_queries
                }
                checkpoint_manager.save_checkpoint(checkpoint_data)
                
                # Continue with next query instead of stopping
                continue
        
        # Final cleanup
        checkpoint_manager.clear_checkpoint()
        logger.info(f"Processing completed. Failed queries: {len(failed_queries)}")
        
        if failed_queries:
            # Save failed queries summary
            failed_queries_path = Path(config.output_folder) / 'failed_queries.json'
            with open(failed_queries_path, 'w') as f:
                json.dump(failed_queries, f, indent=2)
            logger.info(f"Failed queries saved to: {failed_queries_path}")
        
    except Exception as e:
        logger.error(f"Critical error in main processing: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# %%
if __name__ == "__main__":
    main()
