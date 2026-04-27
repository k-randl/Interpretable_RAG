import string
import re
from typing import List, Dict
from collections import Counter
import logging

logger = logging.getLogger(__name__)

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None
    logger.warning("rouge_score library not installed. ROUGE-L metrics will be 0.0.")

try:
    from bert_score import score as bert_scorer
except ImportError:
    bert_scorer = None
    logger.warning("bert_score library not installed. BERTScore will be 0.0.")

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return str(text).lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_metrics(prediction: str, ground_truths: List[str]) -> Dict[str, float]:
    if not ground_truths:
        return {'exact_match': 0.0, 'f1_score': 0.0, 'rougeL': 0.0}
        
    em_scores = [exact_match_score(prediction, gt) for gt in ground_truths]
    f1_scores = [f1_score(prediction, gt) for gt in ground_truths]
    
    best_em = max(em_scores) if em_scores else 0.0
    best_f1 = max(f1_scores) if f1_scores else 0.0
    
    best_rougeL = 0.0
    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(gt, prediction)['rougeL'].fmeasure for gt in ground_truths]
        best_rougeL = max(rouge_scores) if rouge_scores else 0.0

    best_bert_score = 0.0
    if bert_scorer is not None and prediction.strip():
        # bert_score takes lists of strings. We can score prediction against all ground truths.
        # It returns P, R, F1. We take F1.
        try:
            # We want the max score across all ground truths
            P, R, F1 = bert_scorer([prediction] * len(ground_truths), ground_truths, lang="en", verbose=False)
            best_bert_score = F1.max().item()
        except Exception as e:
            logger.error(f"Error computing BERTScore: {e}")
            best_bert_score = 0.0

    return {
        'exact_match': best_em,
        'f1_score': best_f1,
        'rougeL': best_rougeL,
        'bert_score': best_bert_score
    }
