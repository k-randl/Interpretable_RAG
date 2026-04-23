import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
except ImportError:
    logger.warning("Hugging Face 'datasets' library is not installed. Please install it using 'pip install datasets'.")

class DatasetLoader:
    def __init__(self, dataset_name: str, split: str = 'validation'):
        """
        Initializes the DatasetLoader.
        :param dataset_name: 'musique' or 'nq'
        :param split: Data split to load (e.g. 'validation', 'test')
        """
        self.dataset_name = dataset_name.lower()
        self.split = split

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the dataset and normalizes it to a common format:
        {'query_id': str, 'query': str, 'answers': List[str]}
        """
        if 'musique' in self.dataset_name:
            return self._load_musique()
        elif 'nq' in self.dataset_name or 'natural_questions' in self.dataset_name:
            return self._load_nq()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. Expected 'musique' or 'nq'.")

    def _load_musique(self) -> List[Dict[str, Any]]:
        """Loads MuSiQue dataset."""
        logger.info(f"Loading MuSiQue dataset ({self.split} split)...")
        dataset = load_dataset('bdsaglam/musique', split=self.split)
        normalized = []
        for i, item in enumerate(dataset):
            # Normalizing answer aliases if present, otherwise fallback to standard answer
            answers = [item['answer']] if isinstance(item.get('answer'), str) else item.get('answer_aliases', [str(item.get('answer'))])
            
            normalized.append({
                'query_id': item.get('id', f'musique_{i}'),
                'query': item['question'],
                'answers': answers
            })
        return normalized

    def _load_nq(self) -> List[Dict[str, Any]]:
        """Loads Natural Questions (Open) dataset."""
        logger.info(f"Loading Natural Questions dataset ({self.split} split)...")
        dataset = load_dataset('nq_open', split=self.split)
        normalized = []
        for i, item in enumerate(dataset):
            answers = item.get('answer', [])
            if isinstance(answers, str):
                answers = [answers]
                
            normalized.append({
                'query_id': item.get('id', f'nq_{i}'),
                'query': item['question'],
                'answers': answers
            })
        return normalized
