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
        :param dataset_name: 'musique', 'nq', or 'qampari'
        :param split: Data split to load (e.g. 'validation', 'test')
        """
        self.dataset_name = dataset_name.lower()
        self.split = split

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the dataset and normalizes it to a common format:
        {'query_id': str, 'query': str, 'answers': List[str], 'paragraphs': List[str]}
        """
        if 'musique' in self.dataset_name:
            return self._load_musique()
        elif 'nq' in self.dataset_name or 'natural_questions' in self.dataset_name:
            return self._load_nq()
        elif 'qampari' in self.dataset_name:
            return self._load_qampari()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. Expected 'musique', 'nq', or 'qampari'.")

    def _load_qampari(self) -> List[Dict[str, Any]]:
        """Loads QAMPARI dataset (from LOFT version)."""
        logger.info(f"Loading QAMPARI dataset ({self.split} split)...")
        # Try 128k first to get more queries, fallback to 32k
        try:
            dataset = load_dataset('f20180301/loft-rag-qampari-128k', split=self.split)
        except:
            dataset = load_dataset('f20180301/loft-rag-qampari-32k', split=self.split)
        
        normalized = []
        for i, item in enumerate(dataset):
            # Parse query from the instruction-heavy question field
            query = item['question']
            if 'query: ' in query:
                query = query.split('query: ')[-1].strip()
            
            # Context parsing
            raw_context = item['context']
            doc_splits = raw_context.split('ID: ')
            paragraphs = []
            for split in doc_splits:
                if 'CONTENT: ' in split:
                    content = split.split('CONTENT: ')[-1].strip()
                    paragraphs.append(content)
            
            normalized.append({
                'query_id': f'qampari_{i}',
                'query': query,
                'answers': item['answers'],
                'paragraphs': paragraphs
            })
        return normalized

    def _load_musique(self) -> List[Dict[str, Any]]:
        """Loads MuSiQue dataset."""
        logger.info(f"Loading MuSiQue dataset ({self.split} split)...")
        dataset = load_dataset('bdsaglam/musique', split=self.split)
        normalized = []
        for i, item in enumerate(dataset):
            answers = [item['answer']] if isinstance(item.get('answer'), str) else item.get('answer_aliases', [str(item.get('answer'))])
            
            normalized.append({
                'query_id': item.get('id', f'musique_{i}'),
                'query': item['question'],
                'answers': answers,
                'paragraphs': [p['paragraph_text'] for p in item['paragraphs']]
            })
        return normalized

    def _load_nq(self) -> List[Dict[str, Any]]:
        """Loads Natural Questions (Open) dataset."""
        logger.info(f"Loading Natural Questions dataset ({self.split} split)...")
        dataset = load_dataset('google-research-datasets/natural_questions', split=self.split)
        normalized = []
        for i, item in enumerate(dataset):
            answers = item['annotations']['answer']
            
            normalized.append({
                'query_id': item.get('id', f'nq_{i}'),
                'query': item['question'],
                'answers': answers
            })
        return normalized
