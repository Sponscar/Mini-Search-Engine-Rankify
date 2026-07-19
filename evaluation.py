"""
Rankify - IR Evaluation Module
Implements advanced Information Retrieval evaluation metrics:
- MAP (Mean Average Precision)
- MRR (Mean Reciprocal Rank)
- NDCG@k (Normalized Discounted Cumulative Gain)
- Precision@k, Recall@k, F1@k
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


class IREvaluator:
    """Information Retrieval Evaluator with modern metrics."""

    def __init__(self, ground_truth_path: str = 'ground_truth.json'):
        """
        Initialize evaluator with ground truth data.

        Args:
            ground_truth_path: Path to ground truth JSON file
        """
        self.ground_truth_path = ground_truth_path
        self.ground_truth = None
        self.queries = []
        self.relevance_levels = {}
        
        self.load_ground_truth()
    
    def load_ground_truth(self):
        """Load ground truth queries and relevance judgments."""
        try:
            with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.ground_truth = data
            self.queries = data['queries']
            self.relevance_levels = data.get('relevance_levels', {})
            
            print(f"[Evaluator] Loaded {len(self.queries)} ground truth queries")
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Ground truth file not found: {self.ground_truth_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in ground truth file: {e}")
    
    def get_relevance_dict(self, query_id: str) -> Dict[str, int]:
        """
        Get relevance dictionary for a query.

        Args:
            query_id: Query identifier

        Returns:
            Dict mapping doc_id to relevance score (2=highly, 1=partial, 0=not)
        """
        query_data = next((q for q in self.queries if q['query_id'] == query_id), None)
        
        if query_data is None:
            raise ValueError(f"Query {query_id} not found in ground truth")
        
        relevance_dict = {}
        relevant_docs = query_data['relevant_docs']
        
        # Highly relevant = 2
        for doc_id in relevant_docs.get('highly_relevant', []):
            relevance_dict[doc_id] = 2
        
        # Partially relevant = 1
        for doc_id in relevant_docs.get('partially_relevant', []):
            relevance_dict[doc_id] = 1
        
        # Not relevant = 0 (implicit, don't need to add)
        
        return relevance_dict
    
    def precision_at_k(self, results: List[Dict], relevance_dict: Dict[str, int], k: int = 10) -> float:
        """
        Calculate Precision@k.

        Args:
            results: Search results (list of dicts with 'doc_id')
            relevance_dict: Ground truth relevance mapping
            k: Cutoff rank

        Returns:
            Precision@k value
        """
        if not results or k == 0:
            return 0.0
        
        top_k = results[:k]
        relevant_count = sum(1 for r in top_k if relevance_dict.get(r['doc_id'], 0) > 0)
        
        return relevant_count / k
    
    def recall_at_k(self, results: List[Dict], relevance_dict: Dict[str, int], k: int = 10) -> float:
        """
        Calculate Recall@k.

        Args:
            results: Search results
            relevance_dict: Ground truth relevance mapping
            k: Cutoff rank

        Returns:
            Recall@k value
        """
        total_relevant = sum(1 for score in relevance_dict.values() if score > 0)
        
        if total_relevant == 0:
            return 0.0
        
        top_k = results[:k]
        relevant_retrieved = sum(1 for r in top_k if relevance_dict.get(r['doc_id'], 0) > 0)
        
        return relevant_retrieved / total_relevant
    
    def f1_at_k(self, results: List[Dict], relevance_dict: Dict[str, int], k: int = 10) -> float:
        """
        Calculate F1@k.

        Args:
            results: Search results
            relevance_dict: Ground truth relevance mapping
            k: Cutoff rank

        Returns:
            F1@k value
        """
        precision = self.precision_at_k(results, relevance_dict, k)
        recall = self.recall_at_k(results, relevance_dict, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def average_precision(self, results: List[Dict], relevance_dict: Dict[str, int]) -> float:
        """
        Calculate Average Precision (AP) for a single query.

        Args:
            results: Search results
            relevance_dict: Ground truth relevance mapping

        Returns:
            Average Precision value
        """
        total_relevant = sum(1 for score in relevance_dict.values() if score > 0)
        
        if total_relevant == 0:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for rank, result in enumerate(results, 1):
            doc_id = result['doc_id']
            if relevance_dict.get(doc_id, 0) > 0:
                relevant_count += 1
                precision_at_rank = relevant_count / rank
                precision_sum += precision_at_rank
        
        return precision_sum / total_relevant
    
    def mean_average_precision(self, results_dict: Dict[str, List[Dict]]) -> float:
        """
        Calculate Mean Average Precision (MAP) across all queries.

        Args:
            results_dict: Dict mapping query_id to search results

        Returns:
            MAP value
        """
        ap_scores = []
        
        for query_id, results in results_dict.items():
            relevance_dict = self.get_relevance_dict(query_id)
            ap = self.average_precision(results, relevance_dict)
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def reciprocal_rank(self, results: List[Dict], relevance_dict: Dict[str, int]) -> float:
        """
        Calculate Reciprocal Rank (RR) for a single query.

        Args:
            results: Search results
            relevance_dict: Ground truth relevance mapping

        Returns:
            Reciprocal Rank value
        """
        for rank, result in enumerate(results, 1):
            doc_id = result['doc_id']
            if relevance_dict.get(doc_id, 0) > 0:
                return 1.0 / rank
        
        return 0.0
    
    def mean_reciprocal_rank(self, results_dict: Dict[str, List[Dict]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) across all queries.

        Args:
            results_dict: Dict mapping query_id to search results

        Returns:
            MRR value
        """
        rr_scores = []
        
        for query_id, results in results_dict.items():
            relevance_dict = self.get_relevance_dict(query_id)
            rr = self.reciprocal_rank(results, relevance_dict)
            rr_scores.append(rr)
        
        return np.mean(rr_scores) if rr_scores else 0.0
    
    def dcg_at_k(self, results: List[Dict], relevance_dict: Dict[str, int], k: int = 10) -> float:
        """
        Calculate Discounted Cumulative Gain at k.

        Args:
            results: Search results
            relevance_dict: Ground truth relevance mapping
            k: Cutoff rank

        Returns:
            DCG@k value
        """
        dcg = 0.0
        
        for rank, result in enumerate(results[:k], 1):
            doc_id = result['doc_id']
            relevance = relevance_dict.get(doc_id, 0)
            dcg += relevance / np.log2(rank + 1)
        
        return dcg
    
    def ndcg_at_k(self, results: List[Dict], relevance_dict: Dict[str, int], k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.

        Args:
            results: Search results
            relevance_dict: Ground truth relevance mapping
            k: Cutoff rank

        Returns:
            NDCG@k value
        """
        dcg = self.dcg_at_k(results, relevance_dict, k)
        
        # Ideal DCG: sort by relevance descending
        ideal_relevances = sorted(relevance_dict.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal_relevances, 1))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_query(self, query_id: str, results: List[Dict], k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Evaluate a single query with multiple metrics.

        Args:
            query_id: Query identifier
            results: Search results
            k_values: List of k values for @k metrics

        Returns:
            Dict with all metrics
        """
        relevance_dict = self.get_relevance_dict(query_id)
        
        metrics = {
            'query_id': query_id,
            'ap': self.average_precision(results, relevance_dict),
            'rr': self.reciprocal_rank(results, relevance_dict)
        }
        
        for k in k_values:
            metrics[f'p@{k}'] = self.precision_at_k(results, relevance_dict, k)
            metrics[f'r@{k}'] = self.recall_at_k(results, relevance_dict, k)
            metrics[f'f1@{k}'] = self.f1_at_k(results, relevance_dict, k)
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(results, relevance_dict, k)
        
        return metrics
    
    def evaluate_system(self, search_engine, k_values: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Evaluate a search engine on all ground truth queries.

        Args:
            search_engine: Search engine instance with search() method
            k_values: List of k values for @k metrics

        Returns:
            DataFrame with per-query metrics
        """
        results_list = []
        
        for query_data in self.queries:
            query_id = query_data['query_id']
            query_text = query_data['query_text']
            
            # Perform search
            search_results = search_engine.search(query_text)
            results = search_results.get('results', [])
            
            # Evaluate
            metrics = self.evaluate_query(query_id, results, k_values)
            metrics['query_text'] = query_text
            
            results_list.append(metrics)
        
        df = pd.DataFrame(results_list)
        
        # Add aggregate metrics
        aggregate = {'query_id': 'MEAN', 'query_text': 'Average across all queries'}
        aggregate['ap'] = df['ap'].mean()
        aggregate['rr'] = df['rr'].mean()
        
        for k in k_values:
            aggregate[f'p@{k}'] = df[f'p@{k}'].mean()
            aggregate[f'r@{k}'] = df[f'r@{k}'].mean()
            aggregate[f'f1@{k}'] = df[f'f1@{k}'].mean()
            aggregate[f'ndcg@{k}'] = df[f'ndcg@{k}'].mean()
        
        df = pd.concat([df, pd.DataFrame([aggregate])], ignore_index=True)
        
        return df


if __name__ == '__main__':
    # Test evaluation
    print("=" * 70)
    print("  IR EVALUATION MODULE TEST")
    print("=" * 70)
    
    evaluator = IREvaluator()
    
    print(f"\nGround truth queries: {len(evaluator.queries)}")
    for q in evaluator.queries:
        print(f"  - {q['query_id']}: {q['query_text']}")
