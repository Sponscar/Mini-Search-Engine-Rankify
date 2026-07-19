"""
Rankify - Hybrid Search Engine
Combines traditional TF-IDF and semantic search with multiple fusion strategies.
"""

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict


class HybridSearchEngine:
    """
    Hybrid search engine combining TF-IDF and semantic approaches.
    Supports multiple fusion strategies: linear combination, RRF, and reranking.
    """

    def __init__(self, tfidf_engine, semantic_engine):
        """
        Initialize hybrid engine with both traditional and semantic engines.

        Args:
            tfidf_engine: Instance of MiniSearchEngine (TF-IDF)
            semantic_engine: Instance of SemanticSearchEngine
        """
        self.tfidf_engine = tfidf_engine
        self.semantic_engine = semantic_engine
        
        print("[Hybrid Engine] Initialized with TF-IDF and Semantic engines")
    
    def search_traditional(self, query: str) -> Dict:
        """Search using traditional TF-IDF method."""
        results = self.tfidf_engine.search(query)
        results['method'] = 'traditional'
        return results
    
    def search_semantic(self, query: str) -> Dict:
        """Search using semantic embeddings method."""
        results = self.semantic_engine.search(query)
        results['method'] = 'semantic'
        return results
    
    def search_hybrid(self, query: str, method: str = 'rrf', alpha: float = 0.5, top_k: Optional[int] = None) -> Dict:
        """
        Perform hybrid search using specified fusion method.

        Args:
            query: Search query string
            method: Fusion method ('linear', 'rrf', 'rerank')
            alpha: Weight for linear combination (0=semantic only, 1=tfidf only)
            top_k: Number of results to return

        Returns:
            Dict with hybrid search results
        """
        # Get results from both methods
        tfidf_results = self.tfidf_engine.search(query)['results']
        semantic_results = self.semantic_engine.search(query)['results']
        
        # Apply fusion strategy
        if method == 'linear':
            fused_results = self._linear_fusion(tfidf_results, semantic_results, alpha)
        elif method == 'rrf':
            fused_results = self._reciprocal_rank_fusion([tfidf_results, semantic_results])
        elif method == 'rerank':
            fused_results = self._semantic_reranking(tfidf_results, semantic_results)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        
        # Apply top_k if specified
        if top_k is not None:
            fused_results = fused_results[:top_k]
        
        return {
            'query_original': query,
            'query_tokens': self.tfidf_engine.preprocess(query),
            'results': fused_results,
            'total_results': len(fused_results),
            'method': f'hybrid_{method}',
            'fusion_params': {'method': method, 'alpha': alpha}
        }
    
    def _linear_fusion(self, tfidf_results: List[Dict], semantic_results: List[Dict], alpha: float) -> List[Dict]:
        """
        Linear combination: score = alpha * tfidf_score + (1-alpha) * semantic_score

        Args:
            tfidf_results: TF-IDF search results
            semantic_results: Semantic search results
            alpha: Weight for TF-IDF (0 to 1)

        Returns:
            Fused results sorted by combined score
        """
        # Normalize scores to [0, 1]
        tfidf_normalized = self._normalize_scores(tfidf_results)
        semantic_normalized = self._normalize_scores(semantic_results)
        
        # Create score dictionaries
        tfidf_scores = {r['doc_id']: r['score_normalized'] for r in tfidf_normalized}
        semantic_scores = {r['doc_id']: r['score_normalized'] for r in semantic_normalized}
        
        # Get all unique doc_ids
        all_docs = set(tfidf_scores.keys()) | set(semantic_scores.keys())
        
        # Compute combined scores
        combined = []
        for doc_id in all_docs:
            tfidf_score = tfidf_scores.get(doc_id, 0.0)
            semantic_score = semantic_scores.get(doc_id, 0.0)
            
            combined_score = alpha * tfidf_score + (1 - alpha) * semantic_score
            
            # Get document text (prefer from tfidf results)
            doc_text = None
            for r in tfidf_results:
                if r['doc_id'] == doc_id:
                    doc_text = r['document']
                    break
            if doc_text is None:
                for r in semantic_results:
                    if r['doc_id'] == doc_id:
                        doc_text = r['document']
                        break
            
            combined.append({
                'rank': 0,  # Will be filled after sorting
                'doc_id': doc_id,
                'score': round(combined_score, 6),
                'tfidf_score': round(tfidf_score, 6),
                'semantic_score': round(semantic_score, 6),
                'document': doc_text
            })
        
        # Sort by combined score
        combined.sort(key=lambda x: x['score'], reverse=True)
        
        # Assign ranks
        for i, result in enumerate(combined):
            result['rank'] = i + 1
        
        return combined
    
    def _reciprocal_rank_fusion(self, results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF): score = sum(1 / (k + rank))

        Args:
            results_list: List of result lists from different methods
            k: Constant for RRF (default: 60)

        Returns:
            Fused results sorted by RRF score
        """
        rrf_scores = defaultdict(float)
        doc_texts = {}
        
        for results in results_list:
            for result in results:
                doc_id = result['doc_id']
                rank = result['rank']
                
                rrf_scores[doc_id] += 1.0 / (k + rank)
                
                if doc_id not in doc_texts:
                    doc_texts[doc_id] = result['document']
        
        # Create fused results
        fused = []
        for doc_id, score in rrf_scores.items():
            fused.append({
                'rank': 0,  # Will be filled after sorting
                'doc_id': doc_id,
                'score': round(score, 6),
                'document': doc_texts[doc_id]
            })
        
        # Sort by RRF score
        fused.sort(key=lambda x: x['score'], reverse=True)
        
        # Assign ranks
        for i, result in enumerate(fused):
            result['rank'] = i + 1
        
        return fused
    
    def _semantic_reranking(self, tfidf_results: List[Dict], semantic_results: List[Dict], top_k: int = 20) -> List[Dict]:
        """
        Semantic reranking: Use TF-IDF for initial retrieval, semantic for reranking top-k.

        Args:
            tfidf_results: TF-IDF search results
            semantic_results: Semantic search results
            top_k: Number of TF-IDF results to rerank

        Returns:
            Reranked results
        """
        # Get top-k from TF-IDF
        top_tfidf = tfidf_results[:top_k]
        top_doc_ids = {r['doc_id'] for r in top_tfidf}
        
        # Get semantic scores for these docs
        semantic_scores = {r['doc_id']: r['score'] for r in semantic_results}
        
        # Rerank by semantic scores
        reranked = []
        for result in top_tfidf:
            doc_id = result['doc_id']
            semantic_score = semantic_scores.get(doc_id, 0.0)
            
            reranked.append({
                'rank': 0,  # Will be filled after sorting
                'doc_id': doc_id,
                'score': round(semantic_score, 6),
                'tfidf_score': round(result['score'], 6),
                'semantic_score': round(semantic_score, 6),
                'document': result['document']
            })
        
        # Sort by semantic score
        reranked.sort(key=lambda x: x['score'], reverse=True)
        
        # Assign ranks
        for i, result in enumerate(reranked):
            result['rank'] = i + 1
        
        return reranked
    
    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """
        Min-max normalize scores to [0, 1] range.

        Args:
            results: Search results with 'score' field

        Returns:
            Results with added 'score_normalized' field
        """
        if not results:
            return results
        
        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            for r in results:
                r['score_normalized'] = 1.0
        else:
            for r in results:
                r['score_normalized'] = (r['score'] - min_score) / (max_score - min_score)
        
        return results
    
    def compare_methods(self, query: str, top_k: int = 10) -> Dict:
        """
        Compare all search methods side-by-side for a query.

        Args:
            query: Search query
            top_k: Number of results per method

        Returns:
            Dict with results from all methods
        """
        results = {
            'query': query,
            'traditional': self.search_traditional(query)['results'][:top_k],
            'semantic': self.search_semantic(query)['results'][:top_k],
            'reranking': self.search_hybrid(query, method='rerank')['results'][:top_k]
        }
        
        return results
    
    def search(self, query: str, method: str = 'hybrid_rrf', **kwargs) -> Dict:
        """
        Unified search interface.

        Args:
            query: Search query
            method: Search method ('traditional', 'semantic', or 'hybrid_*')
            **kwargs: Additional parameters for hybrid search

        Returns:
            Search results
        """
        if method == 'traditional':
            return self.search_traditional(query)
        elif method == 'semantic':
            return self.search_semantic(query)
        elif method.startswith('hybrid'):
            fusion_method = method.replace('hybrid_', '')
            return self.search_hybrid(query, method=fusion_method, **kwargs)
        else:
            raise ValueError(f"Unknown search method: {method}")


if __name__ == '__main__':
    import os
    from search_engine import MiniSearchEngine
    from semantic_engine import SemanticSearchEngine
    
    print("=" * 70)
    print("  HYBRID SEARCH ENGINE TEST")
    print("=" * 70)
    
    csv_path = 'data green economy.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        exit(1)
    
    # Initialize engines
    print("\n[1] Initializing TF-IDF engine...")
    tfidf_engine = MiniSearchEngine()
    tfidf_engine.load_documents(csv_path)
    
    print("\n[2] Initializing Semantic engine...")
    semantic_engine = SemanticSearchEngine()
    semantic_engine.load_documents(csv_path)
    
    print("\n[3] Creating Hybrid engine...")
    hybrid_engine = HybridSearchEngine(tfidf_engine, semantic_engine)
    
    # Test query
    query = "ekonomi hijau"
    
    print(f"\n{'='*70}")
    print(f"Query: '{query}'")
    print(f"{'='*70}")
    
    # Compare methods
    comparison = hybrid_engine.compare_methods(query, top_k=5)
    
    for method_name, results in comparison.items():
        if method_name == 'query':
            continue
        
        print(f"\n--- {method_name.upper()} ---")
        for r in results:
            print(f"  Rank {r['rank']}: {r['doc_id']} | Score: {r['score']:.6f}")
