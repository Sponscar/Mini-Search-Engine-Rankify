"""
Rankify - Semantic Search Engine
Module implementing semantic search using pre-trained transformer models.
Supports IndoBERT, SBERT, and LaBSE for generating document embeddings.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SemanticSearchEngine:
    """
    Semantic Search Engine using sentence transformers.
    Generates embeddings for documents and queries, performs similarity search.
    """

    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='./model_cache'):
        """
        Initialize semantic search engine.

        Args:
            model_name: HuggingFace model identifier
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.embedding_dim = None
        
        # Document storage
        self.documents_raw = []
        self.doc_ids = []
        self.document_embeddings = None
        self.N = 0
        
        # Cache file path
        self.embeddings_cache_path = 'embeddings_cache.pkl'
        
    def load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"[Semantic Engine] Loading model: {self.model_name}")
            print(f"[Semantic Engine] This may take a few minutes on first run...")
            
            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            print(f"[Semantic Engine] Model loaded successfully!")
            print(f"[Semantic Engine] Embedding dimension: {self.embedding_dim}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Please run: pip install sentence-transformers"
            )
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if self.model is None:
            self.load_model()
        
        print(f"[Semantic Engine] Generating embeddings for {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        print(f"[Semantic Engine] Embeddings generated: {embeddings.shape}")
        return embeddings
    
    def load_documents(self, csv_path: str, use_cache: bool = True):
        """
        Load documents from CSV and generate/load embeddings.

        Args:
            csv_path: Path to CSV file with documents
            use_cache: Whether to use cached embeddings if available
        """
        print(f"[Semantic Engine] Loading documents from: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        self.doc_ids = df.iloc[:, 0].dropna().tolist()
        self.documents_raw = df.iloc[:, 1].dropna().tolist()
        self.N = len(self.documents_raw)
        
        print(f"[Semantic Engine] Loaded {self.N} documents")
        
        # Try to load cached embeddings
        if use_cache and os.path.exists(self.embeddings_cache_path):
            try:
                print(f"[Semantic Engine] Loading cached embeddings...")
                with open(self.embeddings_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Verify cache validity
                if (cache_data['model_name'] == self.model_name and
                    len(cache_data['doc_ids']) == self.N and
                    cache_data['doc_ids'] == self.doc_ids):
                    
                    self.document_embeddings = cache_data['embeddings']
                    self.embedding_dim = self.document_embeddings.shape[1]
                    
                    print(f"[Semantic Engine] Loaded cached embeddings: {self.document_embeddings.shape}")
                    print(f"[Semantic Engine] Cache is valid, skipping re-computation!")
                    return
                else:
                    print(f"[Semantic Engine] Cache invalid (different model or documents), regenerating...")
            
            except Exception as e:
                print(f"[Semantic Engine] Error loading cache: {e}, regenerating...")
        
        # Generate embeddings
        self.document_embeddings = self.generate_embeddings(self.documents_raw)
        
        # Cache embeddings
        self._cache_embeddings()
    
    def _cache_embeddings(self):
        """Save embeddings to cache file."""
        try:
            cache_data = {
                'model_name': self.model_name,
                'doc_ids': self.doc_ids,
                'embeddings': self.document_embeddings,
                'embedding_dim': self.embedding_dim,
                'N': self.N
            }
            
            with open(self.embeddings_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            cache_size_mb = os.path.getsize(self.embeddings_cache_path) / (1024 * 1024)
            print(f"[Semantic Engine] Embeddings cached to: {self.embeddings_cache_path}")
            print(f"[Semantic Engine] Cache size: {cache_size_mb:.2f} MB")
        
        except Exception as e:
            print(f"[Semantic Engine] Warning: Could not cache embeddings: {e}")
    
    def semantic_search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Perform semantic search using query embeddings.

        Args:
            query: Search query string
            top_k: Number of top results to return (None = all with score > 0)

        Returns:
            List of dicts with doc_id, score, document, rank
        """
        if self.document_embeddings is None:
            raise RuntimeError("Documents not loaded. Call load_documents() first.")
        
        if self.model is None:
            self.load_model()
        
        # Generate query embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        # Compute cosine similarities (dot product since embeddings are normalized)
        similarities = np.dot(self.document_embeddings, query_embedding)
        
        # Create results
        results = []
        for doc_idx, score in enumerate(similarities):
            if score > 0:  # Only include positive similarity
                results.append({
                    'rank': 0,  # Will be filled after sorting
                    'doc_id': str(self.doc_ids[doc_idx]),
                    'score': float(score),
                    'document': self.documents_raw[doc_idx],
                    'embedding_similarity': float(score)
                })
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Assign ranks
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        # Apply top_k if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def search(self, query_text: str, top_k: Optional[int] = None) -> Dict:
        """
        Search interface compatible with MiniSearchEngine.

        Args:
            query_text: Search query
            top_k: Number of results to return

        Returns:
            Dict with query info and results
        """
        results = self.semantic_search(query_text, top_k)
        
        return {
            'query_original': query_text,
            'query_tokens': query_text.split(),  # No preprocessing for semantic search
            'results': results,
            'total_results': len(results),
            'method': 'semantic',
            'model': self.model_name
        }
    
    def get_similarity_matrix(self) -> np.ndarray:
        """
        Get document-document similarity matrix.

        Returns:
            numpy array of shape (N, N) with pairwise similarities
        """
        if self.document_embeddings is None:
            raise RuntimeError("Documents not loaded.")
        
        return np.dot(self.document_embeddings, self.document_embeddings.T)
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'total_documents': self.N,
            'embeddings_cached': os.path.exists(self.embeddings_cache_path),
            'cache_path': self.embeddings_cache_path
        }


if __name__ == '__main__':
    # Test the semantic engine
    print("=" * 70)
    print("  SEMANTIC SEARCH ENGINE TEST")
    print("=" * 70)
    
    engine = SemanticSearchEngine()
    csv_path = 'data green economy.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        exit(1)
    
    engine.load_documents(csv_path)
    
    # Test queries
    test_queries = [
        "ekonomi hijau",
        "green economy",
        "energi terbarukan",
        "bamboo innovation"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print(f"{'='*70}")
        
        results = engine.search(query)
        
        print(f"Total results: {results['total_results']}")
        print(f"\nTop 5 results:")
        for r in results['results'][:5]:
            print(f"  Rank {r['rank']}: {r['doc_id']} | Score: {r['score']:.6f}")
            print(f"    {r['document'][:80]}...")
    
    print(f"\n{'='*70}")
    print("  STATISTICS")
    print(f"{'='*70}")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
