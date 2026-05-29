"""
Rankify - Mini Search Engine
Core search engine module implementing Information Retrieval concepts:
- Pre-processing (case folding, punctuation removal, stop-word removal, stemming)
- Inverted Index
- TF-IDF with Log Frequency Weighting
- Vector Space Model with Cosine Similarity
"""

import math
import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


class MiniSearchEngine:
    """Mini Search Engine with TF-IDF and Cosine Similarity."""

    def __init__(self):
        # Initialize Sastrawi stemmer and stopword remover
        stemmer_factory = StemmerFactory()
        self.stemmer = stemmer_factory.create_stemmer()

        stopword_factory = StopWordRemoverFactory()
        self.stop_words = set(stopword_factory.get_stop_words())

        # Add common English stop words since dataset is mixed
        english_stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'just', 'because', 'but', 'and', 'or', 'if', 'while', 'about',
            'against', 'up', 'down', 'it', 'its', 'this', 'that', 'these', 'those',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'am'
        }
        self.stop_words.update(english_stopwords)

        # Storage
        self.documents_raw = []       # Original documents
        self.documents_clean = []     # Pre-processed documents (list of token lists)
        self.doc_ids = []             # Document IDs (D1, D2, ...)
        self.inverted_index = {}      # {term: set(doc_indices)}
        self.vocabulary = set()       # All unique terms
        self.N = 0                    # Total number of documents

    # =========================================================================
    # 1. PRE-PROCESSING
    # =========================================================================

    def preprocess(self, text):
        """
        Full pre-processing pipeline:
        1. Case Folding — convert to lowercase
        2. Punctuation Removal — remove all non-alphanumeric characters
        3. Tokenization — split into words
        4. Stop-word Removal — remove common/unimportant words
        5. Stemming — reduce words to base form (Sastrawi for Indonesian)

        Args:
            text (str): Raw text input

        Returns:
            list: List of cleaned, stemmed tokens
        """
        # Step 1: Case Folding
        text = text.lower()

        # Step 2: Punctuation Removal (keep only letters, numbers, spaces)
        text = re.sub(r'[^a-z0-9\s]', '', text)

        # Step 3: Tokenization
        tokens = text.split()

        # Step 4: Stop-word Removal
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]

        # Step 5: Stemming (Sastrawi)
        tokens = [self.stemmer.stem(t) for t in tokens]

        # Remove empty tokens after stemming
        tokens = [t for t in tokens if len(t) > 1]

        return tokens

    # =========================================================================
    # 2. LOAD DATA & BUILD INDEX
    # =========================================================================

    def load_documents(self, csv_path):
        """
        Load documents from CSV file and build the index.

        Args:
            csv_path (str): Path to CSV file with documents in column index 1
        """
        df = pd.read_csv(csv_path)

        self.doc_ids = df.iloc[:, 0].dropna().tolist()
        self.documents_raw = df.iloc[:, 1].dropna().tolist()
        self.N = len(self.documents_raw)

        # Pre-process all documents
        self.documents_clean = []
        for doc in self.documents_raw:
            tokens = self.preprocess(str(doc))
            self.documents_clean.append(tokens)

        # Build inverted index
        self.build_inverted_index()

        print(f"[Rankify] Loaded {self.N} documents.")
        print(f"[Rankify] Vocabulary size: {len(self.vocabulary)} terms.")
        print(f"[Rankify] Inverted index built successfully.")

    def build_inverted_index(self):
        """
        Build inverted index: {term: set(doc_indices)}
        Also builds the vocabulary set.
        """
        self.inverted_index = {}
        self.vocabulary = set()

        for doc_idx, tokens in enumerate(self.documents_clean):
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.vocabulary.add(token)
                if token not in self.inverted_index:
                    self.inverted_index[token] = set()
                self.inverted_index[token].add(doc_idx)

    # =========================================================================
    # 3. TF-IDF COMPUTATION
    # =========================================================================

    def compute_tf(self, term, tokens):
        """
        Compute Term Frequency with Log Frequency Weighting.
        TF = 1 + log10(raw_tf) if raw_tf > 0, else 0

        Args:
            term (str): The term to compute TF for
            tokens (list): List of tokens (document or query)

        Returns:
            float: Log-weighted TF value
        """
        raw_tf = tokens.count(term)
        if raw_tf > 0:
            return 1 + math.log10(raw_tf)
        return 0.0

    def compute_idf(self, term):
        """
        Compute Inverse Document Frequency.
        IDF = log10(N / df)

        Args:
            term (str): The term to compute IDF for

        Returns:
            float: IDF value
        """
        df = len(self.inverted_index.get(term, set()))
        if df == 0:
            return 0.0
        return math.log10(self.N / df)

    def compute_tfidf_vector(self, tokens):
        """
        Compute the TF-IDF vector for a list of tokens.

        Args:
            tokens (list): List of tokens

        Returns:
            dict: {term: tfidf_weight} for all terms with non-zero weight
        """
        vector = {}
        unique_terms = set(tokens)
        for term in unique_terms:
            tf = self.compute_tf(term, tokens)
            idf = self.compute_idf(term)
            weight = tf * idf
            if weight > 0:
                vector[term] = weight
        return vector

    # =========================================================================
    # 4. COSINE SIMILARITY
    # =========================================================================

    def cosine_similarity(self, vec_q, vec_d):
        """
        Compute Cosine Similarity between two TF-IDF vectors.

        cosine_sim = Σ(wQ × wD) / (|Q| × |D|)

        where |Q| and |D| are L2 norms (Euclidean lengths) of vectors.

        Args:
            vec_q (dict): Query TF-IDF vector
            vec_d (dict): Document TF-IDF vector

        Returns:
            float: Cosine similarity score (0 to 1)
        """
        # Dot product: only shared terms contribute
        dot_product = 0.0
        for term in vec_q:
            if term in vec_d:
                dot_product += vec_q[term] * vec_d[term]

        # L2 norms
        norm_q = math.sqrt(sum(w ** 2 for w in vec_q.values()))
        norm_d = math.sqrt(sum(w ** 2 for w in vec_d.values()))

        if norm_q == 0 or norm_d == 0:
            return 0.0

        return dot_product / (norm_q * norm_d)

    # =========================================================================
    # 5. SEARCH (RANKED RETRIEVAL)
    # =========================================================================

    def search(self, query_text):
        """
        Search for documents matching the query using Cosine Similarity.

        Steps:
        1. Pre-process the query
        2. Compute TF-IDF vector for the query
        3. For each document, compute cosine similarity with the query
        4. Rank documents by similarity score (descending)

        Args:
            query_text (str): User's search query

        Returns:
            dict: {
                'query_original': str,
                'query_tokens': list,
                'results': list of dicts with doc info and scores,
                'total_results': int,
                'preprocessing_details': dict
            }
        """
        # Pre-process query
        query_tokens = self.preprocess(query_text)

        if not query_tokens:
            return {
                'query_original': query_text,
                'query_tokens': [],
                'results': [],
                'total_results': 0,
                'preprocessing_details': {
                    'original': query_text,
                    'after_preprocessing': [],
                    'message': 'Query kosong setelah pre-processing.'
                }
            }

        # Compute query TF-IDF vector
        vec_query = self.compute_tfidf_vector(query_tokens)

        # Compute similarity with each document
        results = []
        for doc_idx in range(self.N):
            vec_doc = self.compute_tfidf_vector(self.documents_clean[doc_idx])
            score = self.cosine_similarity(vec_query, vec_doc)

            if score > 0:
                results.append({
                    'rank': 0,  # Will be filled after sorting
                    'doc_id': str(self.doc_ids[doc_idx]),
                    'score': round(score, 6),
                    'document': self.documents_raw[doc_idx],
                    'tokens_clean': self.documents_clean[doc_idx],
                    'tf_idf_details': {
                        term: {
                            'tf': self.compute_tf(term, self.documents_clean[doc_idx]),
                            'idf': self.compute_idf(term),
                            'tfidf': round(self.compute_tf(term, self.documents_clean[doc_idx]) * self.compute_idf(term), 6)
                        }
                        for term in query_tokens if term in set(self.documents_clean[doc_idx])
                    }
                })

        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)

        # Assign ranks
        for i, result in enumerate(results):
            result['rank'] = i + 1

        return {
            'query_original': query_text,
            'query_tokens': query_tokens,
            'results': results,
            'total_results': len(results),
            'preprocessing_details': {
                'original': query_text,
                'case_folded': query_text.lower(),
                'after_punctuation_removal': re.sub(r'[^a-z0-9\s]', '', query_text.lower()),
                'after_stopword_removal': [t for t in query_text.lower().split() if t not in self.stop_words and len(t) > 1],
                'after_stemming': query_tokens
            }
        }

    # =========================================================================
    # 6. ENGINE STATISTICS
    # =========================================================================

    def get_stats(self):
        """Get search engine statistics for display."""
        return {
            'total_documents': self.N,
            'vocabulary_size': len(self.vocabulary),
            'inverted_index_size': len(self.inverted_index),
            'avg_doc_length': round(
                sum(len(doc) for doc in self.documents_clean) / max(self.N, 1), 1
            ),
            'sample_index': {
                term: {
                    'df': len(docs),
                    'doc_ids': [str(self.doc_ids[i]) for i in sorted(docs)[:10]]
                }
                for term, docs in sorted(
                    self.inverted_index.items(),
                    key=lambda x: len(x[1]),
                    reverse=True
                )[:20]
            }
        }
