"""
Rankify - Reranking Classifier
Classification using Retrieve & Rerank approach with Cross-Encoder.
Retrieves similar documents via SBERT, then re-ranks using Cross-Encoder
for more accurate sentiment classification.
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.metrics.pairwise import cosine_similarity


class RerankingClassifier:
    """
    Classifier using Retrieve & Rerank approach.
    1. Retrieve: Find similar documents using SBERT bi-encoder
    2. Rerank: Re-score using Cross-Encoder for better accuracy
    3. Vote: Weighted KNN voting from top-K reranked documents
    """

    def __init__(self, sbert_model_name='paraphrase-multilingual-MiniLM-L12-v2',
                 cross_encoder_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
                 top_k_retrieve=15, top_k_rerank=7,
                 sbert_model=None):
        """
        Initialize models.
        
        Args:
            sbert_model_name: SBERT bi-encoder for initial retrieval
            cross_encoder_name: Cross-encoder for reranking
            top_k_retrieve: Number of documents to retrieve initially
            top_k_rerank: Number of top documents after reranking for voting
            sbert_model: Existing SentenceTransformer model (to share with SBERTClassifier)
        """
        if sbert_model:
            print(f"[Reranking] Using shared SBERT model")
            self.sbert_model = sbert_model
        else:
            print(f"[Reranking] Loading SBERT model: {sbert_model_name}")
            self.sbert_model = SentenceTransformer(sbert_model_name)
        
        print(f"[Reranking] Loading Cross-Encoder: {cross_encoder_name}")
        self.cross_encoder = CrossEncoder(cross_encoder_name)
        
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        
        self.label_map = {'Positif': 1, 'Negatif': 0}
        self.label_map_inv = {1: 'Positif', 0: 'Negatif'}
        
        # Data storage
        self.train_texts = None
        self.train_labels = None
        self.train_embeddings = None
        self.test_texts = None
        self.test_labels = None
        
        print("[Reranking] Models loaded successfully.")

    def load_data(self, csv_path):
        """Load dataset and create embeddings for training set."""
        print("[Reranking] Loading data...")
        df = pd.read_csv(csv_path, header=1)
        df.columns = ['id', 'text', 'label', 'extra']
        df = df[['id', 'text', 'label']].dropna(subset=['text', 'label'])
        df['label'] = df['label'].str.strip()

        texts = df['text'].tolist()
        labels = df['label'].tolist()
        y = np.array([self.label_map[l] for l in labels])

        # Train/test split (same seed as other classifiers for fair comparison)
        (self.train_texts, self.test_texts,
         self.train_labels, self.test_labels) = train_test_split(
            texts, y,
            test_size=0.2, random_state=42, stratify=y
        )

        pos_count = sum(1 for l in labels if l == 'Positif')
        neg_count = sum(1 for l in labels if l == 'Negatif')
        print(f"[Reranking] Dataset: {len(texts)} docs ({pos_count} Positif, {neg_count} Negatif)")
        print(f"[Reranking] Train: {len(self.train_labels)}, Test: {len(self.test_labels)}")

    def train(self):
        """
        Train by computing SBERT embeddings for all training documents.
        Cross-encoder doesn't need separate training (pre-trained).
        """
        print("\n[Reranking] Computing SBERT embeddings for training set...")
        self.train_embeddings = self.sbert_model.encode(
            self.train_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"[Reranking] Embeddings shape: {self.train_embeddings.shape}")
        
        # Evaluate on training set
        train_acc = self._evaluate_on_set(self.train_texts, self.train_labels)
        print(f"[Reranking] Training accuracy: {train_acc:.4f}")

    def predict(self, text):
        """
        Classify text using Retrieve & Rerank.
        
        Steps:
        1. Encode input text with SBERT
        2. Retrieve top-K similar documents via cosine similarity
        3. Re-rank using Cross-Encoder
        4. Weighted voting from top-K reranked documents
        
        Returns: dict with prediction, confidence, probabilities
        """
        # Step 1: Encode input
        query_embedding = self.sbert_model.encode([text], convert_to_numpy=True)
        
        # Step 2: Retrieve top-K via cosine similarity
        similarities = cosine_similarity(query_embedding, self.train_embeddings)[0]
        top_k_indices = np.argsort(similarities)[::-1][:self.top_k_retrieve]
        
        # Step 3: Re-rank using Cross-Encoder
        pairs = [(text, self.train_texts[i]) for i in top_k_indices]
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Sort by cross-encoder score (descending)
        reranked_order = np.argsort(cross_scores)[::-1]
        top_reranked = reranked_order[:self.top_k_rerank]
        
        # Step 4: Weighted voting
        pos_score = 0.0
        neg_score = 0.0
        total_weight = 0.0
        
        for rank, idx in enumerate(top_reranked):
            original_idx = top_k_indices[idx]
            label = self.train_labels[original_idx]
            # Weight by cross-encoder score (higher = more relevant)
            weight = max(cross_scores[idx], 0.01)  # Prevent zero/negative weights
            
            if label == 1:  # Positif
                pos_score += weight
            else:  # Negatif
                neg_score += weight
            total_weight += weight
        
        # Convert to probabilities
        if total_weight > 0:
            pos_prob = pos_score / total_weight
            neg_prob = neg_score / total_weight
        else:
            pos_prob = 0.5
            neg_prob = 0.5
        
        # Prediction
        if pos_prob >= neg_prob:
            prediction = 'Positif'
            confidence = pos_prob
        else:
            prediction = 'Negatif'
            confidence = neg_prob
        
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'method': 'reranking',
            'probabilities': {
                'Positif': float(pos_prob),
                'Negatif': float(neg_prob)
            },
            'details': {
                'retrieved': int(self.top_k_retrieve),
                'reranked_top': int(self.top_k_rerank)
            }
        }

    def _evaluate_on_set(self, texts, labels):
        """Evaluate on a set of texts."""
        correct = 0
        for text, true_label in zip(texts, labels):
            result = self.predict(text)
            pred_label = self.label_map[result['prediction']]
            if pred_label == true_label:
                correct += 1
        return correct / len(labels)

    def evaluate(self):
        """
        Evaluate on test set.
        Returns: dict with metrics
        """
        print("\n[Reranking] Evaluating on test set...")
        y_true = []
        y_pred = []
        
        for text, true_label in zip(self.test_texts, self.test_labels):
            result = self.predict(text)
            pred_label = self.label_map[result['prediction']]
            y_true.append(true_label)
            y_pred.append(pred_label)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
        report = classification_report(y_true, y_pred,
                                        target_names=['Negatif', 'Positif'],
                                        output_dict=True,
                                        zero_division=0)
        
        print(f"[Reranking] Test Accuracy: {accuracy:.4f}")
        print(f"[Reranking] Test F1-Score: {f1:.4f}")
        
        return {
            'method': 'Reranking (Cross-Encoder)',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm,
            'classification_report': report
        }
