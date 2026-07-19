"""
Rankify - SBERT Classifier
Modern semantic classification using sentence embeddings.
Uses pre-trained SBERT model with centroid-based classification.
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class SBERTClassifier:
    """
    Modern classifier using SBERT embeddings.
    Classification via centroid-based similarity.
    """

    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """Initialize SBERT model."""
        print(f"[SBERT] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Class centroids (average embeddings per class)
        self.centroids = {}
        self.label_map = {'Positif': 1, 'Negatif': 0}
        self.label_map_inv = {1: 'Positif', 0: 'Negatif'}
        
        # Data storage
        self.X_train_embeddings = None
        self.X_test_embeddings = None
        self.y_train = None
        self.y_test = None
        self.texts_train = None
        self.texts_test = None
        
        print(f"[SBERT] Model loaded. Embedding dimension: {self.embedding_dim}")

    def load_data(self, csv_path):
        """Load dataset and create embeddings."""
        print("[SBERT] Loading data...")
        df = pd.read_csv(csv_path, header=1)
        df.columns = ['id', 'text', 'label', 'extra']
        df = df[['id', 'text', 'label']].dropna(subset=['text', 'label'])
        df['label'] = df['label'].str.strip()

        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # Encode labels
        y = np.array([self.label_map[l] for l in labels])
        
        # Train/test split (80/20 stratified)
        (self.texts_train, self.texts_test,
         self.y_train, self.y_test) = train_test_split(
            texts, y,
            test_size=0.2, random_state=42, stratify=y
        )
        
        print("[SBERT] Generating embeddings for training set...")
        self.X_train_embeddings = self.model.encode(
            self.texts_train,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print("[SBERT] Generating embeddings for test set...")
        self.X_test_embeddings = self.model.encode(
            self.texts_test,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        pos_count = sum(1 for l in labels if l == 'Positif')
        neg_count = sum(1 for l in labels if l == 'Negatif')
        print(f"[SBERT] Dataset: {len(texts)} docs ({pos_count} Positif, {neg_count} Negatif)")
        print(f"[SBERT] Train: {len(self.y_train)}, Test: {len(self.y_test)}")
        print(f"[SBERT] Embeddings shape: {self.X_train_embeddings.shape}")

    def train(self):
        """
        Train by computing class centroids.
        Centroid = average embedding of all examples in that class.
        """
        print("\n[SBERT] Training centroid-based classifier...")
        
        for label_name, label_id in self.label_map.items():
            # Get all embeddings for this class
            mask = (self.y_train == label_id)
            class_embeddings = self.X_train_embeddings[mask]
            
            # Compute centroid (mean embedding)
            centroid = np.mean(class_embeddings, axis=0)
            self.centroids[label_id] = centroid
            
            print(f"[SBERT]   {label_name}: {np.sum(mask)} samples, centroid shape {centroid.shape}")
        
        # Evaluate on training set
        train_acc = self._evaluate_embeddings(self.X_train_embeddings, self.y_train)
        print(f"[SBERT] Training accuracy: {train_acc:.4f}")

    def _classify_embedding(self, embedding):
        """
        Classify a single embedding via centroid similarity.
        Returns: (predicted_class, confidence, probabilities)
        """
        similarities = {}
        for label_id, centroid in self.centroids.items():
            # Cosine similarity
            sim = cosine_similarity([embedding], [centroid])[0][0]
            similarities[label_id] = sim
        
        # Convert similarities to probabilities via softmax
        sim_values = np.array(list(similarities.values()))
        exp_sim = np.exp(sim_values - np.max(sim_values))  # numerical stability
        probabilities = exp_sim / np.sum(exp_sim)
        
        # Predicted class = highest similarity
        pred_label_id = max(similarities, key=similarities.get)
        confidence = probabilities[pred_label_id]
        
        proba_dict = {
            self.label_map_inv[label_id]: float(probabilities[i])
            for i, label_id in enumerate(similarities.keys())
        }
        
        return pred_label_id, confidence, proba_dict

    def _evaluate_embeddings(self, embeddings, y_true):
        """Evaluate embeddings against true labels."""
        predictions = []
        for emb in embeddings:
            pred, _, _ = self._classify_embedding(emb)
            predictions.append(pred)
        return accuracy_score(y_true, predictions)

    def predict(self, text):
        """Classify a single text."""
        if not self.centroids:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Encode text
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        
        # Classify
        pred_id, confidence, probabilities = self._classify_embedding(embedding)
        
        return {
            'prediction': self.label_map_inv[pred_id],
            'confidence': float(confidence),
            'probabilities': probabilities
        }

    def predict_batch(self, texts):
        """Classify multiple texts."""
        if not self.centroids:
            raise RuntimeError("Model not trained.")
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        results = []
        
        for i, emb in enumerate(embeddings):
            pred_id, confidence, probabilities = self._classify_embedding(emb)
            results.append({
                'text': texts[i],
                'prediction': self.label_map_inv[pred_id],
                'confidence': float(confidence),
                'probabilities': probabilities
            })
        
        return results

    def evaluate(self):
        """Evaluate on test set and return metrics."""
        if not self.centroids:
            raise RuntimeError("Model not trained.")
        
        print("\n[SBERT] Evaluating on test set...")
        
        # Predict all test samples
        y_pred = []
        y_proba = []
        
        for emb in self.X_test_embeddings:
            pred_id, confidence, proba_dict = self._classify_embedding(emb)
            y_pred.append(pred_id)
            y_proba.append([proba_dict['Negatif'], proba_dict['Positif']])
        
        y_pred = np.array(y_pred)
        y_proba = np.array(y_proba)
        
        metrics = {
            'method': 'SBERT (Centroid)',
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
            'report': classification_report(
                self.y_test, y_pred,
                target_names=['Negatif', 'Positif'],
                zero_division=0
            )
        }
        
        print(f"\n[SBERT] === EVALUATION (Centroid-based) ===")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"\n{metrics['report']}")
        
        return metrics

    def save_model(self, path):
        """Save centroids and label maps."""
        model_data = {
            'centroids': self.centroids,
            'label_map': self.label_map,
            'label_map_inv': self.label_map_inv,
            'embedding_dim': self.embedding_dim
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"[SBERT] Model saved to {path}")

    def load_model(self, path):
        """Load centroids and label maps."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.centroids = model_data['centroids']
        self.label_map = model_data['label_map']
        self.label_map_inv = model_data['label_map_inv']
        self.embedding_dim = model_data['embedding_dim']
        print(f"[SBERT] Model loaded from {path}")


if __name__ == '__main__':
    print("=" * 70)
    print("  SBERT CLASSIFIER TEST")
    print("=" * 70)
    
    clf = SBERTClassifier()
    csv_path = os.path.join(os.path.dirname(__file__), 'data green economy.csv')
    clf.load_data(csv_path)
    clf.train()
    metrics = clf.evaluate()
    
    # Test predictions
    print("\n--- Test Predictions ---")
    test_texts = [
        "Green economy creates new jobs and opportunities",
        "The green economy policy has failed completely",
        "Renewable energy is the future of sustainable development"
    ]
    
    for text in test_texts:
        result = clf.predict(text)
        print(f"  Text: {text[:60]}...")
        print(f"  -> {result['prediction']} ({result['confidence']:.2%})")
        print(f"     Probabilities: {result['probabilities']}")
