"""
Rankify - Traditional Classifier
TF-IDF + ML Classifiers (SVM, Naive Bayes) for sentiment classification.
Baseline method for comparison with modern SBERT approach.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pickle


class TraditionalClassifier:
    """
    Traditional ML classifier using TF-IDF features.
    Supports SVM, Naive Bayes, and Logistic Regression.
    """

    def __init__(self):
        # Preprocessing tools
        stemmer_factory = StemmerFactory()
        self.stemmer = stemmer_factory.create_stemmer()

        stopword_factory = StopWordRemoverFactory()
        self.stop_words = set(stopword_factory.get_stop_words())
        self.stop_words.update({
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'and', 'or', 'but', 'not', 'it', 'its', 'this', 'that',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom'
        })

        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )

        # Classifiers
        self.classifiers = {
            'svm': SVC(kernel='linear', probability=True, random_state=42),
            'nb': MultinomialNB(alpha=1.0),
            'lr': LogisticRegression(max_iter=1000, random_state=42)
        }

        self.trained_model = None
        self.model_name = None
        self.label_map = {'Positif': 1, 'Negatif': 0}
        self.label_map_inv = {1: 'Positif', 0: 'Negatif'}

        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.docs_train = None
        self.docs_test = None

    def preprocess(self, text):
        """Preprocess text: lowercase, remove punctuation, stem."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        tokens = [self.stemmer.stem(t) for t in tokens]
        tokens = [t for t in tokens if len(t) > 1]
        return ' '.join(tokens)

    def load_data(self, csv_path):
        """Load dataset from CSV and prepare train/test split."""
        print("[Traditional] Loading data...")
        df = pd.read_csv(csv_path, header=1)
        df.columns = ['id', 'text', 'label', 'extra']
        df = df[['id', 'text', 'label']].dropna(subset=['text', 'label'])
        df['label'] = df['label'].str.strip()

        texts = df['text'].tolist()
        labels = df['label'].tolist()
        doc_ids = df['id'].tolist()

        # Preprocess texts
        processed_texts = [self.preprocess(str(t)) for t in texts]

        # Encode labels
        y = np.array([self.label_map[l] for l in labels])

        # Train/test split (80/20 stratified)
        (self.docs_train, self.docs_test,
         self.X_train_raw, self.X_test_raw,
         self.y_train, self.y_test) = train_test_split(
            processed_texts, processed_texts, y,
            test_size=0.2, random_state=42, stratify=y
        )

        # Store raw texts for display
        (_, _, self.texts_train, self.texts_test, _, _) = train_test_split(
            texts, texts, y,
            test_size=0.2, random_state=42, stratify=y
        )

        # TF-IDF features
        self.X_train = self.vectorizer.fit_transform(self.X_train_raw)
        self.X_test = self.vectorizer.transform(self.X_test_raw)

        pos_count = sum(1 for l in labels if l == 'Positif')
        neg_count = sum(1 for l in labels if l == 'Negatif')
        print(f"[Traditional] Dataset: {len(texts)} docs ({pos_count} Positif, {neg_count} Negatif)")
        print(f"[Traditional] Train: {len(self.y_train)}, Test: {len(self.y_test)}")
        print(f"[Traditional] TF-IDF features: {self.X_train.shape[1]}")

    def train(self, method='svm'):
        """Train classifier."""
        if method not in self.classifiers:
            raise ValueError(f"Unknown method: {method}. Use: {list(self.classifiers.keys())}")

        self.model_name = method
        self.trained_model = self.classifiers[method]

        print(f"\n[Traditional] Training {method.upper()} classifier...")
        self.trained_model.fit(self.X_train, self.y_train)

        # Training accuracy
        train_acc = self.trained_model.score(self.X_train, self.y_train)
        print(f"[Traditional] Training accuracy: {train_acc:.4f}")

    def predict(self, text):
        """Classify a single text."""
        if self.trained_model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        processed = self.preprocess(text)
        features = self.vectorizer.transform([processed])
        pred = self.trained_model.predict(features)[0]
        proba = self.trained_model.predict_proba(features)[0]

        return {
            'prediction': self.label_map_inv[pred],
            'confidence': float(max(proba)),
            'probabilities': {
                'Positif': float(proba[1]),
                'Negatif': float(proba[0])
            }
        }

    def evaluate(self):
        """Evaluate on test set and return metrics."""
        if self.trained_model is None:
            raise RuntimeError("Model not trained.")

        y_pred = self.trained_model.predict(self.X_test)
        y_proba = self.trained_model.predict_proba(self.X_test)

        metrics = {
            'method': f'Traditional ({self.model_name.upper()})',
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

        print(f"\n[Traditional] === EVALUATION ({self.model_name.upper()}) ===")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"\n{metrics['report']}")

        return metrics

    def cross_validate(self, method='svm', cv=5):
        """Cross-validation for more reliable evaluation."""
        print(f"\n[Traditional] Cross-validation ({cv}-fold, {method.upper()})...")
        clf = self.classifiers[method]

        # Use all data for cross-validation
        all_X = self.vectorizer.transform(self.X_train_raw + self.X_test_raw)
        all_y = np.concatenate([self.y_train, self.y_test])

        scores = cross_val_score(clf, all_X, all_y, cv=cv, scoring='f1')
        print(f"  F1 scores: {[f'{s:.4f}' for s in scores]}")
        print(f"  Mean F1:   {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return {'mean_f1': scores.mean(), 'std_f1': scores.std(), 'scores': scores.tolist()}


if __name__ == '__main__':
    print("=" * 70)
    print("  TRADITIONAL CLASSIFIER TEST")
    print("=" * 70)

    clf = TraditionalClassifier()
    csv_path = os.path.join(os.path.dirname(__file__), 'data green economy.csv')
    clf.load_data(csv_path)

    for method in ['svm', 'nb', 'lr']:
        clf.train(method)
        clf.evaluate()

    # Test prediction
    print("\n--- Test Predictions ---")
    test_texts = [
        "Green economy creates new jobs and opportunities",
        "The green economy policy has failed completely"
    ]
    clf.train('svm')
    for text in test_texts:
        result = clf.predict(text)
        print(f"  Text: {text[:60]}...")
        print(f"  -> {result['prediction']} ({result['confidence']:.2%})")
