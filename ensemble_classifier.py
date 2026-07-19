"""
Rankify - Hybrid Classifier (Ensemble)
Combines Traditional (TF-IDF + ML), SBERT, and Reranking (Cross-Encoder) classifiers.
Supports soft voting, hard voting, and weighted combination.
"""

import numpy as np
from traditional_classifier import TraditionalClassifier
from sbert_classifier import SBERTClassifier
from reranking_classifier import RerankingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


class EnsembleClassifier:
    """
    Hybrid classifier combining traditional, SBERT, and reranking methods.
    """

    def __init__(self, traditional_clf=None, sbert_clf=None, reranking_clf=None):
        """
        Initialize ensemble with three classifiers.
        
        Args:
            traditional_clf: TraditionalClassifier instance
            sbert_clf: SBERTClassifier instance
            reranking_clf: RerankingClassifier instance
        """
        self.traditional_clf = traditional_clf or TraditionalClassifier()
        self.sbert_clf = sbert_clf or SBERTClassifier()
        self.reranking_clf = reranking_clf or RerankingClassifier()
        
        self.label_map = {'Positif': 1, 'Negatif': 0}
        self.label_map_inv = {1: 'Positif', 0: 'Negatif'}
        
        # Weights for weighted combination
        self.weights = {
            'traditional': 0.3,
            'sbert': 0.35,
            'reranking': 0.35
        }

    def load_data(self, csv_path):
        """Load data for all classifiers."""
        print("\n[Hybrid] Loading data for all classifiers...")
        self.traditional_clf.load_data(csv_path)
        self.sbert_clf.load_data(csv_path)
        self.reranking_clf.load_data(csv_path)

    def train(self, traditional_method='svm'):
        """Train all classifiers."""
        print("\n[Hybrid] Training Traditional classifier...")
        self.traditional_clf.train(method=traditional_method)
        
        print("\n[Hybrid] Training SBERT classifier...")
        self.sbert_clf.train()
        
        print("\n[Hybrid] Training Reranking classifier...")
        self.reranking_clf.train()

    def predict_hard_voting(self, text):
        """
        Hard voting: majority vote from all three classifiers.
        If tie, use reranking prediction (most accurate).
        """
        trad_result = self.traditional_clf.predict(text)
        sbert_result = self.sbert_clf.predict(text)
        rerank_result = self.reranking_clf.predict(text)
        
        predictions = [
            trad_result['prediction'],
            sbert_result['prediction'],
            rerank_result['prediction']
        ]
        
        # Count votes
        pos_votes = predictions.count('Positif')
        neg_votes = predictions.count('Negatif')
        
        if pos_votes > neg_votes:
            final_pred = 'Positif'
        elif neg_votes > pos_votes:
            final_pred = 'Negatif'
        else:
            # Tie: use reranking (cross-encoder is most accurate)
            final_pred = rerank_result['prediction']
        
        # Average confidence
        confidence = (trad_result['confidence'] + sbert_result['confidence'] + 
                     rerank_result['confidence']) / 3
        
        return {
            'prediction': final_pred,
            'confidence': float(confidence),
            'method': 'hard_voting',
            'individual_predictions': {
                'traditional': trad_result,
                'sbert': sbert_result,
                'reranking': rerank_result
            }
        }

    def predict_soft_voting(self, text):
        """
        Soft voting: average probabilities from all three classifiers.
        """
        trad_result = self.traditional_clf.predict(text)
        sbert_result = self.sbert_clf.predict(text)
        rerank_result = self.reranking_clf.predict(text)
        
        # Average probabilities from all 3
        avg_proba = {
            'Positif': (trad_result['probabilities']['Positif'] + 
                       sbert_result['probabilities']['Positif'] +
                       rerank_result['probabilities']['Positif']) / 3,
            'Negatif': (trad_result['probabilities']['Negatif'] + 
                       sbert_result['probabilities']['Negatif'] +
                       rerank_result['probabilities']['Negatif']) / 3
        }
        
        final_pred = max(avg_proba, key=avg_proba.get)
        confidence = avg_proba[final_pred]
        
        return {
            'prediction': final_pred,
            'confidence': float(confidence),
            'probabilities': avg_proba,
            'method': 'soft_voting',
            'individual_predictions': {
                'traditional': trad_result,
                'sbert': sbert_result,
                'reranking': rerank_result
            }
        }

    def predict_weighted(self, text):
        """
        Weighted combination: weighted average of probabilities.
        """
        w_trad = self.weights['traditional']
        w_sbert = self.weights['sbert']
        w_rerank = self.weights['reranking']
        
        # Normalize weights
        total = w_trad + w_sbert + w_rerank
        w_trad /= total
        w_sbert /= total
        w_rerank /= total
        
        trad_result = self.traditional_clf.predict(text)
        sbert_result = self.sbert_clf.predict(text)
        rerank_result = self.reranking_clf.predict(text)
        
        weighted_proba = {
            'Positif': (w_trad * trad_result['probabilities']['Positif'] + 
                       w_sbert * sbert_result['probabilities']['Positif'] +
                       w_rerank * rerank_result['probabilities']['Positif']),
            'Negatif': (w_trad * trad_result['probabilities']['Negatif'] + 
                       w_sbert * sbert_result['probabilities']['Negatif'] +
                       w_rerank * rerank_result['probabilities']['Negatif'])
        }
        
        final_pred = max(weighted_proba, key=weighted_proba.get)
        confidence = weighted_proba[final_pred]
        
        return {
            'prediction': final_pred,
            'confidence': float(confidence),
            'probabilities': weighted_proba,
            'method': f'weighted (trad={w_trad:.2f}, sbert={w_sbert:.2f}, rerank={w_rerank:.2f})',
            'weights_used': {'traditional': w_trad, 'sbert': w_sbert, 'reranking': w_rerank},
            'individual_predictions': {
                'traditional': trad_result,
                'sbert': sbert_result,
                'reranking': rerank_result
            }
        }

    def predict(self, text, method='soft_voting', **kwargs):
        """
        Unified predict interface.
        
        Args:
            text: Input text to classify
            method: 'hard_voting', 'soft_voting', or 'weighted'
        """
        if method == 'hard_voting':
            return self.predict_hard_voting(text)
        elif method == 'soft_voting':
            return self.predict_soft_voting(text)
        elif method == 'weighted':
            return self.predict_weighted(text)
        else:
            raise ValueError(f"Unknown method: {method}")

    def evaluate(self, method='soft_voting', **kwargs):
        """
        Evaluate ensemble on test set.
        Uses the test set from traditional_clf (same split with random_state=42).
        """
        print(f"\n[Hybrid] Evaluating with method: {method}")
        
        y_true = self.traditional_clf.y_test
        y_pred = []
        
        for text in self.traditional_clf.texts_test:
            result = self.predict(text, method=method, **kwargs)
            pred_label = result['prediction']
            pred_id = self.label_map[pred_label]
            y_pred.append(pred_id)
        
        y_pred = np.array(y_pred)
        
        metrics = {
            'method': f'Hybrid ({method})',
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'report': classification_report(
                y_true, y_pred,
                target_names=['Negatif', 'Positif'],
                zero_division=0
            )
        }
        
        print(f"\n[Hybrid] === EVALUATION ({method}) ===")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"\n{metrics['report']}")
        
        return metrics

    def compare_all_methods(self):
        """Compare all ensemble methods side-by-side."""
        print("\n" + "=" * 70)
        print("  HYBRID METHOD COMPARISON")
        print("=" * 70)
        
        methods = [
            ('hard_voting', {}),
            ('soft_voting', {}),
            ('weighted', {}),
        ]
        
        results = []
        for method, kwargs in methods:
            metrics = self.evaluate(method=method, **kwargs)
            results.append(metrics)
        
        print("\n" + "=" * 70)
        print("  SUMMARY TABLE")
        print("=" * 70)
        print(f"{'Method':<30} {'Accuracy':>10} {'F1-Score':>10}")
        print("-" * 70)
        for metrics in results:
            print(f"{metrics['method']:<30} {metrics['accuracy']:>10.4f} {metrics['f1']:>10.4f}")
        print("=" * 70)
        
        return results


if __name__ == '__main__':
    import os
    
    print("=" * 70)
    print("  HYBRID CLASSIFIER TEST")
    print("=" * 70)
    
    ensemble = EnsembleClassifier()
    csv_path = os.path.join(os.path.dirname(__file__), 'data green economy.csv')
    ensemble.load_data(csv_path)
    ensemble.train(traditional_method='svm')
    
    ensemble.compare_all_methods()
    
    print("\n--- Test Predictions ---")
    test_texts = [
        "Green economy creates new jobs and opportunities",
        "The green economy policy has failed completely"
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        for method in ['hard_voting', 'soft_voting', 'weighted']:
            result = ensemble.predict(text, method=method)
            print(f"  {method:15s}: {result['prediction']} ({result['confidence']:.2%})")
