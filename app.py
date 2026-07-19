"""
Rankify - Search Engine & Sentiment Classifier
Flask Web Application
Supports both IR (search) and text classification functionality.
"""

import os
from flask import Flask, render_template, request, jsonify
from search_engine import MiniSearchEngine
from traditional_classifier import TraditionalClassifier
from sbert_classifier import SBERTClassifier
from reranking_classifier import RerankingClassifier
from ensemble_classifier import EnsembleClassifier

app = Flask(__name__)

# Initialize search engine (TF-IDF only)
engine = MiniSearchEngine()

# Initialize classifiers
traditional_clf = TraditionalClassifier()
sbert_clf = SBERTClassifier()
reranking_clf = RerankingClassifier(sbert_model=sbert_clf.model)
ensemble_clf = EnsembleClassifier(traditional_clf, sbert_clf, reranking_clf)

# Load documents
csv_path = os.path.join(os.path.dirname(__file__), 'data green economy.csv')
engine.load_documents(csv_path)

# Load and train classifiers
print("[App] Loading classifiers...")
traditional_clf.load_data(csv_path)
sbert_clf.load_data(csv_path)
reranking_clf.load_data(csv_path)
ensemble_clf.load_data(csv_path)

print("[App] Training classifiers...")
traditional_clf.train(method='svm')
sbert_clf.train()
reranking_clf.train()
ensemble_clf.train(traditional_method='svm')
print("[App] Classifiers ready!")


@app.route('/')
def index():
    """Serve the main search page."""
    return render_template('index.html')


@app.route('/api/search', methods=['POST'])
def api_search():
    """
    Search API endpoint.
    Accepts JSON: {"query": "search terms"}
    Returns ranked search results with TF-IDF details.
    """
    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({
            'error': 'Query tidak boleh kosong.',
            'results': [],
            'total_results': 0
        }), 400

    results = engine.search(query)
    return jsonify(results)


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Return search engine statistics."""
    stats = engine.get_stats()
    return jsonify(stats)


@app.route('/api/inverted-index', methods=['GET'])
def api_inverted_index():
    """Return the full inverted index for display."""
    index_data = {}
    for term, doc_indices in sorted(engine.inverted_index.items()):
        index_data[term] = {
            'df': len(doc_indices),
            'documents': [str(engine.doc_ids[i]) for i in sorted(doc_indices)]
        }
    return jsonify({
        'total_terms': len(index_data),
        'index': index_data
    })


@app.route('/api/preprocessing-demo', methods=['POST'])
def api_preprocessing_demo():
    """
    Show step-by-step preprocessing of input text.
    Useful for demonstrating the preprocessing pipeline.
    """
    import re

    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Teks tidak boleh kosong.'}), 400

    # Step-by-step
    step1 = text.lower()
    step2 = re.sub(r'[^a-z0-9\s]', '', step1)
    step3 = step2.split()
    step4 = [t for t in step3 if t not in engine.stop_words and len(t) > 1]
    step5 = [engine.stemmer.stem(t) for t in step4]
    step5 = [t for t in step5 if len(t) > 1]

    return jsonify({
        'original': text,
        'steps': [
            {'name': 'Case Folding', 'result': step1},
            {'name': 'Punctuation Removal', 'result': step2},
            {'name': 'Tokenization', 'result': step3},
            {'name': 'Stop-word Removal', 'result': step4},
            {'name': 'Stemming (Sastrawi)', 'result': step5}
        ]
    })


@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    """Return information about loaded models."""
    return jsonify({
        'tfidf_stats': engine.get_stats()
    })


# ============================================================================
# CLASSIFICATION API ENDPOINTS
# ============================================================================

@app.route('/api/classify', methods=['POST'])
def api_classify():
    """
    Classify text sentiment.
    Accepts JSON: {"text": "...", "method": "traditional|sbert|ensemble"}
    Returns: {"prediction": "Positif/Negatif", "confidence": 0.XX, "probabilities": {...}}
    """
    data = request.get_json()
    text = data.get('text', '').strip()
    method = data.get('method', 'ensemble').lower()
    
    if not text:
        return jsonify({'error': 'Text tidak boleh kosong.'}), 400
    
    if method not in ['traditional', 'sbert', 'reranking', 'hybrid', 'ensemble']:
        return jsonify({'error': 'Method harus: traditional, sbert, reranking, atau hybrid'}), 400
    
    try:
        if method == 'traditional':
            result = traditional_clf.predict(text)
        elif method == 'sbert':
            result = sbert_clf.predict(text)
        elif method == 'reranking':
            result = reranking_clf.predict(text)
        elif method in ['hybrid', 'ensemble']:
            ensemble_method = data.get('ensemble_method', 'soft_voting')
            result = ensemble_clf.predict(text, method=ensemble_method)
        
        result['method_used'] = method
        
        # Classify ALL documents in dataset, sorted by search relevance
        import pandas as pd
        df = pd.read_csv(csv_path, header=1)
        df.columns = ['id', 'text', 'label', 'extra']
        df = df[['id', 'text', 'label']].dropna(subset=['text', 'label'])
        
        # Get search scores for ranking
        search_data = engine.search(text)
        score_map = {}
        for doc in search_data.get('results', []):
            score_map[doc.get('doc_id', '')] = doc.get('score', 0)
        
        relevant_docs = []
        for _, row in df.iterrows():
            doc_text = str(row['text']).strip()
            doc_id = str(row['id']).strip()
            true_label = str(row['label']).strip()
            if not doc_text:
                continue
            
            # Classify with selected method
            if method == 'traditional':
                doc_result = traditional_clf.predict(doc_text)
            elif method == 'sbert':
                doc_result = sbert_clf.predict(doc_text)
            elif method == 'reranking':
                doc_result = reranking_clf.predict(doc_text)
            elif method in ['hybrid', 'ensemble']:
                doc_result = ensemble_clf.predict(doc_text, method='soft_voting')
            
            relevant_docs.append({
                'doc_id': doc_id,
                'text': doc_text[:200],
                'true_label': true_label,
                'similarity': score_map.get(doc_id, 0),
                'prediction': doc_result['prediction'],
                'confidence': float(doc_result['confidence']),
                'probabilities': doc_result.get('probabilities', {})
            })
        
        # Sort by similarity descending (most relevant first)
        relevant_docs.sort(key=lambda x: x['similarity'], reverse=True)
        
        result['relevant_docs'] = relevant_docs
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Classification error: {str(e)}'}), 500


@app.route('/api/classify-batch', methods=['POST'])
def api_classify_batch():
    """
    Classify multiple texts at once.
    Accepts JSON: {"texts": [...], "method": "traditional|sbert|ensemble"}
    """
    data = request.get_json()
    texts = data.get('texts', [])
    method = data.get('method', 'ensemble').lower()
    
    if not texts or not isinstance(texts, list):
        return jsonify({'error': 'texts harus berupa array non-kosong'}), 400
    
    if method not in ['traditional', 'sbert', 'reranking', 'hybrid', 'ensemble']:
        return jsonify({'error': 'Method harus: traditional, sbert, reranking, atau hybrid'}), 400
    
    try:
        results = []
        for text in texts:
            if method == 'traditional':
                result = traditional_clf.predict(text)
            elif method == 'sbert':
                result = sbert_clf.predict(text)
            elif method == 'reranking':
                result = reranking_clf.predict(text)
            elif method in ['hybrid', 'ensemble']:
                result = ensemble_clf.predict(text, method='soft_voting')
            
            result['text'] = text[:100]  # Truncate for response
            results.append(result)
        
        return jsonify({
            'method': method,
            'total': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': f'Batch classification error: {str(e)}'}), 500


@app.route('/api/classification-stats', methods=['GET'])
def api_classification_stats():
    """
    Return classification model statistics and performance.
    """
    try:
        # Evaluate all models on test set
        trad_metrics = traditional_clf.evaluate()
        sbert_metrics = sbert_clf.evaluate()
        ensemble_metrics = ensemble_clf.evaluate(method='soft_voting')
        
        return jsonify({
            'traditional': {
                'method': trad_metrics['method'],
                'accuracy': trad_metrics['accuracy'],
                'precision': trad_metrics['precision'],
                'recall': trad_metrics['recall'],
                'f1': trad_metrics['f1'],
                'confusion_matrix': trad_metrics['confusion_matrix']
            },
            'sbert': {
                'method': sbert_metrics['method'],
                'accuracy': sbert_metrics['accuracy'],
                'precision': sbert_metrics['precision'],
                'recall': sbert_metrics['recall'],
                'f1': sbert_metrics['f1'],
                'confusion_matrix': sbert_metrics['confusion_matrix']
            },
            'ensemble': {
                'method': ensemble_metrics['method'],
                'accuracy': ensemble_metrics['accuracy'],
                'precision': ensemble_metrics['precision'],
                'recall': ensemble_metrics['recall'],
                'f1': ensemble_metrics['f1'],
                'confusion_matrix': ensemble_metrics['confusion_matrix']
            },
            'dataset': {
                'train_size': len(traditional_clf.y_train),
                'test_size': len(traditional_clf.y_test)
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Stats error: {str(e)}'}), 500


@app.route('/api/compare-classifiers', methods=['POST'])
def api_compare_classifiers():
    """
    Compare all classification methods on a single text.
    Accepts JSON: {"text": "..."}
    """
    data = request.get_json()
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'Text tidak boleh kosong.'}), 400
    
    try:
        comparison = {
            'text': text,
            'predictions': {
                'traditional': traditional_clf.predict(text),
                'sbert': sbert_clf.predict(text),
                'reranking': reranking_clf.predict(text),
                'hybrid_soft': ensemble_clf.predict(text, method='soft_voting'),
                'hybrid_hard': ensemble_clf.predict(text, method='hard_voting'),
                'hybrid_weighted': ensemble_clf.predict(text, method='weighted')
            }
        }
        
        return jsonify(comparison)
    
    except Exception as e:
        return jsonify({'error': f'Comparison error: {str(e)}'}), 500


@app.route('/api/evaluation', methods=['GET'])
def api_evaluation():
    """Run full evaluation: IR metrics for search + classification metrics."""
    try:
        from evaluation import IREvaluator

        # --- IR Evaluation (TF-IDF Search) ---
        evaluator = IREvaluator()
        df = evaluator.evaluate_system(engine, k_values=[5, 10])
        mean_row = df[df['query_id'] == 'MEAN'].iloc[0]

        ir_metrics = {
            'map': float(mean_row['ap']),
            'mrr': float(mean_row['rr']),
            'ndcg_5': float(mean_row['ndcg@5']),
            'ndcg_10': float(mean_row['ndcg@10']),
            'p_5': float(mean_row['p@5']),
            'p_10': float(mean_row['p@10']),
        }

        # Per-query details
        per_query = []
        for _, row in df[df['query_id'] != 'MEAN'].iterrows():
            per_query.append({
                'query_id': row['query_id'],
                'query_text': row['query_text'],
                'ap': float(row['ap']),
                'rr': float(row['rr']),
                'ndcg_5': float(row['ndcg@5']),
                'ndcg_10': float(row['ndcg@10']),
            })

        # --- Classification Evaluation ---
        trad_metrics = traditional_clf.evaluate()
        sbert_metrics = sbert_clf.evaluate()
        rerank_metrics = reranking_clf.evaluate()
        hybrid_metrics = ensemble_clf.evaluate(method='soft_voting')

        def clean_metrics(m):
            return {
                'accuracy': m['accuracy'],
                'precision': m['precision'],
                'recall': m['recall'],
                'f1': m['f1'],
                'confusion_matrix': m['confusion_matrix']
            }

        # Per-document predictions on test set
        label_inv = {1: 'Positif', 0: 'Negatif'}
        per_doc = []
        for i, text in enumerate(traditional_clf.texts_test):
            true_label = label_inv[int(traditional_clf.y_test[i])]
            trad_pred = traditional_clf.predict(text)
            sbert_pred = sbert_clf.predict(text)
            rerank_pred = reranking_clf.predict(text)
            hybrid_pred = ensemble_clf.predict(text, method='soft_voting')
            per_doc.append({
                'text': text[:120],
                'true_label': true_label,
                'traditional': trad_pred['prediction'],
                'sbert': sbert_pred['prediction'],
                'reranking': rerank_pred['prediction'],
                'hybrid': hybrid_pred['prediction'],
            })

        return jsonify({
            'ir': {
                'method': 'TF-IDF Search',
                'metrics': ir_metrics,
                'per_query': per_query,
                'num_queries': len(per_query)
            },
            'classification': {
                'traditional': {**clean_metrics(trad_metrics), 'method': trad_metrics['method']},
                'sbert': {**clean_metrics(sbert_metrics), 'method': sbert_metrics['method']},
                'reranking': {**clean_metrics(rerank_metrics), 'method': rerank_metrics['method']},
                'hybrid': {**clean_metrics(hybrid_metrics), 'method': hybrid_metrics['method']},
                'per_document': per_doc,
            },
            'dataset': {
                'total_docs': 50,
                'train_size': len(traditional_clf.y_train),
                'test_size': len(traditional_clf.y_test)
            }
        })

    except Exception as e:
        return jsonify({'error': f'Evaluation error: {str(e)}'}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Rankify - Search Engine & Sentiment Classifier")
    print("  Buka browser: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
