"""
Rankify - Mini Search Engine
Flask Web Application
"""

import os
from flask import Flask, render_template, request, jsonify
from search_engine import MiniSearchEngine

app = Flask(__name__)

# Initialize search engine
engine = MiniSearchEngine()

# Load documents
csv_path = os.path.join(os.path.dirname(__file__), 'data green economy.csv')
engine.load_documents(csv_path)


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


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Rankify - Mini Search Engine")
    print("  Buka browser: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
