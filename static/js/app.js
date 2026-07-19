/**
 * Rankify — Search Engine & Sentiment Classifier
 * Frontend JavaScript — Search interaction, classification, rendering, animations
 */

// =========================================================================
// DOM Elements
// =========================================================================
const searchInput = document.getElementById('search-input');
const searchBtn = document.getElementById('search-btn');
const btnText = searchBtn.querySelector('.btn-text');
const btnLoader = searchBtn.querySelector('.btn-loader');
const tabsNav = document.getElementById('tabs-nav');
const resultCountBadge = document.getElementById('result-count');
const methodSelector = document.getElementById('method-selector');
const searchMethodSelect = document.getElementById('search-method');
const compareBtn = document.getElementById('compare-btn');

const panels = {
    results: document.getElementById('panel-results'),
    preprocessing: document.getElementById('panel-preprocessing'),
    'inverted-index': document.getElementById('panel-inverted-index'),
    stats: document.getElementById('panel-stats')
};

// Classification Elements
const modeSwitcher = document.querySelectorAll('.mode-btn');
const searchSection = document.getElementById('search-section');
const classifySection = document.getElementById('classify-section');
const classifyInput = document.getElementById('classify-input');
const classifyBtn = document.getElementById('classify-btn');
const classifyResult = document.getElementById('classify-result');
const methodButtons = document.querySelectorAll('.method-btn');
const showComparisonBtn = document.getElementById('show-comparison-btn');
const evalSection = document.getElementById('eval-section');
const runEvalBtn = document.getElementById('run-evaluation-btn');
const evalResults = document.getElementById('eval-results');

// State
let currentResults = null;
let invertedIndexData = null;
let currentClassificationMethod = 'hybrid';
let lastClassificationResult = null;

// =========================================================================
// Event Listeners
// =========================================================================

// Search on button click
searchBtn.addEventListener('click', performSearch);

// Search on Enter key
searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') performSearch();
});

// Hint chips
document.querySelectorAll('.hint-chip').forEach(chip => {
    chip.addEventListener('click', () => {
        searchInput.value = chip.dataset.query;
        performSearch();
    });
});

// Mode Switcher
modeSwitcher.forEach(btn => {
    btn.addEventListener('click', () => {
        const mode = btn.dataset.mode;
        
        // Update active state
        modeSwitcher.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Toggle sections
        if (mode === 'search') {
            searchSection.style.display = 'block';
            classifySection.style.display = 'none';
            if (evalSection) evalSection.style.display = 'none';
            tabsNav.style.display = currentResults ? 'flex' : 'none';
            // Hide classification panels
            Object.values(panels).forEach(p => p.style.display = 'none');
            if (currentResults) switchTab('results');
        } else if (mode === 'classify') {
            searchSection.style.display = 'none';
            classifySection.style.display = 'block';
            if (evalSection) evalSection.style.display = 'none';
            tabsNav.style.display = 'none';
            Object.values(panels).forEach(p => p.style.display = 'none');
        } else if (mode === 'evaluation') {
            searchSection.style.display = 'none';
            classifySection.style.display = 'none';
            if (evalSection) evalSection.style.display = 'block';
            tabsNav.style.display = 'none';
            Object.values(panels).forEach(p => p.style.display = 'none');
        }
    });
});

// Classification Method Selector
methodButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        methodButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentClassificationMethod = btn.dataset.method;
    });
});

// Classification Button
classifyBtn.addEventListener('click', performClassification);

// Classify on Ctrl+Enter
classifyInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) performClassification();
});

// Show comparison button
if (showComparisonBtn) {
    showComparisonBtn.addEventListener('click', showAllModelComparison);
}

// Evaluation button
if (runEvalBtn) {
    runEvalBtn.addEventListener('click', runEvaluation);
}

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        const tabName = tab.dataset.tab;
        switchTab(tabName);
    });
});

// Compare methods button
if (compareBtn) {
    compareBtn.addEventListener('click', async () => {
        const query = searchInput.value.trim();
        if (!query) return;

        setLoading(true);
        try {
            const response = await fetch('/api/compare-methods', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, top_k: 10 })
            });
            const data = await response.json();
            renderComparison(data);
            switchTab('results');
        } catch (error) {
            console.error('Comparison error:', error);
        } finally {
            setLoading(false);
        }
    });
}


// Inverted index filter
document.getElementById('index-filter').addEventListener('input', (e) => {
    filterInvertedIndex(e.target.value);
});

// =========================================================================
// Search Function
// =========================================================================

async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) {
        searchInput.focus();
        searchInput.classList.add('shake');
        setTimeout(() => searchInput.classList.remove('shake'), 500);
        return;
    }

    // Show loading state
    setLoading(true);

    try {
        // Use TF-IDF search (traditional)
        const endpoint = '/api/search';
        const body = { query };

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        const data = await response.json();
        currentResults = data;

        // Show tabs
        tabsNav.style.display = 'flex';

        // Update result count
        resultCountBadge.textContent = data.total_results;

        // Render results
        renderResults(data);
        renderPreprocessing(data.preprocessing_details);

        // Load inverted index & stats (only once)
        if (!invertedIndexData) {
            loadInvertedIndex();
            loadStats();
        }

        // Switch to results tab
        switchTab('results');

    } catch (error) {
        console.error('Search error:', error);
        renderError('Terjadi kesalahan saat menghubungi server.');
    } finally {
        setLoading(false);
    }
}

// =========================================================================
// Rendering Functions
// =========================================================================

function renderResults(data) {
    const container = document.getElementById('results-container');
    const title = document.getElementById('results-title');
    const subtitle = document.getElementById('results-subtitle');

    title.textContent = `Hasil Pencarian`;
    subtitle.textContent = `Ditemukan ${data.total_results} dokumen relevan untuk "${data.query_original}" · Query tokens: [${data.query_tokens.join(', ')}]`;

    if (data.total_results === 0) {
        container.innerHTML = `
            <div class="no-results">
                <div class="no-results-icon">📭</div>
                <h3>Tidak ada dokumen relevan</h3>
                <p>Coba gunakan kata kunci lain atau periksa ejaan Anda.</p>
            </div>
        `;
        return;
    }

    let html = '';
    data.results.forEach((result, idx) => {
        const rankClass = idx === 0 ? 'top-1' : idx === 1 ? 'top-2' : idx === 2 ? 'top-3' : '';
        const highlightedText = highlightTerms(result.document, data.query_tokens, data.query_original);
        const scorePercent = Math.min(result.score * 100, 100);
        const delay = idx * 0.08;

        // TF-IDF detail table
        let tfidfHtml = '';
        const details = result.tf_idf_details;
        if (details && Object.keys(details).length > 0) {
            tfidfHtml = `
                <button class="tfidf-toggle" onclick="toggleTfidf(this)">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                        <polyline points="6 9 12 15 18 9"></polyline>
                    </svg>
                    Detail TF-IDF
                </button>
                <div class="tfidf-details">
                    <table class="tfidf-table">
                        <thead>
                            <tr>
                                <th>Term</th>
                                <th>TF (Log)</th>
                                <th>IDF</th>
                                <th>TF-IDF</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${Object.entries(details).map(([term, vals]) => `
                                <tr>
                                    <td><strong>${escapeHtml(term)}</strong></td>
                                    <td>${vals.tf.toFixed(4)}</td>
                                    <td>${vals.idf.toFixed(4)}</td>
                                    <td><strong>${vals.tfidf.toFixed(6)}</strong></td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        }

        html += `
            <div class="result-card" style="animation-delay: ${delay}s">
                <div class="result-card-header">
                    <div class="result-rank">
                        <div class="rank-badge ${rankClass}">${result.rank}</div>
                        <span class="doc-id">${escapeHtml(result.doc_id)}</span>
                    </div>
                    <div class="result-score">
                        <div>
                            <span class="score-label">Cosine Similarity</span>
                            <div class="score-value">${result.score.toFixed(6)}</div>
                        </div>
                    </div>
                </div>
                <div class="score-bar-container">
                    <div class="score-bar" style="width: 0%" data-target="${scorePercent}"></div>
                </div>
                <div class="result-text" style="margin-top: 14px">${highlightedText}</div>
                ${tfidfHtml}
            </div>
        `;
    });

    container.innerHTML = html;

    // Animate score bars
    requestAnimationFrame(() => {
        document.querySelectorAll('.score-bar').forEach(bar => {
            bar.style.width = bar.dataset.target + '%';
        });
    });
}

function renderPreprocessing(details) {
    if (!details) return;
    const container = document.getElementById('preprocessing-container');

    const steps = [
        { name: 'Teks Asli (Original)', result: details.original },
        { name: 'Case Folding', result: details.case_folded },
        { name: 'Punctuation Removal', result: details.after_punctuation_removal },
        { name: 'Stop-word Removal', result: formatTokens(details.after_stopword_removal) },
        { name: 'Stemming (Sastrawi)', result: formatTokens(details.after_stemming) }
    ];

    let html = '';
    steps.forEach((step, idx) => {
        const delay = idx * 0.1;
        html += `
            <div class="preprocessing-step" style="animation-delay: ${delay}s">
                <div class="step-header">
                    <div class="step-number">${idx + 1}</div>
                    <div class="step-name">${step.name}</div>
                </div>
                <div class="step-result">${typeof step.result === 'string' ? escapeHtml(step.result) : step.result}</div>
            </div>
        `;
        if (idx < steps.length - 1) {
            html += '<div class="step-arrow">↓</div>';
        }
    });

    container.innerHTML = html;
}

async function loadInvertedIndex() {
    try {
        const response = await fetch('/api/inverted-index');
        const data = await response.json();
        invertedIndexData = data;
        renderInvertedIndex(data.index);
    } catch (error) {
        console.error('Inverted index error:', error);
    }
}

function renderInvertedIndex(indexData, filter = '') {
    const container = document.getElementById('inverted-index-container');
    const entries = Object.entries(indexData);
    const filtered = filter
        ? entries.filter(([term]) => term.includes(filter.toLowerCase()))
        : entries;

    // Limit display to 100 items for performance
    const displayed = filtered.slice(0, 100);

    let html = `<p class="panel-desc">Menampilkan ${displayed.length} dari ${filtered.length} terms (total: ${entries.length})</p>`;
    html += '<div class="index-grid">';

    displayed.forEach(([term, info]) => {
        html += `
            <div class="index-item">
                <div class="index-term">${escapeHtml(term)}</div>
                <div class="index-df">DF: ${info.df} dokumen</div>
                <div class="index-docs">
                    ${info.documents.map(d => `<span class="index-doc-tag">${escapeHtml(d)}</span>`).join('')}
                </div>
            </div>
        `;
    });

    html += '</div>';
    container.innerHTML = html;
}

function filterInvertedIndex(value) {
    if (invertedIndexData) {
        renderInvertedIndex(invertedIndexData.index, value);
    }
}

async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        renderStats(data);
    } catch (error) {
        console.error('Stats error:', error);
    }
}

function renderStats(data) {
    const container = document.getElementById('stats-container');

    let html = `
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">${data.total_documents}</div>
                <div class="stat-label">Total Dokumen</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${data.vocabulary_size}</div>
                <div class="stat-label">Vocabulary Size</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${data.inverted_index_size}</div>
                <div class="stat-label">Index Terms</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${data.avg_doc_length}</div>
                <div class="stat-label">Rata-rata Token/Doc</div>
            </div>
        </div>
        <h3 style="color: var(--text-primary); margin-bottom: 16px; font-size: 16px;">Top 20 Terms (by Document Frequency)</h3>
        <div class="index-grid">
    `;

    if (data.sample_index) {
        Object.entries(data.sample_index).forEach(([term, info]) => {
            html += `
                <div class="index-item">
                    <div class="index-term">${escapeHtml(term)}</div>
                    <div class="index-df">DF: ${info.df} dokumen</div>
                    <div class="index-docs">
                        ${info.doc_ids.map(d => `<span class="index-doc-tag">${escapeHtml(d)}</span>`).join('')}
                    </div>
                </div>
            `;
        });
    }

    html += '</div>';
    container.innerHTML = html;
}

// =========================================================================
// UI Helper Functions
// =========================================================================

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === tabName);
    });

    // Show/hide panels
    Object.entries(panels).forEach(([name, panel]) => {
        panel.style.display = name === tabName ? 'block' : 'none';
    });
}

function setLoading(loading) {
    btnText.style.display = loading ? 'none' : 'inline';
    btnLoader.style.display = loading ? 'inline-flex' : 'none';
    searchBtn.disabled = loading;
}

function toggleTfidf(button) {
    const details = button.nextElementSibling;
    const isOpen = details.classList.contains('open');
    details.classList.toggle('open');

    // Rotate arrow
    const svg = button.querySelector('svg');
    svg.style.transform = isOpen ? 'rotate(0deg)' : 'rotate(180deg)';
    svg.style.transition = 'transform 0.3s ease';
}

function highlightTerms(text, queryTokens, originalQuery) {
    let html = escapeHtml(text);

    // Highlight each query token in the text
    const wordsToHighlight = [
        ...queryTokens,
        ...originalQuery.toLowerCase().split(/\s+/)
    ];

    const uniqueWords = [...new Set(wordsToHighlight)].filter(w => w.length > 1);

    uniqueWords.forEach(word => {
        const regex = new RegExp(`(${escapeRegex(word)})`, 'gi');
        html = html.replace(regex, '<span class="highlight">$1</span>');
    });

    return html;
}

function formatTokens(tokens) {
    if (!tokens || !Array.isArray(tokens)) return '';
    return tokens.map(t => `<span style="
        display: inline-block;
        background: rgba(124, 58, 237, 0.15);
        color: #a78bfa;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 2px;
        font-size: 12px;
    ">${escapeHtml(t)}</span>`).join(' ');
}

function escapeHtml(str) {
    if (typeof str !== 'string') return String(str);
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function renderError(message) {
    const container = document.getElementById('results-container');
    container.innerHTML = `
        <div class="no-results">
            <div class="no-results-icon">⚠️</div>
            <h3>Error</h3>
            <p>${escapeHtml(message)}</p>
        </div>
    `;
    tabsNav.style.display = 'flex';
    switchTab('results');
}

function renderComparison(data) {
    const container = document.getElementById('results-container');
    const methods = ['traditional', 'semantic', 'reranking'];
    
    let html = `<div class="comparison-header">
        <h2>Perbandingan Metode</h2>
        <p>Query: "${escapeHtml(data.query)}"</p>
    </div><div class="comparison-grid">`;
    
    methods.forEach(method => {
        const results = data[method] || [];
        const label = method.replace('_', ' ').toUpperCase();
        html += `<div class="comparison-column"><h3>${label}</h3><div class="comparison-results">`;
        results.slice(0, 5).forEach(r => {
            html += `<div class="result-card mini">
                <span class="result-rank">#${r.rank}</span>
                <span class="result-id">${escapeHtml(r.doc_id)}</span>
                <span class="result-score">${r.score.toFixed(4)}</span>
                <p>${escapeHtml(r.document.substring(0, 60))}...</p>
            </div>`;
        });
        html += `</div></div>`;
    });
    html += `</div>`;
    container.innerHTML = html;
    tabsNav.style.display = 'flex';
    resultCountBadge.textContent = 'Compare';
}

// =========================================================================
// CLASSIFICATION FUNCTIONS
// =========================================================================

async function performClassification() {
    const text = classifyInput.value.trim();
    
    if (!text) {
        alert('Mohon masukkan teks untuk diklasifikasi.');
        return;
    }
    
    // Show loading
    classifyBtn.disabled = true;
    classifyBtn.querySelector('.btn-text').style.display = 'none';
    classifyBtn.querySelector('.btn-loader').style.display = 'inline-block';
    
    try {
        const response = await fetch('/api/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                method: currentClassificationMethod
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            lastClassificationResult = data;
            displayClassificationResult(data);
        } else {
            alert('Error: ' + (data.error || 'Classification failed'));
        }
    } catch (error) {
        console.error('Classification error:', error);
        alert('Terjadi kesalahan saat klasifikasi.');
    } finally {
        // Hide loading
        classifyBtn.disabled = false;
        classifyBtn.querySelector('.btn-text').style.display = 'inline';
        classifyBtn.querySelector('.btn-loader').style.display = 'none';
    }
}

function displayClassificationResult(data) {
    classifyResult.style.display = 'block';
    
    // Display relevant documents
    const relevantDocsContainer = document.getElementById('relevant-docs');
    if (data.relevant_docs && data.relevant_docs.length > 0) {
        const posCount = data.relevant_docs.filter(d => d.prediction === 'Positif').length;
        const negCount = data.relevant_docs.filter(d => d.prediction === 'Negatif').length;
        const methodName = currentClassificationMethod === 'hybrid' ? 'Hybrid (SBERT + Reranking + Ensemble)' : 'Traditional (TF-IDF + SVM)';
        let docsHtml = `
            <div class="docs-header">
                <h4>📄 Klasifikasi Seluruh Dataset</h4>
                <span class="docs-method">${methodName}</span>
            </div>
            <div class="docs-summary">
                <div class="summary-stat">
                    <span class="summary-count positive">${posCount}</span>
                    <span class="summary-label">Positif</span>
                </div>
                <div class="summary-stat">
                    <span class="summary-count negative">${negCount}</span>
                    <span class="summary-label">Negatif</span>
                </div>
                <div class="summary-stat">
                    <span class="summary-count total">${data.relevant_docs.length}</span>
                    <span class="summary-label">Total</span>
                </div>
            </div>
            <p class="eval-subtitle" style="margin-bottom:1rem;">
                Diurutkan berdasarkan relevansi dengan query "<strong>${classifyInput.value.trim()}</strong>"
            </p>
            <div class="relevant-docs-list">`;
        
        data.relevant_docs.forEach((doc, i) => {
            const conf = (doc.confidence * 100).toFixed(1);
            const sim = doc.similarity > 0 ? (doc.similarity * 100).toFixed(1) + '%' : '-';
            const isMatch = doc.prediction === doc.true_label;
            docsHtml += `
                <div class="relevant-doc-item">
                    <div class="doc-rank">#${i + 1}</div>
                    <div class="doc-content">
                        <p class="doc-text">${doc.text}</p>
                        <div class="doc-meta">
                            <span class="label-badge label-${doc.true_label.toLowerCase()}">Asli: ${doc.true_label}</span>
                            <span class="label-badge label-${doc.prediction.toLowerCase()} ${!isMatch ? 'label-wrong' : ''}">Prediksi: ${doc.prediction}</span>
                            <span class="doc-conf">${conf}%</span>
                            <span class="doc-sim">Relevansi: ${sim}</span>
                            ${isMatch ? '<span style="color:var(--accent-1)">✓</span>' : '<span style="color:#dc2626">✗</span>'}
                        </div>
                    </div>
                </div>`;
        });
        
        docsHtml += '</div>';
        relevantDocsContainer.innerHTML = docsHtml;
        relevantDocsContainer.style.display = 'block';
    } else {
        relevantDocsContainer.style.display = 'none';
    }
    
    // Scroll to result
    classifyResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function showAllModelComparison() {
    const text = classifyInput.value.trim();
    
    if (!text) return;
    
    showComparisonBtn.disabled = true;
    showComparisonBtn.textContent = 'Loading...';
    
    try {
        const response = await fetch('/api/compare-classifiers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayModelComparison(data.predictions);
            showComparisonBtn.textContent = 'Hide Comparison';
            showComparisonBtn.onclick = hideModelComparison;
        } else {
            alert('Error: ' + (data.error || 'Comparison failed'));
        }
    } catch (error) {
        console.error('Comparison error:', error);
        alert('Terjadi kesalahan.');
    } finally {
        showComparisonBtn.disabled = false;
    }
}

function displayModelComparison(predictions) {
    const container = document.getElementById('comparison-results');
    
    let html = '<h4>All Model Predictions:</h4><div class="model-comparison-grid">';
    
    const models = [
        { key: 'traditional', name: 'Traditional (TF-IDF + SVM)' },
        { key: 'sbert', name: 'SBERT (Centroid Similarity)' },
        { key: 'reranking', name: 'Reranking (Cross-Encoder)' },
        { key: 'hybrid_soft', name: 'Hybrid (Soft Voting)' },
        { key: 'hybrid_hard', name: 'Hybrid (Hard Voting)' },
        { key: 'hybrid_weighted', name: 'Hybrid (Weighted)' }
    ];
    
    models.forEach(model => {
        const pred = predictions[model.key];
        if (!pred) return;
        
        const isPositive = pred.prediction === 'Positif';
        const confidence = (pred.confidence * 100).toFixed(1);
        
        html += `
            <div class="model-card">
                <h5>${model.name}</h5>
                <div class="model-prediction ${isPositive ? 'positive' : 'negative'}">
                    ${pred.prediction}
                </div>
                <div class="model-confidence">${confidence}%</div>
                ${pred.probabilities ? `
                    <div class="model-probs">
                        <div>Positif: ${(pred.probabilities.Positif * 100).toFixed(1)}%</div>
                        <div>Negatif: ${(pred.probabilities.Negatif * 100).toFixed(1)}%</div>
                    </div>
                ` : ''}
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
    container.style.display = 'block';
}

function hideModelComparison() {
    const container = document.getElementById('comparison-results');
    container.style.display = 'none';
    showComparisonBtn.textContent = 'Show All Model Predictions';
    showComparisonBtn.onclick = showAllModelComparison;
}

// =========================================================================
// EVALUATION FUNCTIONS
// =========================================================================

async function runEvaluation() {
    runEvalBtn.disabled = true;
    runEvalBtn.querySelector('.btn-text').style.display = 'none';
    runEvalBtn.querySelector('.btn-loader').style.display = 'inline-block';

    try {
        const response = await fetch('/api/evaluation');
        const data = await response.json();

        if (response.ok) {
            renderEvaluation(data);
        } else {
            alert('Error: ' + (data.error || 'Evaluation failed'));
        }
    } catch (error) {
        console.error('Evaluation error:', error);
        alert('Terjadi kesalahan saat evaluasi.');
    } finally {
        runEvalBtn.disabled = false;
        runEvalBtn.querySelector('.btn-text').style.display = 'inline';
        runEvalBtn.querySelector('.btn-loader').style.display = 'none';
    }
}

function renderEvaluation(data) {
    evalResults.style.display = 'block';
    const ir = data.ir;
    const clf = data.classification;
    const ds = data.dataset;

    let html = '';

    // Dataset Info
    html += `<div class="eval-card">
        <h4>📁 Dataset Info</h4>
        <p>Total dokumen: <strong>${ds.total_docs}</strong> | Train: <strong>${ds.train_size}</strong> | Test: <strong>${ds.test_size}</strong></p>
    </div>`;

    // IR Metrics
    html += `<div class="eval-card">
        <h4>🔍 Search Engine — TF-IDF (Tradisional)</h4>
        <p class="eval-subtitle">Evaluasi dengan ${ir.num_queries} ground-truth queries</p>
        <div class="metrics-grid">
            ${metricBar('MAP', ir.metrics.map)}
            ${metricBar('MRR', ir.metrics.mrr)}
            ${metricBar('NDCG@5', ir.metrics.ndcg_5)}
            ${metricBar('NDCG@10', ir.metrics.ndcg_10)}
        </div>
        <details class="eval-details">
            <summary>Detail per Query</summary>
            <table class="eval-table">
                <tr><th>Query</th><th>Teks</th><th>AP</th><th>RR</th><th>NDCG@10</th></tr>
                ${ir.per_query.map(q => `<tr>
                    <td>${q.query_id}</td>
                    <td>${q.query_text}</td>
                    <td>${q.ap.toFixed(4)}</td>
                    <td>${q.rr.toFixed(4)}</td>
                    <td>${q.ndcg_10.toFixed(4)}</td>
                </tr>`).join('')}
            </table>
        </details>
    </div>`;

    // Classification Metrics
    html += `<div class="eval-card">
        <h4>🧠 Classification — Sentiment Analysis</h4>
        <p class="eval-subtitle">Evaluasi pada test set (${ds.test_size} dokumen)</p>
        <table class="eval-table">
            <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr>
            ${classRow(clf.traditional)}
            ${classRow(clf.sbert)}
            ${classRow(clf.reranking)}
            ${classRow(clf.hybrid)}
        </table>
    </div>`;

    // Visual Comparison
    html += `<div class="eval-card">
        <h4>📊 Perbandingan Visual</h4>
        <div class="visual-comparison">
            <div class="comparison-col">
                <h5>Search Engine (IR)</h5>
                ${metricBar('MAP', ir.metrics.map, 'var(--accent-2)')}
                ${metricBar('MRR', ir.metrics.mrr, 'var(--accent-2)')}
                ${metricBar('NDCG@10', ir.metrics.ndcg_10, 'var(--accent-2)')}
            </div>
            <div class="comparison-col">
                <h5>Classification (F1-Score)</h5>
                ${metricBar('Traditional', clf.traditional.f1, '#f59e0b')}
                ${metricBar('SBERT', clf.sbert.f1, 'var(--accent-3)')}
                ${metricBar('Reranking', clf.reranking.f1, 'var(--accent-2)')}
                ${metricBar('Hybrid', clf.hybrid.f1, 'var(--accent-1)')}
            </div>
        </div>
    </div>`;

    // Per-document predictions
    if (clf.per_document && clf.per_document.length > 0) {
        html += `<div class="eval-card">
            <h4>📝 Detail Prediksi per Dokumen (Test Set)</h4>
            <p class="eval-subtitle">Perbandingan prediksi setiap model pada ${clf.per_document.length} dokumen test</p>
            <div style="overflow-x:auto;">
            <table class="eval-table">
                <tr>
                    <th>No</th>
                    <th>Teks</th>
                    <th>Label Asli</th>
                    <th>Traditional</th>
                    <th>SBERT</th>
                    <th>Reranking</th>
                    <th>Hybrid</th>
                </tr>
                ${clf.per_document.map((doc, i) => `<tr>
                    <td>${i + 1}</td>
                    <td style="max-width:250px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${doc.text}">${doc.text}</td>
                    <td><span class="label-badge label-${doc.true_label.toLowerCase()}">${doc.true_label}</span></td>
                    <td><span class="label-badge label-${doc.traditional.toLowerCase()} ${doc.traditional !== doc.true_label ? 'label-wrong' : ''}">${doc.traditional}</span></td>
                    <td><span class="label-badge label-${doc.sbert.toLowerCase()} ${doc.sbert !== doc.true_label ? 'label-wrong' : ''}">${doc.sbert}</span></td>
                    <td><span class="label-badge label-${doc.reranking.toLowerCase()} ${doc.reranking !== doc.true_label ? 'label-wrong' : ''}">${doc.reranking}</span></td>
                    <td><span class="label-badge label-${doc.hybrid.toLowerCase()} ${doc.hybrid !== doc.true_label ? 'label-wrong' : ''}">${doc.hybrid}</span></td>
                </tr>`).join('')}
            </table>
            </div>
        </div>`;
    }

    evalResults.innerHTML = html;
    evalResults.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function metricBar(label, value, color) {
    const pct = (value * 100).toFixed(1);
    const barColor = color || (value >= 0.7 ? 'var(--success)' : value >= 0.4 ? 'var(--warning)' : 'var(--danger)');
    return `<div class="metric-item">
        <span class="metric-label">${label}</span>
        <div class="metric-bar-bg">
            <div class="metric-bar-fill" style="width:${pct}%;background:${barColor}"></div>
        </div>
        <span class="metric-value">${pct}%</span>
    </div>`;
}

function classRow(m) {
    return `<tr>
        <td>${m.method}</td>
        <td>${(m.accuracy * 100).toFixed(1)}%</td>
        <td>${(m.precision * 100).toFixed(1)}%</td>
        <td>${(m.recall * 100).toFixed(1)}%</td>
        <td><strong>${(m.f1 * 100).toFixed(1)}%</strong></td>
    </tr>`;
}
