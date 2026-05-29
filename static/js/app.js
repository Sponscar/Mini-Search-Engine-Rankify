/**
 * Rankify — Mini Search Engine
 * Frontend JavaScript — Search interaction, rendering, animations
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

const panels = {
    results: document.getElementById('panel-results'),
    preprocessing: document.getElementById('panel-preprocessing'),
    'inverted-index': document.getElementById('panel-inverted-index'),
    stats: document.getElementById('panel-stats')
};

// State
let currentResults = null;
let invertedIndexData = null;

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

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        const tabName = tab.dataset.tab;
        switchTab(tabName);
    });
});

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
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
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
