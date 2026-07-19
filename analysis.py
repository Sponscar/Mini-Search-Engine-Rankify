"""
Rankify - Analisis & Laporan UTS
Script untuk menghasilkan data analisis:
a. Analisis Bobot IDF
b. Analisis Efek Normalisasi (Cosine vs Non-Cosine)
c. Evaluasi Sistem (Precision, Recall, F-Measure)
"""

import math
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
from search_engine import MiniSearchEngine

# Initialize engine
engine = MiniSearchEngine()
csv_path = os.path.join(os.path.dirname(__file__), 'data green economy.csv')
engine.load_documents(csv_path)

print("=" * 70)
print("  BAGIAN A: ANALISIS BOBOT IDF")
print("=" * 70)

# Pilih 2 kata kunci dari topik "green economy"
keyword1 = "ekonomi"  # kata umum (sering muncul)
keyword2 = "bambu"    # kata jarang (jarang muncul)

# Preprocess keywords to get stemmed versions
kw1_stemmed = engine.preprocess(keyword1)
kw2_stemmed = engine.preprocess(keyword2)
print(f"\nKata kunci 1: '{keyword1}' -> setelah stemming: {kw1_stemmed}")
print(f"Kata kunci 2: '{keyword2}' -> setelah stemming: {kw2_stemmed}")

# Get the stemmed terms
term1 = kw1_stemmed[0] if kw1_stemmed else keyword1
term2 = kw2_stemmed[0] if kw2_stemmed else keyword2

# Count df for each term
df1 = len(engine.inverted_index.get(term1, set()))
df2 = len(engine.inverted_index.get(term2, set()))
N = engine.N

print(f"\n--- Perhitungan IDF untuk '{term1}' ---")
print(f"  N (total dokumen) = {N}")
print(f"  df (dokumen mengandung '{term1}') = {df1}")
if df1 > 0:
    idf1 = math.log10(N / df1)
    print(f"  IDF = log10(N / df)")
    print(f"      = log10({N} / {df1})")
    print(f"      = log10({N/df1:.4f})")
    print(f"      = {idf1:.6f}")
else:
    idf1 = 0
    print(f"  IDF = 0 (term tidak ditemukan dalam corpus)")

print(f"\n--- Perhitungan IDF untuk '{term2}' ---")
print(f"  N (total dokumen) = {N}")
print(f"  df (dokumen mengandung '{term2}') = {df2}")
if df2 > 0:
    idf2 = math.log10(N / df2)
    print(f"  IDF = log10(N / df)")
    print(f"      = log10({N} / {df2})")
    print(f"      = log10({N/df2:.4f})")
    print(f"      = {idf2:.6f}")
else:
    idf2 = 0
    print(f"  IDF = 0 (term tidak ditemukan dalam corpus)")

print(f"\n--- Perbandingan ---")
print(f"  IDF('{term1}') = {idf1:.6f}  (muncul di {df1} dokumen)")
print(f"  IDF('{term2}') = {idf2:.6f}  (muncul di {df2} dokumen)")

# Show which documents contain each term
if term1 in engine.inverted_index:
    docs1 = sorted(engine.inverted_index[term1])
    print(f"\n  Dokumen mengandung '{term1}': {[str(engine.doc_ids[i]) for i in docs1]}")
if term2 in engine.inverted_index:
    docs2 = sorted(engine.inverted_index[term2])
    print(f"  Dokumen mengandung '{term2}': {[str(engine.doc_ids[i]) for i in docs2]}")

# Also show a few more terms for richer analysis
print(f"\n--- Tabel IDF Beberapa Term ---")
print(f"  {'Term':<20} {'df':<6} {'N/df':<12} {'IDF':>10}")
print(f"  {'-'*20} {'-'*6} {'-'*12} {'-'*10}")

sample_terms = []
# Get some terms with varying df
for term, docs in sorted(engine.inverted_index.items(), key=lambda x: len(x[1])):
    df = len(docs)
    if df not in [d[1] for d in sample_terms]:
        sample_terms.append((term, df))
    if len(sample_terms) >= 8:
        break

# Add our keywords
for t, d in [(term1, df1), (term2, df2)]:
    if (t, d) not in sample_terms:
        sample_terms.append((t, d))

sample_terms.sort(key=lambda x: x[1])

for term, df in sample_terms:
    if df > 0:
        idf_val = math.log10(N / df)
        print(f"  {term:<20} {df:<6} {N/df:<12.4f} {idf_val:>10.6f}")


print("\n\n" + "=" * 70)
print("  BAGIAN B: ANALISIS EFEK NORMALISASI")
print("=" * 70)

query_text = "ekonomi hijau"
query_tokens = engine.preprocess(query_text)
print(f"\nQuery: '{query_text}' -> tokens: {query_tokens}")

# Compute TF-IDF vectors for query
vec_query = engine.compute_tfidf_vector(query_tokens)
print(f"Vektor query: {vec_query}")

print(f"\n--- Perbandingan: Cosine Similarity vs Dot Product ---")
print(f"{'Rank':<6} {'Doc ID':<8} {'Cosine Sim':>12} {'Dot Product':>13} {'|D| (Norm)':>12} {'Pjg Dok':>10}")
print(f"{'-'*6} {'-'*8} {'-'*12} {'-'*13} {'-'*12} {'-'*10}")

comparison_results = []

for doc_idx in range(engine.N):
    vec_doc = engine.compute_tfidf_vector(engine.documents_clean[doc_idx])
    
    # Cosine Similarity
    cosine_score = engine.cosine_similarity(vec_query, vec_doc)
    
    # Dot Product (tanpa normalisasi)
    dot_product = 0.0
    for term in vec_query:
        if term in vec_doc:
            dot_product += vec_query[term] * vec_doc[term]
    
    # Norm of document vector
    norm_d = math.sqrt(sum(w ** 2 for w in vec_doc.values()))
    
    # Document length (jumlah token)
    doc_length = len(engine.documents_clean[doc_idx])
    
    if cosine_score > 0 or dot_product > 0:
        comparison_results.append({
            'doc_id': str(engine.doc_ids[doc_idx]),
            'cosine': cosine_score,
            'dot_product': dot_product,
            'norm_d': norm_d,
            'doc_length': doc_length,
            'raw_text': engine.documents_raw[doc_idx][:80]
        })

# Sort by cosine (default ranking)
comparison_results.sort(key=lambda x: x['cosine'], reverse=True)

# Add cosine rank
for i, r in enumerate(comparison_results):
    r['cosine_rank'] = i + 1

# Sort by dot product for dot rank
dot_sorted = sorted(comparison_results, key=lambda x: x['dot_product'], reverse=True)
for i, r in enumerate(dot_sorted):
    r['dot_rank'] = i + 1

# Print sorted by cosine
comparison_results.sort(key=lambda x: x['cosine'], reverse=True)
for r in comparison_results:
    print(f"{r['cosine_rank']:<6} {r['doc_id']:<8} {r['cosine']:>12.6f} {r['dot_product']:>13.6f} {r['norm_d']:>12.4f} {r['doc_length']:>10}")

# Show ranking difference
print(f"\n--- Perbedaan Ranking: Cosine vs Dot Product ---")
print(f"{'Doc ID':<8} {'Rank Cosine':>12} {'Rank DotProd':>13} {'Perbedaan':>10} {'Pjg Dok':>10}")
print(f"{'-'*8} {'-'*12} {'-'*13} {'-'*10} {'-'*10}")

for r in comparison_results:
    diff = r['dot_rank'] - r['cosine_rank']
    marker = ""
    if diff < 0:
        marker = " (naik)"
    elif diff > 0:
        marker = " (turun)"
    print(f"{r['doc_id']:<8} {r['cosine_rank']:>12} {r['dot_rank']:>13} {diff:>+10}{marker:>10}")


print("\n\n" + "=" * 70)
print("  BAGIAN C: EVALUASI SISTEM")
print("=" * 70)

# ===== SKENARIO 1 =====
query1 = "energi terbarukan"
print(f"\n{'='*50}")
print(f"  SKENARIO 1: Query = '{query1}'")
print(f"{'='*50}")

results1 = engine.search(query1)
print(f"Query tokens: {results1['query_tokens']}")
print(f"Jumlah hasil: {results1['total_results']}")

print(f"\nHasil Pencarian (Top 10):")
for r in results1['results'][:10]:
    print(f"  Rank {r['rank']}: {r['doc_id']} | Score: {r['score']:.6f} | {r['document'][:80]}...")

# Ground Truth untuk query "energi terbarukan"
# Dokumen yang BENAR-BENAR relevan tentang energi terbarukan
ground_truth_1 = {'D4', 'D8', 'D10', 'D14', 'D18', 'D25', 'D35', 'D50'}

print(f"\nGround Truth (dokumen yang benar-benar relevan):")
print(f"  {sorted(ground_truth_1)}")

# Retrieved documents (top results with score > 0)
retrieved_1 = set()
for r in results1['results']:
    retrieved_1.add(r['doc_id'])

print(f"Retrieved (dokumen yang ditemukan sistem):")
print(f"  {sorted(retrieved_1)}")

# Calculate metrics
relevant_retrieved_1 = ground_truth_1 & retrieved_1
precision1 = len(relevant_retrieved_1) / len(retrieved_1) if retrieved_1 else 0
recall1 = len(relevant_retrieved_1) / len(ground_truth_1) if ground_truth_1 else 0
f_measure1 = (2 * precision1 * recall1) / (precision1 + recall1) if (precision1 + recall1) > 0 else 0

print(f"\nRelevant & Retrieved: {sorted(relevant_retrieved_1)}")
print(f"  |Relevant & Retrieved| = {len(relevant_retrieved_1)}")
print(f"  |Retrieved|           = {len(retrieved_1)}")
print(f"  |Relevant|            = {len(ground_truth_1)}")
print(f"\n  Precision = |Relevant & Retrieved| / |Retrieved|")
print(f"            = {len(relevant_retrieved_1)} / {len(retrieved_1)}")
print(f"            = {precision1:.4f} ({precision1*100:.1f}%)")
print(f"\n  Recall    = |Relevant & Retrieved| / |Relevant|")
print(f"            = {len(relevant_retrieved_1)} / {len(ground_truth_1)}")
print(f"            = {recall1:.4f} ({recall1*100:.1f}%)")
print(f"\n  F-Measure = 2 * Precision * Recall / (Precision + Recall)")
print(f"            = 2 * {precision1:.4f} * {recall1:.4f} / ({precision1:.4f} + {recall1:.4f})")
print(f"            = {f_measure1:.4f} ({f_measure1*100:.1f}%)")


# ===== SKENARIO 2 =====
query2 = "green economy"
print(f"\n\n{'='*50}")
print(f"  SKENARIO 2: Query = '{query2}'")
print(f"{'='*50}")

results2 = engine.search(query2)
print(f"Query tokens: {results2['query_tokens']}")
print(f"Jumlah hasil: {results2['total_results']}")

print(f"\nHasil Pencarian (Top 10):")
for r in results2['results'][:10]:
    print(f"  Rank {r['rank']}: {r['doc_id']} | Score: {r['score']:.6f} | {r['document'][:80]}...")

# Ground Truth untuk query "green economy"
# Dokumen yang BENAR-BENAR relevan tentang green economy secara eksplisit
ground_truth_2 = {'D5', 'D11', 'D12', 'D14', 'D15', 'D17', 'D19', 'D20', 'D21', 'D22', 
                  'D24', 'D25', 'D27', 'D28', 'D29', 'D30', 'D32', 'D37', 'D38', 'D42',
                  'D43', 'D45', 'D46', 'D47', 'D49'}

print(f"\nGround Truth (dokumen yang benar-benar relevan):")
print(f"  {sorted(ground_truth_2)}")

retrieved_2 = set()
for r in results2['results']:
    retrieved_2.add(r['doc_id'])

print(f"Retrieved (dokumen yang ditemukan sistem):")
print(f"  {sorted(retrieved_2)}")

relevant_retrieved_2 = ground_truth_2 & retrieved_2
precision2 = len(relevant_retrieved_2) / len(retrieved_2) if retrieved_2 else 0
recall2 = len(relevant_retrieved_2) / len(ground_truth_2) if ground_truth_2 else 0
f_measure2 = (2 * precision2 * recall2) / (precision2 + recall2) if (precision2 + recall2) > 0 else 0

print(f"\nRelevant & Retrieved: {sorted(relevant_retrieved_2)}")
print(f"  |Relevant & Retrieved| = {len(relevant_retrieved_2)}")
print(f"  |Retrieved|           = {len(retrieved_2)}")
print(f"  |Relevant|            = {len(ground_truth_2)}")
print(f"\n  Precision = |Relevant & Retrieved| / |Retrieved|")
print(f"            = {len(relevant_retrieved_2)} / {len(retrieved_2)}")
print(f"            = {precision2:.4f} ({precision2*100:.1f}%)")
print(f"\n  Recall    = |Relevant & Retrieved| / |Relevant|")
print(f"            = {len(relevant_retrieved_2)} / {len(ground_truth_2)}")
print(f"            = {recall2:.4f} ({recall2*100:.1f}%)")
print(f"\n  F-Measure = 2 * Precision * Recall / (Precision + Recall)")
print(f"            = 2 * {precision2:.4f} * {recall2:.4f} / ({precision2:.4f} + {recall2:.4f})")
print(f"            = {f_measure2:.4f} ({f_measure2*100:.1f}%)")

print("\n\n" + "=" * 70)
print("  RINGKASAN EVALUASI")
print("=" * 70)
print(f"\n{'Metrik':<15} {'Skenario 1':>15} {'Skenario 2':>15}")
print(f"{'-'*15} {'-'*15} {'-'*15}")
print(f"{'Query':<15} {query1:>15} {query2:>15}")
print(f"{'Precision':<15} {precision1:>14.1%} {precision2:>14.1%}")
print(f"{'Recall':<15} {recall1:>14.1%} {recall2:>14.1%}")
print(f"{'F-Measure':<15} {f_measure1:>14.1%} {f_measure2:>14.1%}")
print()


print("\n\n" + "=" * 70)
print("  BAGIAN D: EVALUASI SEMANTIC & HYBRID SEARCH")
print("=" * 70)

# Initialize semantic and hybrid engines
print("\n[1] Initializing Semantic Engine...")
from semantic_engine import SemanticSearchEngine
from hybrid_engine import HybridSearchEngine
from evaluation import IREvaluator

semantic_engine = SemanticSearchEngine()
semantic_engine.load_documents(csv_path)

print("\n[2] Initializing Hybrid Engine...")
hybrid_engine = HybridSearchEngine(engine, semantic_engine)

print("\n[3] Loading Evaluator...")
evaluator = IREvaluator('ground_truth.json')

print("\n" + "=" * 70)
print("  PERBANDINGAN METODE: Traditional vs Semantic vs Hybrid")
print("=" * 70)

# Evaluate all systems
print("\n[Evaluating Traditional TF-IDF...]")
results_traditional = evaluator.evaluate_system(engine, k_values=[5, 10])

print("\n[Evaluating Semantic Search...]")
results_semantic = evaluator.evaluate_system(semantic_engine, k_values=[5, 10])

print("\n[Evaluating Semantic Reranking...]")
# Use reranking method instead of RRF for requirement compliance
class RerankWrapper:
    def __init__(self, hybrid_eng):
        self.hybrid_eng = hybrid_eng
    def search(self, query):
        return self.hybrid_eng.search_hybrid(query, method='rerank')

rerank_engine = RerankWrapper(hybrid_engine)
results_hybrid = evaluator.evaluate_system(rerank_engine, k_values=[5, 10])

# Display results
print("\n" + "=" * 70)
print("  HASIL EVALUASI - MEAN AVERAGE PRECISION (MAP)")
print("=" * 70)

mean_row_trad = results_traditional[results_traditional['query_id'] == 'MEAN'].iloc[0]
mean_row_sem = results_semantic[results_semantic['query_id'] == 'MEAN'].iloc[0]
mean_row_hyb = results_hybrid[results_hybrid['query_id'] == 'MEAN'].iloc[0]

print(f"\n{'Metrik':<20} {'Traditional':>15} {'Semantic':>15} {'Reranking':>15}")
print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15}")
print(f"{'MAP':<20} {mean_row_trad['ap']:>14.4f} {mean_row_sem['ap']:>14.4f} {mean_row_hyb['ap']:>14.4f}")
print(f"{'MRR':<20} {mean_row_trad['rr']:>14.4f} {mean_row_sem['rr']:>14.4f} {mean_row_hyb['rr']:>14.4f}")
print(f"{'P@5':<20} {mean_row_trad['p@5']:>14.4f} {mean_row_sem['p@5']:>14.4f} {mean_row_hyb['p@5']:>14.4f}")
print(f"{'P@10':<20} {mean_row_trad['p@10']:>14.4f} {mean_row_sem['p@10']:>14.4f} {mean_row_hyb['p@10']:>14.4f}")
print(f"{'R@5':<20} {mean_row_trad['r@5']:>14.4f} {mean_row_sem['r@5']:>14.4f} {mean_row_hyb['r@5']:>14.4f}")
print(f"{'R@10':<20} {mean_row_trad['r@10']:>14.4f} {mean_row_sem['r@10']:>14.4f} {mean_row_hyb['r@10']:>14.4f}")
print(f"{'NDCG@5':<20} {mean_row_trad['ndcg@5']:>14.4f} {mean_row_sem['ndcg@5']:>14.4f} {mean_row_hyb['ndcg@5']:>14.4f}")
print(f"{'NDCG@10':<20} {mean_row_trad['ndcg@10']:>14.4f} {mean_row_sem['ndcg@10']:>14.4f} {mean_row_hyb['ndcg@10']:>14.4f}")

# Per-query comparison
print("\n" + "=" * 70)
print("  PERBANDINGAN PER QUERY (MAP)")
print("=" * 70)

print(f"\n{'Query ID':<10} {'Query':<35} {'Trad':>10} {'Sem':>10} {'Rerank':>10}")
print(f"{'-'*10} {'-'*35} {'-'*10} {'-'*10} {'-'*10}")

for idx in range(len(results_traditional) - 1):  # Exclude MEAN row
    q_trad = results_traditional.iloc[idx]
    q_sem = results_semantic.iloc[idx]
    q_hyb = results_hybrid.iloc[idx]
    
    query_text_short = q_trad['query_text'][:33] + '..' if len(q_trad['query_text']) > 35 else q_trad['query_text']
    
    print(f"{q_trad['query_id']:<10} {query_text_short:<35} {q_trad['ap']:>10.4f} {q_sem['ap']:>10.4f} {q_hyb['ap']:>10.4f}")

# Identify best method
print("\n" + "=" * 70)
print("  KESIMPULAN")
print("=" * 70)

map_scores = {
    'Traditional': mean_row_trad['ap'],
    'Semantic': mean_row_sem['ap'],
    'Semantic Reranking': mean_row_hyb['ap']
}

best_method = max(map_scores, key=map_scores.get)
print(f"\nMetode terbaik berdasarkan MAP: {best_method} (MAP = {map_scores[best_method]:.4f})")

improvement_sem = ((mean_row_sem['ap'] - mean_row_trad['ap']) / mean_row_trad['ap'] * 100)
improvement_hyb = ((mean_row_hyb['ap'] - mean_row_trad['ap']) / mean_row_trad['ap'] * 100)

print(f"\nPeningkatan Semantic vs Traditional: {improvement_sem:+.2f}%")
print(f"Peningkatan Reranking vs Traditional: {improvement_hyb:+.2f}%")

print("\n" + "=" * 70)
print("  ANALISIS SELESAI")
print("=" * 70)
print("\nSemua metrik telah dihitung:")
print("  [OK] Traditional TF-IDF (Baseline)")
print("  [OK] Semantic Search (SBERT Multilingual)")
print("  [OK] Semantic Reranking (Hybrid Method)")
print("\nMetrik evaluasi: MAP, MRR, P@k, R@k, NDCG@k")
print("Ground truth queries: 7 queries dengan relevance judgments")
print()
