"""
Rankify - Perbandingan Sistem Tradisional vs Modern
Script untuk mengevaluasi dan membandingkan:
  1. Search Engine (TF-IDF) - Metrik IR: MAP, MRR, NDCG
  2. Classification (SBERT + Ensemble) - Metrik: Accuracy, Precision, Recall, F1
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from search_engine import MiniSearchEngine
from traditional_classifier import TraditionalClassifier
from sbert_classifier import SBERTClassifier
from ensemble_classifier import EnsembleClassifier
from evaluation import IREvaluator


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title):
    print(f"\n--- {title} ---")


def run_comparison():
    csv_path = os.path.join(os.path.dirname(__file__), 'data green economy.csv')

    # ==================================================================
    # BAGIAN 1: EVALUASI SEARCH ENGINE (TF-IDF Tradisional)
    # ==================================================================
    print_header("BAGIAN 1: EVALUASI SEARCH ENGINE (TF-IDF)")

    # Load search engine
    engine = MiniSearchEngine()
    engine.load_documents(csv_path)

    # Load evaluator with ground truth
    evaluator = IREvaluator()

    print(f"\nJumlah ground truth queries: {len(evaluator.queries)}")
    for q in evaluator.queries:
        hr = len(q['relevant_docs']['highly_relevant'])
        pr = len(q['relevant_docs']['partially_relevant'])
        print(f"  {q['query_id']}: \"{q['query_text']}\" "
              f"({hr} highly relevant, {pr} partially relevant)")

    # Evaluate TF-IDF search
    print_subheader("Hasil Evaluasi TF-IDF Search")
    df = evaluator.evaluate_system(engine, k_values=[5, 10])

    # Print per-query results
    print(f"\n{'Query':<6} {'AP':>8} {'RR':>8} {'P@5':>8} {'P@10':>8} "
          f"{'R@5':>8} {'R@10':>8} {'NDCG@5':>8} {'NDCG@10':>8}")
    print("-" * 70)

    for _, row in df.iterrows():
        qid = str(row['query_id'])
        print(f"{qid:<6} {row['ap']:>8.4f} {row['rr']:>8.4f} "
              f"{row['p@5']:>8.4f} {row['p@10']:>8.4f} "
              f"{row['r@5']:>8.4f} {row['r@10']:>8.4f} "
              f"{row['ndcg@5']:>8.4f} {row['ndcg@10']:>8.4f}")

    # Extract aggregate metrics
    mean_row = df[df['query_id'] == 'MEAN'].iloc[0]
    map_score = mean_row['ap']
    mrr_score = mean_row['rr']
    ndcg5 = mean_row['ndcg@5']
    ndcg10 = mean_row['ndcg@10']

    print_subheader("Ringkasan Metrik IR (Search Engine)")
    print(f"  MAP  (Mean Average Precision)  : {map_score:.4f}")
    print(f"  MRR  (Mean Reciprocal Rank)    : {mrr_score:.4f}")
    print(f"  NDCG@5                         : {ndcg5:.4f}")
    print(f"  NDCG@10                        : {ndcg10:.4f}")

    # ==================================================================
    # BAGIAN 2: EVALUASI CLASSIFICATION (Modern)
    # ==================================================================
    print_header("BAGIAN 2: EVALUASI CLASSIFICATION (Modern)")

    # --- Traditional Classifier (TF-IDF + SVM) ---
    print_subheader("2a. Traditional Classifier (TF-IDF + SVM)")
    trad_clf = TraditionalClassifier()
    trad_clf.load_data(csv_path)

    trad_results = {}
    for method in ['svm', 'nb', 'lr']:
        trad_clf.train(method)
        metrics = trad_clf.evaluate()
        trad_results[method] = metrics

    # --- SBERT Classifier ---
    print_subheader("2b. SBERT Classifier (Semantic Embeddings)")
    sbert_clf = SBERTClassifier()
    sbert_clf.load_data(csv_path)
    sbert_clf.train()
    sbert_metrics = sbert_clf.evaluate()

    # --- Ensemble Classifier ---
    print_subheader("2c. Ensemble Classifier")
    ensemble_clf = EnsembleClassifier(trad_clf, sbert_clf)
    trad_clf.train('svm')  # retrain with SVM for ensemble
    ensemble_metrics = ensemble_clf.evaluate(method='soft_voting')

    # ==================================================================
    # BAGIAN 3: TABEL PERBANDINGAN
    # ==================================================================
    print_header("BAGIAN 3: PERBANDINGAN TRADISIONAL vs MODERN")

    print_subheader("3a. Metrik Search Engine (Information Retrieval)")
    print(f"\n  {'Metrik':<35} {'TF-IDF (Tradisional)':>20}")
    print(f"  {'-'*55}")
    print(f"  {'MAP (Mean Average Precision)':<35} {map_score:>20.4f}")
    print(f"  {'MRR (Mean Reciprocal Rank)':<35} {mrr_score:>20.4f}")
    print(f"  {'NDCG@5':<35} {ndcg5:>20.4f}")
    print(f"  {'NDCG@10':<35} {ndcg10:>20.4f}")

    print_subheader("3b. Metrik Classification (Sentiment Analysis)")
    svm = trad_results['svm']
    print(f"\n  {'Metrik':<15} {'Trad (SVM)':>12} {'SBERT':>12} {'Ensemble':>12}")
    print(f"  {'-'*51}")
    print(f"  {'Accuracy':<15} {svm['accuracy']:>12.4f} "
          f"{sbert_metrics['accuracy']:>12.4f} "
          f"{ensemble_metrics['accuracy']:>12.4f}")
    print(f"  {'Precision':<15} {svm['precision']:>12.4f} "
          f"{sbert_metrics['precision']:>12.4f} "
          f"{ensemble_metrics['precision']:>12.4f}")
    print(f"  {'Recall':<15} {svm['recall']:>12.4f} "
          f"{sbert_metrics['recall']:>12.4f} "
          f"{ensemble_metrics['recall']:>12.4f}")
    print(f"  {'F1-Score':<15} {svm['f1']:>12.4f} "
          f"{sbert_metrics['f1']:>12.4f} "
          f"{ensemble_metrics['f1']:>12.4f}")

    print_subheader("3c. Perbandingan Semua Classifier")
    print(f"\n  {'Model':<30} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1-Score':>10}")
    print(f"  {'-'*70}")
    for method_name, m in trad_results.items():
        label = f"Traditional ({method_name.upper()})"
        print(f"  {label:<30} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f}")
    print(f"  {'SBERT (Centroid)':<30} {sbert_metrics['accuracy']:>10.4f} "
          f"{sbert_metrics['precision']:>10.4f} "
          f"{sbert_metrics['recall']:>10.4f} "
          f"{sbert_metrics['f1']:>10.4f}")
    print(f"  {'Ensemble (Soft Voting)':<30} {ensemble_metrics['accuracy']:>10.4f} "
          f"{ensemble_metrics['precision']:>10.4f} "
          f"{ensemble_metrics['recall']:>10.4f} "
          f"{ensemble_metrics['f1']:>10.4f}")

    # ==================================================================
    # BAGIAN 4: KESIMPULAN
    # ==================================================================
    print_header("BAGIAN 4: KESIMPULAN")

    print("""
  SEARCH ENGINE (Tradisional - TF-IDF):
  - Menggunakan TF-IDF + Cosine Similarity untuk ranking dokumen
  - Dievaluasi dengan metrik IR standar (MAP, MRR, NDCG)
  - Cocok untuk tugas pencarian dokumen (Information Retrieval)

  CLASSIFICATION (Modern - SBERT):
  - Menggunakan SBERT (Sentence-BERT) untuk semantic embeddings
  - Klasifikasi sentimen menggunakan centroid-based similarity
  - Dievaluasi dengan Accuracy, Precision, Recall, F1-Score
  - Memahami konteks dan makna kalimat

  PERBANDINGAN:
  - Tradisional (TF-IDF): Berbasis frekuensi kata, cepat, interpretable
  - Modern (SBERT): Berbasis deep learning, memahami semantik
  - Ensemble: Menggabungkan keduanya untuk hasil terbaik
    """)

    print("=" * 70)
    print("  EVALUASI SELESAI")
    print("=" * 70)


if __name__ == '__main__':
    run_comparison()
