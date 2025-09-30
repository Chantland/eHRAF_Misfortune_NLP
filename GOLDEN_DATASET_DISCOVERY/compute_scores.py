"""
Step 1: Compute ALL scores and save to file
Run this ONCE after embeddings are in Pinecone
Then use the saved scores for instant threshold testing
"""

import pandas as pd
import numpy as np
import os
from golden_dataset_finder import GoldenDatasetFinder
from dotenv import load_dotenv
from tqdm import tqdm
import pickle

load_dotenv()

# Configuration
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

DATA_PATH = "sample_dataset.xlsx"
INDEX_NAME = "hraf-misfortune-test"
NAMESPACE = "test"
OUTPUT_FILE = "cached_scores.pkl"


def main():
    print("=" * 70)
    print("STEP 1: Compute and Cache All Scores")
    print("=" * 70)

    # Load data
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH, header=1)
    print(f"   Loaded {len(df)} passages")

    # Initialize
    finder = GoldenDatasetFinder(
        voyage_api_key=VOYAGE_API_KEY,
        pinecone_api_key=PINECONE_API_KEY,
        index_name=INDEX_NAME,
        region="us-east-1"
    )

    label_columns = finder._auto_detect_label_columns(df)
    valid_mask = df['Passage'].notna()
    embedded_indices = df[valid_mask].index.tolist()

    print(f"   {len(embedded_indices)} passages with embeddings")
    print(f"   {len(label_columns)} label columns")

    # Estimate cost
    num_labels_to_rerank = sum(df[col].sum() for col in label_columns)
    est_cost = (num_labels_to_rerank * 200 / 1000) * 0.00005  # Rough estimate
    print(f"\n💰 Estimated reranking cost: ${est_cost:.2f}")

    response = input("\nCompute all scores? This will:\n"
                     "  1. Calculate consistency (1 min, free)\n"
                     "  2. Calculate rerank scores (~$3-5)\n"
                     "  3. Save to file for instant reuse\n"
                     "Proceed? (y/n): ")

    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Step 1: Calculate consistency scores
    print("\n📊 Step 1: Calculating consistency scores...")
    consistency_scores = {}

    for idx in tqdm(embedded_indices, desc="Consistency"):
        try:
            similar = finder.find_similar_passages(idx, k=15, namespace=NAMESPACE)
            consistency = finder.calculate_label_consistency(
                idx, similar, label_columns, namespace=NAMESPACE
            )
            active_labels = [l for l in label_columns if df.loc[idx, l] == 1]
            if active_labels:
                consistency_scores[idx] = {
                    'avg': np.mean([consistency[l] for l in active_labels]),
                    'by_label': {l: consistency[l] for l in active_labels}
                }
        except Exception as e:
            print(f"Error on {idx}: {e}")
            consistency_scores[idx] = {'avg': 0.0, 'by_label': {}}

    print(f"   ✅ Calculated consistency for {len(consistency_scores)} passages")

    # Step 2: Calculate rerank scores
    print("\n🎯 Step 2: Calculating rerank scores...")
    rerank_scores = {label: {} for label in label_columns}
    passages = df['Passage'].tolist()

    for label in tqdm(label_columns, desc="Reranking"):
        label_indices = [idx for idx in embedded_indices
                         if idx in df.index and df.loc[idx, label] == 1]
        if not label_indices:
            continue

        label_passages = [passages[idx] for idx in label_indices]
        scores = finder.rerank_passages_for_label(label_passages, label)

        for idx, score in zip(label_indices, scores):
            rerank_scores[label][idx] = score

    print(f"   ✅ Calculated rerank scores for {len(label_columns)} labels")

    # Step 3: Combine and save
    print("\n💾 Step 3: Saving scores to file...")

    # Create combined score table
    score_data = {
        'passage_idx': [],
        'consistency_avg': [],
        'rerank_avg': [],
        'num_labels': []
    }

    for idx in embedded_indices:
        active_labels = [l for l in label_columns if df.loc[idx, l] == 1]
        if not active_labels:
            continue

        cons_avg = consistency_scores.get(idx, {}).get('avg', 0.0)
        rerank_values = [rerank_scores[l].get(idx, 0.0) for l in active_labels]
        rerank_avg = np.mean(rerank_values) if rerank_values else 0.0

        score_data['passage_idx'].append(idx)
        score_data['consistency_avg'].append(cons_avg)
        score_data['rerank_avg'].append(rerank_avg)
        score_data['num_labels'].append(len(active_labels))

    scores_df = pd.DataFrame(score_data)

    # Save both detailed and summary
    cache = {
        'df_summary': scores_df,
        'consistency_detailed': consistency_scores,
        'rerank_detailed': rerank_scores,
        'label_columns': label_columns,
        'embedded_indices': embedded_indices
    }

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(cache, f)

    # Also save as Excel for easy viewing
    scores_df.to_excel("cached_scores.xlsx", index=False)

    print(f"   ✅ Saved to {OUTPUT_FILE} and cached_scores.xlsx")

    # Show summary
    print("\n" + "=" * 70)
    print("SCORE SUMMARY")
    print("=" * 70)
    print(f"\nProcessed {len(scores_df)} passages with labels")
    print(f"\nConsistency scores:")
    print(f"   Min:    {scores_df['consistency_avg'].min():.3f}")
    print(f"   Median: {scores_df['consistency_avg'].median():.3f}")
    print(f"   Mean:   {scores_df['consistency_avg'].mean():.3f}")
    print(f"   Max:    {scores_df['consistency_avg'].max():.3f}")

    print(f"\nRerank scores:")
    print(f"   Min:    {scores_df['rerank_avg'].min():.3f}")
    print(f"   Median: {scores_df['rerank_avg'].median():.3f}")
    print(f"   Mean:   {scores_df['rerank_avg'].mean():.3f}")
    print(f"   Max:    {scores_df['rerank_avg'].max():.3f}")

    print("\n✅ Done! Now run 'python test_thresholds.py' to test thresholds instantly")


if __name__ == "__main__":
    main()