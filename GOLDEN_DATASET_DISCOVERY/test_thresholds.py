"""
Step 2: Test different thresholds instantly using cached scores
Run this as many times as you want - it's instant and free!
"""

import pandas as pd
import pickle
import os

CACHED_SCORES_FILE = "cached_scores.pkl"
DATA_PATH = "sample_dataset.xlsx"

def test_threshold_combination(scores_df, min_consistency, min_rerank):
    """Filter golden passages based on thresholds"""
    filtered = scores_df[
        (scores_df['consistency_avg'] >= min_consistency) &
        (scores_df['rerank_avg'] >= min_rerank)
    ].copy()

    filtered['composite'] = 0.5 * filtered['consistency_avg'] + 0.5 * filtered['rerank_avg']
    return filtered.sort_values('composite', ascending=False)


def main():
    print("="*70)
    print("INSTANT THRESHOLD TESTING - Uses Cached Scores")
    print("="*70)

    # Load cached scores
    if not os.path.exists(CACHED_SCORES_FILE):
        print(f"\n❌ Error: {CACHED_SCORES_FILE} not found!")
        print("   Run 'python compute_and_save_scores.py' first")
        return

    print(f"\nLoading cached scores from {CACHED_SCORES_FILE}...")
    with open(CACHED_SCORES_FILE, 'rb') as f:
        cache = pickle.load(f)

    scores_df = cache['df_summary']
    print(f"   ✅ Loaded scores for {len(scores_df)} passages")

    # Load original data for label distribution
    df = pd.read_excel(DATA_PATH, header=1)
    label_columns = cache['label_columns']

    print("\n" + "="*70)
    print("THRESHOLD TESTING MATRIX")
    print("="*70)

    # Test grid of thresholds
    consistency_thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    rerank_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    print("\nGolden passage counts by threshold combination:\n")
    print(f"{'Consistency →':>15}", end='')
    for c_thresh in consistency_thresholds:
        print(f"{c_thresh:>7.2f}", end='')
    print()
    print(f"{'Rerank ↓':>15}", end='')
    print("-" * (7 * len(consistency_thresholds)))

    results = []
    for r_thresh in rerank_thresholds:
        print(f"{r_thresh:>15.2f}", end='')
        for c_thresh in consistency_thresholds:
            golden = test_threshold_combination(scores_df, c_thresh, r_thresh)
            count = len(golden)
            print(f"{count:>7d}", end='')

            if count > 0:
                results.append({
                    'consistency': c_thresh,
                    'rerank': r_thresh,
                    'count': count,
                    'pct': count / len(scores_df) * 100,
                    'mean_consistency': golden['consistency_avg'].mean(),
                    'mean_rerank': golden['rerank_avg'].mean(),
                    'mean_composite': golden['composite'].mean()
                })
        print()

    # Show detailed results for interesting combinations
    print("\n" + "="*70)
    print("RECOMMENDED THRESHOLD COMBINATIONS")
    print("="*70)

    # Find combinations that give 5-15% golden set
    good_combos = [r for r in results if 5 <= r['pct'] <= 15]
    good_combos.sort(key=lambda x: x['mean_composite'], reverse=True)

    if good_combos:
        print("\nTop combinations (5-15% of dataset, sorted by quality):\n")
        print(f"{'Consistency':>12} {'Rerank':>8} {'Count':>6} {'%':>6} {'Mean Composite':>15}")
        print("-" * 60)

        for combo in good_combos[:5]:
            print(f"{combo['consistency']:>12.2f} {combo['rerank']:>8.2f} "
                  f"{combo['count']:>6d} {combo['pct']:>5.1f}% "
                  f"{combo['mean_composite']:>15.3f}")

        # Recommend the best one
        best = good_combos[0]
        print(f"\n💡 RECOMMENDED THRESHOLDS:")
        print(f"   MIN_CONSISTENCY = {best['consistency']}")
        print(f"   MIN_RERANK_SCORE = {best['rerank']}")
        print(f"   Expected: {best['count']} passages ({best['pct']:.1f}%)")
        print(f"   Quality: Composite score {best['mean_composite']:.3f}")
    else:
        print("\n⚠️  No combinations found in 5-15% range")
        print("Showing all non-empty combinations:")
        results.sort(key=lambda x: x['mean_composite'], reverse=True)
        for combo in results[:10]:
            print(f"   {combo['consistency']:.2f}/{combo['rerank']:.2f}: "
                  f"{combo['count']} passages ({combo['pct']:.1f}%)")

    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE TESTING")
    print("="*70)

    while True:
        print("\nEnter thresholds to test (or 'q' to quit):")
        cons_input = input("  Consistency threshold (0.0-1.0): ")
        if cons_input.lower() == 'q':
            break

        rerank_input = input("  Rerank threshold (0.0-1.0): ")
        if rerank_input.lower() == 'q':
            break

        try:
            min_cons = float(cons_input)
            min_rerank = float(rerank_input)

            golden = test_threshold_combination(scores_df, min_cons, min_rerank)

            if len(golden) == 0:
                print(f"\n  ❌ No passages meet criteria")
                continue

            print(f"\n  ✅ Found {len(golden)} passages ({len(golden)/len(scores_df)*100:.1f}%)")
            print(f"     Mean consistency: {golden['consistency_avg'].mean():.3f}")
            print(f"     Mean rerank:      {golden['rerank_avg'].mean():.3f}")
            print(f"     Mean composite:   {golden['composite'].mean():.3f}")

            # Show label distribution
            print(f"\n     Label distribution in golden set:")
            for label in label_columns[:8]:  # Show top 8
                if label in df.columns:
                    golden_with_label = sum(df.loc[idx, label] == 1
                                          for idx in golden['passage_idx']
                                          if idx in df.index)
                    total_with_label = df[label].sum()
                    pct = (golden_with_label / total_with_label * 100) if total_with_label > 0 else 0
                    print(f"       {label:30s}: {golden_with_label:3d}/{int(total_with_label):3d} ({pct:5.1f}%)")

            save = input("\n     Save this golden set to Excel? (y/n): ")
            if save.lower() == 'y':
                # Merge with original data
                golden_indices = golden['passage_idx'].tolist()
                golden_full = df.loc[golden_indices].copy()

                # Add confidence scores
                for idx in golden_indices:
                    score_row = golden[golden['passage_idx'] == idx].iloc[0]
                    golden_full.loc[idx, 'confidence_consistency'] = score_row['consistency_avg']
                    golden_full.loc[idx, 'confidence_rerank'] = score_row['rerank_avg']
                    golden_full.loc[idx, 'confidence_composite'] = score_row['composite']

                filename = f"golden_set_{min_cons:.2f}_{min_rerank:.2f}.xlsx"
                golden_full.to_excel(filename, index=False)
                print(f"     ✅ Saved to {filename}")

        except ValueError:
            print("  ❌ Invalid input. Please enter numbers between 0.0 and 1.0")


if __name__ == "__main__":
    main()