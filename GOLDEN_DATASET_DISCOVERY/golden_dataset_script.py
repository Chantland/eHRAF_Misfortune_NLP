"""
Test Script with Lower Thresholds - Get Results Immediately
Runs with more permissive thresholds to identify golden dataset
"""

import pandas as pd
import os
from golden_dataset_finder import GoldenDatasetFinder
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CONFIGURATION - LOWER THRESHOLDS
# =============================================================================

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

SAMPLE_DATA_PATH = "sample_dataset.xlsx"
OUTPUT_PATH = "golden_dataset_LOWER_THRESHOLDS.xlsx"

INDEX_NAME = "hraf-misfortune-test"
CLOUD = "aws"
REGION = "us-east-1"

# LOWER THRESHOLDS - More permissive
MIN_CONSISTENCY = 0.55  # Lowered from 0.7
MIN_RERANK_SCORE = 0.50  # Lowered from 0.6
TARGET_SIZE = 150
K_SIMILAR = 15


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("HRAF Golden Dataset Finder - LOWER THRESHOLDS")
    print("=" * 70)

    # Load data
    print(f"\n📂 Loading sample data from: {SAMPLE_DATA_PATH}")
    df = pd.read_excel(SAMPLE_DATA_PATH, header=1)
    print(f"   ✅ Loaded {len(df)} passages")

    # Check for missing passages
    passage_col = 'Passage'
    missing_passages = df[passage_col].isna().sum()
    if missing_passages > 0:
        print(f"\n⚠️  Found {missing_passages} passages with missing text")

    # Initialize finder
    print(f"\n🔧 Initializing Golden Dataset Finder...")
    finder = GoldenDatasetFinder(
        voyage_api_key=VOYAGE_API_KEY,
        pinecone_api_key=PINECONE_API_KEY,
        index_name=INDEX_NAME,
        cloud=CLOUD,
        region=REGION
    )
    print("   ✅ Initialization successful")

    # Run with lower thresholds
    print(f"\n🔍 Identifying golden passages...")
    print(f"   - Minimum consistency: {MIN_CONSISTENCY} ⬇️ (was 0.7)")
    print(f"   - Minimum rerank score: {MIN_RERANK_SCORE} ⬇️ (was 0.6)")
    print(f"   - Target size: {TARGET_SIZE}")
    print(f"   - Checking {K_SIMILAR} similar passages per query")

    try:
        golden_df = finder.identify_golden_dataset(
            df=df,
            passage_column='Passage',
            min_consistency=MIN_CONSISTENCY,
            min_rerank_score=MIN_RERANK_SCORE,
            target_size=TARGET_SIZE,
            k_similar=K_SIMILAR,
            namespace="test",
            recompute_embeddings=False  # Use existing embeddings
        )

        # Analyze results
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        if len(golden_df) == 0:
            print("\n⚠️  Still no golden passages! Data quality may be very low.")
            print("   Consider lowering thresholds further or investigating data.")
            return

        print(f"\n📊 Golden Dataset Statistics:")
        print(f"   - Total golden passages: {len(golden_df)}")
        print(f"   - Percentage of sample: {len(golden_df) / len(df) * 100:.1f}%")
        print(f"   - Mean consistency: {golden_df['confidence_consistency'].mean():.3f}")
        print(f"   - Mean rerank score: {golden_df['confidence_rerank'].mean():.3f}")
        print(f"   - Mean composite score: {golden_df['confidence_composite'].mean():.3f}")

        # Score ranges
        print(f"\n📈 Score Ranges in Golden Set:")
        print(
            f"   Consistency:  {golden_df['confidence_consistency'].min():.3f} - {golden_df['confidence_consistency'].max():.3f}")
        print(
            f"   Rerank:       {golden_df['confidence_rerank'].min():.3f} - {golden_df['confidence_rerank'].max():.3f}")
        print(
            f"   Composite:    {golden_df['confidence_composite'].min():.3f} - {golden_df['confidence_composite'].max():.3f}")

        # Auto-detect label columns
        label_cols = finder._auto_detect_label_columns(df)

        # Label distribution
        print(f"\n📈 Label Distribution in Golden Set:")
        for col in label_cols[:10]:  # Show first 10
            if col in golden_df.columns:
                golden_positive = golden_df[col].sum()
                original_positive = df[col].sum()
                percentage = (golden_positive / original_positive * 100) if original_positive > 0 else 0
                print(f"   - {col:30s}: {int(golden_positive):3d}/{int(original_positive):3d} ({percentage:5.1f}%)")

        # Top 5 passages
        print(f"\n⭐ Top 5 Most Confident Passages:")
        top_5 = golden_df.nlargest(5, 'confidence_composite')
        for i, (idx, row) in enumerate(top_5.iterrows(), 1):
            passage_preview = str(row['Passage'])[:80] + "..." if len(str(row['Passage'])) > 80 else str(row['Passage'])
            print(
                f"\n   {i}. Composite: {row['confidence_composite']:.3f} | Consistency: {row['confidence_consistency']:.3f} | Rerank: {row['confidence_rerank']:.3f}")
            print(f"      {passage_preview}")

        # Save results
        print(f"\n💾 Saving results to: {OUTPUT_PATH}")
        golden_df.to_excel(OUTPUT_PATH, index=False)
        print("   ✅ Results saved successfully")

        print("\n" + "=" * 70)
        print("✅ Test run completed successfully!")
        print("=" * 70)

        print("\n📝 Next Steps:")
        print("   1. Review the golden_dataset_LOWER_THRESHOLDS.xlsx file")
        print("   2. Check if the passages look like high-quality labels")
        print("   3. Adjust thresholds if needed")
        print("   4. Run on full 10k dataset with calibrated thresholds")

    except Exception as e:
        print(f"\n❌ Error during golden dataset identification: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()