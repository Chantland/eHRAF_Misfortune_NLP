#%% md
# # HRAF Golden Dataset Discovery - Interactive Notebook
# 
# Interactive Jupyter notebook for exploring HRAF passages, testing custom queries,
# and identifying high-confidence golden dataset passages.
#%%
# ============================================================================
# SECTION 1: Setup and Imports
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML, Markdown
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed
import pickle
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Import custom classes
from golden_dataset_finder import GoldenDatasetFinder

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✅ Imports successful")
#%%
# ============================================================================
# SECTION 2: Configuration
# ============================================================================

# API Keys
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Paths
DATA_PATH = "../GOLDEN_DATASET/data/sample_dataset.xlsx"
CACHED_SCORES_FILE = "../GOLDEN_DATASET/cached_scores.pkl"

# Pinecone settings
INDEX_NAME = "hraf-misfortune-test"
NAMESPACE = "test"
REGION = "us-east-1"

print(f"📂 Data path: {DATA_PATH}")
print(f"💾 Cache file: {CACHED_SCORES_FILE}")
print(f"🔧 Pinecone index: {INDEX_NAME}")


#%%
# ============================================================================
# SECTION 3: Load Data and Initialize
# ============================================================================

print("\n" + "="*70)
print("Loading Data")
print("="*70)

# Load dataset
df = pd.read_excel(DATA_PATH, header=1)
print(f"✅ Loaded {len(df)} passages")
print(f"   Columns: {len(df.columns)}")
print(f"   Missing passages: {df['Passage'].isna().sum()}")

# Initialize finder
finder = GoldenDatasetFinder(
    voyage_api_key=VOYAGE_API_KEY,
    pinecone_api_key=PINECONE_API_KEY,
    index_name=INDEX_NAME,
    region=REGION
)

# Auto-detect label columns
label_columns = finder._auto_detect_label_columns(df)
print(f"\n✅ Found {len(label_columns)} label columns:")
for i, label in enumerate(label_columns, 1):
    positive_count = df[label].sum()
    print(f"   {i:2d}. {label:30s} ({int(positive_count):4d} positive, {positive_count/len(df)*100:5.1f}%)")

#%%
# ============================================================================
# SECTION 4: Load or Compute Scores
# ============================================================================

print("\n" + "="*70)
print("Score Management")
print("="*70)

def load_cached_scores():
    """Load pre-computed scores if available"""
    if os.path.exists(CACHED_SCORES_FILE):
        with open(CACHED_SCORES_FILE, 'rb') as f:
            cache = pickle.load(f)
        print(f"✅ Loaded cached scores for {len(cache['df_summary'])} passages")
        return cache
    else:
        print(f"⚠️  No cached scores found at {CACHED_SCORES_FILE}")
        print("   Run compute_scores.py first, or use the compute function below")
        return None

def compute_and_cache_scores(k_similar=15):
    """Compute all scores and save to cache"""
    print(f"\n📊 Computing scores for {len(df)} passages...")
    print(f"   This will take several minutes and cost ~$3-5")

    from tqdm import tqdm

    # Filter valid passages
    valid_mask = df['Passage'].notna()
    valid_df = df[valid_mask]
    embedded_indices = valid_df.index.tolist()

    # Step 1: Consistency scores
    print(f"\n1️⃣  Calculating consistency scores (k={k_similar})...")
    consistency_scores = {}

    for idx in tqdm(embedded_indices):
        try:
            similar = finder.find_similar_passages(idx, k=k_similar, namespace=NAMESPACE)
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
            consistency_scores[idx] = {'avg': 0.0, 'by_label': {}}

    # Step 2: Rerank scores
    print(f"\n2️⃣  Calculating rerank scores...")
    rerank_scores = {label: {} for label in label_columns}
    passages = df['Passage'].tolist()

    for label in tqdm(label_columns):
        label_indices = [idx for idx in embedded_indices if df.loc[idx, label] == 1]
        if not label_indices:
            continue

        label_passages = [passages[idx] for idx in label_indices]
        scores = finder.rerank_passages_for_label(label_passages, label)

        for idx, score in zip(label_indices, scores):
            rerank_scores[label][idx] = score

    # Step 3: Combine and save
    print(f"\n3️⃣  Combining and saving...")
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

    cache = {
        'df_summary': scores_df,
        'consistency_detailed': consistency_scores,
        'rerank_detailed': rerank_scores,
        'label_columns': label_columns,
        'embedded_indices': embedded_indices
    }

    with open(CACHED_SCORES_FILE, 'wb') as f:
        pickle.dump(cache, f)

    scores_df.to_excel("cached_scores.xlsx", index=False)

    print(f"\n✅ Saved to {CACHED_SCORES_FILE} and cached_scores.xlsx")
    return cache

# Try to load cached scores
cache = load_cached_scores()
#%%
# ============================================================================
# SECTION 5: Interactive Data Exploration
# ============================================================================

print("\n" + "="*70)
print("Interactive Exploration Tools")
print("="*70)

def explore_passage(passage_idx):
    """Display detailed information about a specific passage"""
    if passage_idx not in df.index:
        print(f"❌ Passage {passage_idx} not found")
        return

    row = df.loc[passage_idx]

    # Basic info
    print(f"\n{'='*70}")
    print(f"PASSAGE {passage_idx}")
    print(f"{'='*70}\n")

    # Passage text
    text = row['Passage']
    print(f"📄 TEXT ({len(str(text))} characters):")
    print(f"{'-'*70}")
    print(text)
    print(f"{'-'*70}\n")

    # Active labels
    active_labels = [l for l in label_columns if row[l] == 1]
    print(f"🏷️  LABELS ({len(active_labels)}):")
    for label in active_labels:
        print(f"   ✓ {label}")

    # Scores (if available)
    if cache is not None:
        scores_df = cache['df_summary']
        if passage_idx in scores_df['passage_idx'].values:
            score_row = scores_df[scores_df['passage_idx'] == passage_idx].iloc[0]
            print(f"\n📊 CONFIDENCE SCORES:")
            print(f"   Consistency: {score_row['consistency_avg']:.3f}")
            print(f"   Rerank:      {score_row['rerank_avg']:.3f}")
            print(f"   Composite:   {(score_row['consistency_avg'] + score_row['rerank_avg'])/2:.3f}")

            # Per-label scores
            consistency_detailed = cache['consistency_detailed']
            rerank_detailed = cache['rerank_detailed']

            if passage_idx in consistency_detailed:
                print(f"\n📈 PER-LABEL SCORES:")
                for label in active_labels:
                    cons = consistency_detailed[passage_idx]['by_label'].get(label, 0)
                    rerank = rerank_detailed[label].get(passage_idx, 0)
                    print(f"   {label:30s}: Consistency={cons:.3f}, Rerank={rerank:.3f}")

    # Similar passages
    try:
        print(f"\n🔍 SIMILAR PASSAGES:")
        similar = finder.find_similar_passages(passage_idx, k=5, namespace=NAMESPACE)
        for i, sim in enumerate(similar, 1):
            sim_idx = sim['passage_idx']
            sim_text = df.loc[sim_idx, 'Passage']
            preview = str(sim_text)[:100] + "..." if len(str(sim_text)) > 100 else str(sim_text)
            print(f"\n   {i}. Passage {sim_idx} (similarity: {sim['similarity']:.3f})")
            print(f"      {preview}")
    except Exception as e:
        print(f"\n⚠️  Could not find similar passages: {e}")

# Interactive passage explorer
passage_explorer = widgets.IntText(
    value=0,
    description='Passage ID:',
    disabled=False
)

def on_explore_click(passage_idx):
    explore_passage(passage_idx)

interact(on_explore_click, passage_idx=passage_explorer)
#%%
# ============================================================================
# SECTION 6: Custom Query Search
# ============================================================================

print("\n" + "="*70)
print("Custom Query Search")
print("="*70)

def search_passages_by_query(query_text, top_k=10):
    """Search passages using custom query via reranking"""
    print(f"\n🔍 Searching for: '{query_text}'")
    print(f"   Top {top_k} results:\n")

    # Get all valid passages
    valid_passages = df[df['Passage'].notna()]['Passage'].tolist()
    valid_indices = df[df['Passage'].notna()].index.tolist()

    # Use reranker
    from voyageai import Client
    voyage = Client(api_key=VOYAGE_API_KEY)

    result = voyage.rerank(
        query=query_text,
        documents=valid_passages,
        model="rerank-2.5",
        top_k=top_k
    )

    # Display results
    for i, doc in enumerate(result.results, 1):
        idx = valid_indices[doc.index]
        score = doc.relevance_score
        text = valid_passages[doc.index]
        preview = text[:150] + "..." if len(text) > 150 else text

        # Get labels
        active_labels = [l for l in label_columns if df.loc[idx, l] == 1]
        labels_str = ", ".join(active_labels) if active_labels else "No labels"

        print(f"{i:2d}. Passage {idx} | Score: {score:.3f}")
        print(f"    Labels: {labels_str}")
        print(f"    {preview}")
        print()

# Interactive query search
query_input = widgets.Text(
    value='People seeking help from shamans for illness',
    placeholder='Enter your search query',
    description='Query:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='80%')
)

top_k_slider = widgets.IntSlider(
    value=10,
    min=5,
    max=50,
    step=5,
    description='Results:',
    style={'description_width': 'initial'}
)

search_button = widgets.Button(
    description='Search',
    button_style='primary'
)

def on_search_click(b):
    search_passages_by_query(query_input.value, top_k_slider.value)

search_button.on_click(on_search_click)

display(widgets.VBox([query_input, top_k_slider, search_button]))

#%%

# ============================================================================
# SECTION 7: Score Visualization
# ============================================================================

print("\n" + "="*70)
print("Score Distribution Visualization")
print("="*70)

if cache is not None:
    scores_df = cache['df_summary']

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Consistency distribution
    axes[0, 0].hist(scores_df['consistency_avg'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(scores_df['consistency_avg'].median(), color='red',
                       linestyle='--', label=f'Median: {scores_df["consistency_avg"].median():.3f}')
    axes[0, 0].set_xlabel('Consistency Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Consistency Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Rerank distribution
    axes[0, 1].hist(scores_df['rerank_avg'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].axvline(scores_df['rerank_avg'].median(), color='red',
                       linestyle='--', label=f'Median: {scores_df["rerank_avg"].median():.3f}')
    axes[0, 1].set_xlabel('Rerank Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Rerank Score Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Scatter plot
    axes[1, 0].scatter(scores_df['consistency_avg'], scores_df['rerank_avg'],
                      alpha=0.5, s=20)
    axes[1, 0].set_xlabel('Consistency Score')
    axes[1, 0].set_ylabel('Rerank Score')
    axes[1, 0].set_title('Consistency vs Rerank Scores')
    axes[1, 0].grid(alpha=0.3)

    # 4. Composite score distribution
    composite = (scores_df['consistency_avg'] + scores_df['rerank_avg']) / 2
    axes[1, 1].hist(composite, bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].axvline(composite.median(), color='red',
                       linestyle='--', label=f'Median: {composite.median():.3f}')
    axes[1, 1].set_xlabel('Composite Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Composite Score Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary statistics
    print("\n📊 SCORE STATISTICS:")
    print(f"\nConsistency:")
    print(f"   Min:    {scores_df['consistency_avg'].min():.3f}")
    print(f"   25th:   {scores_df['consistency_avg'].quantile(0.25):.3f}")
    print(f"   Median: {scores_df['consistency_avg'].median():.3f}")
    print(f"   75th:   {scores_df['consistency_avg'].quantile(0.75):.3f}")
    print(f"   Max:    {scores_df['consistency_avg'].max():.3f}")

    print(f"\nRerank:")
    print(f"   Min:    {scores_df['rerank_avg'].min():.3f}")
    print(f"   25th:   {scores_df['rerank_avg'].quantile(0.25):.3f}")
    print(f"   Median: {scores_df['rerank_avg'].median():.3f}")
    print(f"   75th:   {scores_df['rerank_avg'].quantile(0.75):.3f}")
    print(f"   Max:    {scores_df['rerank_avg'].max():.3f}")
else:
    print("⚠️  No cached scores available. Run compute_and_cache_scores() first.")


#%%

# ============================================================================
# SECTION 8: Interactive Threshold Testing
# ============================================================================

print("\n" + "="*70)
print("Interactive Threshold Testing")
print("="*70)

if cache is not None:
    scores_df = cache['df_summary']

    def test_thresholds(min_consistency, min_rerank, show_passages=True, show_labels=True):
        """Test threshold combination and show results"""
        # Filter golden passages
        golden = scores_df[
            (scores_df['consistency_avg'] >= min_consistency) &
            (scores_df['rerank_avg'] >= min_rerank)
        ].copy()

        if len(golden) == 0:
            print(f"❌ No passages meet criteria")
            print(f"   Try lowering thresholds")
            return

        golden['composite'] = (golden['consistency_avg'] + golden['rerank_avg']) / 2
        golden = golden.sort_values('composite', ascending=False)

        # Summary statistics
        print(f"\n{'='*70}")
        print(f"GOLDEN DATASET SUMMARY")
        print(f"{'='*70}\n")
        print(f"✅ Found {len(golden)} passages ({len(golden)/len(scores_df)*100:.1f}% of dataset)")
        print(f"\n📊 Quality Metrics:")
        print(f"   Mean consistency: {golden['consistency_avg'].mean():.3f}")
        print(f"   Mean rerank:      {golden['rerank_avg'].mean():.3f}")
        print(f"   Mean composite:   {golden['composite'].mean():.3f}")

        if show_labels:
            # Label distribution
            print(f"\n🏷️  Label Distribution:")
            golden_indices = golden['passage_idx'].tolist()

            for label in label_columns[:12]:  # Show top 12
                golden_count = sum(df.loc[idx, label] == 1 for idx in golden_indices if idx in df.index)
                total_count = df[label].sum()
                pct = (golden_count / total_count * 100) if total_count > 0 else 0
                print(f"   {label:30s}: {golden_count:3d}/{int(total_count):3d} ({pct:5.1f}%)")

        if show_passages:
            # Top passages
            print(f"\n⭐ Top 5 Most Confident Passages:")
            for i, (_, row) in enumerate(golden.head(5).iterrows(), 1):
                idx = int(row['passage_idx'])
                passage_text = df.loc[idx, 'Passage']
                preview = str(passage_text)[:100] + "..." if len(str(passage_text)) > 100 else str(passage_text)

                active_labels = [l for l in label_columns if df.loc[idx, l] == 1]

                print(f"\n   {i}. Passage {idx}")
                print(f"      Composite: {row['composite']:.3f} | "
                      f"Consistency: {row['consistency_avg']:.3f} | "
                      f"Rerank: {row['rerank_avg']:.3f}")
                print(f"      Labels: {', '.join(active_labels)}")
                print(f"      {preview}")

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Score distributions
        ax1.scatter(scores_df['consistency_avg'], scores_df['rerank_avg'],
                   alpha=0.3, s=20, label='All passages', color='gray')
        ax1.scatter(golden['consistency_avg'], golden['rerank_avg'],
                   alpha=0.7, s=30, label='Golden set', color='gold')
        ax1.axvline(min_consistency, color='red', linestyle='--', alpha=0.7)
        ax1.axhline(min_rerank, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Consistency Score')
        ax1.set_ylabel('Rerank Score')
        ax1.set_title('Golden Dataset Selection')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Composite score histogram
        all_composite = (scores_df['consistency_avg'] + scores_df['rerank_avg']) / 2
        ax2.hist(all_composite, bins=50, alpha=0.5, label='All passages', edgecolor='black')
        ax2.hist(golden['composite'], bins=30, alpha=0.7, label='Golden set',
                color='gold', edgecolor='black')
        ax2.axvline(golden['composite'].mean(), color='red', linestyle='--',
                   label=f'Golden mean: {golden["composite"].mean():.3f}')
        ax2.set_xlabel('Composite Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Composite Score Distribution')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        return golden

    # Interactive widgets
    consistency_slider = widgets.FloatSlider(
        value=0.55,
        min=0.0,
        max=1.0,
        step=0.05,
        description='Min Consistency:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )

    rerank_slider = widgets.FloatSlider(
        value=0.50,
        min=0.0,
        max=1.0,
        step=0.05,
        description='Min Rerank:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )

    show_passages_check = widgets.Checkbox(
        value=True,
        description='Show top passages',
        style={'description_width': 'initial'}
    )

    show_labels_check = widgets.Checkbox(
        value=True,
        description='Show label distribution',
        style={'description_width': 'initial'}
    )

    test_button = widgets.Button(
        description='Test Thresholds',
        button_style='success'
    )

    save_button = widgets.Button(
        description='Save Golden Set',
        button_style='primary'
    )

    current_golden = None

    def on_test_click(b):
        global current_golden
        current_golden = test_thresholds(
            consistency_slider.value,
            rerank_slider.value,
            show_passages_check.value,
            show_labels_check.value
        )

    def on_save_click(b):
        if current_golden is None or len(current_golden) == 0:
            print("❌ No golden dataset to save. Test thresholds first.")
            return

        # Merge with original data
        golden_indices = current_golden['passage_idx'].tolist()
        golden_full = df.loc[golden_indices].copy()

        # Add confidence scores
        for idx in golden_indices:
            score_row = current_golden[current_golden['passage_idx'] == idx].iloc[0]
            golden_full.loc[idx, 'confidence_consistency'] = score_row['consistency_avg']
            golden_full.loc[idx, 'confidence_rerank'] = score_row['rerank_avg']
            golden_full.loc[idx, 'confidence_composite'] = score_row['composite']

        filename = f"golden_set_{consistency_slider.value:.2f}_{rerank_slider.value:.2f}.xlsx"
        golden_full.to_excel(filename, index=False)
        print(f"\n✅ Saved {len(golden_full)} passages to {filename}")

    test_button.on_click(on_test_click)
    save_button.on_click(on_save_click)

    display(widgets.VBox([
        consistency_slider,
        rerank_slider,
        show_passages_check,
        show_labels_check,
        widgets.HBox([test_button, save_button])
    ]))
else:
    print("⚠️  No cached scores available. Run compute_and_cache_scores() first.")


#%%

# ============================================================================
# SECTION 9: Quick Functions
# ============================================================================

print("\n" + "="*70)
print("Quick Access Functions")
print("="*70)

def quick_summary():
    """Print quick summary of dataset and scores"""
    print(f"\n📊 DATASET SUMMARY")
    print(f"   Total passages: {len(df)}")
    print(f"   Valid passages: {df['Passage'].notna().sum()}")
    print(f"   Labels: {len(label_columns)}")

    if cache is not None:
        scores_df = cache['df_summary']
        print(f"\n📈 SCORE SUMMARY")
        print(f"   Passages with scores: {len(scores_df)}")
        print(f"   Avg consistency: {scores_df['consistency_avg'].mean():.3f}")
        print(f"   Avg rerank: {scores_df['rerank_avg'].mean():.3f}")

def export_golden_percentile(percentile=90):
    """Export top N percentile of passages as golden set"""
    if cache is None:
        print("❌ No cached scores available")
        return

    scores_df = cache['df_summary']
    scores_df['composite'] = (scores_df['consistency_avg'] + scores_df['rerank_avg']) / 2

    threshold = np.percentile(scores_df['composite'], percentile)
    golden = scores_df[scores_df['composite'] >= threshold]

    golden_indices = golden['passage_idx'].tolist()
    golden_full = df.loc[golden_indices].copy()

    for idx in golden_indices:
        score_row = golden[golden['passage_idx'] == idx].iloc[0]
        golden_full.loc[idx, 'confidence_composite'] = score_row['composite']

    filename = f"golden_set_top{100-percentile}pct.xlsx"
    golden_full.to_excel(filename, index=False)
    print(f"✅ Exported top {100-percentile}% ({len(golden_full)} passages) to {filename}")

print("\nAvailable functions:")
print("   • quick_summary() - Dataset and score overview")
print("   • export_golden_percentile(90) - Export top 10% as golden set")
print("   • explore_passage(idx) - Detailed passage examination")
print("   • search_passages_by_query('text', 10) - Custom query search")
print("   • compute_and_cache_scores() - Compute all scores (costs $3-5)")

print("\n" + "="*70)
print("✅ Notebook ready! Use the interactive widgets above to explore.")
print("="*70)