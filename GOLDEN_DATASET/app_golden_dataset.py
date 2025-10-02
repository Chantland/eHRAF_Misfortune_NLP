"""
HRAF Golden Dataset Discovery - With Directory Navigator
Run with: streamlit run app_golden_dataset.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from pathlib import Path
from datetime import datetime
import io
from tqdm import tqdm

from discovery_architecture import GoldenDatasetFinder
from dotenv import load_dotenv

# Page config
st.set_page_config(
    page_title="HRAF Golden Dataset Discovery",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.df = None
    st.session_state.finder = None
    st.session_state.label_columns = None
    st.session_state.cache = None
    st.session_state.golden_dataset = None
    st.session_state.tier1_dataset = None
    st.session_state.tier2_dataset = None
    st.session_state.inference_dataset = None
    st.session_state.passage_col = None
    st.session_state.selected_file = None
    st.session_state.namespace = None
    st.session_state.current_directory = Path.cwd()
    st.session_state.browse_mode = "quick"

# Configuration
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "hraf-misfortune-test"
REGION = "us-east-1"

# Directory structure
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cached_scores"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Functions
def get_namespace_from_filename(filepath):
    """Generate a clean namespace name from filepath"""
    filename = Path(filepath).stem
    namespace = filename.lower()
    namespace = ''.join(c if c.isalnum() or c == '_' else '_' for c in namespace)
    namespace = namespace[:63]
    return namespace


def get_xlsx_files_in_directory(directory: Path):
    """Get all .xlsx files in specified directory"""
    if not directory.exists() or not directory.is_dir():
        return []

    try:
        xlsx_files = list(directory.glob('*.xlsx'))
        return [f for f in xlsx_files if not f.name.startswith('~') and not f.name.startswith('.')]
    except PermissionError:
        return []


def get_subdirectories(directory: Path):
    """Get all subdirectories in specified directory"""
    if not directory.exists() or not directory.is_dir():
        return []

    try:
        subdirs = [d for d in directory.iterdir() if d.is_dir() and not d.name.startswith('.')]
        return sorted(subdirs, key=lambda x: x.name.lower())
    except PermissionError:
        return []


def render_directory_browser():
    """Render directory browser in sidebar"""
    current_dir = st.session_state.current_directory

    st.markdown("**Current Directory:**")
    path_parts = list(current_dir.parts)

    breadcrumb_cols = st.columns([1, 3])

    with breadcrumb_cols[0]:
        if st.button("⬆️ Parent", key="go_up", disabled=current_dir == current_dir.parent):
            st.session_state.current_directory = current_dir.parent
            st.rerun()

    with breadcrumb_cols[1]:
        st.text_input("Path:", str(current_dir), key="path_display", disabled=True, label_visibility="collapsed")

    quick_nav_col1, quick_nav_col2 = st.columns(2)
    with quick_nav_col1:
        if st.button("🏠 Home", key="go_home"):
            st.session_state.current_directory = Path.home()
            st.rerun()
    with quick_nav_col2:
        if st.button("📂 data/", key="go_data"):
            st.session_state.current_directory = DATA_DIR
            st.rerun()

    st.markdown("---")

    subdirs = get_subdirectories(current_dir)
    if subdirs:
        st.markdown("**📁 Folders:**")
        for subdir in subdirs[:10]:
            if st.button(f"📁 {subdir.name}", key=f"dir_{subdir}"):
                st.session_state.current_directory = subdir
                st.rerun()

        if len(subdirs) > 10:
            st.caption(f"... and {len(subdirs) - 10} more folders")

    st.markdown("---")

    xlsx_files = get_xlsx_files_in_directory(current_dir)

    if xlsx_files:
        st.markdown("**📊 Excel Files:**")
        file_options = {f.name: str(f) for f in xlsx_files}
        selected_name = st.selectbox(
            "Select file:",
            options=list(file_options.keys()),
            key="file_selector",
            label_visibility="collapsed"
        )
        selected_file = file_options[selected_name]
        return selected_file
    else:
        st.info("No .xlsx files in this directory")
        return None


def get_cache_filename(xlsx_file):
    """Generate cache filename in cached_scores directory"""
    xlsx_path = Path(xlsx_file)
    cache_name = xlsx_path.stem + '_cached_scores.pkl'
    return str(CACHE_DIR / cache_name)


def detect_passage_column(df):
    """Auto-detect which column contains passage text"""
    possible_names = ['Passage', 'passage', 'Text', 'text', 'Content', 'content']

    df.columns = [str(col).strip() for col in df.columns]

    for name in possible_names:
        if name in df.columns:
            return name

    for name in possible_names:
        for col in df.columns:
            if col.lower() == name.lower():
                return col

    for col in df.columns:
        try:
            if df[col].dtype == 'object':
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    avg_length = non_null.astype(str).str.len().mean()
                    if avg_length > 100:
                        return col
        except:
            continue

    return None


def load_data(filename, header_row=1):
    """Load Excel data"""
    try:
        df = pd.read_excel(filename, header=header_row)
        st.info(f"📋 Header row {header_row}: {list(df.columns)[:5]}...")

        passage_col = detect_passage_column(df)
        if not passage_col:
            st.error("❌ Could not find passage column")
            st.warning(f"Columns: {', '.join([str(c) for c in df.columns[:10]])}")
            return None, None, None, None, None, None

        finder = GoldenDatasetFinder(
            voyage_api_key=VOYAGE_API_KEY,
            pinecone_api_key=PINECONE_API_KEY,
            index_name=INDEX_NAME,
            region=REGION
        )

        label_columns = finder._auto_detect_label_columns(df)

        namespace = get_namespace_from_filename(filename)
        st.info(f"📦 Using namespace: '{namespace}'")

        cache_file = get_cache_filename(filename)
        cache = None
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            st.success(f"✅ Loaded cached scores from: {Path(cache_file).name}")

        return df, finder, label_columns, cache, passage_col, namespace

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None, None, None


def compute_scores_for_dataset(df, finder, label_columns, passage_col, namespace, k_similar=15):
    """Compute consistency and rerank scores for all passages"""
    import time

    valid_mask = df[passage_col].notna()
    valid_df = df[valid_mask]
    embedded_indices = valid_df.index.tolist()

    st.info(f"📊 Computing scores for {len(embedded_indices)} passages...")

    st.write("### Step 1: Checking Embeddings in Pinecone")

    try:
        test_fetch = finder.index.fetch(ids=[f"passage_0"], namespace=namespace)
        vectors_dict = finder._get_vectors_from_fetch(test_fetch)
        has_embeddings = len(vectors_dict) > 0
    except:
        has_embeddings = False

    if not has_embeddings:
        st.warning("⚠️ No embeddings found in Pinecone. Creating embeddings first...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        batch_size = 16
        total_batches = (len(valid_df) + batch_size - 1) // batch_size

        for i in range(0, len(valid_df), batch_size):
            batch_df = valid_df.iloc[i:i + batch_size]
            batch_texts = batch_df[passage_col].tolist()
            batch_texts = [str(text) if pd.notna(text) else "" for text in batch_texts]

            if not any(batch_texts):
                continue

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = finder.voyage.embed(
                        texts=batch_texts,
                        model="voyage-3-large",
                        input_type="document"
                    )
                    embeddings = result.embeddings
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        status_text.warning(f"⚠️ Retry {attempt+1}/{max_retries} after error: {str(e)[:100]}")
                        time.sleep(wait_time)
                    else:
                        st.error(f"❌ Failed after {max_retries} attempts on batch {i//batch_size + 1}")
                        st.error(f"Error: {str(e)}")
                        raise

            vectors = []
            for j, embedding in enumerate(embeddings):
                original_idx = valid_df.index[i + j]
                text = batch_texts[j]
                passage_id = f"passage_{original_idx}"

                metadata = {
                    'text_preview': text[:1000],
                    'passage_idx': int(original_idx),
                    'text_length': len(text)
                }

                for label in label_columns:
                    if label in batch_df.columns:
                        val = batch_df.iloc[j][label]
                        metadata[f"label_{label}"] = int(val) if pd.notna(val) else 0

                vectors.append({
                    'id': passage_id,
                    'values': embedding,
                    'metadata': metadata
                })

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    finder.index.upsert(vectors=vectors, namespace=namespace)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        st.error(f"❌ Failed to upsert batch {i//batch_size + 1}")
                        raise

            progress = (i + batch_size) / len(valid_df)
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Embedding batch {(i//batch_size)+1}/{total_batches}...")

            time.sleep(0.5)

        st.success("✅ Embeddings created and stored in Pinecone!")
    else:
        st.success("✅ Embeddings already exist in Pinecone")

    st.write("### Step 2: Calculating Consistency Scores")
    consistency_scores = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx_num, idx in enumerate(embedded_indices):
        try:
            similar = finder.find_similar_passages(idx, k=k_similar, namespace=namespace)
            consistency = finder.calculate_label_consistency(
                idx, similar, label_columns, namespace=namespace
            )
            active_labels = [l for l in label_columns if df.loc[idx, l] == 1]
            if active_labels:
                consistency_scores[idx] = {
                    'avg': np.mean([consistency[l] for l in active_labels]),
                    'by_label': {l: consistency[l] for l in active_labels}
                }
        except Exception as e:
            st.warning(f"Error on passage {idx}: {e}")
            consistency_scores[idx] = {'avg': 0.0, 'by_label': {}}

        if (idx_num + 1) % 10 == 0:
            progress = (idx_num + 1) / len(embedded_indices)
            progress_bar.progress(progress)
            status_text.text(f"Consistency: {idx_num + 1}/{len(embedded_indices)} passages...")

    progress_bar.progress(1.0)
    st.success(f"✅ Calculated consistency for {len(consistency_scores)} passages")

    st.write("### Step 3: Calculating Rerank Scores")
    rerank_scores = {label: {} for label in label_columns}
    passages = df[passage_col].tolist()

    progress_bar = st.progress(0)
    status_text = st.empty()

    for label_num, label in enumerate(label_columns):
        label_indices = [idx for idx in embedded_indices if df.loc[idx, label] == 1]
        if not label_indices:
            continue

        label_passages = [passages[idx] for idx in label_indices]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                scores = finder.rerank_passages_for_label(label_passages, label)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    status_text.warning(f"⚠️ Retry {attempt+1}/{max_retries} for label {label}")
                    time.sleep(wait_time)
                else:
                    st.error(f"❌ Failed reranking for label: {label}")
                    st.error(f"Error: {str(e)}")
                    scores = [0.0] * len(label_passages)

        for idx, score in zip(label_indices, scores):
            rerank_scores[label][idx] = score

        progress = (label_num + 1) / len(label_columns)
        progress_bar.progress(progress)
        status_text.text(f"Reranking: {label} ({label_num + 1}/{len(label_columns)})...")

        time.sleep(0.3)

    progress_bar.progress(1.0)
    st.success(f"✅ Calculated rerank scores for {len(label_columns)} labels")

    st.write("### Step 4: Creating Summary")

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
        'embedded_indices': embedded_indices,
        'computed_date': datetime.now().isoformat(),
        'namespace': namespace
    }

    st.success(f"✅ Score computation complete! Processed {len(scores_df)} passages")

    return cache


def create_tiered_datasets(df, scores_df, label_columns, tier1_cons, tier1_rerank, tier1_size_pct=20, tier2_size_pct=30):
    """Create 3 tiers"""
    valid_indices = scores_df['passage_idx'].tolist()
    scores_df = scores_df.copy()
    scores_df['composite'] = (scores_df['consistency_avg'] + scores_df['rerank_avg']) / 2

    tier1_mask = (scores_df['consistency_avg'] >= tier1_cons) & (scores_df['rerank_avg'] >= tier1_rerank)
    tier1_indices = scores_df[tier1_mask]['passage_idx'].tolist()

    total = len(valid_indices)
    tier1_target = int(total * tier1_size_pct / 100)
    tier2_target = int(total * tier2_size_pct / 100)

    if len(tier1_indices) < tier1_target:
        scores_sorted = scores_df.sort_values('composite', ascending=False)
        tier1_indices = scores_sorted.head(tier1_target)['passage_idx'].tolist()
    elif len(tier1_indices) > tier1_target:
        tier1_scores = scores_df[scores_df['passage_idx'].isin(tier1_indices)]
        tier1_scores = tier1_scores.sort_values('composite', ascending=False)
        tier1_indices = tier1_scores.head(tier1_target)['passage_idx'].tolist()

    remaining_indices = [idx for idx in valid_indices if idx not in tier1_indices]
    remaining_scores = scores_df[scores_df['passage_idx'].isin(remaining_indices)]

    tier2_scored_count = int(tier2_target * 0.7)
    tier2_random_count = tier2_target - tier2_scored_count

    tier2_scored = remaining_scores.sort_values('composite', ascending=False).head(tier2_scored_count)
    tier2_scored_indices = tier2_scored['passage_idx'].tolist()

    tier2_pool = [idx for idx in remaining_indices if idx not in tier2_scored_indices]
    tier2_random_indices = np.random.choice(tier2_pool, size=min(tier2_random_count, len(tier2_pool)), replace=False).tolist()

    tier2_indices = tier2_scored_indices + tier2_random_indices
    inference_indices = [idx for idx in remaining_indices if idx not in tier2_indices]

    tier1_df = df.loc[tier1_indices].copy()
    tier2_df = df.loc[tier2_indices].copy()
    inference_df = df.loc[inference_indices].copy()

    for idx in tier1_indices:
        if idx in scores_df['passage_idx'].values:
            score_row = scores_df[scores_df['passage_idx'] == idx].iloc[0]
            tier1_df.loc[idx, 'confidence_composite'] = score_row['composite']

    for idx in tier2_indices:
        if idx in scores_df['passage_idx'].values:
            score_row = scores_df[scores_df['passage_idx'] == idx].iloc[0]
            tier2_df.loc[idx, 'confidence_composite'] = score_row['composite']

    for idx in inference_indices:
        if idx in scores_df['passage_idx'].values:
            score_row = scores_df[scores_df['passage_idx'] == idx].iloc[0]
            inference_df.loc[idx, 'confidence_composite'] = score_row['composite']

    return tier1_df, tier2_df, inference_df


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🔍 HRAF Golden Dataset")
    st.markdown("---")

    st.markdown("### 📂 File Selection")

    browse_mode = st.radio(
        "Selection mode:",
        ["Quick (data/ folder)", "Browse directories"],
        key="browse_mode_selector",
        horizontal=True
    )

    st.session_state.browse_mode = "browse" if "Browse" in browse_mode else "quick"

    st.markdown("---")

    if st.session_state.browse_mode == "quick":
        xlsx_files = get_xlsx_files_in_directory(DATA_DIR)

        if not xlsx_files:
            st.error("No .xlsx files in `data/` folder!")
            st.info("Switch to 'Browse directories' or add files to `data/`")
            selected_file = None
        else:
            file_options = {f.name: str(f) for f in xlsx_files}
            selected_name = st.selectbox(
                "Select file:",
                options=list(file_options.keys()),
                key="quick_file_selector"
            )
            selected_file = file_options[selected_name]
    else:
        selected_file = render_directory_browser()

    if selected_file:
        st.markdown("---")
        cache_file = get_cache_filename(selected_file)
        if os.path.exists(cache_file):
            cache_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            st.success(f"✅ Scores cached")
            st.caption(f"Updated: {cache_mod_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.warning("⚠️ No cached scores")
            st.caption("Use 'Compute Scores' to generate")

    st.markdown("---")

    if selected_file:
        header_row = st.number_input("Header row:", min_value=0, max_value=5, value=1,
                                     help="0=first row, 1=second row")

        with st.expander("⚙️ Manual Column"):
            manual_col = st.text_input("Passage column:", placeholder="e.g., Passage")

        if st.button("Load Data", type="primary"):
            with st.spinner("Loading..."):
                df, finder, label_columns, cache, passage_col, namespace = load_data(selected_file, header_row)

                if df is not None and passage_col is None and manual_col:
                    if manual_col in df.columns:
                        passage_col = manual_col
                        st.success(f"✅ Using: '{passage_col}'")

                if df is not None and passage_col is not None:
                    st.session_state.df = df
                    st.session_state.finder = finder
                    st.session_state.label_columns = label_columns
                    st.session_state.cache = cache
                    st.session_state.passage_col = passage_col
                    st.session_state.selected_file = selected_file
                    st.session_state.namespace = namespace
                    st.session_state.initialized = True
                    st.success("✅ Loaded!")
                    st.info(f"📄 Column: '{passage_col}'")
    else:
        st.info("👆 Select a file first")

    st.markdown("---")

    if st.session_state.initialized:
        st.markdown("### ✅ Loaded")
        st.caption(f"📄 {Path(st.session_state.selected_file).name}")
        st.caption(f"📦 {st.session_state.get('namespace', 'N/A')}")
    else:
        st.info("👆 Load data to begin")


# ============================================================================
# MAIN CONTENT
# ============================================================================

if not st.session_state.initialized:
    st.markdown("# 🔍 HRAF Golden Dataset Discovery")
    st.markdown("""
    ### Welcome!
    
    This tool identifies high-quality passages for NLP training.
    
    **Features:**
    - **Browse and select datasets** from any directory
    - **Compute quality scores** for your passages
    - Analyze passage quality using consistency + rerank metrics
    - Create tiered datasets (Golden/Training/Inference)
    - Search and explore passages
    - Export results
    
    **Get Started:**
    1. Select "Quick" mode to use files in `data/` folder
    2. Or select "Browse directories" to navigate your filesystem
    3. Select your Excel file
    4. Click "Load Data"
    5. Go to "Compute Scores" if you don't have cached scores
    
    👈 **Start in the sidebar!**
    """)

else:
    # ============================================================================
    # DATA LOADED - Show navigation and page content
    # ============================================================================

    st.markdown("# 🔍 HRAF Golden Dataset Discovery")

    page = st.radio(
        "Navigate:",
        ["📊 Overview", "💻 Compute Scores", "🔍 Search", "⚙️ Thresholds", "📦 Tiers", "💾 Export"],
        horizontal=True,
        label_visibility="visible"
    )

    st.markdown("---")

    # ============================================================================
    # PAGE CONTENT
    # ============================================================================

    if page == "📊 Overview":
        st.markdown("## 📊 Dataset Overview")

        df = st.session_state.df
        cache = st.session_state.cache
        label_columns = st.session_state.label_columns
        passage_col = st.session_state.get('passage_col', 'Passage')
        namespace = st.session_state.get('namespace', 'unknown')

        st.info(f"**File:** {Path(st.session_state.selected_file).name}  |  **Namespace:** `{namespace}`")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", len(df))
        with col2:
            st.metric("Valid", df[passage_col].notna().sum())
        with col3:
            st.metric("Labels", len(label_columns))
        with col4:
            if cache:
                st.metric("With Scores", len(cache['df_summary']))
            else:
                st.metric("With Scores", "N/A")

        if cache:
            st.markdown("### 📈 Score Statistics")
            scores_df = cache['df_summary']

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Consistency**")
                st.write(f"Min: {scores_df['consistency_avg'].min():.3f}")
                st.write(f"25th percentile: {scores_df['consistency_avg'].quantile(0.25):.3f}")
                st.write(f"Median: {scores_df['consistency_avg'].median():.3f}")
                st.write(f"Mean: {scores_df['consistency_avg'].mean():.3f}")
                st.write(f"75th percentile: {scores_df['consistency_avg'].quantile(0.75):.3f}")
                st.write(f"Max: {scores_df['consistency_avg'].max():.3f}")

                st.markdown("**Passages above threshold:**")
                for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    count = (scores_df['consistency_avg'] >= thresh).sum()
                    pct = count / len(scores_df) * 100
                    st.write(f"  ≥ {thresh}: {count} ({pct:.1f}%)")

            with col2:
                st.markdown("**Rerank**")
                st.write(f"Min: {scores_df['rerank_avg'].min():.3f}")
                st.write(f"25th percentile: {scores_df['rerank_avg'].quantile(0.25):.3f}")
                st.write(f"Median: {scores_df['rerank_avg'].median():.3f}")
                st.write(f"Mean: {scores_df['rerank_avg'].mean():.3f}")
                st.write(f"75th percentile: {scores_df['rerank_avg'].quantile(0.75):.3f}")
                st.write(f"Max: {scores_df['rerank_avg'].max():.3f}")

                st.markdown("**Passages above threshold:**")
                for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    count = (scores_df['rerank_avg'] >= thresh).sum()
                    pct = count / len(scores_df) * 100
                    st.write(f"  ≥ {thresh}: {count} ({pct:.1f}%)")

            if scores_df['consistency_avg'].median() < 0.4:
                st.warning("""
                ⚠️ **Low Consistency Detected**
                
                Your consistency scores are low (median < 0.4), which suggests:
                - Similar passages have different labels
                - High inter-rater disagreement in original labeling
                - Inconsistent coding between RAs
                
                **Recommendation:** 
                - Use **Rerank scores** more heavily (they're more reliable)
                - Or use **Composite score only** (average of both)
                - Lower your consistency threshold to 0.2-0.4 instead of 0.5+
                """)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.hist(scores_df['consistency_avg'], bins=50, edgecolor='black', alpha=0.7)
            ax1.axvline(scores_df['consistency_avg'].median(), color='red', linestyle='--',
                       label=f'Median: {scores_df["consistency_avg"].median():.3f}')
            ax1.axvline(0.5, color='orange', linestyle=':', label='Typical threshold (0.5)')
            ax1.set_xlabel('Consistency Score')
            ax1.set_title('Consistency Distribution')
            ax1.legend()
            ax1.grid(alpha=0.3)

            ax2.hist(scores_df['rerank_avg'], bins=50, edgecolor='black', alpha=0.7, color='green')
            ax2.axvline(scores_df['rerank_avg'].median(), color='red', linestyle='--',
                       label=f'Median: {scores_df["rerank_avg"].median():.3f}')
            ax2.axvline(0.5, color='orange', linestyle=':', label='Typical threshold (0.5)')
            ax2.set_xlabel('Rerank Score')
            ax2.set_title('Rerank Distribution')
            ax2.legend()
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("### 🏷️ Label Distribution")
            label_stats = []
            for label in label_columns:
                count = df[label].sum()
                pct = (count / len(df)) * 100
                label_stats.append({
                    'Label': label,
                    'Count': int(count),
                    'Percentage': f"{pct:.1f}%"
                })
            label_stats_df = pd.DataFrame(label_stats)
            st.dataframe(label_stats_df, width='stretch')
        else:
            st.warning("⚠️ No cached scores available. Go to 'Compute Scores' to generate them.")

    elif page == "💻 Compute Scores":
        st.markdown("## 💻 Compute Quality Scores")

        df = st.session_state.df
        finder = st.session_state.finder
        label_columns = st.session_state.label_columns
        passage_col = st.session_state.get('passage_col', 'Passage')
        selected_file = st.session_state.selected_file
        namespace = st.session_state.get('namespace', 'default')
        cache_file = get_cache_filename(selected_file)

        st.info(f"**File:** {Path(selected_file).name}  |  **Namespace:** `{namespace}`")

        st.markdown("""
        ### What This Does
        
        Computes two quality scores for each passage:
        
        1. **Consistency Score** - Agreement with similar passages (0-1)
           - Uses vector similarity to find 15 similar passages
           - Measures label agreement across similar passages
           
        2. **Rerank Score** - Relevance to label definition (0-1)
           - Uses VoyageAI reranker to score passage-label fit
           - Higher score = passage clearly demonstrates the label
        
        ### Output Files
        
        Results will be saved to:
        - **Cache:** `{}`
        - **Excel:** `{}`
        
        **Note:** Embeddings are stored in Pinecone namespace `{}` for data isolation.
        """.format(cache_file, cache_file.replace('.pkl', '.xlsx'), namespace))

        num_passages = df[passage_col].notna().sum()
        num_labels_to_rerank = sum(df[col].sum() for col in label_columns)

        avg_passage_length = df[passage_col].dropna().astype(str).str.len().mean()
        est_tokens = (num_passages * avg_passage_length) / 4
        embedding_cost = (est_tokens / 1_000_000) * 0.10

        rerank_cost = (num_labels_to_rerank * 200 / 1000) * 0.00005

        total_cost = embedding_cost + rerank_cost

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Passages", num_passages)
        with col2:
            st.metric("Labels", len(label_columns))
        with col3:
            st.metric("Est. Cost", f"${total_cost:.2f}")

        st.info(f"💡 Estimated time: {int(num_passages / 200)} - {int(num_passages / 150)} minutes")

        if os.path.exists(cache_file):
            cache_date = datetime.fromtimestamp(os.path.getmtime(cache_file))
            st.warning(f"⚠️ Cached scores already exist (computed: {cache_date.strftime('%Y-%m-%d %H:%M')})")
            st.write("You can recompute to update scores with current data.")

        with st.expander("⚙️ Advanced Settings"):
            k_similar = st.slider("Number of similar passages to check:", 5, 30, 15,
                                 help="More passages = more reliable consistency score but slower")

        col1, col2 = st.columns([1, 3])
        with col1:
            compute_button = st.button("🚀 Compute Scores", type="primary")
        with col2:
            if os.path.exists(cache_file):
                if st.button("🔄 Recompute (Overwrite)", type="secondary"):
                    compute_button = True

        if compute_button:
            try:
                st.markdown("---")

                with st.spinner("Computing scores..."):
                    cache = compute_scores_for_dataset(
                        df, finder, label_columns, passage_col,
                        namespace, k_similar
                    )

                st.write("### Step 5: Saving Cache")
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache, f)

                excel_name = Path(cache_file).stem + '.xlsx'
                excel_file = CACHE_DIR / excel_name
                cache['df_summary'].to_excel(excel_file, index=False)

                st.success(f"✅ Saved cache to: {cache_file}")
                st.success(f"✅ Saved Excel summary to: {excel_file}")

                st.session_state.cache = cache

                st.markdown("---")
                st.markdown("### 📊 Score Summary")
                scores_df = cache['df_summary']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Passages Scored", len(scores_df))
                with col2:
                    st.metric("Avg Consistency", f"{scores_df['consistency_avg'].mean():.3f}")
                with col3:
                    st.metric("Avg Rerank", f"{scores_df['rerank_avg'].mean():.3f}")

                st.success("✅ Score computation complete! Go to 'Overview' or 'Thresholds' to see results.")

            except Exception as e:
                st.error(f"❌ Error during computation: {e}")
                import traceback
                with st.expander("📋 Full Error Details"):
                    st.code(traceback.format_exc())

                st.markdown("---")
                st.markdown("### 🔧 Troubleshooting")
                st.markdown("""
                **Common issues:**
                
                1. **Network Timeout / Connection Error**
                   - Your internet connection was interrupted
                   - VoyageAI servers are experiencing issues
                   - **Solution:** Click "🔄 Recompute" to retry
                
                2. **Rate Limit**
                   - Too many requests to VoyageAI
                   - **Solution:** Wait 1-2 minutes, then retry
                
                3. **Out of API Credits**
                   - Check your VoyageAI account
                   - **Solution:** Add credits and retry
                
                **Your progress is saved!**
                - Embeddings already in Pinecone won't be recomputed
                - Click "🔄 Recompute" to continue from where it failed
                """)

                try:
                    test_fetch = finder.index.fetch(ids=[f"passage_0"], namespace=namespace)
                    vectors_dict = finder._get_vectors_from_fetch(test_fetch)
                    if len(vectors_dict) > 0:
                        st.success("✅ Step 1 (Embeddings) is complete - this won't be redone")
                except:
                    st.warning("⚠️ Step 1 (Embeddings) may need to be completed")

    elif page == "🔍 Search":
        st.markdown("## 🔍 Search Passages")

        df = st.session_state.df
        passage_col = st.session_state.get('passage_col', 'Passage')
        label_columns = st.session_state.label_columns
        cache = st.session_state.cache

        query = st.text_input("Search query:", placeholder="e.g., shamans healing illness")
        top_k = st.number_input("Results:", 5, 50, 10)

        if st.button("Search", type="primary"):
            if query:
                with st.spinner("Searching..."):
                    from voyageai import Client
                    voyage = Client(api_key=VOYAGE_API_KEY)

                    valid_mask = df[passage_col].notna()
                    valid_passages = df[valid_mask][passage_col].tolist()
                    valid_indices = df[valid_mask].index.tolist()

                    result = voyage.rerank(query=query, documents=valid_passages, model="rerank-2.5", top_k=top_k)

                    st.markdown(f"### Top {len(result.results)} Results")

                    for i, doc in enumerate(result.results, 1):
                        idx = valid_indices[doc.index]
                        score = doc.relevance_score
                        text = valid_passages[doc.index]

                        active_labels = [l for l in label_columns if df.loc[idx, l] == 1]

                        confidence_str = ""
                        if cache:
                            scores_df = cache['df_summary']
                            if idx in scores_df['passage_idx'].values:
                                score_row = scores_df[scores_df['passage_idx'] == idx].iloc[0]
                                confidence_str = f" | Confidence: {(score_row['consistency_avg'] + score_row['rerank_avg'])/2:.3f}"

                        with st.expander(f"#{i} - Passage {idx} | Relevance: {score:.3f}{confidence_str}"):
                            st.markdown(f"**Labels:** {', '.join(active_labels) if active_labels else 'None'}")
                            st.markdown("---")
                            st.write(text)

    elif page == "⚙️ Thresholds":
        st.markdown("## ⚙️ Configure Thresholds")

        df = st.session_state.df
        cache = st.session_state.cache
        label_columns = st.session_state.label_columns
        passage_col = st.session_state.get('passage_col', 'Passage')

        if not cache:
            st.error("⚠️ No scores available. Go to 'Compute Scores' to generate them.")
            st.stop()

        scores_df = cache['df_summary']

        with st.expander("📖 How to Choose Thresholds", expanded=False):
            st.markdown("""
            **Goal:** Balance quality vs. quantity
            
            **For High-Quality Data (consistency > 0.5):**
            - Set both thresholds at median or higher
            - Use composite score (average of both)
            
            **For Noisy Data (consistency < 0.4):**
            - **Option 1:** Use rerank score only (more reliable)
            - **Option 2:** Use composite but lower consistency threshold
            - **Option 3:** Weight rerank more heavily (70/30)
            
            **Your data quality:**
            - Consistency median: {:.3f}
            - Rerank median: {:.3f}
            """.format(scores_df['consistency_avg'].median(), scores_df['rerank_avg'].median()))

        st.markdown("### 🎯 Scoring Strategy")

        strategy = st.radio(
            "How to calculate quality score:",
            [
                "Composite (50/50 average)",
                "Rerank Only (ignore consistency)",
                "Weighted (70% rerank, 30% consistency)",
                "Custom weighting"
            ],
            help="Choose based on your data quality"
        )

        if strategy == "Custom weighting":
            rerank_weight = st.slider("Rerank weight:", 0.0, 1.0, 0.7, 0.05)
            consistency_weight = 1.0 - rerank_weight
            st.caption(f"Consistency weight: {consistency_weight:.2f}")

        st.markdown("---")

        if strategy == "Rerank Only (ignore consistency)":
            st.markdown("### 🎚️ Adjust Threshold")
            min_rerank = st.slider("Min Rerank Score", 0.0, 1.0,
                                  float(scores_df['rerank_avg'].quantile(0.3)), 0.05)
            st.caption(f"Percentile: {(scores_df['rerank_avg'] <= min_rerank).mean()*100:.1f}%")

            min_cons = 0.0

            golden = scores_df[scores_df['rerank_avg'] >= min_rerank].copy()
            golden['composite'] = golden['rerank_avg']

        else:
            st.markdown("### 🎚️ Adjust Thresholds")
            col1, col2 = st.columns(2)

            default_cons = max(0.3, float(scores_df['consistency_avg'].quantile(0.25)))
            default_rerank = float(scores_df['rerank_avg'].quantile(0.4))

            with col1:
                min_cons = st.slider("Min Consistency", 0.0, 1.0, default_cons, 0.05)
                st.caption(f"Percentile: {(scores_df['consistency_avg'] <= min_cons).mean()*100:.1f}%")
            with col2:
                min_rerank = st.slider("Min Rerank", 0.0, 1.0, default_rerank, 0.05)
                st.caption(f"Percentile: {(scores_df['rerank_avg'] <= min_rerank).mean()*100:.1f}%")

            golden = scores_df[
                (scores_df['consistency_avg'] >= min_cons) &
                (scores_df['rerank_avg'] >= min_rerank)
            ].copy()

            if strategy == "Composite (50/50 average)":
                golden['composite'] = (golden['consistency_avg'] + golden['rerank_avg']) / 2
            elif strategy == "Weighted (70% rerank, 30% consistency)":
                golden['composite'] = 0.7 * golden['rerank_avg'] + 0.3 * golden['consistency_avg']
            elif strategy == "Custom weighting":
                golden['composite'] = rerank_weight * golden['rerank_avg'] + consistency_weight * golden['consistency_avg']

        if len(golden) == 0:
            st.error("❌ No passages meet criteria! Lower your thresholds.")

            st.markdown("### 💡 Suggestions")
            st.write("Try these threshold combinations:")

            suggestions = [
                (0.2, 0.3, "Very inclusive"),
                (0.3, 0.4, "Moderate"),
                (0.4, 0.5, "Conservative")
            ]

            for cons, rerank, desc in suggestions:
                test_golden = scores_df[
                    (scores_df['consistency_avg'] >= cons) &
                    (scores_df['rerank_avg'] >= rerank)
                ]
                if len(test_golden) > 0:
                    pct = len(test_golden) / len(scores_df) * 100
                    st.write(f"- Consistency ≥ {cons}, Rerank ≥ {rerank} ({desc}): **{len(test_golden)} passages ({pct:.1f}%)**")

        else:
            golden = golden.sort_values('composite', ascending=False)
            st.session_state.golden_dataset = golden

            st.markdown("---")
            st.markdown("### ✅ Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Golden Passages", len(golden))
            with col2:
                st.metric("Percentage", f"{len(golden)/len(scores_df)*100:.1f}%")
            with col3:
                st.metric("Avg Quality", f"{golden['composite'].mean():.3f}")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            if strategy == "Rerank Only (ignore consistency)":
                ax1.hist(scores_df['rerank_avg'], bins=50, alpha=0.5, edgecolor='black', label='All', color='lightgray')
                ax1.hist(golden['rerank_avg'], bins=30, alpha=0.7, label='Golden', color='gold', edgecolor='black')
                ax1.axvline(min_rerank, color='red', linestyle='--', alpha=0.7, label=f'Threshold: {min_rerank:.2f}')
                ax1.set_xlabel('Rerank Score')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Rerank Score Distribution')
                ax1.legend()
                ax1.grid(alpha=0.3)

                ax2.hist(golden['composite'], bins=30, alpha=0.7, color='gold', edgecolor='black')
                ax2.axvline(golden['composite'].mean(), color='red', linestyle='--',
                           label=f'Mean: {golden["composite"].mean():.3f}')
                ax2.set_xlabel('Quality Score (Rerank)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Golden Set Quality')
                ax2.legend()
                ax2.grid(alpha=0.3)
            else:
                ax1.scatter(scores_df['consistency_avg'], scores_df['rerank_avg'],
                           alpha=0.3, s=20, color='gray', label='All')
                ax1.scatter(golden['consistency_avg'], golden['rerank_avg'],
                           alpha=0.7, s=30, color='gold', label='Golden')
                ax1.axvline(min_cons, color='red', linestyle='--', alpha=0.7)
                ax1.axhline(min_rerank, color='blue', linestyle='--', alpha=0.7)
                ax1.set_xlabel('Consistency Score')
                ax1.set_ylabel('Rerank Score')
                ax1.set_title('Golden Dataset Selection')
                ax1.legend()
                ax1.grid(alpha=0.3)

                all_comp = scores_df['composite'] if 'composite' in scores_df else (scores_df['consistency_avg'] + scores_df['rerank_avg']) / 2
                ax2.hist(all_comp, bins=50, alpha=0.5, edgecolor='black', label='All')
                ax2.hist(golden['composite'], bins=30, alpha=0.7, color='gold', edgecolor='black', label='Golden')
                ax2.axvline(golden['composite'].mean(), color='red', linestyle='--',
                           label=f'Golden mean: {golden["composite"].mean():.3f}')
                ax2.set_xlabel('Composite Score')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Score Distribution')
                ax2.legend()
                ax2.grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("### 🏷️ Label Distribution in Golden Set")
            golden_indices = golden['passage_idx'].tolist()
            label_dist = []
            for label in label_columns:
                golden_count = sum(df.loc[idx, label] == 1 for idx in golden_indices if idx in df.index)
                total_count = df[label].sum()
                pct = (golden_count / total_count * 100) if total_count > 0 else 0
                label_dist.append({
                    'Label': label,
                    'Golden': int(golden_count),
                    'Total': int(total_count),
                    'Coverage': f"{pct:.1f}%"
                })
            label_dist_df = pd.DataFrame(label_dist)
            st.dataframe(label_dist_df, width='stretch')

    elif page == "📦 Tiers":
        st.markdown("## 📦 Create Dataset Tiers")

        df = st.session_state.df
        cache = st.session_state.cache
        label_columns = st.session_state.label_columns

        if not cache:
            st.error("⚠️ No scores available")
            st.stop()

        scores_df = cache['df_summary']

        with st.expander("📖 Understanding Tiers", expanded=False):
            st.markdown("""
            **Training Strategy:**
            
            1. **Tier 1 (Golden)** - Train first on highest quality data
            2. **Tier 2 (Training)** - Add for robustness and generalization
            3. **Inference** - NEVER train on this! Only for testing
            """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Quality Thresholds:**")
            tier1_cons = st.slider("Tier 1 Consistency", 0.0, 1.0, 0.60, 0.05)
            tier1_rerank = st.slider("Tier 1 Rerank", 0.0, 1.0, 0.50, 0.05)
        with col2:
            st.markdown("**Target Sizes:**")
            tier1_pct = st.slider("Tier 1 %", 10, 40, 20, 5)
            tier2_pct = st.slider("Tier 2 %", 20, 50, 30, 5)
            inference_pct = 100 - tier1_pct - tier2_pct
            st.info(f"Inference: {inference_pct}% (auto)")

        if st.button("🎯 Generate Tiers", type="primary"):
            with st.spinner("Creating tiers..."):
                tier1, tier2, inference = create_tiered_datasets(df, scores_df, label_columns,
                                                                 tier1_cons, tier1_rerank, tier1_pct, tier2_pct)
                st.session_state.tier1_dataset = tier1
                st.session_state.tier2_dataset = tier2
                st.session_state.inference_dataset = inference
                st.success("✅ Tiers created!")

        if st.session_state.tier1_dataset is not None:
            tier1 = st.session_state.tier1_dataset
            tier2 = st.session_state.tier2_dataset
            inference = st.session_state.inference_dataset

            st.markdown("### 📊 Tier Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### 🏆 Tier 1")
                st.metric("Passages", len(tier1))
                st.metric("%", f"{len(tier1)/len(df)*100:.1f}%")
            with col2:
                st.markdown("#### 📚 Tier 2")
                st.metric("Passages", len(tier2))
                st.metric("%", f"{len(tier2)/len(df)*100:.1f}%")
            with col3:
                st.markdown("#### 🎯 Inference")
                st.metric("Passages", len(inference))
                st.metric("%", f"{len(inference)/len(df)*100:.1f}%")

    elif page == "💾 Export":
        st.markdown("## 💾 Export Results")

        df = st.session_state.df

        if st.session_state.golden_dataset is not None:
            st.markdown("### 🏆 Golden Dataset")
            golden = st.session_state.golden_dataset
            st.info(f"{len(golden)} passages")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            golden_indices = golden['passage_idx'].tolist()
            golden_full = df.loc[golden_indices].copy()
            for idx in golden_indices:
                score_row = golden[golden['passage_idx'] == idx].iloc[0]
                golden_full.loc[idx, 'confidence_composite'] = score_row['composite']

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                golden_full.to_excel(writer, index=False)

            st.download_button(
                label="📥 Download Golden Dataset",
                data=output.getvalue(),
                file_name=f"golden_dataset_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        if st.session_state.tier1_dataset is not None:
            st.markdown("---")
            st.markdown("### 📦 Tiered Datasets")

            tier1 = st.session_state.tier1_dataset
            tier2 = st.session_state.tier2_dataset
            inference = st.session_state.inference_dataset

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                tier1.to_excel(writer, sheet_name='Tier1_Golden', index=False)
                tier2.to_excel(writer, sheet_name='Tier2_Training', index=False)
                inference.to_excel(writer, sheet_name='Inference', index=False)

                summary = pd.DataFrame({
                    'Tier': ['Tier 1', 'Tier 2', 'Inference', 'Total'],
                    'Count': [len(tier1), len(tier2), len(inference), len(tier1)+len(tier2)+len(inference)],
                    'Percentage': [
                        f"{len(tier1)/(len(tier1)+len(tier2)+len(inference))*100:.1f}%",
                        f"{len(tier2)/(len(tier1)+len(tier2)+len(inference))*100:.1f}%",
                        f"{len(inference)/(len(tier1)+len(tier2)+len(inference))*100:.1f}%",
                        "100.0%"
                    ]
                })
                summary.to_excel(writer, sheet_name='Summary', index=False)

            st.download_button(
                label="📥 Download All Tiers (Multi-Sheet Excel)",
                data=output.getvalue(),
                file_name=f"tiered_datasets_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>HRAF Golden Dataset Discovery | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)