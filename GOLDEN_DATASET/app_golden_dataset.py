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


def _render_search_results(results, df, passage_col, label_columns, cache, namespace, finder):
    """Helper function to render search results with model inference"""

    # Load model if available
    model_loaded = (
            st.session_state.get('model_loader') is not None and
            st.session_state.model_loader.is_loaded()
    )

    # Batch inference option
    if model_loaded and len(results) > 1:
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            if st.button("🤖 Run Inference on All Results", type="primary", key="batch_inference"):
                st.session_state.run_batch_inference = True
                st.rerun()

        with col2:
            if st.session_state.get('batch_inference_results'):
                if st.button("📊 Show Batch Summary", type="secondary"):
                    _show_batch_summary(st.session_state.batch_inference_results, label_columns)

        with col3:
            if st.session_state.get('batch_inference_results'):
                if st.button("🗑️ Clear Batch Results", type="secondary"):
                    st.session_state.batch_inference_results = None
                    st.session_state.run_batch_inference = False
                    st.rerun()

        st.markdown("---")

    # Run batch inference if triggered
    if st.session_state.get('run_batch_inference'):
        with st.spinner(f"Running inference on {len(results)} passages..."):
            batch_results = _run_batch_inference(results, df, passage_col, label_columns)
            st.session_state.batch_inference_results = batch_results
            st.session_state.run_batch_inference = False
            st.success(f"✅ Completed inference on {len(results)} passages!")
            st.rerun()

    for i, result in enumerate(results, 1):
        idx = result['passage_idx']

        # Build score display
        score_parts = []
        if 'vector_score' in result:
            score_parts.append(f"Vector: {result['vector_score']:.3f}")
        if 'rerank_score' in result:
            score_parts.append(f"Rerank: {result['rerank_score']:.3f}")
        if 'combined_score' in result:
            score_parts.append(f"Combined: {result['combined_score']:.3f}")

        score_str = " | ".join(score_parts)

        # Get confidence if available
        confidence_str = ""
        if cache:
            scores_df = cache['df_summary']
            if idx in scores_df['passage_idx'].values:
                score_row = scores_df[scores_df['passage_idx'] == idx].iloc[0]
                conf = (score_row['consistency_avg'] + score_row['rerank_avg']) / 2
                confidence_str = f" | Quality: {conf:.3f}"

        # Get passage text
        text = df.loc[idx, passage_col] if idx in df.index else "N/A"
        if pd.isna(text):
            text = "N/A"

        # Get labels
        active_labels = [l for l in label_columns if idx in df.index and df.loc[idx, l] == 1]

        with st.expander(f"#{i} - Passage {idx} | {score_str}{confidence_str}"):
            st.markdown(f"**Labels:** {', '.join(active_labels) if active_labels else 'None'}")
            st.markdown("---")

            # Show text
            preview_length = 1500
            if len(text) > preview_length:
                st.write(text[:preview_length] + "...")
                with st.expander("Show full text"):
                    st.write(text)
            else:
                st.write(text)

            st.markdown("---")

            # Action buttons
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                # Store intent to search similar, but don't execute here
                if st.button("🔍 Find Similar", key=f"similar_{idx}_{i}"):
                    st.info(f"💡 To find similar passages: Go to 'Similar to Passage' mode and enter index {idx}")

            with col2:
                if model_loaded:
                    # Check if batch inference has been run
                    batch_results = st.session_state.get('batch_inference_results', {})
                    has_batch_result = idx in batch_results

                    # Use checkbox to show inference
                    show_inference = st.checkbox(
                        "Show Inference",
                        key=f"show_infer_{idx}_{i}",
                        value=has_batch_result  # Auto-show if batch was run
                    )

            # Show inference results if checkbox is checked
            if model_loaded and st.session_state.get(f"show_infer_{idx}_{i}", False):
                st.markdown("---")

                # Use batch result if available, otherwise run inference
                batch_results = st.session_state.get('batch_inference_results', {})
                if idx in batch_results:
                    _display_inference_result(
                        batch_results[idx]['result'],
                        batch_results[idx]['actual_labels'],
                        label_columns
                    )
                else:
                    _run_inference_on_passage(idx, text, active_labels, label_columns)

            with col3:
                # Copy button using clipboard
                if st.button("📋", key=f"copy_{idx}_{i}", help="Copy text"):
                    st.code(text, language=None)


def _run_inference_on_passage(idx, text, actual_labels_list, label_columns):
    """Run model inference and compare to actual labels"""

    if not st.session_state.model_loader.is_loaded():
        st.warning("No model loaded")
        return

    with st.spinner("Running inference..."):
        try:
            result = st.session_state.model_loader.predict_passage(text)

            # Build actual labels dict - need to map between different naming conventions
            df = st.session_state.df
            actual_labels_dict = _build_actual_labels_dict(idx, df, label_columns, result['probabilities'].keys())

            _display_inference_result(result, actual_labels_dict, label_columns)

        except Exception as e:
            st.error(f"Inference error: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


def _build_actual_labels_dict(idx, df, label_columns, model_labels):
    """
    Build actual labels dict with proper name mapping between model and dataframe

    Model uses: EVENT, ACTION_Shaman_Medium_Healer, etc.
    DataFrame might use: Illness, Shaman_Medium_Healer, etc.
    """
    actual_labels = {}

    # First pass: Get all labels from dataframe
    df_labels = {}
    for col in label_columns:
        if col in df.columns and idx in df.index:
            val = df.loc[idx, col]
            if pd.notna(val):
                df_labels[col] = int(val)

    # Second pass: Map to model label names
    for model_label in model_labels:
        found = False

        # Try exact match first
        if model_label in df_labels:
            actual_labels[model_label] = df_labels[model_label]
            found = True

        # Try without prefix (ACTION_Shaman -> Shaman_Medium_Healer)
        if not found and '_' in model_label:
            parts = model_label.split('_', 1)
            if len(parts) > 1:
                suffix = parts[1]
                if suffix in df_labels:
                    actual_labels[model_label] = df_labels[suffix]
                    found = True

        # For main categories (EVENT, CAUSE, ACTION), infer from sublabels
        if not found and model_label in ['EVENT', 'CAUSE', 'ACTION']:
            # Check if any sublabel is present
            has_sublabel = False

            if model_label == 'EVENT':
                sublabels = ['Illness', 'Accident', 'Other']
            elif model_label == 'CAUSE':
                sublabels = ['Just_Happens', 'Material_Physical', 'Spirits_Gods',
                             'Witchcraft_Sorcery', 'Rule_Violation_Taboo', 'Other.1']
            elif model_label == 'ACTION':
                sublabels = ['Physical_Material', 'Technical_Specialist', 'Divination',
                             'Shaman_Medium_Healer', 'Priest_High_Religion', 'Other.2']
            else:
                sublabels = []

            for sublabel in sublabels:
                if sublabel in df_labels and df_labels[sublabel] == 1:
                    has_sublabel = True
                    break

            actual_labels[model_label] = 1 if has_sublabel else 0
            found = True

        # Default to 0 if not found
        if not found:
            actual_labels[model_label] = 0

    return actual_labels


def _display_inference_result(result, actual_labels_dict, label_columns):
    """Display inference results with comparison to actual labels"""

    st.markdown("#### 🤖 Model Predictions")

    # Get comparison
    from model_inference import compare_predictions_to_labels
    comparison = compare_predictions_to_labels(result['predictions'], actual_labels_dict)

    # Build comparison table
    comparison_data = []
    for label in sorted(result['probabilities'].keys()):
        pred_prob = result['probabilities'][label]
        is_predicted = label in result['predicted_labels']
        pred_str = f"✓ {pred_prob:.2f}" if is_predicted else f"  {pred_prob:.2f}"

        # Get actual value
        actual_val = actual_labels_dict.get(label, 0)
        actual_str = "✓" if actual_val == 1 else "—"

        # Get comparison
        comp = comparison.get(label, "")
        if "True Positive" in comp:
            comp_str = "✓ Match"
            comp_color = "🟢"
        elif "True Negative" in comp:
            comp_str = "✓ Match"
            comp_color = "⚪"
        elif "False Positive" in comp:
            comp_str = "✗ Over-predicted"
            comp_color = "🔴"
        elif "False Negative" in comp:
            comp_str = "✗ Missed"
            comp_color = "🟡"
        else:
            comp_str = "—"
            comp_color = ""

        comparison_data.append({
            'Label': label,
            'Predicted': pred_str,
            'Actual': actual_str,
            'Result': f"{comp_color} {comp_str}".strip()
        })

    st.dataframe(
        pd.DataFrame(comparison_data),
        hide_index=True,
        use_container_width=True
    )

    # Summary stats
    tp = sum(1 for c in comparison.values() if "True Positive" in c)
    tn = sum(1 for c in comparison.values() if "True Negative" in c)
    fp = sum(1 for c in comparison.values() if "False Positive" in c)
    fn = sum(1 for c in comparison.values() if "False Negative" in c)

    # Explanation of metrics
    with st.expander("📊 What do these metrics mean?"):
        st.markdown("""
        **Confusion Matrix Metrics:**

        - **TP (True Positive)**: Model predicted ✓ AND actual was ✓ → **Correct positive** 🟢
        - **TN (True Negative)**: Model predicted — AND actual was — → **Correct negative** ⚪
        - **FP (False Positive)**: Model predicted ✓ BUT actual was — → **Over-predicted** 🔴
        - **FN (False Negative)**: Model predicted — BUT actual was ✓ → **Missed** 🟡

        **Good model indicators:**
        - High TP and TN (correctly identifying both positive and negative cases)
        - Low FP and FN (few mistakes)
        """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🟢 TP", tp, help="True Positives: Correctly predicted positive")
    with col2:
        st.metric("⚪ TN", tn, help="True Negatives: Correctly predicted negative")
    with col3:
        st.metric("🔴 FP", fp, help="False Positives: Incorrectly predicted positive")
    with col4:
        st.metric("🟡 FN", fn, help="False Negatives: Missed actual positives")


def _run_batch_inference(results, df, passage_col, label_columns):
    """Run inference on all search results"""
    batch_results = {}

    for result in results:
        idx = result['passage_idx']

        if idx not in df.index:
            continue

        text = df.loc[idx, passage_col]
        if pd.isna(text) or not isinstance(text, str):
            continue

        try:
            # Run inference
            inference_result = st.session_state.model_loader.predict_passage(text)

            # Build actual labels
            actual_labels_dict = _build_actual_labels_dict(
                idx, df, label_columns,
                inference_result['probabilities'].keys()
            )

            batch_results[idx] = {
                'result': inference_result,
                'actual_labels': actual_labels_dict
            }

        except Exception as e:
            print(f"Error on passage {idx}: {e}")
            continue

    return batch_results


def _show_batch_summary(batch_results, label_columns):
    """Show summary statistics across all batch inference results"""
    st.markdown("### 📊 Batch Inference Summary")

    from model_inference import compare_predictions_to_labels

    # Aggregate metrics across all passages
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    label_stats = {}

    for idx, data in batch_results.items():
        result = data['result']
        actual_labels = data['actual_labels']

        comparison = compare_predictions_to_labels(result['predictions'], actual_labels)

        tp = sum(1 for c in comparison.values() if "True Positive" in c)
        tn = sum(1 for c in comparison.values() if "True Negative" in c)
        fp = sum(1 for c in comparison.values() if "False Positive" in c)
        fn = sum(1 for c in comparison.values() if "False Negative" in c)

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        # Per-label stats
        for label, comp in comparison.items():
            if label not in label_stats:
                label_stats[label] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

            if "True Positive" in comp:
                label_stats[label]['tp'] += 1
            elif "True Negative" in comp:
                label_stats[label]['tn'] += 1
            elif "False Positive" in comp:
                label_stats[label]['fp'] += 1
            elif "False Negative" in comp:
                label_stats[label]['fn'] += 1

    # Overall metrics
    st.markdown("#### Overall Performance")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🟢 Total TP", total_tp)
    with col2:
        st.metric("⚪ Total TN", total_tn)
    with col3:
        st.metric("🔴 Total FP", total_fp)
    with col4:
        st.metric("🟡 Total FN", total_fn)

    # Calculate overall metrics
    total_predictions = total_tp + total_tn + total_fp + total_fn
    if total_predictions > 0:
        accuracy = (total_tp + total_tn) / total_predictions
        if (total_tp + total_fp) > 0:
            precision = total_tp / (total_tp + total_fp)
        else:
            precision = 0
        if (total_tp + total_fn) > 0:
            recall = total_tp / (total_tp + total_fn)
        else:
            recall = 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}")
        with col2:
            st.metric("Precision", f"{precision:.3f}")
        with col3:
            st.metric("Recall", f"{recall:.3f}")

    # Per-label breakdown
    st.markdown("#### Per-Label Performance")

    label_breakdown = []
    for label, stats in sorted(label_stats.items()):
        tp = stats['tp']
        tn = stats['tn']
        fp = stats['fp']
        fn = stats['fn']

        total = tp + tn + fp + fn
        if total > 0:
            acc = (tp + tn) / total
        else:
            acc = 0

        label_breakdown.append({
            'Label': label,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'Accuracy': f"{acc:.3f}"
        })

    st.dataframe(
        pd.DataFrame(label_breakdown),
        hide_index=True,
        use_container_width=True
    )


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
        ["📊 Overview", "💻 Compute Scores", "🔍 Search", "⚙️ Thresholds", "📦 Tiers", "🤖 Model Inference", "💾 Export"],
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

        st.markdown("## 🔍 Enhanced Search")

        df = st.session_state.df

        passage_col = st.session_state.get('passage_col', 'Passage')

        label_columns = st.session_state.label_columns

        cache = st.session_state.cache

        finder = st.session_state.finder

        namespace = st.session_state.get('namespace', 'main')

        st.markdown("""
        
            **Search Strategy:**
        
            1. **Vector Search** - Fast Pinecone similarity search (100+ candidates)
        
            2. **Reranking** - Precise VoyageAI rerank on top results 
        
            3. **Instruction-Following** - Guide reranker with natural language instructions
        
            """)

        # Search mode selection

        search_mode = st.radio(

            "Search mode:",

            ["📝 Text Query", "🔍 Similar to Passage", "🏷️ Label Semantic Search"],

            horizontal=True,

            key="search_mode_radio"

        )

        # Clear results when changing modes

        if 'last_search_mode' not in st.session_state:
            st.session_state.last_search_mode = search_mode

        if st.session_state.last_search_mode != search_mode:
            st.session_state.search_results = None

            st.session_state.search_mode = None

            st.session_state.last_search_mode = search_mode

        # Add clear results button

        if st.session_state.get('search_results'):

            if st.button("🗑️ Clear Results", type="secondary"):
                st.session_state.search_results = None

                st.session_state.search_mode = None

                st.rerun()

        st.markdown("---")

        # ========================================================================

        # MODE 1: Text Query

        # ========================================================================

        if search_mode == "📝 Text Query":

            st.markdown("### 📝 Search by Text Query")

            query = st.text_input(

                "Search query:",

                placeholder="e.g., shamans healing illness with spirits",

                key="text_query"

            )

            col1, col2 = st.columns(2)

            with col1:

                label_filter = st.selectbox(

                    "Filter by label (optional):",

                    ["None"] + label_columns,

                    key="label_filter"

                )

                label_filter = None if label_filter == "None" else label_filter

            with col2:

                top_k_results = st.number_input(

                    "Number of results:",

                    min_value=1,

                    max_value=50,

                    value=10,

                    key="top_k_results"

                )

            with st.expander("⚙️ Advanced Options"):

                col1, col2 = st.columns(2)

                with col1:

                    top_k_vector = st.slider(

                        "Vector search candidates:",

                        min_value=10,

                        max_value=200,

                        value=100,

                        help="More candidates = better recall but slower"

                    )

                    min_similarity = st.slider(

                        "Minimum similarity:",

                        min_value=0.0,

                        max_value=1.0,

                        value=0.3,

                        step=0.05

                    )

                with col2:

                    use_rerank = st.checkbox("Use reranking", value=True)

                    if use_rerank:

                        instruction = st.text_area(

                            "Reranker instruction (optional):",

                            placeholder="e.g., Prioritize passages with detailed descriptions",

                            height=100,

                            help="Guide the reranker with natural language instructions"

                        )

                    else:

                        instruction = None

            if st.button("🔍 Search", type="primary", key="search_text"):

                if not query:

                    st.warning("Please enter a search query")

                else:

                    with st.spinner("Searching..."):

                        try:

                            results = finder.search_with_filters(

                                query=query,

                                namespace=namespace,

                                label_filter=label_filter,

                                top_k_vector=top_k_vector,

                                top_k_rerank=top_k_results if use_rerank else len(df),

                                rerank_instruction=instruction if use_rerank else None,

                                min_similarity=min_similarity

                            )

                            if not results:

                                st.warning("No results found. Try lowering the similarity threshold or removing filters.")

                                st.session_state.search_results = None

                            else:

                                st.success(f"Found {len(results)} results")

                                # Store results in session state

                                st.session_state.search_results = results

                                st.session_state.search_mode = "text_query"


                        except Exception as e:

                            st.error(f"Search error: {e}")

                            st.session_state.search_results = None

                            import traceback

                            with st.expander("Error details"):

                                st.code(traceback.format_exc())

            # Render results if they exist in session state

            if st.session_state.get('search_results') and st.session_state.get('search_mode') == "text_query":
                _render_search_results(

                    st.session_state.search_results, df, passage_col, label_columns, cache, namespace, finder

                )


        # ========================================================================

        # MODE 2: Similar to Passage

        # ========================================================================

        elif search_mode == "🔍 Similar to Passage":

            st.markdown("### 🔍 Find Similar Passages")

            st.caption("Uses pure vector similarity - fast and no API costs")

            col1, col2 = st.columns(2)

            with col1:

                passage_idx = st.number_input(

                    "Passage index:",

                    min_value=0,

                    max_value=len(df) - 1,

                    value=0,

                    key="similar_idx"

                )

            with col2:

                k_similar = st.number_input(

                    "Number of results:",

                    min_value=1,

                    max_value=50,

                    value=10,

                    key="k_similar"

                )

            label_filter = st.selectbox(

                "Filter by label (optional):",

                ["None"] + label_columns,

                key="label_filter_similar"

            )

            label_filter = None if label_filter == "None" else label_filter

            # Show reference passage

            if passage_idx in df.index and passage_col in df.columns:

                with st.expander(f"📄 Reference Passage {passage_idx}", expanded=True):

                    ref_text = df.loc[passage_idx, passage_col]

                    if pd.notna(ref_text):

                        st.write(ref_text[:500] + "..." if len(ref_text) > 500 else ref_text)

                        active_labels = [l for l in label_columns if df.loc[passage_idx, l] == 1]

                        if active_labels:
                            st.markdown(f"**Labels:** {', '.join(active_labels)}")

            if st.button("🔍 Find Similar", type="primary", key="search_similar"):

                with st.spinner("Finding similar passages..."):

                    try:

                        results = finder.search_similar_to_passage(

                            passage_idx=passage_idx,

                            namespace=namespace,

                            k=k_similar,

                            label_filter=label_filter

                        )

                        if not results:

                            st.warning("No similar passages found")

                            st.session_state.search_results = None

                        else:

                            st.success(f"Found {len(results)} similar passages")

                            # Convert to standard format for rendering

                            formatted_results = []

                            for r in results:
                                formatted_results.append({

                                    'passage_idx': r['passage_idx'],

                                    'vector_score': r['similarity'],

                                    'combined_score': r['similarity'],

                                    'metadata': r['metadata']

                                })

                            # Store results in session state

                            st.session_state.search_results = formatted_results

                            st.session_state.search_mode = "similar"


                    except Exception as e:

                        st.error(f"Search error: {e}")

                        st.session_state.search_results = None

            # Render results if they exist in session state

            if st.session_state.get('search_results') and st.session_state.get('search_mode') == "similar":
                _render_search_results(

                    st.session_state.search_results, df, passage_col, label_columns, cache, namespace, finder

                )


        # ========================================================================

        # MODE 3: Label Semantic Search

        # ========================================================================

        else:  # Label Semantic Search

            st.markdown("### 🏷️ Label Semantic Search")

            st.caption("Find passages most relevant to a label's semantic meaning")

            col1, col2 = st.columns(2)

            with col1:

                selected_label = st.selectbox(

                    "Select label:",

                    label_columns,

                    key="semantic_label"

                )

            with col2:

                top_k_semantic = st.number_input(

                    "Number of results:",

                    min_value=1,

                    max_value=50,

                    value=10,

                    key="top_k_semantic"

                )

            # Show label description

            if selected_label in finder.LABEL_QUERIES:
                st.info(f"**Label description:** {finder.LABEL_QUERIES[selected_label]}")

            if st.button("🔍 Search", type="primary", key="search_semantic"):

                with st.spinner("Searching..."):

                    try:

                        results = finder.search_by_label_semantic(

                            label=selected_label,

                            namespace=namespace,

                            top_k_vector=100,

                            top_k_rerank=top_k_semantic

                        )

                        if not results:

                            st.warning("No results found")

                            st.session_state.search_results = None

                        else:

                            st.success(f"Found {len(results)} results")

                            # Store results in session state

                            st.session_state.search_results = results

                            st.session_state.search_mode = "semantic"


                    except Exception as e:

                        st.error(f"Search error: {e}")

                        st.session_state.search_results = None

            # Render results if they exist in session state

            if st.session_state.get('search_results') and st.session_state.get('search_mode') == "semantic":

                _render_search_results(

                    st.session_state.search_results, df, passage_col, label_columns, cache, namespace, finder

                )


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

    elif page == "🤖 Model Inference":

        st.markdown("## 🤖 Model Inference")

        df = st.session_state.df

        passage_col = st.session_state.get('passage_col', 'Passage')

        label_columns = st.session_state.label_columns

        # Initialize model loader and browser state

        if 'model_loader' not in st.session_state:
            from model_inference import HRAFModelLoader

            st.session_state.model_loader = HRAFModelLoader()

            st.session_state.loaded_model_path = None

            st.session_state.model_browse_directory = Path("./models").resolve()

        st.markdown("""
        
            Test your trained models on passages from the dataset to see how they perform.
        
            This helps you:
        
            - Validate model predictions on specific passages
        
            - Compare predictions to actual labels
        
            - Identify which passages your model struggles with
        
            - Test different models on the same passages
        
            """)

        # Model Selection with Browser

        st.markdown("### 📁 Select Model")

        browse_mode_model = st.radio(

            "Selection mode:",

            ["Quick (./models folder)", "Browse directories"],

            key="model_browse_mode",

            horizontal=True

        )

        selected_model_path = None

        if browse_mode_model == "Quick (./models folder)":

            # Quick mode - show models in results folder

            from model_inference import find_model_directories

            model_dirs = find_model_directories("./models")

            if not model_dirs:

                st.warning("⚠️ No trained models found in `./models/` directory")

                st.info("Train a model using the hierarchical training notebook first, or use Browse mode")

            else:

                col1, col2 = st.columns([3, 1])

                with col1:

                    model_options = {str(m.parent.name if m.name == "final_model" else m.name): str(m) for m in model_dirs}

                    selected_model_name = st.selectbox(

                        "Available models:",

                        options=list(model_options.keys()),

                        key="model_selector_quick"

                    )

                    selected_model_path = model_options[selected_model_name]

                with col2:

                    st.write("")  # Spacing

                    st.write("")

                    if st.button("🔄 Load Model", type="primary", key="load_quick"):

                        with st.spinner("Loading model..."):

                            success = st.session_state.model_loader.load_model(selected_model_path)

                            if success:

                                st.session_state.loaded_model_path = selected_model_path

                                st.success("✅ Model loaded!")

                            else:

                                st.error("❌ Failed to load model")


        else:

            # Browse mode - directory navigator

            current_dir = st.session_state.model_browse_directory

            st.markdown("**Current Directory:**")

            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:

                if st.button("⬆️ Parent", key="model_go_up", disabled=current_dir == current_dir.parent):
                    st.session_state.model_browse_directory = current_dir.parent

                    st.rerun()

            with col2:

                st.text_input("Path:", str(current_dir), key="model_path_display", disabled=True, label_visibility="collapsed")

            with col3:

                if st.button("🏠 Home", key="model_go_home"):
                    st.session_state.model_browse_directory = Path.home()

                    st.rerun()

            # Quick navigation

            if st.button("📂 ./models", key="model_go_results"):
                st.session_state.model_browse_directory = Path("./models").resolve()

                st.rerun()

            st.markdown("---")

            # Show subdirectories

            subdirs = get_subdirectories(current_dir)

            if subdirs:

                st.markdown("**📁 Folders:**")

                for subdir in subdirs[:10]:

                    if st.button(f"📁 {subdir.name}", key=f"model_dir_{subdir}"):
                        st.session_state.model_browse_directory = subdir

                        st.rerun()

                if len(subdirs) > 10:
                    st.caption(f"... and {len(subdirs) - 10} more folders")

            st.markdown("---")

            # Look for model files in current directory

            has_config = (current_dir / "config.json").exists()

            has_model = (current_dir / "pytorch_model.bin").exists() or (current_dir / "model.safetensors").exists()

            if has_config and has_model:

                st.success(f"✅ Model found in current directory!")

                selected_model_path = str(current_dir)

                if st.button("🔄 Load This Model", type="primary", key="load_browse"):

                    with st.spinner("Loading model..."):

                        success = st.session_state.model_loader.load_model(selected_model_path)

                        if success:

                            st.session_state.loaded_model_path = selected_model_path

                            st.success("✅ Model loaded!")

                        else:

                            st.error("❌ Failed to load model")

            else:

                # Show what's in current directory

                model_subdirs = []

                try:

                    for item in current_dir.iterdir():

                        if item.is_dir():

                            item_has_config = (item / "config.json").exists()

                            item_has_model = (item / "pytorch_model.bin").exists() or (item / "model.safetensors").exists()

                            if item_has_config and item_has_model:
                                model_subdirs.append(item)

                    if model_subdirs:

                        st.info(f"Found {len(model_subdirs)} model(s) in subdirectories. Navigate into a folder to load.")

                    else:

                        st.info(
                            "No model found in current directory. Navigate to a folder containing `config.json` and model weights.")

                except (FileNotFoundError, PermissionError) as e:

                    st.error(f"Cannot access directory: {e}")

                    st.info("The directory may not exist. Navigate to an existing directory or create the results folder.")

        # Show loaded model info

        if st.session_state.model_loader.is_loaded():

            st.markdown("---")

            st.markdown("### ℹ️ Loaded Model")

            model_info = st.session_state.model_loader.get_model_info()

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric("Model", Path(
                    st.session_state.loaded_model_path).parent.name if st.session_state.loaded_model_path else "Unknown")

            if model_info:

                with col2:

                    test_f1 = model_info.get('test_results', {}).get('eval_f1_micro', 'N/A')

                    if test_f1 != 'N/A':

                        st.metric("Test F1", f"{test_f1:.3f}")

                    else:

                        st.metric("Test F1", "N/A")

                with col3:

                    config = model_info.get('config', {})

                    hierarchy = "Hierarchical" if config.get('use_hierarchy') else "Flat"

                    if config.get('gated_hierarchy'):
                        hierarchy += " (Gated)"

                    st.metric("Architecture", hierarchy)

                with st.expander("📋 Model Configuration"):

                    st.json(config)

            st.markdown("---")

            # Inference Section

            st.markdown("### 🎯 Test Predictions")

            inference_mode = st.radio(

                "Select passages:",

                ["From Dataset", "Custom Text"],

                horizontal=True

            )

            if inference_mode == "From Dataset":

                # Select passages from dataset

                st.markdown("**Select passages to test:**")

                # Filter options

                col1, col2 = st.columns(2)

                with col1:

                    filter_by = st.selectbox(

                        "Filter by:",

                        ["All passages", "Has specific label", "Random sample", "By index range"]

                    )

                with col2:

                    if filter_by == "Has specific label":

                        filter_label = st.selectbox("Label:", label_columns)

                    elif filter_by == "Random sample":

                        sample_size = st.number_input("Sample size:", 1, 100, 10)

                    elif filter_by == "By index range":

                        start_idx = st.number_input("Start index:", 0, len(df) - 1, 0)

                        end_idx = st.number_input("End index:", start_idx + 1, len(df), min(start_idx + 10, len(df)))

                # Get filtered passages

                if filter_by == "All passages":

                    available_indices = df.index.tolist()

                elif filter_by == "Has specific label":

                    available_indices = df[df[filter_label] == 1].index.tolist()

                elif filter_by == "Random sample":

                    available_indices = df.sample(n=min(sample_size, len(df))).index.tolist()

                elif filter_by == "By index range":

                    available_indices = df.iloc[start_idx:end_idx].index.tolist()

                st.info(f"Found {len(available_indices)} passages")

                num_to_show = st.slider("Number to test:", 1, min(20, len(available_indices)), 5)

                if st.button("🔮 Predict", type="primary"):

                    selected_indices = available_indices[:num_to_show]

                    for idx in selected_indices:

                        passage_text = df.loc[idx, passage_col]

                        # Skip if passage is NaN or empty

                        if pd.isna(passage_text) or not isinstance(passage_text, str):
                            st.warning(f"⚠️ Passage {idx} has no text, skipping...")

                            continue

                        # Build actual_labels dict with error handling for missing columns and NaN values

                        actual_labels = {}

                        for col in label_columns:

                            if col in df.columns:

                                val = df.loc[idx, col]

                                # Handle NaN - treat as 0

                                if pd.isna(val):

                                    actual_labels[col] = 0

                                else:

                                    actual_labels[col] = int(val)

                            else:

                                # Try without prefix (EVENT_Illness -> Illness)

                                if '_' in col:

                                    suffix = col.split('_', 1)[1]

                                    if suffix in df.columns:

                                        val = df.loc[idx, suffix]

                                        if pd.isna(val):

                                            actual_labels[suffix] = 0

                                        else:

                                            actual_labels[suffix] = int(val)

                        with st.expander(f"📄 Passage {idx}"):

                            st.markdown("**Text:**")

                            st.write(passage_text[:500] + "..." if len(passage_text) > 500 else passage_text)

                            # Get predictions

                            with st.spinner("Predicting..."):

                                result = st.session_state.model_loader.predict_passage(passage_text)

                            # Compare

                            from model_inference import compare_predictions_to_labels

                            comparison = compare_predictions_to_labels(result['predictions'], actual_labels)

                            # Build unified label set (use model's label format)

                            all_model_labels = result['probabilities'].keys()

                            # Infer main categories from actual sublabels

                            ra_coded_labels = set()

                            for label, val in actual_labels.items():

                                if val == 1:

                                    ra_coded_labels.add(label)

                                    # Infer main category

                                    if label in ['Illness', 'Accident', 'Other']:

                                        ra_coded_labels.add('EVENT')

                                    elif label in ['Just_Happens', 'Material_Physical', 'Spirits_Gods',

                                                   'Witchcraft_Sorcery', 'Rule_Violation_Taboo']:

                                        ra_coded_labels.add('CAUSE')

                                    elif label in ['Physical_Material', 'Technical_Specialist', 'Divination',

                                                   'Shaman_Medium_Healer', 'Priest_High_Religion', 'Other.2']:

                                        ra_coded_labels.add('ACTION')

                            # Create comparison data

                            comparison_data = []

                            for label in sorted(all_model_labels):

                                # Predicted

                                pred_prob = result['probabilities'][label]

                                is_predicted = label in result['predicted_labels']

                                pred_str = f"✓ {pred_prob:.2f}" if is_predicted else f"  {pred_prob:.2f}"

                                # RA Coded (handle name mismatches)

                                ra_label = None

                                if label in ra_coded_labels:

                                    ra_label = label

                                elif '_' in label:

                                    suffix = label.split('_', 1)[1]

                                    if suffix in ra_coded_labels:
                                        ra_label = suffix

                                ra_str = f"✓ {ra_label}" if ra_label else "—"

                                # Comparison

                                comp = comparison.get(label, "")

                                if "True Positive" in comp:

                                    comp_str = "✓ Match"

                                    comp_color = "🟢"

                                elif "True Negative" in comp:

                                    comp_str = "✓ Match"

                                    comp_color = "⚪"

                                elif "False Positive" in comp:

                                    comp_str = "✗ Over-predicted"

                                    comp_color = "🔴"

                                elif "False Negative" in comp:

                                    comp_str = "✗ Missed"

                                    comp_color = "🟡"

                                else:

                                    comp_str = "—"

                                    comp_color = ""

                                comparison_data.append({

                                    'Label': label,

                                    'Predicted': pred_str,

                                    'RA Coded': ra_str,

                                    'Result': f"{comp_color} {comp_str}".strip()

                                })

                            # Display as dataframe

                            st.dataframe(

                                pd.DataFrame(comparison_data),

                                hide_index=True,

                                use_container_width=True

                            )


            else:  # Custom Text

                st.markdown("**Enter custom passage text:**")

                custom_text = st.text_area(

                    "Passage:",

                    placeholder="Enter text to test...",

                    height=150

                )

                use_optimal = st.checkbox("Use optimal thresholds from training", value=True)

                if not use_optimal:

                    threshold = st.slider("Threshold:", 0.0, 1.0, 0.5, 0.05)

                else:

                    threshold = 0.5

                if st.button("🔮 Predict", type="primary") and custom_text:

                    with st.spinner("Predicting..."):

                        result = st.session_state.model_loader.predict_passage(

                            custom_text,

                            use_optimal_thresholds=use_optimal,

                            default_threshold=threshold

                        )

                    st.markdown("### Results")

                    if result['predicted_labels']:

                        st.markdown("**Predicted Labels:**")

                        for label in result['predicted_labels']:
                            prob = result['probabilities'][label]

                            st.write(f"✓ **{label}** (confidence: {prob:.3f})")

                    else:

                        st.info("No labels predicted above threshold")

                    with st.expander("📊 All Probabilities"):

                        prob_df = pd.DataFrame([

                            {"Label": k, "Probability": v, "Predicted": result['predictions'][k]}

                            for k, v in result['probabilities'].items()

                        ]).sort_values('Probability', ascending=False)

                        st.dataframe(prob_df, use_container_width=True)


        else:

            st.info("👆 Load a model to start testing predictions")

    elif page == "💾 Export":

        st.markdown("## 💾 Export Results")


        df = st.session_state.df

        cache = st.session_state.cache

        label_columns = st.session_state.label_columns

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


        # Export Score Results

        if cache is not None:

            st.markdown("### 📊 Score Results")

            scores_df = cache['df_summary']

            st.info(f"{len(scores_df)} passages with computed scores")


            col1, col2 = st.columns(2)


            with col1:

                st.markdown("**Summary Scores (Simple)**")

                st.caption("Passage index + average scores")


                output_summary = io.BytesIO()

                scores_df.to_excel(output_summary, index=False, engine='openpyxl')


                st.download_button(

                    label="📥 Download Score Summary",

                    data=output_summary.getvalue(),

                    file_name=f"score_summary_{timestamp}.xlsx",

                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                    help="Simple table with passage indices and average scores"

                )


            with col2:

                st.markdown("**Detailed Scores (Per-Label)**")

                st.caption("Includes consistency & rerank per label")


                # Build detailed export

                detailed_rows = []

                for idx in scores_df['passage_idx'].tolist():

                    row_data = {

                        'passage_idx': idx,

                        'consistency_avg': scores_df[scores_df['passage_idx'] == idx]['consistency_avg'].iloc[0],

                        'rerank_avg': scores_df[scores_df['passage_idx'] == idx]['rerank_avg'].iloc[0],

                        'num_labels': scores_df[scores_df['passage_idx'] == idx]['num_labels'].iloc[0]

                    }


                    # Add per-label consistency scores

                    consistency_detailed = cache.get('consistency_detailed', {})

                    if idx in consistency_detailed:

                        for label in label_columns:

                            if label in consistency_detailed[idx].get('by_label', {}):

                                row_data[f'consistency_{label}'] = consistency_detailed[idx]['by_label'][label]


                    # Add per-label rerank scores

                    rerank_detailed = cache.get('rerank_detailed', {})

                    for label in label_columns:

                        if label in rerank_detailed and idx in rerank_detailed[label]:

                            row_data[f'rerank_{label}'] = rerank_detailed[label][idx]


                    detailed_rows.append(row_data)


                detailed_df = pd.DataFrame(detailed_rows)


                output_detailed = io.BytesIO()

                detailed_df.to_excel(output_detailed, index=False, engine='openpyxl')


                st.download_button(

                    label="📥 Download Detailed Scores",

                    data=output_detailed.getvalue(),

                    file_name=f"score_detailed_{timestamp}.xlsx",

                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                    help="Full breakdown with per-label consistency and rerank scores"

                )


            st.markdown("---")


        if st.session_state.golden_dataset is not None:

            st.markdown("### 🏆 Golden Dataset")

            golden = st.session_state.golden_dataset

            st.info(f"{len(golden)} passages")


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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>HRAF Golden Dataset Discovery | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)