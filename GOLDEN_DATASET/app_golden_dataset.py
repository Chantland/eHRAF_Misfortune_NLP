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
    st.session_state.current_page = "📊 Overview"

# Initialize model loader state
if 'model_loader' not in st.session_state:
    from model_inference import HRAFModelLoader
    st.session_state.model_loader = HRAFModelLoader()
    st.session_state.loaded_model_path = None
    st.session_state.model_browse_directory = Path("./models").resolve()
    st.session_state.model_browse_mode = "quick"

# Configuration
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "hraf-misfortune-test"
REGION = "us-east-1"

# Directory structure
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cached_scores"
MODEL_DIR = Path("models")

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


def render_directory_browser(key_prefix="data"):
    """Render directory browser in sidebar"""
    if key_prefix == "data":
        current_dir = st.session_state.current_directory
    else:
        current_dir = st.session_state.model_browse_directory

    breadcrumb_cols = st.columns([1, 4])

    with breadcrumb_cols[0]:
        if st.button("⬆️", key=f"{key_prefix}_go_up", disabled=current_dir == current_dir.parent, help="Parent directory"):
            if key_prefix == "data":
                st.session_state.current_directory = current_dir.parent
            else:
                st.session_state.model_browse_directory = current_dir.parent
            st.rerun()

    with breadcrumb_cols[1]:
        st.text_input("", str(current_dir), key=f"{key_prefix}_path_display", disabled=True, label_visibility="collapsed")

    quick_nav_col1, quick_nav_col2 = st.columns(2)
    with quick_nav_col1:
        if st.button("🏠", key=f"{key_prefix}_go_home", help="Home directory"):
            if key_prefix == "data":
                st.session_state.current_directory = Path.home()
            else:
                st.session_state.model_browse_directory = Path.home()
            st.rerun()
    with quick_nav_col2:
        default_dir = DATA_DIR if key_prefix == "data" else MODEL_DIR
        dir_name = "data/" if key_prefix == "data" else "models/"
        if st.button(f"📂", key=f"{key_prefix}_go_default", help=f"Go to {dir_name}"):
            if key_prefix == "data":
                st.session_state.current_directory = default_dir
            else:
                st.session_state.model_browse_directory = default_dir
            st.rerun()

    subdirs = get_subdirectories(current_dir)
    if subdirs:
        for subdir in subdirs[:8]:
            if st.button(f"📁 {subdir.name}", key=f"{key_prefix}_dir_{subdir}", use_container_width=True):
                if key_prefix == "data":
                    st.session_state.current_directory = subdir
                else:
                    st.session_state.model_browse_directory = subdir
                st.rerun()

        if len(subdirs) > 8:
            st.caption(f"... +{len(subdirs) - 8} more")

    if key_prefix == "data":
        xlsx_files = get_xlsx_files_in_directory(current_dir)
        if xlsx_files:
            file_options = {f.name: str(f) for f in xlsx_files}
            selected_name = st.selectbox(
                "📊 Select file:",
                options=list(file_options.keys()),
                key=f"{key_prefix}_file_selector",
                label_visibility="collapsed"
            )
            selected_file = file_options[selected_name]
            return selected_file
        else:
            st.info("No .xlsx files here")
            return None
    else:
        # Model directory - check for model files
        has_config = (current_dir / "config.json").exists()
        has_model = (current_dir / "pytorch_model.bin").exists() or (current_dir / "model.safetensors").exists()

        if has_config and has_model:
            st.success("✅ Model found")
            return str(current_dir)
        else:
            model_subdirs = []
            try:
                for item in current_dir.iterdir():
                    if item.is_dir():
                        item_has_config = (item / "config.json").exists()
                        item_has_model = (item / "pytorch_model.bin").exists() or (item / "model.safetensors").exists()
                        if item_has_config and item_has_model:
                            model_subdirs.append(item)

                if model_subdirs:
                    st.caption(f"{len(model_subdirs)} model(s) in subdirs")
                else:
                    st.caption("No model here")
            except (FileNotFoundError, PermissionError):
                st.error("Can't access")

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


def load_data(filename, header_row=1, passage_col_override=None):
    """Load Excel data"""
    try:
        df = pd.read_excel(filename, header=header_row)

        passage_col = passage_col_override if passage_col_override else detect_passage_column(df)
        if not passage_col:
            return None, None, None, None, None, None

        finder = GoldenDatasetFinder(
            voyage_api_key=VOYAGE_API_KEY,
            pinecone_api_key=PINECONE_API_KEY,
            index_name=INDEX_NAME,
            region=REGION
        )

        label_columns = finder._auto_detect_label_columns(df)

        namespace = get_namespace_from_filename(filename)

        cache_file = get_cache_filename(filename)
        cache = None
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)

        return df, finder, label_columns, cache, passage_col, namespace

    except Exception as e:
        st.error(f"Error: {e}")
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
    model_loaded = st.session_state.model_loader.is_loaded()

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
                if st.button("🔍 Find Similar", key=f"similar_{idx}_{i}"):
                    st.info(f"💡 To find similar passages: Go to 'Similar to Passage' mode and enter index {idx}")

            with col2:
                if model_loaded:
                    batch_results = st.session_state.get('batch_inference_results', {})
                    has_batch_result = idx in batch_results

                    show_inference = st.checkbox(
                        "Show Inference",
                        key=f"show_infer_{idx}_{i}",
                        value=has_batch_result
                    )

            if model_loaded and st.session_state.get(f"show_infer_{idx}_{i}", False):
                st.markdown("---")

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

            df = st.session_state.df
            actual_labels_dict = _build_actual_labels_dict(idx, df, label_columns, result['probabilities'].keys())

            _display_inference_result(result, actual_labels_dict, label_columns)

        except Exception as e:
            st.error(f"Inference error: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


def _build_actual_labels_dict(idx, df, label_columns, model_labels):
    """Build actual labels dict with proper name mapping between model and dataframe"""
    actual_labels = {}

    df_labels = {}
    for col in label_columns:
        if col in df.columns and idx in df.index:
            val = df.loc[idx, col]
            if pd.notna(val):
                df_labels[col] = int(val)

    for model_label in model_labels:
        found = False

        if model_label in df_labels:
            actual_labels[model_label] = df_labels[model_label]
            found = True

        if not found and '_' in model_label:
            parts = model_label.split('_', 1)
            if len(parts) > 1:
                suffix = parts[1]
                if suffix in df_labels:
                    actual_labels[model_label] = df_labels[suffix]
                    found = True

        if not found and model_label in ['EVENT', 'CAUSE', 'ACTION']:
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

        if not found:
            actual_labels[model_label] = 0

    return actual_labels


def _display_inference_result(result, actual_labels_dict, label_columns):
    """Display inference results with comparison to actual labels"""

    st.markdown("#### 🤖 Model Predictions")

    from model_inference import compare_predictions_to_labels
    comparison = compare_predictions_to_labels(result['predictions'], actual_labels_dict)

    comparison_data = []
    for label in sorted(result['probabilities'].keys()):
        pred_prob = result['probabilities'][label]
        is_predicted = label in result['predicted_labels']
        pred_str = f"✓ {pred_prob:.2f}" if is_predicted else f"  {pred_prob:.2f}"

        actual_val = actual_labels_dict.get(label, 0)
        actual_str = "✓" if actual_val == 1 else "—"

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

    tp = sum(1 for c in comparison.values() if "True Positive" in c)
    tn = sum(1 for c in comparison.values() if "True Negative" in c)
    fp = sum(1 for c in comparison.values() if "False Positive" in c)
    fn = sum(1 for c in comparison.values() if "False Negative" in c)

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
            inference_result = st.session_state.model_loader.predict_passage(text)

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
    st.markdown("## 🔍 HRAF Dataset Tool")

    # ========================================================================
    # DATA SELECTION
    # ========================================================================
    with st.container():
        st.markdown("### 📂 Data")

        browse_mode = st.radio(
            "",
            ["Quick", "Browse"],
            key="browse_mode_selector",
            horizontal=True,
            label_visibility="collapsed"
        )

        st.session_state.browse_mode = "browse" if "Browse" in browse_mode else "quick"

        if st.session_state.browse_mode == "quick":
            xlsx_files = get_xlsx_files_in_directory(DATA_DIR)

            if not xlsx_files:
                st.warning("No .xlsx files in `data/`")
                selected_file = None
            else:
                file_options = {f.name: str(f) for f in xlsx_files}
                selected_name = st.selectbox(
                    "File:",
                    options=list(file_options.keys()),
                    key="quick_file_selector",
                    label_visibility="collapsed"
                )
                selected_file = file_options[selected_name]
        else:
            selected_file = render_directory_browser(key_prefix="data")

        if selected_file and st.button("📂 Load", type="primary", use_container_width=True):
            # Quick load without showing settings
            with st.spinner("Loading..."):
                df, finder, label_columns, cache, passage_col, namespace = load_data(selected_file, header_row=1)

                if df is not None and passage_col is not None:
                    st.session_state.df = df
                    st.session_state.finder = finder
                    st.session_state.label_columns = label_columns
                    st.session_state.cache = cache
                    st.session_state.passage_col = passage_col
                    st.session_state.selected_file = selected_file
                    st.session_state.namespace = namespace
                    st.session_state.initialized = True

                    # Auto-navigate to Compute if no cache
                    if cache is None:
                        st.session_state.current_page = "💻 Compute Scores"

                    st.rerun()

    st.divider()

    # ========================================================================
    # DATA STATUS
    # ========================================================================
    if st.session_state.initialized:
        with st.container():
            df = st.session_state.df
            cache = st.session_state.cache

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Passages", len(df), label_visibility="visible")
            with col2:
                if cache:
                    st.metric("Scored", len(cache['df_summary']))
                else:
                    st.metric("Scored", "—")

            with st.expander("📋 Details"):
                st.caption(f"**File:** {Path(st.session_state.selected_file).name}")
                st.caption(f"**Namespace:** {st.session_state.get('namespace', 'N/A')}")
                st.caption(f"**Column:** {st.session_state.get('passage_col', 'N/A')}")
                st.caption(f"**Labels:** {len(st.session_state.label_columns)}")

                if cache:
                    cache_file = get_cache_filename(st.session_state.selected_file)
                    if os.path.exists(cache_file):
                        cache_date = datetime.fromtimestamp(os.path.getmtime(cache_file))
                        st.caption(f"**Computed:** {cache_date.strftime('%Y-%m-%d %H:%M')}")

    st.divider()

    # ========================================================================
    # MODEL SELECTION
    # ========================================================================
    with st.container():
        st.markdown("### 🤖 Model")

        model_browse_mode = st.radio(
            "",
            ["Quick", "Browse"],
            key="model_browse_mode_selector",
            horizontal=True,
            label_visibility="collapsed"
        )

        st.session_state.model_browse_mode = "browse" if "Browse" in model_browse_mode else "quick"

        selected_model_path = None

        if st.session_state.model_browse_mode == "quick":
            from model_inference import find_model_directories
            model_dirs = find_model_directories("./models")

            if not model_dirs:
                st.warning("No models in `./models/`")
            else:
                model_options = {str(m.parent.name if m.name == "final_model" else m.name): str(m) for m in model_dirs}
                selected_model_name = st.selectbox(
                    "Model:",
                    options=list(model_options.keys()),
                    key="model_selector_quick",
                    label_visibility="collapsed"
                )
                selected_model_path = model_options[selected_model_name]

                if st.button("🔄 Load", type="primary", key="load_model_quick", use_container_width=True):
                    with st.spinner("Loading..."):
                        success = st.session_state.model_loader.load_model(selected_model_path)
                        if success:
                            st.session_state.loaded_model_path = selected_model_path
                            st.rerun()
        else:
            selected_model_path = render_directory_browser(key_prefix="model")

            if selected_model_path and st.button("🔄 Load", type="primary", key="load_model_browse", use_container_width=True):
                with st.spinner("Loading..."):
                    success = st.session_state.model_loader.load_model(selected_model_path)
                    if success:
                        st.session_state.loaded_model_path = selected_model_path
                        st.rerun()

    # Show loaded model status
    if st.session_state.model_loader.is_loaded():
        with st.container():
            model_info = st.session_state.model_loader.get_model_info()

            col1, col2 = st.columns(2)

            with col1:
                if model_info:
                    test_f1 = model_info.get('test_results', {}).get('eval_f1_micro', None)
                    if test_f1:
                        st.metric("F1", f"{test_f1:.3f}")
                    else:
                        st.metric("F1", "—")

            with col2:
                if model_info:
                    config = model_info.get('config', {})
                    hierarchy = "Hier" if config.get('use_hierarchy') else "Flat"
                    st.metric("Type", hierarchy)

            with st.expander("📋 Config"):
                if st.session_state.loaded_model_path:
                    model_name = Path(st.session_state.loaded_model_path).parent.name
                    st.caption(f"**Model:** {model_name}")

                if model_info:
                    st.json(model_info.get('config', {}))


# ============================================================================
# MAIN CONTENT
# ============================================================================

if not st.session_state.initialized:
    st.markdown("# 🔍 HRAF Golden Dataset Discovery")
    st.markdown("""
    ### Welcome!
    
    This tool identifies high-quality passages for NLP training.
    
    **Get Started:**
    1. 👈 Select a dataset in the sidebar
    2. Click "Load" to load your data
    3. Follow the workflow through each page
    
    **Features:**
    - Compute quality scores for passages
    - Load trained models for inference testing
    - Search and explore passages
    - Create tiered training datasets
    - Export results
    """)

else:
    # ============================================================================
    # DATA LOADED - Show navigation and page content
    # ============================================================================

    st.markdown("# 🔍 HRAF Golden Dataset Discovery")

    # Use session state for current page if set, otherwise default
    if st.session_state.current_page:
        default_index = ["📊 Overview", "💻 Compute Scores", "🔍 Search", "⚙️ Thresholds", "📦 Tiers", "🤖 Model Inference", "💾 Export"].index(st.session_state.current_page)
    else:
        default_index = 0

    page = st.radio(
        "Navigate:",
        ["📊 Overview", "💻 Compute Scores", "🔍 Search", "⚙️ Thresholds", "📦 Tiers", "🤖 Model Inference", "💾 Export"],
        horizontal=True,
        label_visibility="visible",
        index=default_index
    )

    # Update current page
    st.session_state.current_page = page

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
                st.metric("With Scores", "—")

        if cache:
            st.markdown("### 📈 Score Statistics")
            scores_df = cache['df_summary']

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Consistency**")
                st.write(f"Median: {scores_df['consistency_avg'].median():.3f}")
                st.write(f"Mean: {scores_df['consistency_avg'].mean():.3f}")

                st.markdown("**Passages ≥ threshold:**")
                for thresh in [0.3, 0.5, 0.7]:
                    count = (scores_df['consistency_avg'] >= thresh).sum()
                    pct = count / len(scores_df) * 100
                    st.write(f"  {thresh}: {count} ({pct:.1f}%)")

            with col2:
                st.markdown("**Rerank**")
                st.write(f"Median: {scores_df['rerank_avg'].median():.3f}")
                st.write(f"Mean: {scores_df['rerank_avg'].mean():.3f}")

                st.markdown("**Passages ≥ threshold:**")
                for thresh in [0.3, 0.5, 0.7]:
                    count = (scores_df['rerank_avg'] >= thresh).sum()
                    pct = count / len(scores_df) * 100
                    st.write(f"  {thresh}: {count} ({pct:.1f}%)")

            if scores_df['consistency_avg'].median() < 0.4:
                st.warning("""
                ⚠️ **Low Consistency Detected**
                
                Median consistency < 0.4 suggests high inter-rater disagreement.
                
                **Recommendation:** Use rerank scores more heavily or lower consistency thresholds.
                """)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.hist(scores_df['consistency_avg'], bins=50, edgecolor='black', alpha=0.7)
            ax1.axvline(scores_df['consistency_avg'].median(), color='red', linestyle='--',
                       label=f'Median: {scores_df["consistency_avg"].median():.3f}')
            ax1.set_xlabel('Consistency Score')
            ax1.set_title('Consistency Distribution')
            ax1.legend()
            ax1.grid(alpha=0.3)

            ax2.hist(scores_df['rerank_avg'], bins=50, edgecolor='black', alpha=0.7, color='green')
            ax2.axvline(scores_df['rerank_avg'].median(), color='red', linestyle='--',
                       label=f'Median: {scores_df["rerank_avg"].median():.3f}')
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
            st.dataframe(label_stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("💡 No scores computed yet. Go to 'Compute Scores' to generate them.")

    elif page == "💻 Compute Scores":
        st.markdown("## 💻 Compute Quality Scores")

        df = st.session_state.df
        finder = st.session_state.finder
        all_label_columns = st.session_state.label_columns
        selected_file = st.session_state.selected_file
        namespace = st.session_state.get('namespace', 'default')
        cache_file = get_cache_filename(selected_file)

        # Configuration section
        st.markdown("### ⚙️ Configuration")

        col1, col2 = st.columns(2)

        with col1:
            header_row = st.number_input(
                "Header row:",
                min_value=0,
                max_value=5,
                value=1,
                help="0=first row, 1=second row"
            )

            # Detect passage column with current settings
            temp_df = pd.read_excel(selected_file, header=header_row)
            detected_col = detect_passage_column(temp_df)

            passage_col_override = st.text_input(
                "Passage column:",
                value=detected_col if detected_col else "",
                placeholder="e.g., Passage",
                help="Auto-detected, but you can override"
            )

        with col2:
            st.markdown("**Select labels to compute:**")
            selected_labels = st.multiselect(
                "Labels:",
                options=all_label_columns,
                default=all_label_columns,
                label_visibility="collapsed",
                help="Choose which labels to include in computation"
            )

        # Reload data if settings changed
        if st.button("↻ Apply Settings", type="secondary"):
            with st.spinner("Reloading..."):
                df, finder, label_columns, cache, passage_col, namespace = load_data(
                    selected_file,
                    header_row=header_row,
                    passage_col_override=passage_col_override if passage_col_override else None
                )

                if df is not None:
                    # Filter to selected labels
                    label_columns = [l for l in label_columns if l in selected_labels]

                    st.session_state.df = df
                    st.session_state.finder = finder
                    st.session_state.label_columns = label_columns
                    st.session_state.cache = cache
                    st.session_state.passage_col = passage_col
                    st.session_state.namespace = namespace
                    st.success("✅ Settings applied!")
                    st.rerun()

        st.markdown("---")

        # Use current label columns (filtered if settings were applied)
        label_columns = st.session_state.label_columns
        passage_col = st.session_state.get('passage_col', 'Passage')

        st.markdown("""
        ### What This Does
        
        Computes two quality scores for each passage:
        
        1. **Consistency Score** - Agreement with similar passages (0-1)
        2. **Rerank Score** - Relevance to label definition (0-1)
        
        Results saved to `data/cached_scores/`
        """)

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
            st.warning(f"⚠️ Cached scores exist (from {cache_date.strftime('%Y-%m-%d %H:%M')})")

        with st.expander("⚙️ Advanced Settings"):
            k_similar = st.slider("Similar passages to check:", 5, 30, 15)

        col1, col2 = st.columns([1, 3])
        with col1:
            compute_button = st.button("🚀 Compute", type="primary")
        with col2:
            if os.path.exists(cache_file):
                if st.button("🔄 Recompute", type="secondary"):
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

                st.success(f"✅ Saved to: {Path(cache_file).name}")

                st.session_state.cache = cache

                st.markdown("---")
                st.markdown("### 📊 Summary")
                scores_df = cache['df_summary']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Scored", len(scores_df))
                with col2:
                    st.metric("Avg Consistency", f"{scores_df['consistency_avg'].mean():.3f}")
                with col3:
                    st.metric("Avg Rerank", f"{scores_df['rerank_avg'].mean():.3f}")

                st.success("✅ Complete! Go to Overview or Thresholds to see results.")

            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                with st.expander("📋 Error Details"):
                    st.code(traceback.format_exc())

    elif page == "🔍 Search":
        st.markdown("## 🔍 Enhanced Search")

        df = st.session_state.df
        passage_col = st.session_state.get('passage_col', 'Passage')
        label_columns = st.session_state.label_columns
        cache = st.session_state.cache
        finder = st.session_state.finder
        namespace = st.session_state.get('namespace', 'main')

        search_mode = st.radio(
            "Search mode:",
            ["📝 Text Query", "🔍 Similar to Passage", "🏷️ Label Semantic"],
            horizontal=True,
            key="search_mode_radio"
        )

        if 'last_search_mode' not in st.session_state:
            st.session_state.last_search_mode = search_mode

        if st.session_state.last_search_mode != search_mode:
            st.session_state.search_results = None
            st.session_state.search_mode = None
            st.session_state.last_search_mode = search_mode

        if st.session_state.get('search_results'):
            if st.button("🗑️ Clear", type="secondary"):
                st.session_state.search_results = None
                st.session_state.search_mode = None
                st.rerun()

        st.markdown("---")

        if search_mode == "📝 Text Query":
            st.markdown("### 📝 Search by Text Query")
            query = st.text_input(
                "Query:",
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
                    "Results:",
                    min_value=1,
                    max_value=50,
                    value=10,
                    key="top_k_results"
                )

            with st.expander("⚙️ Advanced"):
                col1, col2 = st.columns(2)
                with col1:
                    top_k_vector = st.slider("Vector candidates:", 10, 200, 100)
                    min_similarity = st.slider("Min similarity:", 0.0, 1.0, 0.3, 0.05)

                with col2:
                    use_rerank = st.checkbox("Use reranking", value=True)
                    if use_rerank:
                        instruction = st.text_area(
                            "Reranker instruction:",
                            placeholder="e.g., Prioritize detailed descriptions",
                            height=80
                        )
                    else:
                        instruction = None

            if st.button("🔍 Search", type="primary", key="search_text"):
                if not query:
                    st.warning("Enter a query")
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
                                st.warning("No results found")
                                st.session_state.search_results = None
                            else:
                                st.success(f"Found {len(results)} results")
                                st.session_state.search_results = results
                                st.session_state.search_mode = "text_query"

                        except Exception as e:
                            st.error(f"Search error: {e}")

            if st.session_state.get('search_results') and st.session_state.get('search_mode') == "text_query":
                _render_search_results(
                    st.session_state.search_results, df, passage_col, label_columns, cache, namespace, finder
                )

        elif search_mode == "🔍 Similar to Passage":
            st.markdown("### 🔍 Find Similar Passages")

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
                    "Results:",
                    min_value=1,
                    max_value=50,
                    value=10,
                    key="k_similar"
                )

            label_filter = st.selectbox(
                "Filter (optional):",
                ["None"] + label_columns,
                key="label_filter_similar"
            )
            label_filter = None if label_filter == "None" else label_filter

            if passage_idx in df.index and passage_col in df.columns:
                with st.expander(f"📄 Reference: Passage {passage_idx}", expanded=True):
                    ref_text = df.loc[passage_idx, passage_col]
                    if pd.notna(ref_text):
                        st.write(ref_text[:500] + "..." if len(ref_text) > 500 else ref_text)
                        active_labels = [l for l in label_columns if df.loc[passage_idx, l] == 1]
                        if active_labels:
                            st.markdown(f"**Labels:** {', '.join(active_labels)}")

            if st.button("🔍 Find Similar", type="primary", key="search_similar"):
                with st.spinner("Finding..."):
                    try:
                        results = finder.search_similar_to_passage(
                            passage_idx=passage_idx,
                            namespace=namespace,
                            k=k_similar,
                            label_filter=label_filter
                        )

                        if not results:
                            st.warning("No similar passages")
                            st.session_state.search_results = None
                        else:
                            st.success(f"Found {len(results)}")
                            formatted_results = []
                            for r in results:
                                formatted_results.append({
                                    'passage_idx': r['passage_idx'],
                                    'vector_score': r['similarity'],
                                    'combined_score': r['similarity'],
                                    'metadata': r['metadata']
                                })
                            st.session_state.search_results = formatted_results
                            st.session_state.search_mode = "similar"

                    except Exception as e:
                        st.error(f"Error: {e}")

            if st.session_state.get('search_results') and st.session_state.get('search_mode') == "similar":
                _render_search_results(
                    st.session_state.search_results, df, passage_col, label_columns, cache, namespace, finder
                )

        else:  # Label Semantic
            st.markdown("### 🏷️ Label Semantic Search")

            col1, col2 = st.columns(2)
            with col1:
                selected_label = st.selectbox(
                    "Label:",
                    label_columns,
                    key="semantic_label"
                )

            with col2:
                top_k_semantic = st.number_input(
                    "Results:",
                    min_value=1,
                    max_value=50,
                    value=10,
                    key="top_k_semantic"
                )

            if selected_label in finder.LABEL_QUERIES:
                st.info(f"**Definition:** {finder.LABEL_QUERIES[selected_label]}")

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
                            st.warning("No results")
                            st.session_state.search_results = None
                        else:
                            st.success(f"Found {len(results)}")
                            st.session_state.search_results = results
                            st.session_state.search_mode = "semantic"

                    except Exception as e:
                        st.error(f"Error: {e}")

            if st.session_state.get('search_results') and st.session_state.get('search_mode') == "semantic":
                _render_search_results(
                    st.session_state.search_results, df, passage_col, label_columns, cache, namespace, finder
                )

    elif page == "⚙️ Thresholds":
        st.markdown("## ⚙️ Configure Thresholds")

        df = st.session_state.df
        cache = st.session_state.cache
        label_columns = st.session_state.label_columns

        if not cache:
            st.error("⚠️ No scores. Go to Compute Scores first.")
            st.stop()

        scores_df = cache['df_summary']

        with st.expander("📖 How to Choose"):
            st.markdown("""
            **Goal:** Balance quality vs. quantity
            
            **High-Quality Data (consistency > 0.5):**
            - Use composite score (50/50 average)
            
            **Noisy Data (consistency < 0.4):**
            - Use rerank only OR weight rerank heavily (70/30)
            
            **Your data:**
            - Consistency median: {:.3f}
            - Rerank median: {:.3f}
            """.format(scores_df['consistency_avg'].median(), scores_df['rerank_avg'].median()))

        strategy = st.radio(
            "Scoring strategy:",
            [
                "Composite (50/50)",
                "Rerank Only",
                "Weighted (70% rerank)",
                "Custom"
            ],
            horizontal=True
        )

        if strategy == "Custom":
            rerank_weight = st.slider("Rerank weight:", 0.0, 1.0, 0.7, 0.05)
            consistency_weight = 1.0 - rerank_weight

        st.markdown("---")

        if strategy == "Rerank Only":
            min_rerank = st.slider("Min Rerank", 0.0, 1.0,
                                  float(scores_df['rerank_avg'].quantile(0.3)), 0.05)
            min_cons = 0.0
            golden = scores_df[scores_df['rerank_avg'] >= min_rerank].copy()
            golden['composite'] = golden['rerank_avg']

        else:
            col1, col2 = st.columns(2)

            default_cons = max(0.3, float(scores_df['consistency_avg'].quantile(0.25)))
            default_rerank = float(scores_df['rerank_avg'].quantile(0.4))

            with col1:
                min_cons = st.slider("Min Consistency", 0.0, 1.0, default_cons, 0.05)
            with col2:
                min_rerank = st.slider("Min Rerank", 0.0, 1.0, default_rerank, 0.05)

            golden = scores_df[
                (scores_df['consistency_avg'] >= min_cons) &
                (scores_df['rerank_avg'] >= min_rerank)
            ].copy()

            if strategy == "Composite (50/50)":
                golden['composite'] = (golden['consistency_avg'] + golden['rerank_avg']) / 2
            elif strategy == "Weighted (70% rerank)":
                golden['composite'] = 0.7 * golden['rerank_avg'] + 0.3 * golden['consistency_avg']
            elif strategy == "Custom":
                golden['composite'] = rerank_weight * golden['rerank_avg'] + consistency_weight * golden['consistency_avg']

        if len(golden) == 0:
            st.error("❌ No passages meet criteria!")
        else:
            golden = golden.sort_values('composite', ascending=False)
            st.session_state.golden_dataset = golden

            st.markdown("---")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Golden", len(golden))
            with col2:
                st.metric("%", f"{len(golden)/len(scores_df)*100:.1f}%")
            with col3:
                st.metric("Avg Quality", f"{golden['composite'].mean():.3f}")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            if strategy == "Rerank Only":
                ax1.hist(scores_df['rerank_avg'], bins=50, alpha=0.5, edgecolor='black', label='All', color='lightgray')
                ax1.hist(golden['rerank_avg'], bins=30, alpha=0.7, label='Golden', color='gold', edgecolor='black')
                ax1.axvline(min_rerank, color='red', linestyle='--', alpha=0.7)
                ax1.set_xlabel('Rerank Score')
                ax1.set_title('Selection')
                ax1.legend()

                ax2.hist(golden['composite'], bins=30, alpha=0.7, color='gold', edgecolor='black')
                ax2.axvline(golden['composite'].mean(), color='red', linestyle='--')
                ax2.set_xlabel('Quality Score')
                ax2.set_title('Golden Set')
            else:
                ax1.scatter(scores_df['consistency_avg'], scores_df['rerank_avg'],
                           alpha=0.3, s=20, color='gray', label='All')
                ax1.scatter(golden['consistency_avg'], golden['rerank_avg'],
                           alpha=0.7, s=30, color='gold', label='Golden')
                ax1.axvline(min_cons, color='red', linestyle='--', alpha=0.7)
                ax1.axhline(min_rerank, color='blue', linestyle='--', alpha=0.7)
                ax1.set_xlabel('Consistency')
                ax1.set_ylabel('Rerank')
                ax1.set_title('Selection')
                ax1.legend()

                all_comp = (scores_df['consistency_avg'] + scores_df['rerank_avg']) / 2
                ax2.hist(all_comp, bins=50, alpha=0.5, edgecolor='black', label='All')
                ax2.hist(golden['composite'], bins=30, alpha=0.7, color='gold', edgecolor='black', label='Golden')
                ax2.axvline(golden['composite'].mean(), color='red', linestyle='--')
                ax2.set_xlabel('Composite')
                ax2.set_title('Distribution')
                ax2.legend()

            plt.tight_layout()
            st.pyplot(fig)

    elif page == "📦 Tiers":
        st.markdown("## 📦 Create Dataset Tiers")

        df = st.session_state.df
        cache = st.session_state.cache
        label_columns = st.session_state.label_columns

        if not cache:
            st.error("⚠️ No scores")
            st.stop()

        scores_df = cache['df_summary']

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
            with st.spinner("Creating..."):
                tier1, tier2, inference = create_tiered_datasets(df, scores_df, label_columns,
                                                                 tier1_cons, tier1_rerank, tier1_pct, tier2_pct)
                st.session_state.tier1_dataset = tier1
                st.session_state.tier2_dataset = tier2
                st.session_state.inference_dataset = inference
                st.success("✅ Created!")

        if st.session_state.tier1_dataset is not None:
            tier1 = st.session_state.tier1_dataset
            tier2 = st.session_state.tier2_dataset
            inference = st.session_state.inference_dataset

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### 🏆 Tier 1")
                st.metric("Count", len(tier1))
                st.metric("%", f"{len(tier1)/len(df)*100:.1f}%")
            with col2:
                st.markdown("#### 📚 Tier 2")
                st.metric("Count", len(tier2))
                st.metric("%", f"{len(tier2)/len(df)*100:.1f}%")
            with col3:
                st.markdown("#### 🎯 Inference")
                st.metric("Count", len(inference))
                st.metric("%", f"{len(inference)/len(df)*100:.1f}%")

    elif page == "🤖 Model Inference":
        st.markdown("## 🤖 Model Inference Testing")

        df = st.session_state.df
        passage_col = st.session_state.get('passage_col', 'Passage')
        label_columns = st.session_state.label_columns

        if not st.session_state.model_loader.is_loaded():
            st.warning("⚠️ No model loaded")
            st.info("👈 Load a model in the sidebar first")
            st.stop()

        st.markdown("Test your model on passages to validate predictions.")

        inference_mode = st.radio(
            "Mode:",
            ["From Dataset", "Custom Text"],
            horizontal=True
        )

        if inference_mode == "From Dataset":
            col1, col2 = st.columns(2)

            with col1:
                filter_by = st.selectbox(
                    "Filter:",
                    ["All", "Has label", "Random", "Index range"]
                )

            with col2:
                if filter_by == "Has label":
                    filter_label = st.selectbox("Label:", label_columns)
                elif filter_by == "Random":
                    sample_size = st.number_input("Size:", 1, 100, 10)
                elif filter_by == "Index range":
                    start_idx = st.number_input("Start:", 0, len(df) - 1, 0)
                    end_idx = st.number_input("End:", start_idx + 1, len(df), min(start_idx + 10, len(df)))

            if filter_by == "All":
                available_indices = df.index.tolist()
            elif filter_by == "Has label":
                available_indices = df[df[filter_label] == 1].index.tolist()
            elif filter_by == "Random":
                available_indices = df.sample(n=min(sample_size, len(df))).index.tolist()
            elif filter_by == "Index range":
                available_indices = df.iloc[start_idx:end_idx].index.tolist()

            num_to_show = st.slider("Test:", 1, min(20, len(available_indices)), 5)

            if st.button("🔮 Predict", type="primary"):
                selected_indices = available_indices[:num_to_show]

                for idx in selected_indices:
                    passage_text = df.loc[idx, passage_col]

                    if pd.isna(passage_text) or not isinstance(passage_text, str):
                        st.warning(f"⚠️ Passage {idx} has no text")
                        continue

                    actual_labels = {}
                    for col in label_columns:
                        if col in df.columns:
                            val = df.loc[idx, col]
                            actual_labels[col] = 0 if pd.isna(val) else int(val)

                    with st.expander(f"📄 Passage {idx}"):
                        st.markdown("**Text:**")
                        st.write(passage_text[:500] + "..." if len(passage_text) > 500 else passage_text)

                        with st.spinner("Predicting..."):
                            result = st.session_state.model_loader.predict_passage(passage_text)

                        from model_inference import compare_predictions_to_labels
                        comparison = compare_predictions_to_labels(result['predictions'], actual_labels)

                        all_model_labels = result['probabilities'].keys()

                        ra_coded_labels = set()
                        for label, val in actual_labels.items():
                            if val == 1:
                                ra_coded_labels.add(label)
                                if label in ['Illness', 'Accident', 'Other']:
                                    ra_coded_labels.add('EVENT')
                                elif label in ['Just_Happens', 'Material_Physical', 'Spirits_Gods',
                                               'Witchcraft_Sorcery', 'Rule_Violation_Taboo']:
                                    ra_coded_labels.add('CAUSE')
                                elif label in ['Physical_Material', 'Technical_Specialist', 'Divination',
                                               'Shaman_Medium_Healer', 'Priest_High_Religion', 'Other.2']:
                                    ra_coded_labels.add('ACTION')

                        comparison_data = []
                        for label in sorted(all_model_labels):
                            pred_prob = result['probabilities'][label]
                            is_predicted = label in result['predicted_labels']
                            pred_str = f"✓ {pred_prob:.2f}" if is_predicted else f"  {pred_prob:.2f}"

                            ra_label = None
                            if label in ra_coded_labels:
                                ra_label = label
                            elif '_' in label:
                                suffix = label.split('_', 1)[1]
                                if suffix in ra_coded_labels:
                                    ra_label = suffix

                            ra_str = f"✓" if ra_label else "—"

                            comp = comparison.get(label, "")
                            if "True Positive" in comp:
                                comp_str = "✓"
                                comp_color = "🟢"
                            elif "True Negative" in comp:
                                comp_str = "✓"
                                comp_color = "⚪"
                            elif "False Positive" in comp:
                                comp_str = "✗ FP"
                                comp_color = "🔴"
                            elif "False Negative" in comp:
                                comp_str = "✗ FN"
                                comp_color = "🟡"
                            else:
                                comp_str = "—"
                                comp_color = ""

                            comparison_data.append({
                                'Label': label,
                                'Pred': pred_str,
                                'Actual': ra_str,
                                'Result': f"{comp_color} {comp_str}".strip()
                            })

                        st.dataframe(
                            pd.DataFrame(comparison_data),
                            hide_index=True,
                            use_container_width=True
                        )

        else:  # Custom Text
            custom_text = st.text_area(
                "Passage:",
                placeholder="Enter text...",
                height=150
            )

            use_optimal = st.checkbox("Use optimal thresholds", value=True)

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

                if result['predicted_labels']:
                    st.markdown("**Predicted:**")
                    for label in result['predicted_labels']:
                        prob = result['probabilities'][label]
                        st.write(f"✓ **{label}** ({prob:.3f})")
                else:
                    st.info("No labels predicted")

                with st.expander("📊 All Probabilities"):
                    prob_df = pd.DataFrame([
                        {"Label": k, "Probability": v}
                        for k, v in result['probabilities'].items()
                    ]).sort_values('Probability', ascending=False)
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    elif page == "💾 Export":
        st.markdown("## 💾 Export Results")

        df = st.session_state.df
        cache = st.session_state.cache
        label_columns = st.session_state.label_columns
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if cache is not None:
            st.markdown("### 📊 Score Results")
            scores_df = cache['df_summary']

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Summary**")
                output_summary = io.BytesIO()
                scores_df.to_excel(output_summary, index=False, engine='openpyxl')

                st.download_button(
                    label="📥 Download Summary",
                    data=output_summary.getvalue(),
                    file_name=f"scores_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            with col2:
                st.markdown("**Detailed**")
                detailed_rows = []
                for idx in scores_df['passage_idx'].tolist():
                    row_data = {
                        'passage_idx': idx,
                        'consistency_avg': scores_df[scores_df['passage_idx'] == idx]['consistency_avg'].iloc[0],
                        'rerank_avg': scores_df[scores_df['passage_idx'] == idx]['rerank_avg'].iloc[0],
                        'num_labels': scores_df[scores_df['passage_idx'] == idx]['num_labels'].iloc[0]
                    }
                    detailed_rows.append(row_data)

                detailed_df = pd.DataFrame(detailed_rows)

                output_detailed = io.BytesIO()
                detailed_df.to_excel(output_detailed, index=False, engine='openpyxl')

                st.download_button(
                    label="📥 Download Detailed",
                    data=output_detailed.getvalue(),
                    file_name=f"scores_detailed_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            st.markdown("---")

        if st.session_state.golden_dataset is not None:
            st.markdown("### 🏆 Golden Dataset")
            golden = st.session_state.golden_dataset

            golden_indices = golden['passage_idx'].tolist()
            golden_full = df.loc[golden_indices].copy()

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                golden_full.to_excel(writer, index=False)

            st.download_button(
                label="📥 Download Golden Dataset",
                data=output.getvalue(),
                file_name=f"golden_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Footer
st.markdown("---")
st.caption("HRAF Golden Dataset Discovery | Built with Streamlit")