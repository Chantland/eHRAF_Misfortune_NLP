"""
HRAF Golden Dataset Discovery - With Directory Navigator
Run with: streamlit run app_golden_dataset.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os
import pickle
from pathlib import Path
from datetime import datetime
import io
from tqdm import tqdm

from discovery_architecture import GoldenDatasetFinder
from model_training import render_training_page
from model_inference import HRAFModelLoader
from chat_assistant import render_chat_page
from data_preparation import render_data_preparation_page


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

# CHANGED: Initialize multi-model support instead of single model
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = {}  # Dictionary: {model_name: HRAFModelLoader}
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

# Configuration presets
TIER_PRESETS = {
    'balanced': {
        'name': 'Balanced (Recommended)',
        'description': 'Good balance between quality and quantity',
        'tier1': {
            'min_consistency': 0.65,
            'max_consistency': 1.00,
            'min_rerank': 0.45,
            'max_rerank': 1.00,
            'target_pct': 12
        },
        'tier2': {
            'min_consistency': 0.45,
            'max_consistency': 0.65,
            'min_rerank': 0.30,
            'max_rerank': 0.45,
            'target_pct': 25
        }
    },
    'conservative': {
        'name': 'Conservative (High Quality)',
        'description': 'Highest quality, smaller training set',
        'tier1': {
            'min_consistency': 0.70,
            'max_consistency': 1.00,
            'min_rerank': 0.50,
            'max_rerank': 1.00,
            'target_pct': 10
        },
        'tier2': {
            'min_consistency': 0.50,
            'max_consistency': 0.70,
            'min_rerank': 0.35,
            'max_rerank': 0.50,
            'target_pct': 22
        }
    },
    'aggressive': {
        'name': 'Aggressive (More Data)',
        'description': 'More data with acceptable quality floor',
        'tier1': {
            'min_consistency': 0.60,
            'max_consistency': 1.00,
            'min_rerank': 0.40,
            'max_rerank': 1.00,
            'target_pct': 15
        },
        'tier2': {
            'min_consistency': 0.40,
            'max_consistency': 0.60,
            'min_rerank': 0.25,
            'max_rerank': 0.40,
            'target_pct': 28
        }
    }
}

# Label targets for critical bottleneck labels
DEFAULT_LABEL_TARGETS = {
    'tier1': {
        'Just_Happens': 300,
        'Technical_Specialist': 250,
        'Divination': 200,
        'Rule_Violation_Taboo': 250,
        'Priest_High_Religion': 150
    },
    'tier2': {
        'Illness': 400,
        'Accident': 250,
        'Other': 200,
        'Material_Physical': 300,
        'Spirits_Gods': 350,
        'Witchcraft_Sorcery': 200,
        'Physical_Material': 400,
        'Shaman_Medium_Healer': 300
    }
}

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
        st.text_input("Current path", str(current_dir), key=f"{key_prefix}_path_display", disabled=True, label_visibility="collapsed")

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
            if st.button(f"📁 {subdir.name}", key=f"{key_prefix}_dir_{subdir}", width='stretch'):
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
                "Select file",
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


def create_advanced_tiered_datasets(
        df,
        scores_df,
        label_columns,
        tier1_config,
        tier2_config,
        label_targets=None
):
    """
    Create tiered datasets with advanced configuration

    Args:
        df: Full dataset
        scores_df: Quality scores dataframe
        label_columns: List of label column names
        tier1_config: Dict with tier 1 parameters
        tier2_config: Dict with tier 2 parameters
        label_targets: Optional dict with per-label targets

    Returns:
        Tuple of (tier1_df, tier2_df, inference_df, metadata)
    """
    valid_indices = scores_df['passage_idx'].tolist()
    scores_df = scores_df.copy()
    scores_df['composite'] = (scores_df['consistency_avg'] + scores_df['rerank_avg']) / 2

    # Tier 1: Elite training data
    tier1_mask = (
            (scores_df['consistency_avg'] >= tier1_config['min_consistency']) &
            (scores_df['consistency_avg'] <= tier1_config['max_consistency']) &
            (scores_df['rerank_avg'] >= tier1_config['min_rerank']) &
            (scores_df['rerank_avg'] <= tier1_config['max_rerank'])
    )

    tier1_candidates = scores_df[tier1_mask].copy()

    # Apply label targeting for Tier 1 if specified
    if label_targets and 'tier1' in label_targets:
        tier1_indices = apply_label_targeting(
            df, tier1_candidates, label_columns,
            label_targets['tier1'], tier1_config['target_size']
        )
    else:
        # Sort by composite score and take top N
        tier1_candidates = tier1_candidates.sort_values('composite', ascending=False)
        target_count = int(len(valid_indices) * tier1_config['target_pct'] / 100)
        tier1_indices = tier1_candidates.head(target_count)['passage_idx'].tolist()

    # Tier 2: Expansion training data
    remaining_indices = [idx for idx in valid_indices if idx not in tier1_indices]
    remaining_scores = scores_df[scores_df['passage_idx'].isin(remaining_indices)]

    tier2_mask = (
            (remaining_scores['consistency_avg'] >= tier2_config['min_consistency']) &
            (remaining_scores['consistency_avg'] <= tier2_config['max_consistency']) &
            (remaining_scores['rerank_avg'] >= tier2_config['min_rerank']) &
            (remaining_scores['rerank_avg'] <= tier2_config['max_rerank'])
    )

    tier2_candidates = remaining_scores[tier2_mask].copy()

    # Apply label targeting for Tier 2 if specified
    if label_targets and 'tier2' in label_targets:
        tier2_indices = apply_label_targeting(
            df, tier2_candidates, label_columns,
            label_targets['tier2'], tier2_config['target_size']
        )
    else:
        # Mixed strategy: 70% by score, 30% random
        tier2_candidates = tier2_candidates.sort_values('composite', ascending=False)
        target_count = int(len(valid_indices) * tier2_config['target_pct'] / 100)

        scored_count = int(target_count * 0.7)
        random_count = target_count - scored_count

        tier2_scored_indices = tier2_candidates.head(scored_count)['passage_idx'].tolist()
        tier2_pool = [idx for idx in remaining_indices if idx not in tier2_scored_indices]
        tier2_random_indices = np.random.choice(
            tier2_pool,
            size=min(random_count, len(tier2_pool)),
            replace=False
        ).tolist()

        tier2_indices = tier2_scored_indices + tier2_random_indices

    # Inference: Everything else
    inference_indices = [idx for idx in valid_indices
                         if idx not in tier1_indices and idx not in tier2_indices]

    # Create dataframes
    tier1_df = df.loc[tier1_indices].copy()
    tier2_df = df.loc[tier2_indices].copy()
    inference_df = df.loc[inference_indices].copy()

    # Add confidence scores
    for idx in tier1_indices:
        if idx in scores_df['passage_idx'].values:
            score_row = scores_df[scores_df['passage_idx'] == idx].iloc[0]
            tier1_df.loc[idx, 'confidence_composite'] = score_row['composite']
            tier1_df.loc[idx, 'confidence_consistency'] = score_row['consistency_avg']
            tier1_df.loc[idx, 'confidence_rerank'] = score_row['rerank_avg']
            tier1_df.loc[idx, 'tier'] = 1

    for idx in tier2_indices:
        if idx in scores_df['passage_idx'].values:
            score_row = scores_df[scores_df['passage_idx'] == idx].iloc[0]
            tier2_df.loc[idx, 'confidence_composite'] = score_row['composite']
            tier2_df.loc[idx, 'confidence_consistency'] = score_row['consistency_avg']
            tier2_df.loc[idx, 'confidence_rerank'] = score_row['rerank_avg']
            tier2_df.loc[idx, 'tier'] = 2

    for idx in inference_indices:
        if idx in scores_df['passage_idx'].values:
            score_row = scores_df[scores_df['passage_idx'] == idx].iloc[0]
            inference_df.loc[idx, 'confidence_composite'] = score_row['composite']
            inference_df.loc[idx, 'confidence_consistency'] = score_row['consistency_avg']
            inference_df.loc[idx, 'confidence_rerank'] = score_row['rerank_avg']
            inference_df.loc[idx, 'tier'] = 3

    # Generate metadata
    metadata = generate_tier_metadata(
        tier1_df, tier2_df, inference_df,
        label_columns, tier1_config, tier2_config
    )

    return tier1_df, tier2_df, inference_df, metadata


def apply_label_targeting(df, candidates, label_columns, targets, target_size):
    """Apply label-specific targeting to select passages"""
    selected_indices = []
    remaining_candidates = candidates.copy()

    # Priority labels first
    for label, target_count in sorted(targets.items(), key=lambda x: x[1], reverse=True):
        if label not in label_columns:
            continue

        # Find candidates with this label
        label_candidates = []
        for idx in remaining_candidates['passage_idx'].tolist():
            if idx in df.index and df.loc[idx, label] == 1:
                label_candidates.append(idx)

        # Take up to target_count
        selected = label_candidates[:target_count]
        selected_indices.extend(selected)

        # Remove from candidates
        remaining_candidates = remaining_candidates[
            ~remaining_candidates['passage_idx'].isin(selected)
        ]

    # Fill remaining with top-scoring passages
    remaining_needed = target_size - len(selected_indices)
    if remaining_needed > 0:
        remaining_candidates = remaining_candidates.sort_values('composite', ascending=False)
        additional = remaining_candidates.head(remaining_needed)['passage_idx'].tolist()
        selected_indices.extend(additional)

    return selected_indices[:target_size]


def generate_tier_metadata(tier1_df, tier2_df, inference_df, label_columns, tier1_config, tier2_config):
    """Generate comprehensive metadata about tiers"""
    metadata = {
        'created_at': datetime.now().isoformat(),
        'total_passages': len(tier1_df) + len(tier2_df) + len(inference_df),
        'tiers': {}
    }

    for tier_name, tier_df, config in [
        ('tier1', tier1_df, tier1_config),
        ('tier2', tier2_df, tier2_config),
        ('inference', inference_df, None)
    ]:
        tier_meta = {
            'count': len(tier_df),
            'percentage': len(tier_df) / metadata['total_passages'] * 100,
        }

        if config:
            tier_meta['config'] = config

        # Quality statistics
        if 'confidence_consistency' in tier_df.columns:
            tier_meta['quality'] = {
                'consistency_mean': float(tier_df['confidence_consistency'].mean()),
                'consistency_median': float(tier_df['confidence_consistency'].median()),
                'consistency_std': float(tier_df['confidence_consistency'].std()),
                'rerank_mean': float(tier_df['confidence_rerank'].mean()),
                'rerank_median': float(tier_df['confidence_rerank'].median()),
                'rerank_std': float(tier_df['confidence_rerank'].std()),
                'composite_mean': float(tier_df['confidence_composite'].mean()),
                'composite_median': float(tier_df['confidence_composite'].median()),
            }

        # Label distribution
        label_dist = {}
        for label in label_columns:
            if label in tier_df.columns:
                count = int((tier_df[label] == 1).sum())
                label_dist[label] = {
                    'count': count,
                    'percentage': count / len(tier_df) * 100 if len(tier_df) > 0 else 0
                }
        tier_meta['label_distribution'] = label_dist

        metadata['tiers'][tier_name] = tier_meta

    return metadata


def validate_tier_quality(tier1_df, tier2_df, label_columns, targets):
    """Validate tier quality against targets"""
    checks = {
        'tier1': [],
        'tier2': [],
        'warnings': []
    }

    # Tier 1 quality checks
    if 'confidence_consistency' in tier1_df.columns:
        tier1_cons_mean = tier1_df['confidence_consistency'].mean()
        tier1_rerank_mean = tier1_df['confidence_rerank'].mean()

        checks['tier1'].append({
            'check': 'Consistency Mean ≥ 0.70',
            'value': f"{tier1_cons_mean:.3f}",
            'passed': tier1_cons_mean >= 0.70
        })

        checks['tier1'].append({
            'check': 'Rerank Mean ≥ 0.50',
            'value': f"{tier1_rerank_mean:.3f}",
            'passed': tier1_rerank_mean >= 0.50
        })

    # Tier 1 label coverage checks
    if targets and 'tier1' in targets:
        for label, target_count in targets['tier1'].items():
            if label in label_columns:
                actual_count = (tier1_df[label] == 1).sum()
                checks['tier1'].append({
                    'check': f'{label} Count ≥ {target_count}',
                    'value': f"{actual_count}",
                    'passed': actual_count >= target_count
                })

                if actual_count < target_count:
                    checks['warnings'].append(
                        f"Tier 1: {label} has only {actual_count} examples (target: {target_count})"
                    )

    # Tier 2 quality checks
    if 'confidence_consistency' in tier2_df.columns:
        tier2_cons_mean = tier2_df['confidence_consistency'].mean()
        tier2_rerank_mean = tier2_df['confidence_rerank'].mean()

        checks['tier2'].append({
            'check': 'Consistency Mean ≥ 0.50',
            'value': f"{tier2_cons_mean:.3f}",
            'passed': tier2_cons_mean >= 0.50
        })

        checks['tier2'].append({
            'check': 'Rerank Mean ≥ 0.35',
            'value': f"{tier2_rerank_mean:.3f}",
            'passed': tier2_rerank_mean >= 0.35
        })

    # Check all labels are covered
    for label in label_columns:
        tier1_count = (tier1_df[label] == 1).sum()
        tier2_count = (tier2_df[label] == 1).sum()
        combined_count = tier1_count + tier2_count

        if combined_count == 0:
            checks['warnings'].append(f"Label '{label}' has no examples in training tiers!")

    return checks


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
    """Helper function to render search results with multi-model inference"""

    # Check if any models are loaded
    models_loaded = len(st.session_state.loaded_models) > 0

    # Batch inference option
    if models_loaded and len(results) > 1:
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            # Model selection for batch inference
            selected_models = st.multiselect(
                "Select models for batch inference:",
                options=list(st.session_state.loaded_models.keys()),
                default=list(st.session_state.loaded_models.keys()),
                key="batch_models_select"
            )

        with col2:
            if selected_models and st.button("🤖 Run Batch Inference", type="primary", key="batch_inference"):
                st.session_state.run_batch_inference = True
                st.session_state.batch_selected_models = selected_models
                st.rerun()

        with col3:
            if st.session_state.get('batch_inference_results'):
                if st.button("🗑️ Clear Batch Results", type="secondary"):
                    st.session_state.batch_inference_results = None
                    st.session_state.run_batch_inference = False
                    st.rerun()

        st.markdown("---")

    # Run batch inference if triggered
    if st.session_state.get('run_batch_inference'):
        selected_models = st.session_state.get('batch_selected_models', [])
        with st.spinner(f"Running inference on {len(results)} passages with {len(selected_models)} model(s)..."):
            batch_results = _run_batch_inference_multi_model(
                results, df, passage_col, label_columns, selected_models
            )
            st.session_state.batch_inference_results = batch_results
            st.session_state.run_batch_inference = False
            st.success(f"✅ Completed inference on {len(results)} passages with {len(selected_models)} models!")
            st.rerun()

    # Show batch summary if available
    if st.session_state.get('batch_inference_results'):
        _show_multi_model_batch_summary(st.session_state.batch_inference_results, label_columns)
        st.markdown("---")

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
                if models_loaded:
                    batch_results = st.session_state.get('batch_inference_results', {})
                    has_batch_result = idx in batch_results

                    show_inference = st.checkbox(
                        "Show Inference",
                        key=f"show_infer_{idx}_{i}",
                        value=has_batch_result
                    )

            if models_loaded and st.session_state.get(f"show_infer_{idx}_{i}", False):
                st.markdown("---")

                batch_results = st.session_state.get('batch_inference_results', {})
                if idx in batch_results:
                    _display_multi_model_inference_result(
                        batch_results[idx],
                        active_labels,
                        label_columns
                    )
                else:
                    _run_inference_on_passage_multi_model(idx, text, active_labels, label_columns)

            with col3:
                if st.button("📋", key=f"copy_{idx}_{i}", help="Copy text"):
                    st.code(text, language=None)

def _display_multi_model_comparison(model_results, actual_labels, label_columns):
    """Display side-by-side comparison of multiple models"""

    # Get all labels from all models
    all_labels = set()
    for result in model_results.values():
        all_labels.update(result['probabilities'].keys())

    # Create comparison table
    comparison_data = []

    for label in sorted(all_labels):
        row = {'Label': label}

        # Add actual label
        actual_val = actual_labels.get(label, 0)
        row['Actual'] = "✓" if actual_val == 1 else "—"

        # Add each model's prediction and result
        for model_name, result in model_results.items():
            pred_prob = result['probabilities'].get(label, 0)
            is_predicted = label in result['predicted_labels']

            from model_inference import compare_predictions_to_labels
            comparison = compare_predictions_to_labels(result['predictions'], actual_labels)
            comp = comparison.get(label, "")

            if "True Positive" in comp:
                comp_icon = "🟢"
            elif "True Negative" in comp:
                comp_icon = "⚪"
            elif "False Positive" in comp:
                comp_icon = "🔴"
            elif "False Negative" in comp:
                comp_icon = "🟡"
            else:
                comp_icon = ""

            if is_predicted:
                row[f"{model_name}"] = f"{comp_icon} ✓ {pred_prob:.2f}"
            else:
                row[f"{model_name}"] = f"{comp_icon}   {pred_prob:.2f}"

        comparison_data.append(row)

    st.dataframe(
        pd.DataFrame(comparison_data),
        hide_index=True,
        use_container_width=True
    )

    # Calculate and display metrics for each model
    st.markdown("#### Model Performance Metrics")

    metrics_data = []
    for model_name, result in model_results.items():
        from model_inference import compare_predictions_to_labels
        comparison = compare_predictions_to_labels(result['predictions'], actual_labels)

        tp = sum(1 for c in comparison.values() if "True Positive" in c)
        tn = sum(1 for c in comparison.values() if "True Negative" in c)
        fp = sum(1 for c in comparison.values() if "False Positive" in c)
        fn = sum(1 for c in comparison.values() if "False Negative" in c)

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics_data.append({
            'Model': model_name,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'Acc': f"{accuracy:.3f}",
            'Prec': f"{precision:.3f}",
            'Rec': f"{recall:.3f}",
            'F1': f"{f1:.3f}"
        })

    st.dataframe(
        pd.DataFrame(metrics_data),
        hide_index=True,
        use_container_width=True
    )


def load_data(filename, header_row=0, passage_col_override=None):
    """Load Excel data with flexible header handling"""
    try:
        df = pd.read_excel(filename, header=header_row)

        # Flatten column names if multi-level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(map(str, col)).strip('_') for col in df.columns.values]
            print(f"Flattened multi-level columns")

        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]

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


def _run_inference_on_passage_multi_model(idx, text, actual_labels_list, label_columns):
    """Run inference on a single passage with all loaded models"""

    st.markdown("#### 🤖 Model Predictions")

    model_results = {}

    for model_name, loader in st.session_state.loaded_models.items():
        if not loader.is_loaded():
            continue

        with st.spinner(f"Running {model_name}..."):
            try:
                result = loader.predict_passage(text)
                model_results[model_name] = result
            except Exception as e:
                st.error(f"Error with {model_name}: {e}")

    if model_results:
        df = st.session_state.df
        actual_labels_dict = _build_actual_labels_dict(
            idx, df, label_columns,
            list(model_results.values())[0]['probabilities'].keys()
        )

        _display_multi_model_comparison(model_results, actual_labels_dict, label_columns)

def _run_batch_inference_multi_model(results, df, passage_col, label_columns, selected_models):
    """Run inference on all search results with multiple models"""
    batch_results = {}

    for result in results:
        idx = result['passage_idx']

        if idx not in df.index:
            continue

        text = df.loc[idx, passage_col]
        if pd.isna(text) or not isinstance(text, str):
            continue

        passage_results = {
            'actual_labels': _build_actual_labels_dict(idx, df, label_columns, []),
            'model_results': {}
        }

        for model_name in selected_models:
            loader = st.session_state.loaded_models.get(model_name)
            if not loader or not loader.is_loaded():
                continue

            try:
                inference_result = loader.predict_passage(text)
                passage_results['model_results'][model_name] = inference_result
            except Exception as e:
                print(f"Error on passage {idx} with model {model_name}: {e}")
                continue

        if passage_results['model_results']:
            batch_results[idx] = passage_results

    return batch_results


def _display_multi_model_inference_result(passage_data, actual_labels_list, label_columns):
    """Display inference results from multiple models"""

    st.markdown("#### 🤖 Model Predictions Comparison")

    model_results = passage_data['model_results']
    actual_labels = passage_data['actual_labels']

    if not model_results:
        st.warning("No model results available")
        return

    # Create comparison table
    all_labels = set()
    for model_result in model_results.values():
        all_labels.update(model_result['probabilities'].keys())

    comparison_data = []

    for label in sorted(all_labels):
        row = {'Label': label}

        # Add actual label
        actual_val = actual_labels.get(label, 0)
        row['Actual'] = "✓" if actual_val == 1 else "—"

        # Add each model's prediction
        for model_name, result in model_results.items():
            pred_prob = result['probabilities'].get(label, 0)
            is_predicted = label in result['predicted_labels']

            if is_predicted:
                row[f"{model_name}\nPred"] = f"✓ {pred_prob:.2f}"
            else:
                row[f"{model_name}\nPred"] = f"  {pred_prob:.2f}"

        comparison_data.append(row)

    st.dataframe(
        pd.DataFrame(comparison_data),
        hide_index=True,
        width='stretch'
    )

    # Show per-model metrics
    st.markdown("#### 📊 Per-Model Metrics")

    metrics_data = []
    for model_name, result in model_results.items():
        from model_inference import compare_predictions_to_labels
        comparison = compare_predictions_to_labels(result['predictions'], actual_labels)

        tp = sum(1 for c in comparison.values() if "True Positive" in c)
        tn = sum(1 for c in comparison.values() if "True Negative" in c)
        fp = sum(1 for c in comparison.values() if "False Positive" in c)
        fn = sum(1 for c in comparison.values() if "False Negative" in c)

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0

        metrics_data.append({
            'Model': model_name,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'Accuracy': f"{accuracy:.3f}"
        })

    st.dataframe(
        pd.DataFrame(metrics_data),
        hide_index=True,
        use_container_width=True
    )


def _show_multi_model_batch_summary(batch_results, label_columns):
    """Show summary statistics across all batch inference results with multiple models"""
    st.markdown("### 📊 Multi-Model Batch Inference Summary")

    if not batch_results:
        st.warning("No batch results available")
        return

    # Get all models
    model_names = set()
    for passage_data in batch_results.values():
        model_names.update(passage_data['model_results'].keys())

    model_names = sorted(model_names)

    # Calculate metrics per model
    model_stats = {}

    for model_name in model_names:
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0

        for passage_data in batch_results.values():
            if model_name not in passage_data['model_results']:
                continue

            result = passage_data['model_results'][model_name]
            actual_labels = passage_data['actual_labels']

            from model_inference import compare_predictions_to_labels
            comparison = compare_predictions_to_labels(result['predictions'], actual_labels)

            tp = sum(1 for c in comparison.values() if "True Positive" in c)
            tn = sum(1 for c in comparison.values() if "True Negative" in c)
            fp = sum(1 for c in comparison.values() if "False Positive" in c)
            fn = sum(1 for c in comparison.values() if "False Negative" in c)

            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

        model_stats[model_name] = {
            'tp': total_tp,
            'tn': total_tn,
            'fp': total_fp,
            'fn': total_fn
        }

    # Display overall comparison
    st.markdown("#### Overall Performance by Model")

    comparison_data = []
    for model_name, stats in model_stats.items():
        total = stats['tp'] + stats['tn'] + stats['fp'] + stats['fn']
        accuracy = (stats['tp'] + stats['tn']) / total if total > 0 else 0

        precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        comparison_data.append({
            'Model': model_name,
            'TP': stats['tp'],
            'TN': stats['tn'],
            'FP': stats['fp'],
            'FN': stats['fn'],
            'Accuracy': f"{accuracy:.3f}",
            'Precision': f"{precision:.3f}",
            'Recall': f"{recall:.3f}",
            'F1': f"{f1:.3f}"
        })

    st.dataframe(
        pd.DataFrame(comparison_data),
        hide_index=True,
        width='stretch'
    )

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
        width='stretch'
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
        width='stretch'
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
            "Browse mode",
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
                    "Select file",
                    options=list(file_options.keys()),
                    key="quick_file_selector",
                    label_visibility="collapsed"
                )
                selected_file = file_options[selected_name]
        else:
            selected_file = render_directory_browser(key_prefix="data")

        if selected_file and st.button("📂 Load", type="primary", width='stretch'):
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
    # MODEL SELECTION - MULTI-MODEL SUPPORT
    # ========================================================================
    with st.container():
        st.markdown("### 🤖 Models")

        # Show currently loaded models
        if st.session_state.loaded_models:
            st.markdown(f"**Loaded: {len(st.session_state.loaded_models)}**")

            # Display each loaded model with option to unload
            for model_name in list(st.session_state.loaded_models.keys()):
                col1, col2 = st.columns([3, 1])

                with col1:
                    loader = st.session_state.loaded_models[model_name]
                    info = loader.get_model_info()

                    if info:
                        config = info.get('config', {})
                        test_f1 = info.get('test_results', {}).get('eval_f1_micro', None)
                        hierarchy = "Hier" if config.get('use_hierarchy') else "Flat"

                        if test_f1:
                            st.caption(f"**{model_name}** | {hierarchy} | F1: {test_f1:.3f}")
                        else:
                            st.caption(f"**{model_name}** | {hierarchy}")
                    else:
                        st.caption(f"**{model_name}**")

                with col2:
                    if st.button("🗑️", key=f"unload_{model_name}", help="Unload model"):
                        del st.session_state.loaded_models[model_name]
                        st.rerun()

            # Expandable details for each model
            with st.expander("📋 Model Details"):
                selected_model = st.selectbox(
                    "Select model to view:",
                    options=list(st.session_state.loaded_models.keys()),
                    key="model_detail_selector"
                )

                if selected_model:
                    loader = st.session_state.loaded_models[selected_model]
                    info = loader.get_model_info()

                    if info:
                        st.json(info.get('config', {}))

                        # Show optimal thresholds if available
                        if loader.optimal_thresholds:
                            with st.expander("Optimal Thresholds"):
                                thresh_df = pd.DataFrame([
                                    {
                                        'Label': label,
                                        'Threshold': data.get('threshold', 0.5),
                                        'F1': data.get('f1', 0)
                                    }
                                    for label, data in loader.optimal_thresholds.items()
                                ])
                                st.dataframe(thresh_df, hide_index=True)

            st.markdown("---")

        # Model browse and load section
        st.markdown("**Load New Model:**")

        model_browse_mode = st.radio(
            "Browse mode",
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
                    "Select model",
                    options=list(model_options.keys()),
                    key="model_selector_quick",
                    label_visibility="collapsed"
                )
                selected_model_path = model_options[selected_model_name]

                # Input for custom model name
                custom_name = st.text_input(
                    "Model nickname (optional):",
                    value="",
                    key="model_custom_name_quick"
                )

                if st.button("🔄 Load Model", type="primary", key="load_model_quick", width='stretch'):
                    model_name = custom_name or selected_model_name

                    # Check if already loaded
                    if model_name in st.session_state.loaded_models:
                        st.warning(f"Model '{model_name}' already loaded")
                    else:
                        with st.spinner(f"Loading {model_name}..."):
                            from model_inference import HRAFModelLoader

                            new_loader = HRAFModelLoader()
                            success = new_loader.load_model(selected_model_path)

                            if success:
                                st.session_state.loaded_models[model_name] = new_loader
                                st.success(f"✅ Loaded: {model_name}")
                                st.rerun()
                            else:
                                st.error(f"Failed to load {model_name}")
        else:
            selected_model_path = render_directory_browser(key_prefix="model")

            if selected_model_path:
                # Get default name from path
                default_name = Path(selected_model_path).parent.name

                custom_name = st.text_input(
                    "Model nickname:",
                    value=default_name,
                    key="model_custom_name_browse"
                )

                if st.button("🔄 Load Model", type="primary", key="load_model_browse", width='stretch'):
                    model_name = custom_name or default_name

                    if model_name in st.session_state.loaded_models:
                        st.warning(f"Model '{model_name}' already loaded")
                    else:
                        with st.spinner(f"Loading {model_name}..."):
                            from model_inference import HRAFModelLoader

                            new_loader = HRAFModelLoader()
                            success = new_loader.load_model(selected_model_path)

                            if success:
                                st.session_state.loaded_models[model_name] = new_loader
                                st.success(f"✅ Loaded: {model_name}")
                                st.rerun()
                            else:
                                st.error(f"Failed to load {model_name}")


# ============================================================================
# MAIN CONTENT
# ============================================================================

if not st.session_state.initialized:
    st.markdown("# 🔍 HRAF Dataset Tool")
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

    st.markdown("# 🔍 HRAF Dataset Tool")

    # Use session state for current page if set, otherwise default
    if st.session_state.current_page:
        page_list = ["📊 Overview", "💻 Compute Scores", "🔍 Search", "🛠️ Data Prep", "🎓 Train Model", "🤖 Model Inference",
                     "💬 Chat"]
        try:
            default_index = page_list.index(st.session_state.current_page)
        except ValueError:
            default_index = 0
    else:
        default_index = 0

    page = st.radio(
        "Navigate",
        ["📊 Overview", "💻 Compute Scores", "🔍 Search", "🛠️ Data Prep", "🎓 Train Model", "🤖 Model Inference", "💬 Chat"],
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
            st.dataframe(label_stats_df, width='stretch', hide_index=True)
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

        st.info("💡 **Tip**: Files exported from Data Prep have standardized single-row headers")

        col1, col2 = st.columns(2)

        with col1:

            header_row = st.number_input(

                "Header row (0=first row):",

                min_value=0,

                max_value=5,

                value=0,  # Changed default to 0 for single header

                help="Use 0 for exported files, 1 for original HRAF multi-header files"

            )

            # Detect passage column with current settings

            temp_df = pd.read_excel(selected_file, header=header_row)

            # Flatten columns if multi-level

            if isinstance(temp_df.columns, pd.MultiIndex):
                temp_df.columns = ['_'.join(map(str, col)).strip('_') for col in temp_df.columns.values]

            detected_col = detect_passage_column(temp_df)

            passage_col_override = st.text_input(

                "Passage column:",

                value=detected_col if detected_col else "",

                placeholder="e.g., Passage",

                help="Auto-detected, but you can override"

            )

        with col2:

            st.markdown("**Select labels to compute:**")

            # Try to auto-detect labels from temp_df

            auto_labels = []

            for col in temp_df.columns:

                if col != detected_col and temp_df[col].dtype in ['int64', 'float64']:

                    unique_vals = temp_df[col].dropna().unique()

                    if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):

                        if (temp_df[col] == 1).sum() > 0:
                            auto_labels.append(col)

            selected_labels = st.multiselect(

                "Labels to compute",

                options=auto_labels if auto_labels else all_label_columns,

                default=auto_labels if auto_labels else all_label_columns,

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

    elif page == "🔍 Search":
        st.markdown("## 🔍 Enhanced Search")

        df = st.session_state.df
        passage_col = st.session_state.get('passage_col', 'Passage')
        label_columns = st.session_state.label_columns
        cache = st.session_state.cache
        finder = st.session_state.finder
        namespace = st.session_state.get('namespace', 'main')

        search_mode = st.radio(
            "Search mode",
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

        st.markdown("## 📦 Advanced Tier Configuration")
        df = st.session_state.df
        cache = st.session_state.cache
        label_columns = st.session_state.label_columns

        if not cache:
            st.error("⚠️ No scores. Go to Compute Scores first.")
            st.stop()

        scores_df = cache['df_summary']

        st.markdown("""
           Create tiered training datasets with precise quality control and label targeting.

           **Strategy:**
           - **Tier 1**: Elite training data (unambiguous, high-quality examples)
           - **Tier 2**: Expansion data (good quality, broader coverage)
           - **Inference**: Validation/test set (realistic quality distribution)
           """)

        # Configuration tabs
        config_tab, label_tab, validate_tab = st.tabs(["⚙️ Tier Config", "🏷️ Label Targets", "✅ Validate"])
        with config_tab:
            st.markdown("### Configuration Strategy")
            col1, col2 = st.columns([1, 1])

            with col1:
                preset = st.selectbox(
                    "Configuration Preset:",
                    options=['custom'] + list(TIER_PRESETS.keys()),
                    format_func=lambda x: 'Custom' if x == 'custom' else TIER_PRESETS[x]['name'],
                    key='tier_preset'
                )

            with col2:
                if preset != 'custom':
                    st.info(f"**{TIER_PRESETS[preset]['description']}**")

            # Load preset or use custom
            if preset != 'custom':
                tier1_config = TIER_PRESETS[preset]['tier1'].copy()
                tier2_config = TIER_PRESETS[preset]['tier2'].copy()

            else:
                # Initialize custom config

                if 'custom_tier1_config' not in st.session_state:
                    st.session_state.custom_tier1_config = TIER_PRESETS['balanced']['tier1'].copy()
                    st.session_state.custom_tier2_config = TIER_PRESETS['balanced']['tier2'].copy()

                tier1_config = st.session_state.custom_tier1_config
                tier2_config = st.session_state.custom_tier2_config

            st.markdown("---")

            # Tier 1 Configuration
            st.markdown("### 🥇 Tier 1: Elite Training Data")

            col1, col2, col3 = st.columns(3)
            with col1:
                tier1_min_cons = st.slider(
                    "Min Consistency:",
                    0.0, 1.0,
                    tier1_config['min_consistency'],
                    0.05,
                    key='tier1_min_cons'
                )

                tier1_max_cons = st.slider(
                    "Max Consistency:",
                    tier1_min_cons, 1.0,
                    tier1_config['max_consistency'],
                    0.05,
                    key='tier1_max_cons'
                )

            with col2:
                tier1_min_rerank = st.slider(
                    "Min Rerank:",
                    0.0, 1.0,
                    tier1_config['min_rerank'],
                    0.05,
                    key='tier1_min_rerank'
                )

                tier1_max_rerank = st.slider(
                    "Max Rerank:",
                    tier1_min_rerank, 1.0,
                    tier1_config['max_rerank'],
                    0.05,
                    key='tier1_max_rerank'
                )

            with col3:
                tier1_pct = st.slider(
                    "Target Size %:",
                    5, 30,
                    tier1_config['target_pct'],
                    1,
                    key='tier1_pct'
                )

                # Calculate expected count
                expected_tier1 = int(len(scores_df) * tier1_pct / 100)
                st.metric("Expected Count", f"{expected_tier1:,}")

            # Update config
            tier1_config = {
                'min_consistency': tier1_min_cons,
                'max_consistency': tier1_max_cons,
                'min_rerank': tier1_min_rerank,
                'max_rerank': tier1_max_rerank,
                'target_pct': tier1_pct,
                'target_size': expected_tier1
            }

            # Show what passages would qualify
            tier1_candidates = scores_df[
                (scores_df['consistency_avg'] >= tier1_min_cons) &
                (scores_df['consistency_avg'] <= tier1_max_cons) &
                (scores_df['rerank_avg'] >= tier1_min_rerank) &
                (scores_df['rerank_avg'] <= tier1_max_rerank)
                ]

            st.info(
                f"📊 {len(tier1_candidates):,} passages meet Tier 1 criteria ({len(tier1_candidates) / len(scores_df) * 100:.1f}%)")

            st.markdown("---")

            # Tier 2 Configuration
            st.markdown("### 📚 Tier 2: Expansion Training Data")
            col1, col2, col3 = st.columns(3)

            with col1:
                tier2_min_cons = st.slider(
                    "Min Consistency:",
                    0.0, 1.0,
                    tier2_config['min_consistency'],
                    0.05,
                    key='tier2_min_cons'
                )

                tier2_max_cons = st.slider(
                    "Max Consistency:",
                    tier2_min_cons, 1.0,
                    tier2_config['max_consistency'],
                    0.05,
                    key='tier2_max_cons'
                )

            with col2:
                tier2_min_rerank = st.slider(
                    "Min Rerank:",
                    0.0, 1.0,
                    tier2_config['min_rerank'],
                    0.05,
                    key='tier2_min_rerank'
                )

                tier2_max_rerank = st.slider(
                    "Max Rerank:",
                    tier2_min_rerank, 1.0,
                    tier2_config['max_rerank'],
                    0.05,
                    key='tier2_max_rerank'
                )

            with col3:
                tier2_pct = st.slider(
                    "Target Size %:",
                    10, 50,
                    tier2_config['target_pct'],
                    1,
                    key='tier2_pct'
                )

                expected_tier2 = int(len(scores_df) * tier2_pct / 100)
                st.metric("Expected Count", f"{expected_tier2:,}")

            tier2_config = {
                'min_consistency': tier2_min_cons,
                'max_consistency': tier2_max_cons,
                'min_rerank': tier2_min_rerank,
                'max_rerank': tier2_max_rerank,
                'target_pct': tier2_pct,
                'target_size': expected_tier2
            }

            tier2_candidates = scores_df[
                (scores_df['consistency_avg'] >= tier2_min_cons) &
                (scores_df['consistency_avg'] <= tier2_max_cons) &
                (scores_df['rerank_avg'] >= tier2_min_rerank) &
                (scores_df['rerank_avg'] <= tier2_max_rerank)
                ]

            st.info(
                f"📊 {len(tier2_candidates):,} passages meet Tier 2 criteria ({len(tier2_candidates) / len(scores_df) * 100:.1f}%)")

            # Inference set
            inference_pct = 100 - tier1_pct - tier2_pct
            st.markdown(f"### 🎯 Inference: {inference_pct}% (auto)")
            st.caption("Remaining passages used for validation/testing")

        with label_tab:
            st.markdown("### 🏷️ Label Distribution Targets")
            st.markdown("""

               Optionally specify target counts for specific labels in each tier.

               This ensures critical bottleneck labels have sufficient representation.

               """)

            use_label_targets = st.checkbox(
                "Enable Label Targeting",
                value=False,
                help="Prioritize specific labels to meet target counts"
            )

            label_targets = None

            if use_label_targets:
                st.markdown("#### Tier 1 Critical Labels")
                st.caption("Rare or difficult labels that need clear examples")
                tier1_targets = {}

                # Critical labels from chatbot analysis
                critical_labels = ['Just_Happens', 'Technical_Specialist', 'Divination',
                                   'Rule_Violation_Taboo', 'Priest_High_Religion']

                tier1_cols = st.columns(3)
                for i, label in enumerate(critical_labels):
                    if label in label_columns:
                        with tier1_cols[i % 3]:
                            default_val = DEFAULT_LABEL_TARGETS['tier1'].get(label, 100)
                            target = st.number_input(
                                f"{label}:",
                                min_value=0,
                                max_value=1000,
                                value=default_val,
                                step=10,
                                key=f"tier1_target_{label}"
                            )

                            if target > 0:
                                tier1_targets[label] = target

                # Show current label distribution in candidates
                with st.expander("📊 Current Label Availability in Tier 1 Candidates"):
                    avail_data = []

                    for label in critical_labels:
                        if label in label_columns:
                            # Check how many candidates have this label
                            candidates_with_label = 0

                            for idx in tier1_candidates['passage_idx'].tolist():
                                if idx in df.index and df.loc[idx, label] == 1:
                                    candidates_with_label += 1
                            target = tier1_targets.get(label, 0)

                            avail_data.append({
                                'Label': label,
                                'Available': candidates_with_label,
                                'Target': target,
                                'Status': '✅' if candidates_with_label >= target else '⚠️'
                            })

                    st.dataframe(pd.DataFrame(avail_data), hide_index=True, width='stretch')

                st.markdown("---")
                st.markdown("#### Tier 2 Balanced Distribution")
                st.caption("All labels for generalization")
                tier2_targets = {}

                # All other labels
                other_labels = [l for l in label_columns if l not in critical_labels]

                tier2_cols = st.columns(3)
                for i, label in enumerate(other_labels[:9]):  # Limit display

                    with tier2_cols[i % 3]:
                        default_val = DEFAULT_LABEL_TARGETS['tier2'].get(label, 150)

                        target = st.number_input(
                            f"{label}:",
                            min_value=0,
                            max_value=1000,
                            value=default_val,
                            step=10,
                            key=f"tier2_target_{label}"
                        )

                        if target > 0:
                            tier2_targets[label] = target

                if tier1_targets or tier2_targets:
                    label_targets = {}
                    if tier1_targets:
                        label_targets['tier1'] = tier1_targets
                    if tier2_targets:
                        label_targets['tier2'] = tier2_targets

        with validate_tab:
            st.markdown("### ✅ Quality Validation")

            if st.button("🎯 Generate Tiers", type="primary", key="generate_tiers_advanced"):
                with st.spinner("Creating tiered datasets..."):
                    try:
                        tier1, tier2, inference, metadata = create_advanced_tiered_datasets(
                            df, scores_df, label_columns,
                            tier1_config, tier2_config, label_targets
                        )

                        st.session_state.tier1_dataset = tier1
                        st.session_state.tier2_dataset = tier2
                        st.session_state.inference_dataset = inference
                        st.session_state.tier_metadata = metadata

                        st.success("✅ Tiers created successfully!")

                        # Run validation
                        checks = validate_tier_quality(tier1, tier2, label_columns, label_targets)

                        st.markdown("---")
                        st.markdown("### 📊 Quality Validation Results")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### Tier 1 Checks")
                            for check in checks['tier1']:
                                status = "✅" if check['passed'] else "❌"
                                st.markdown(f"{status} {check['check']}: **{check['value']}**")

                        with col2:
                            st.markdown("#### Tier 2 Checks")
                            for check in checks['tier2']:
                                status = "✅" if check['passed'] else "❌"
                                st.markdown(f"{status} {check['check']}: **{check['value']}**")

                        if checks['warnings']:
                            st.markdown("---")
                            st.markdown("#### ⚠️ Warnings")
                            for warning in checks['warnings']:
                                st.warning(warning)


                    except Exception as e:
                        st.error(f"Error creating tiers: {e}")
                        import traceback

                        with st.expander("Error details"):
                            st.code(traceback.format_exc())

            # Show results if available
            if st.session_state.get('tier1_dataset') is not None:
                st.markdown("---")
                st.markdown("### 📈 Tier Statistics")
                tier1 = st.session_state.tier1_dataset
                tier2 = st.session_state.tier2_dataset
                inference = st.session_state.inference_dataset
                metadata = st.session_state.get('tier_metadata', {})

                # Overview metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("#### 🥇 Tier 1")
                    st.metric("Count", len(tier1))
                    st.metric("% of Total", f"{len(tier1) / len(df) * 100:.1f}%")

                    if 'confidence_composite' in tier1.columns:
                        st.metric("Avg Quality", f"{tier1['confidence_composite'].mean():.3f}")

                with col2:
                    st.markdown("#### 📚 Tier 2")
                    st.metric("Count", len(tier2))
                    st.metric("% of Total", f"{len(tier2) / len(df) * 100:.1f}%")

                    if 'confidence_composite' in tier2.columns:
                        st.metric("Avg Quality", f"{tier2['confidence_composite'].mean():.3f}")

                with col3:
                    st.markdown("#### 🎯 Inference")
                    st.metric("Count", len(inference))
                    st.metric("% of Total", f"{len(inference) / len(df) * 100:.1f}%")

                    if 'confidence_composite' in inference.columns:
                        st.metric("Avg Quality", f"{inference['confidence_composite'].mean():.3f}")

                # Label distribution comparison
                st.markdown("---")
                st.markdown("### 🏷️ Label Distribution by Tier")

                dist_data = []
                for label in label_columns:
                    tier1_count = (tier1[label] == 1).sum()
                    tier2_count = (tier2[label] == 1).sum()
                    inference_count = (inference[label] == 1).sum()

                    dist_data.append({
                        'Label': label,
                        'Tier 1': tier1_count,
                        'Tier 2': tier2_count,
                        'Inference': inference_count,
                        'Total': tier1_count + tier2_count + inference_count
                    })

                st.dataframe(pd.DataFrame(dist_data), hide_index=True, width='stretch')

                # Quality distribution visualization
                st.markdown("---")
                st.markdown("### 📊 Quality Distribution")

                if 'confidence_composite' in tier1.columns:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

                    # Tier 1
                    axes[0].hist(tier1['confidence_composite'].dropna(), bins=30,
                                 edgecolor='black', alpha=0.7, color='gold')
                    axes[0].axvline(tier1['confidence_composite'].mean(),
                                    color='red', linestyle='--',
                                    label=f'Mean: {tier1["confidence_composite"].mean():.3f}')
                    axes[0].set_xlabel('Composite Score')
                    axes[0].set_title('Tier 1 Quality Distribution')
                    axes[0].legend()
                    axes[0].grid(alpha=0.3)

                    # Tier 2
                    axes[1].hist(tier2['confidence_composite'].dropna(), bins=30,
                                 edgecolor='black', alpha=0.7, color='skyblue')
                    axes[1].axvline(tier2['confidence_composite'].mean(),
                                    color='red', linestyle='--',
                                    label=f'Mean: {tier2["confidence_composite"].mean():.3f}')
                    axes[1].set_xlabel('Composite Score')
                    axes[1].set_title('Tier 2 Quality Distribution')
                    axes[1].legend()
                    axes[1].grid(alpha=0.3)

                    # Inference
                    axes[2].hist(inference['confidence_composite'].dropna(), bins=30,
                                 edgecolor='black', alpha=0.7, color='lightgray')
                    axes[2].axvline(inference['confidence_composite'].mean(),
                                    color='red', linestyle='--',
                                    label=f'Mean: {inference["confidence_composite"].mean():.3f}')
                    axes[2].set_xlabel('Composite Score')
                    axes[2].set_title('Inference Quality Distribution')
                    axes[2].legend()
                    axes[2].grid(alpha=0.3)

                    plt.tight_layout()

                    st.pyplot(fig)

    elif page == "🛠️ Data Prep":
        render_data_preparation_page(st.session_state)

    elif page == "🤖 Model Inference":

        st.markdown("## 🤖 Model Inference Testing")

        df = st.session_state.df

        passage_col = st.session_state.get('passage_col', 'Passage')

        label_columns = st.session_state.label_columns

        if not st.session_state.loaded_models:
            st.warning("⚠️ No models loaded")

            st.info("👈 Load one or more models in the sidebar first")

            st.stop()

        st.markdown(f"**{len(st.session_state.loaded_models)} model(s) loaded**")

        # Model selection for inference

        selected_models = st.multiselect(

            "Select models to compare:",

            options=list(st.session_state.loaded_models.keys()),

            default=list(st.session_state.loaded_models.keys()),

            key="inference_models_select"

        )

        if not selected_models:
            st.warning("Select at least one model")

            st.stop()

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

                        st.markdown("---")

                        st.markdown("#### 🤖 Model Comparison")

                        # Run inference with all selected models

                        model_results = {}

                        for model_name in selected_models:

                            loader = st.session_state.loaded_models[model_name]

                            with st.spinner(f"Running {model_name}..."):

                                try:

                                    result = loader.predict_passage(passage_text)

                                    model_results[model_name] = result

                                except Exception as e:

                                    st.error(f"Error with {model_name}: {e}")

                        if model_results:
                            _display_multi_model_comparison(

                                model_results, actual_labels, label_columns

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

                model_results = {}

                for model_name in selected_models:

                    loader = st.session_state.loaded_models[model_name]

                    with st.spinner(f"Running {model_name}..."):

                        try:

                            result = loader.predict_passage(

                                custom_text,

                                use_optimal_thresholds=use_optimal,

                                default_threshold=threshold

                            )

                            model_results[model_name] = result

                        except Exception as e:

                            st.error(f"Error with {model_name}: {e}")

                if model_results:

                    st.markdown("### 🤖 Model Comparison")

                    # Create comparison table

                    all_labels = set()

                    for result in model_results.values():
                        all_labels.update(result['probabilities'].keys())

                    comparison_data = []

                    for label in sorted(all_labels):

                        row = {'Label': label}

                        for model_name, result in model_results.items():

                            prob = result['probabilities'].get(label, 0)

                            is_pred = label in result['predicted_labels']

                            if is_pred:

                                row[f"{model_name}"] = f"✓ {prob:.3f}"

                            else:

                                row[f"{model_name}"] = f"  {prob:.3f}"

                        comparison_data.append(row)

                    st.dataframe(

                        pd.DataFrame(comparison_data),

                        hide_index=True,

                        width='stretch'

                    )

                    # Show which labels each model predicted

                    st.markdown("#### Predicted Labels by Model")

                    for model_name, result in model_results.items():

                        if result['predicted_labels']:

                            st.markdown(f"**{model_name}:** {', '.join(result['predicted_labels'])}")

                        else:

                            st.markdown(f"**{model_name}:** None")

    elif page == "💬 Chat":
        render_chat_page(st.session_state)

    elif page == "🎓 Train Model":
        render_training_page(st.session_state)

    elif page == "💾 Export":

        st.markdown("## 💾 Export Results")

        df = st.session_state.df

        cache = st.session_state.cache

        label_columns = st.session_state.label_columns

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Score Results Export

        if cache is not None:

            st.markdown("### 📊 Score Results")

            scores_df = cache['df_summary']

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("**Summary Scores**")

                output_summary = io.BytesIO()

                scores_df.to_excel(output_summary, index=False, engine='openpyxl')

                st.download_button(

                    label="📥 Download Summary",

                    data=output_summary.getvalue(),

                    file_name=f"scores_summary_{timestamp}.xlsx",

                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

                )

            with col2:

                st.markdown("**Detailed Scores**")

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

        # Golden Dataset Export

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

            st.markdown("---")

        # Tiered Datasets Export

        if st.session_state.get('tier1_dataset') is not None:

            st.markdown("### 📦 Tiered Training Datasets")

            tier1 = st.session_state.tier1_dataset

            tier2 = st.session_state.tier2_dataset

            inference = st.session_state.inference_dataset

            metadata = st.session_state.get('tier_metadata', {})

            st.info(f"Tier 1: {len(tier1):,} | Tier 2: {len(tier2):,} | Inference: {len(inference):,}")

            # Export options

            export_format = st.radio(

                "Export format:",

                ["Individual Excel Files", "Multi-Sheet Excel", "CSV Files", "Complete Package"],

                horizontal=True

            )

            if export_format == "Individual Excel Files":

                col1, col2, col3 = st.columns(3)

                with col1:

                    output_tier1 = io.BytesIO()

                    tier1.to_excel(output_tier1, index=False, engine='openpyxl')

                    st.download_button(

                        label="📥 Tier 1",

                        data=output_tier1.getvalue(),

                        file_name=f"tier1_{timestamp}.xlsx",

                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                        key="download_tier1"

                    )

                with col2:

                    output_tier2 = io.BytesIO()

                    tier2.to_excel(output_tier2, index=False, engine='openpyxl')

                    st.download_button(

                        label="📥 Tier 2",

                        data=output_tier2.getvalue(),

                        file_name=f"tier2_{timestamp}.xlsx",

                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                        key="download_tier2"

                    )

                with col3:

                    output_inference = io.BytesIO()

                    inference.to_excel(output_inference, index=False, engine='openpyxl')

                    st.download_button(

                        label="📥 Inference",

                        data=output_inference.getvalue(),

                        file_name=f"inference_{timestamp}.xlsx",

                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                        key="download_inference"

                    )


            elif export_format == "Multi-Sheet Excel":

                output_multi = io.BytesIO()

                with pd.ExcelWriter(output_multi, engine='openpyxl') as writer:

                    tier1.to_excel(writer, sheet_name='Tier_1', index=False)

                    tier2.to_excel(writer, sheet_name='Tier_2', index=False)

                    inference.to_excel(writer, sheet_name='Inference', index=False)

                    # Add metadata sheet

                    if metadata:

                        # Flatten metadata for Excel

                        meta_rows = []

                        for tier_name, tier_meta in metadata.get('tiers', {}).items():
                            meta_rows.append({

                                'Tier': tier_name,

                                'Count': tier_meta.get('count', 0),

                                'Percentage': f"{tier_meta.get('percentage', 0):.2f}%",

                                'Consistency_Mean': f"{tier_meta.get('quality', {}).get('consistency_mean', 0):.3f}",

                                'Rerank_Mean': f"{tier_meta.get('quality', {}).get('rerank_mean', 0):.3f}",

                                'Composite_Mean': f"{tier_meta.get('quality', {}).get('composite_mean', 0):.3f}"

                            })

                        meta_df = pd.DataFrame(meta_rows)

                        meta_df.to_excel(writer, sheet_name='Metadata', index=False)

                st.download_button(

                    label="📥 Download All Tiers (Multi-Sheet)",

                    data=output_multi.getvalue(),

                    file_name=f"tiered_datasets_{timestamp}.xlsx",

                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

                )


            elif export_format == "CSV Files":

                import zipfile

                # Create zip file with all CSVs

                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:

                    # Tier 1

                    tier1_csv = io.StringIO()

                    tier1.to_csv(tier1_csv, index=False)

                    zip_file.writestr(f'tier1_{timestamp}.csv', tier1_csv.getvalue())

                    # Tier 2

                    tier2_csv = io.StringIO()

                    tier2.to_csv(tier2_csv, index=False)

                    zip_file.writestr(f'tier2_{timestamp}.csv', tier2_csv.getvalue())

                    # Inference

                    inference_csv = io.StringIO()

                    inference.to_csv(inference_csv, index=False)

                    zip_file.writestr(f'inference_{timestamp}.csv', inference_csv.getvalue())

                    # Metadata JSON

                    if metadata:
                        metadata_json = json.dumps(metadata, indent=2)

                        zip_file.writestr(f'metadata_{timestamp}.json', metadata_json)

                st.download_button(

                    label="📥 Download All Tiers (CSV + Metadata)",

                    data=zip_buffer.getvalue(),

                    file_name=f"tiered_datasets_{timestamp}.zip",

                    mime="application/zip"

                )


            elif export_format == "Complete Package":

                import zipfile

                st.markdown("**Complete training package includes:**")

                st.markdown("""

                   - All three tiers (Excel)

                   - Detailed metadata (JSON)

                   - Configuration used (JSON)

                   - Label distribution report (Excel)

                   - Quality statistics (Excel)

                   - README with usage instructions

                   """)

                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:

                    # Datasets

                    tier1_excel = io.BytesIO()

                    tier1.to_excel(tier1_excel, index=False, engine='openpyxl')

                    zip_file.writestr(f'datasets/tier1_{timestamp}.xlsx', tier1_excel.getvalue())

                    tier2_excel = io.BytesIO()

                    tier2.to_excel(tier2_excel, index=False, engine='openpyxl')

                    zip_file.writestr(f'datasets/tier2_{timestamp}.xlsx', tier2_excel.getvalue())

                    inference_excel = io.BytesIO()

                    inference.to_excel(inference_excel, index=False, engine='openpyxl')

                    zip_file.writestr(f'datasets/inference_{timestamp}.xlsx', inference_excel.getvalue())

                    # Metadata

                    if metadata:
                        metadata_json = json.dumps(metadata, indent=2)

                        zip_file.writestr(f'metadata/tier_metadata_{timestamp}.json', metadata_json)

                    # Label distribution

                    dist_data = []

                    for label in label_columns:
                        tier1_count = (tier1[label] == 1).sum()

                        tier2_count = (tier2[label] == 1).sum()

                        inference_count = (inference[label] == 1).sum()

                        dist_data.append({

                            'Label': label,

                            'Tier_1': tier1_count,

                            'Tier_2': tier2_count,

                            'Inference': inference_count,

                            'Total': tier1_count + tier2_count + inference_count,

                            'Tier_1_Pct': f"{tier1_count / len(tier1) * 100:.1f}%" if len(tier1) > 0 else "0%",

                            'Tier_2_Pct': f"{tier2_count / len(tier2) * 100:.1f}%" if len(tier2) > 0 else "0%"

                        })

                    dist_df = pd.DataFrame(dist_data)

                    dist_excel = io.BytesIO()

                    dist_df.to_excel(dist_excel, index=False, engine='openpyxl')

                    zip_file.writestr(f'reports/label_distribution_{timestamp}.xlsx', dist_excel.getvalue())

                    # README

                    readme_content = f"""# HRAF Tiered Training Datasets

   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}


   ## Dataset Overview


   - **Tier 1 (Elite)**: {len(tier1):,} passages - High-quality training data

   - **Tier 2 (Expansion)**: {len(tier2):,} passages - Broader coverage

   - **Inference (Validation)**: {len(inference):,} passages - Test set


   ## Usage


   ### Training Protocol


   1. **Stage 1: Foundational Training**

      - Use: tier1_{timestamp}.xlsx

      - Epochs: 5-10

      - Focus: Learn clear patterns from unambiguous examples


   2. **Stage 2: Expansion Training**  

      - Use: tier1_{timestamp}.xlsx + tier2_{timestamp}.xlsx

      - Epochs: 3-5

      - Focus: Generalize to harder cases


   3. **Stage 3: Validation**

      - Use: inference_{timestamp}.xlsx

      - Purpose: Final model evaluation and threshold optimization


   ### Quality Metrics


   Tier 1 Quality:

   - Consistency: {tier1['confidence_consistency'].mean():.3f if 'confidence_consistency' in tier1.columns else 'N/A'}

   - Rerank: {tier1['confidence_rerank'].mean():.3f if 'confidence_rerank' in tier1.columns else 'N/A'}


   Tier 2 Quality:

   - Consistency: {tier2['confidence_consistency'].mean():.3f if 'confidence_consistency' in tier2.columns else 'N/A'}

   - Rerank: {tier2['confidence_rerank'].mean():.3f if 'confidence_rerank' in tier2.columns else 'N/A'}


   ## Files


   - `datasets/`: Training and test data files

   - `metadata/`: Configuration and statistics

   - `reports/`: Label distribution and quality reports

   - `README.md`: This file


   ## Label Columns


   Each dataset contains the following label columns:

   {', '.join(label_columns)}


   ## Notes


   - Passages are marked with 'tier' column (1, 2, or 3)

   - Quality scores included: confidence_composite, confidence_consistency, confidence_rerank

   - See metadata JSON for detailed statistics


   For questions or issues, refer to the HRAF Golden Dataset Discovery tool documentation.

   """

                    zip_file.writestr('README.md', readme_content)

                st.download_button(

                    label="📥 Download Complete Package",

                    data=zip_buffer.getvalue(),

                    file_name=f"hraf_training_package_{timestamp}.zip",

                    mime="application/zip"

                )

            # Show metadata preview

            if metadata:
                with st.expander("📋 View Metadata"):
                    st.json(metadata)

# Footer
st.markdown("---")
st.caption("HRAF Golden Dataset Discovery | Built with Streamlit")