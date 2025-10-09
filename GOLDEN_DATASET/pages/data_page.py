"""
Data Page - Load, Clean, Embed, Score, and Prepare Training Data

Architecture:
- Self-contained page module
- Uses components from components/
- Uses core functionality from core/
- Clean separation of concerns
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, List
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import components
from components.ui_helpers import (
    render_data_summary,
    render_file_browser
)

# Import core functionality
from core import GoldenDatasetFinder
from core import DataAnalyzer, DataExperiment


def render():
    """Main render function for Data page"""

    st.markdown("# 📊 Data Management")
    st.caption("Load, clean, embed, score, and prepare training datasets")

    # Create tabs for different sections
    tabs = st.tabs([
        "📂 Load Data",
        "🧹 Clean & Analyze",
        "🔢 Embed & Score",
        "📦 Create Training Sets"
    ])

    with tabs[0]:
        render_load_data_section()

    with tabs[1]:
        render_clean_analyze_section()

    with tabs[2]:
        render_embed_score_section()

    with tabs[3]:
        render_create_training_sets_section()


# ============================================================================
# SECTION 1: LOAD DATA
# ============================================================================

def render_load_data_section():
    """Load and validate datasets"""

    st.markdown("### 📂 Load Dataset")

    # Show current status
    if st.session_state.get('initialized'):
        df = st.session_state.df
        st.success(f"✅ Dataset loaded: {len(df)} passages")

        col1, col2 = st.columns([3, 1])

        with col1:
            render_data_summary(
                df,
                st.session_state.label_columns,
                st.session_state.passage_col,
                show_distribution=False
            )

        with col2:
            if st.button("🔄 Load Different Dataset"):
                # Reset and reload
                st.session_state.initialized = False
                st.rerun()

        return

    # File selection
    st.markdown("#### Select Data Source")

    source_type = st.radio(
        "Source:",
        ["📁 Browse Files", "🧪 Browse Experiments", "⬆️ Upload File"],
        horizontal=True,
        key="data_source_type"
    )

    if source_type == "📁 Browse Files":
        render_file_browser_loader()

    elif source_type == "🧪 Browse Experiments":
        render_experiment_browser_loader()

    elif source_type == "⬆️ Upload File":
        render_file_upload_loader()


def render_file_browser_loader():
    """Browse and load from file system"""

    data_dir = Path("./data")

    selected_file = render_file_browser(
        data_dir,
        file_types=['.xlsx', '.csv'],
        key_suffix="data_load"
    )

    if selected_file:
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            header_row = st.number_input(
                "Header row:",
                min_value=0,
                max_value=5,
                value=0,
                help="Row containing column names (0 = first row)"
            )

        with col2:
            passage_col = st.text_input(
                "Passage column:",
                value="Passage",
                help="Name of column containing text"
            )

        if st.button("📂 Load Dataset", type="primary"):
            load_dataset_from_file(selected_file, header_row, passage_col)


def render_experiment_browser_loader():
    """Browse and load from saved experiments"""

    experiment = DataExperiment()
    experiments = experiment.list_experiments()

    if not experiments:
        st.info("💡 No experiments found. Create experiments after loading data.")
        return

    # Show experiments
    exp_names = [exp['name'] for exp in experiments]

    selected_exp_name = st.selectbox(
        "Select experiment:",
        exp_names,
        key="exp_load_selector"
    )

    selected_exp = next((e for e in experiments if e['name'] == selected_exp_name), None)

    if selected_exp:
        meta = selected_exp['metadata']

        # Show metadata
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Type", meta.get('experiment_type', 'unknown'))

        with col2:
            stats = meta.get('statistics', {})
            st.metric("Passages", stats.get('num_passages', 'N/A'))

        with col3:
            st.metric("Labels", len(stats.get('label_columns', [])))

        if st.button("📂 Load Experiment", type="primary"):
            load_dataset_from_experiment(selected_exp)


def render_file_upload_loader():
    """Upload and load file"""

    uploaded_file = st.file_uploader(
        "Choose Excel file:",
        type=['xlsx', 'xls'],
        key="data_upload"
    )

    if uploaded_file:
        # Show file info
        st.info(f"📄 {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

        col1, col2 = st.columns(2)

        with col1:
            header_row = st.number_input(
                "Header row:",
                min_value=0,
                max_value=5,
                value=0
            )

        with col2:
            passage_col = st.text_input(
                "Passage column:",
                value="Passage"
            )

        if st.button("📂 Load Dataset", type="primary"):
            # Save temporarily and load
            temp_path = Path(f"./temp/{uploaded_file.name}")
            temp_path.parent.mkdir(exist_ok=True)

            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            load_dataset_from_file(temp_path, header_row, passage_col)

            # Clean up
            temp_path.unlink()


def load_dataset_from_file(filepath: Path, header_row: int, passage_col: str):
    """Load dataset from file"""

    with st.spinner("Loading dataset..."):
        try:
            # Load Excel
            df = pd.read_excel(filepath, header=header_row)

            # Validate passage column
            if passage_col not in df.columns:
                st.error(f"Column '{passage_col}' not found!")
                st.info(f"Available columns: {', '.join(df.columns)}")
                return

            # Auto-detect labels
            label_columns = detect_label_columns(df, passage_col)

            if not label_columns:
                st.error("No binary label columns detected!")
                return

            # Initialize discovery architecture
            from dotenv import load_dotenv
            import os
            load_dotenv()

            finder = GoldenDatasetFinder(
                voyage_api_key=os.getenv("VOYAGE_API_KEY"),
                pinecone_api_key=os.getenv("PINECONE_API_KEY"),
                index_name="hraf-misfortune-test",
                region="us-east-1"
            )

            # Generate namespace
            namespace = filepath.stem.lower().replace(' ', '_')

            # Store in session state
            st.session_state.df = df
            st.session_state.label_columns = label_columns
            st.session_state.passage_col = passage_col
            st.session_state.finder = finder
            st.session_state.namespace = namespace
            st.session_state.selected_file = str(filepath)
            st.session_state.initialized = True

            st.success(f"✅ Loaded: {len(df)} passages, {len(label_columns)} labels")
            st.rerun()

        except Exception as e:
            st.error(f"Error loading dataset: {e}")


def load_dataset_from_experiment(experiment: Dict):
    """Load dataset from saved experiment"""

    with st.spinner("Loading experiment..."):
        try:
            exp_dir = experiment['directory']
            meta = experiment['metadata']

            # Load data file
            data_file = exp_dir / "data.xlsx"

            if not data_file.exists():
                st.error("Experiment data file not found!")
                return

            df = pd.read_excel(data_file)

            # Get metadata
            label_columns = meta['statistics']['label_columns']
            passage_col = meta['statistics']['passage_column']

            # Initialize finder
            from dotenv import load_dotenv
            import os
            load_dotenv()

            finder = GoldenDatasetFinder(
                voyage_api_key=os.getenv("VOYAGE_API_KEY"),
                pinecone_api_key=os.getenv("PINECONE_API_KEY"),
                index_name="hraf-misfortune-test",
                region="us-east-1"
            )

            namespace = meta.get('provenance', {}).get('source_namespace', 'experiment')

            # Store in session state
            st.session_state.df = df
            st.session_state.label_columns = label_columns
            st.session_state.passage_col = passage_col
            st.session_state.finder = finder
            st.session_state.namespace = namespace
            st.session_state.selected_file = str(data_file)
            st.session_state.initialized = True

            st.success(f"✅ Loaded experiment: {len(df)} passages")
            st.rerun()

        except Exception as e:
            st.error(f"Error loading experiment: {e}")


def detect_label_columns(df: pd.DataFrame, passage_col: str) -> List[str]:
    """Auto-detect binary label columns"""

    label_columns = []

    exclude_cols = {passage_col, 'ID', 'Culture', 'Region', 'Description'}

    for col in df.columns:
        if col in exclude_cols:
            continue

        if df[col].dtype in ['int64', 'float64']:
            unique_vals = df[col].dropna().unique()

            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                if (df[col] == 1).sum() > 0:
                    label_columns.append(col)

    return label_columns


# ============================================================================
# SECTION 2: CLEAN & ANALYZE
# ============================================================================

def render_clean_analyze_section():
    """Clean and analyze data quality"""

    if not st.session_state.get('initialized'):
        st.info("💡 Load a dataset first")
        return

    st.markdown("### 🧹 Clean & Analyze")

    df = st.session_state.df
    label_columns = st.session_state.label_columns
    passage_col = st.session_state.passage_col

    # Initialize analyzer
    analyzer = DataAnalyzer(df, label_columns, passage_col)

    # Run analysis
    if st.button("🔎 Analyze Data Quality", type="primary"):
        with st.spinner("Analyzing..."):
            analysis = analyzer.analyze_quality()
            st.session_state.quality_analysis = analysis

    # Show results
    if 'quality_analysis' in st.session_state:
        analysis = st.session_state.quality_analysis

        # Issues
        if analysis['issues']:
            st.markdown("#### ⚠️ Issues Detected")
            for issue in analysis['issues']:
                st.warning(issue)
        else:
            st.success("✅ No major issues detected!")

        # Recommendations
        if analysis['suggestions']:
            with st.expander("💡 Recommendations"):
                for suggestion in analysis['suggestions']:
                    st.info(suggestion)

        # TODO: Add cleaning workflow here


# ============================================================================
# SECTION 3: EMBED & SCORE
# ============================================================================

def render_embed_score_section():
    """Embed passages and calculate quality scores"""

    if not st.session_state.get('initialized'):
        st.info("💡 Load a dataset first")
        return

    st.markdown("### 🔢 Embed & Score")

    # TODO: Implement embedding and scoring workflow
    st.info("🚧 Embedding and scoring interface coming soon")


# ============================================================================
# SECTION 4: CREATE TRAINING SETS
# ============================================================================

def render_create_training_sets_section():
    """Create tiered training datasets"""

    if not st.session_state.get('initialized'):
        st.info("💡 Load a dataset first")
        return

    st.markdown("### 📦 Create Training Sets")

    # TODO: Implement tier creation workflow
    st.info("🚧 Training set creation interface coming soon")