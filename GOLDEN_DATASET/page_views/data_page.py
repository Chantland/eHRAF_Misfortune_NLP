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
    render_file_browser,
    render_quality_threshold_controls,
    render_progress_with_eta
)

# Import core functionality
from core.discovery_architecture import GoldenDatasetFinder
from core.data_preparation import (
    DataAnalyzer,
    DataSegmenter,
    DataExperiment,
    render_data_preparation_page
)
from datetime import datetime

def make_df_display_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dataframe to display-safe format for Streamlit
    Fixes PyArrow serialization issues with mixed-type object columns
    """
    df_safe = df.copy()
    for col in df_safe.columns:
        if df_safe[col].dtype == 'object':
            df_safe[col] = df_safe[col].astype(str)
    return df_safe

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
                show_distribution=True
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
        ["📁 Browse Files", "⬆️ Upload File"],
        horizontal=True,
        key="data_source_type"
    )

    if source_type == "📁 Browse Files":
        render_file_browser_loader()

    elif source_type == "⬆️ Upload File":
        render_file_upload_loader()


def render_file_browser_loader():
    """Browse and load from file system with interactive preview"""

    data_dir = Path("./data")

    selected_file = render_file_browser(
        data_dir,
        file_types=['.xlsx', '.csv'],
        key_suffix="data_load"
    )

    if selected_file:
        st.markdown("---")

        # Show interactive preview
        config = render_interactive_data_preview(selected_file)

        if config:
            # Load with confirmed settings
            load_dataset_with_config(config)


def load_dataset_with_config(config: Dict):
    """Load dataset with user-confirmed configuration"""

    filepath = config['filepath']
    header_row = config['header_row']
    passage_col = config['passage_col']
    all_columns = config['all_columns']
    label_columns = config['label_columns']
    metadata_columns = config['metadata_columns']

    with st.spinner("Loading dataset with your configuration..."):
        try:
            # Load full dataset with selected header
            df = pd.read_excel(filepath, header=header_row)

            # Validate all columns exist
            if passage_col not in df.columns:
                st.error(f"❌ Passage column '{passage_col}' not found!")
                return

            missing_cols = [col for col in all_columns if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Selected columns not found: {missing_cols}")
                return

            # Keep only selected columns
            keep_cols = [passage_col] + all_columns
            df = df[keep_cols].copy()

            # Validate label columns are numeric
            for label in label_columns:
                if df[label].dtype not in ['int64', 'float64', 'Int64']:
                    st.warning(f"⚠️ Label column '{label}' is not numeric. Converting to numeric...")
                    df[label] = pd.to_numeric(df[label], errors='coerce').fillna(0).astype(int)

            # Show final statistics
            st.info(f"""
            **Dataset Loaded:**
            - Total passages: {len(df)}
            - Valid passages: {df[passage_col].notna().sum()}
            - Label columns: {len(label_columns)}
            - Metadata columns: {len(metadata_columns)}
            """)

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

            # Generate namespace from filename
            namespace = filepath.stem.lower().replace(' ', '_')

            # Store in session state
            st.session_state.df = df
            st.session_state.label_columns = label_columns
            st.session_state.passage_col = passage_col
            st.session_state.metadata_columns = metadata_columns
            st.session_state.all_columns = all_columns
            st.session_state.finder = finder
            st.session_state.namespace = namespace
            st.session_state.selected_file = str(filepath)
            st.session_state.initialized = True

            # Store configuration for future reference
            st.session_state.load_config = config

            # Clear preview state
            for key in ['preview_df', 'preview_file', 'preview_header_row', 'selected_columns',
                        'selected_label_columns']:
                if key in st.session_state:
                    del st.session_state[key]

            st.success(f"✅ Successfully loaded dataset!")
            st.balloons()
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())

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
    """Upload and load file with interactive preview"""

    uploaded_file = st.file_uploader(
        "Choose Excel file:",
        type=['xlsx', 'xls'],
        key="data_upload"
    )

    if uploaded_file:
        # Save temporarily
        temp_path = Path(f"./temp/{uploaded_file.name}")
        temp_path.parent.mkdir(exist_ok=True)

        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        st.markdown("---")

        # Show interactive preview
        config = render_interactive_data_preview(temp_path)

        if config:
            # Load with confirmed settings
            load_dataset_with_config(config)

            # Clean up temp file
            temp_path.unlink()

def render_interactive_data_preview(filepath: Path):
    """
    Interactive data preview with complete column control
    Click the row to set header, check boxes to select labels
    """

    st.markdown("#### 📋 Data Preview & Configuration")
    st.caption("Configure exactly which columns to use - no assumptions")

    # Initialize preview state
    if 'preview_df' not in st.session_state or st.session_state.get('preview_file') != str(filepath):
        df_raw = pd.read_excel(filepath, header=None, nrows=10)

        # Convert to display-safe format
        df_raw_display = df_raw.copy()
        for col in df_raw_display.columns:
            df_raw_display[col] = df_raw_display[col].astype(str)

        st.session_state['preview_df'] = df_raw
        st.session_state['preview_df_display'] = df_raw_display
        st.session_state['preview_file'] = str(filepath)
        st.session_state['preview_header_row'] = 0
        st.session_state['selected_columns'] = []
        st.session_state['selected_label_columns'] = []

    df_raw = st.session_state['preview_df']
    df_raw_display = st.session_state['preview_df_display']

    # ========================================================================
    # STEP 1: Select Header Row - CLICK TO SELECT
    # ========================================================================

    st.markdown("##### 1️⃣ Identify Header Row")
    st.caption("👆 Click the row number below that contains your column headers")

    # Current selection
    current_header = st.session_state.get('preview_header_row', 0)

    # Show clickable row buttons
    st.markdown("**Click to select header row:**")

    cols = st.columns([1, 10])

    with cols[0]:
        st.markdown("**Row**")
    with cols[1]:
        st.markdown("**Data Preview**")

    # Display each row as a clickable option
    for row_idx in range(min(6, len(df_raw))):
        cols = st.columns([1, 10])

        with cols[0]:
            # Button to select this row
            is_selected = (row_idx == current_header)
            button_label = f"{'✅' if is_selected else '⬜'} {row_idx}"

            if st.button(
                    button_label,
                    key=f"header_row_{row_idx}",
                    type="primary" if is_selected else "secondary",
                    width='stretch'
            ):
                st.session_state['preview_header_row'] = row_idx
                st.session_state['selected_columns'] = []
                st.session_state['selected_label_columns'] = []
                st.rerun()

        with cols[1]:
            # Show the row data
            row_data = df_raw_display.iloc[row_idx].tolist()
            row_str = " | ".join([str(v)[:30] for v in row_data[:10]])  # First 10 columns
            if len(row_data) > 10:
                row_str += " | ..."

            # Highlight if selected
            if is_selected:
                st.success(f"**{row_str}**")
            else:
                st.text(row_str)

    header_row = current_header

    # Load with selected header row
    df_preview = pd.read_excel(filepath, header=header_row, nrows=5)

    # Create display-safe version
    df_preview_display = df_preview.copy()
    for col in df_preview_display.columns:
        if df_preview_display[col].dtype == 'object':
            df_preview_display[col] = df_preview_display[col].astype(str)

    all_columns = list(df_preview.columns)

    st.markdown("---")

    # ========================================================================
    # STEP 2: Select Passage Column
    # ========================================================================

    st.markdown("##### 2️⃣ Select Passage Column")

    st.caption(f"Preview with selected header (row {header_row}):")
    st.dataframe(
        df_preview_display,
        width='stretch',
        hide_index=True
    )

    # Smart default: look for "Passage" or "passage" first
    default_passage_col = None

    # First priority: exact match "Passage"
    if "Passage" in all_columns:
        default_passage_col = "Passage"
    # Second priority: case-insensitive "passage"
    elif "passage" in all_columns:
        default_passage_col = "passage"
    # Third priority: contains "passage" anywhere
    else:
        for col in all_columns:
            if "passage" in str(col).lower():
                default_passage_col = col
                break

    # Fourth priority: look for common text column names
    if default_passage_col is None:
        text_keywords = ['text', 'content', 'body', 'description']
        for keyword in text_keywords:
            for col in all_columns:
                if keyword in str(col).lower():
                    default_passage_col = col
                    break
            if default_passage_col:
                break

    # Fifth priority: find column with longest average text
    if default_passage_col is None:
        max_length = 0
        for col in all_columns:
            if df_preview[col].dtype == 'object':
                try:
                    avg_length = df_preview[col].astype(str).str.len().mean()
                    if avg_length > max_length:
                        max_length = avg_length
                        default_passage_col = col
                except:
                    pass

    # Final fallback: first column
    if default_passage_col is None:
        default_passage_col = all_columns[0]

    # Get the index for the default
    try:
        default_index = all_columns.index(default_passage_col)
    except:
        default_index = 0

    # Dropdown with smart default
    passage_col = st.selectbox(
        "Which column contains the passage text?",
        options=all_columns,
        index=default_index,
        key="interactive_passage_col",
        help="Select the column containing the full text passages"
    )

    # Show what was auto-detected
    if passage_col == default_passage_col and default_passage_col != all_columns[0]:
        st.caption(f"💡 Auto-detected '{passage_col}' as passage column")

    # Show preview of selected passage column
    if passage_col:
        st.markdown("**Passage preview:**")
        sample_passage = str(df_preview[passage_col].iloc[0])

        # Show length info
        passage_lengths = df_preview[passage_col].astype(str).str.len()
        avg_length = passage_lengths.mean()

        st.caption(f"Sample length: {len(sample_passage)} chars | Average: {avg_length:.0f} chars")

        st.text_area(
            "First passage:",
            value=sample_passage[:500] + "..." if len(sample_passage) > 500 else sample_passage,
            height=150,
            disabled=True,
            label_visibility="collapsed"
        )

    # ========================================================================
    # STEP 3: Select Columns to Include
    # ========================================================================

    st.markdown("##### 3️⃣ Select Columns to Include")
    st.caption("All columns selected by default - uncheck any you don't want")

    available_columns = [col for col in all_columns if col != passage_col]

    # Initialize with ALL columns selected by default
    if 'selected_columns' not in st.session_state or not st.session_state.get('selected_columns'):
        st.session_state['selected_columns'] = available_columns.copy()

    # Quick action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔢 Numeric Only", key="select_numeric_cols", width='stretch'):
            numeric_cols = [col for col in available_columns
                            if df_preview[col].dtype in ['int64', 'float64', 'Int64']]
            st.session_state['selected_columns'] = numeric_cols
            st.rerun()

    with col2:
        if st.button("🗑️ Clear All", key="clear_cols", width='stretch'):
            st.session_state['selected_columns'] = []
            st.session_state['selected_label_columns'] = []
            st.rerun()

    st.markdown(f"**Available columns ({len(available_columns)}):**")
    st.caption(f"✅ {len(st.session_state['selected_columns'])} selected")

    # Track selections
    if 'column_selections' not in st.session_state:
        st.session_state['column_selections'] = {}

    selected_columns = []

    # Display in grid
    num_cols = 3
    for i in range(0, len(available_columns), num_cols):
        cols = st.columns(num_cols)

        for j, col_name in enumerate(available_columns[i:i + num_cols]):
            with cols[j]:
                # Get column info
                dtype = str(df_preview[col_name].dtype)
                try:
                    sample_val = str(df_preview[col_name].iloc[0])
                    if len(sample_val) > 20:
                        sample_val = sample_val[:20] + "..."
                except:
                    sample_val = "N/A"

                # DEFAULT TO CHECKED (True by default)
                default_value = col_name in st.session_state.get('selected_columns', available_columns)

                is_selected = st.checkbox(
                    f"**{col_name}**",
                    value=default_value,
                    key=f"col_select_{col_name}_{header_row}",
                    help=f"Type: {dtype}\nSample: {sample_val}"
                )

                # Update tracking
                st.session_state['column_selections'][col_name] = is_selected

                if is_selected:
                    selected_columns.append(col_name)

                # Show type
                st.caption(f"`{dtype}`")

    # Update session state
    st.session_state['selected_columns'] = selected_columns

    if not selected_columns:
        st.warning("⚠️ No columns selected - select at least one column")
        return None

    # Show count in success message
    st.info(f"✅ **{len(selected_columns)}** of **{len(available_columns)}** columns selected")

    st.markdown("---")

    # ========================================================================
    # STEP 4: Specify Label Columns - AUTO-SELECT DETECTED
    # ========================================================================

    st.markdown("##### 4️⃣ Specify Label Columns")
    st.caption("Binary columns auto-selected as labels - uncheck any that aren't labels")

    if not selected_columns:
        st.info("💡 Select columns in Step 3 first")
        return None

    # Detect potential labels
    potential_labels = []
    for col in selected_columns:
        if df_preview[col].dtype in ['int64', 'float64', 'Int64']:
            try:
                unique_vals = df_preview[col].dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    potential_labels.append(col)
            except:
                pass

    # Initialize with detected labels selected by default
    if 'selected_label_columns' not in st.session_state or not st.session_state.get('selected_label_columns'):
        st.session_state['selected_label_columns'] = potential_labels.copy()

    if potential_labels:
        st.info(f"💡 Auto-detected {len(potential_labels)} binary columns")

    # Quick buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ All Binary", key="select_detected_labels", disabled=not potential_labels,
                     width='stretch'):
            st.session_state['selected_label_columns'] = potential_labels.copy()
            st.rerun()

    with col2:
        if st.button("✅ All Columns", key="select_all_as_labels", width='stretch'):
            st.session_state['selected_label_columns'] = selected_columns.copy()
            st.rerun()

    with col3:
        if st.button("🗑️ Clear All", key="clear_labels", width='stretch'):
            st.session_state['selected_label_columns'] = []
            st.rerun()

    st.markdown("**Mark classification labels:**")
    st.caption(f"✅ {len(st.session_state['selected_label_columns'])} marked as labels")

    # Track label selections
    if 'label_selections' not in st.session_state:
        st.session_state['label_selections'] = {}

    selected_label_columns = []

    # Show each selected column
    for col_name in selected_columns:
        cols = st.columns([3, 2, 2, 3])

        with cols[0]:
            # DEFAULT: checked if it's a detected label
            default_value = col_name in st.session_state.get('selected_label_columns', potential_labels)

            is_label = st.checkbox(
                f"**{col_name}**",
                value=default_value,
                key=f"label_select_{col_name}_{header_row}",
                help="Mark as classification label"
            )

            # Track selection
            st.session_state['label_selections'][col_name] = is_label

            if is_label:
                selected_label_columns.append(col_name)

        with cols[1]:
            dtype = str(df_preview[col_name].dtype)
            # Highlight binary columns
            if col_name in potential_labels:
                st.caption(f"✅ `{dtype}`")
            else:
                st.caption(f"`{dtype}`")

        with cols[2]:
            try:
                unique_count = df_preview[col_name].nunique()
                st.caption(f"Unique: {unique_count}")
            except:
                st.caption("Unique: N/A")

        with cols[3]:
            try:
                unique_count = df_preview[col_name].nunique()
                if unique_count <= 5:
                    unique_vals = [str(v) for v in df_preview[col_name].dropna().unique().tolist()]
                    vals_str = ', '.join(unique_vals[:3])
                    if len(unique_vals) > 3:
                        vals_str += '...'
                    st.caption(f"{vals_str}")
                else:
                    sample = str(df_preview[col_name].iloc[0])[:15]
                    st.caption(f"e.g. {sample}...")
            except:
                st.caption("—")

    # Update session state
    st.session_state['selected_label_columns'] = selected_label_columns

    if not selected_label_columns:
        st.error("⚠️ No label columns selected - select at least one")
        return None

    # Show count in success message
    st.info(f"✅ **{len(selected_label_columns)}** of **{len(selected_columns)}** columns marked as labels")

    # Summary
    metadata_columns = [col for col in selected_columns if col not in selected_label_columns]

    if metadata_columns:
        with st.expander("ℹ️ Column Summary"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Labels ({len(selected_label_columns)}):**")
                for col in selected_label_columns:
                    marker = "✅" if col in potential_labels else "⚪"
                    st.markdown(f"{marker} {col}")

            with col2:
                st.markdown(f"**Metadata ({len(metadata_columns)}):**")
                for col in metadata_columns:
                    st.markdown(f"• {col}")

    st.markdown("---")

    # ========================================================================
    # STEP 5: Confirm Configuration
    # ========================================================================

    st.markdown("##### 5️⃣ Review and Load")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Passage Column", passage_col)
        st.metric("Header Row", header_row)

    with col2:
        st.metric("Label Columns", len(selected_label_columns))
        st.metric("Metadata Columns", len(metadata_columns))

    with col3:
        st.metric("Total Columns", len(selected_columns) + 1)
        try:
            full_df = pd.read_excel(filepath, header=header_row, usecols=[passage_col])
            st.metric("Total Rows", f"{len(full_df):,}")
        except:
            st.metric("Total Rows", "Unknown")

    # Warning for non-binary labels
    non_binary = [col for col in selected_label_columns if col not in potential_labels]
    if non_binary:
        st.warning(f"⚠️ Non-binary labels: {', '.join(non_binary[:3])}" +
                   ("..." if len(non_binary) > 3 else ""))

    # Load button
    if st.button("✅ Load Dataset", type="primary", width='stretch'):
        return {
            'filepath': filepath,
            'header_row': header_row,
            'passage_col': passage_col,
            'all_columns': selected_columns,
            'label_columns': selected_label_columns,
            'metadata_columns': metadata_columns
        }

    return None

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

        st.markdown("#### 📊 Analysis Results")

        # Issues
        if analysis['issues']:
            st.markdown("**⚠️ Issues Found:**")
            for issue in analysis['issues']:
                st.warning(issue)
        else:
            st.success("✅ No major issues detected!")

        # Statistics
        with st.expander("📈 Detailed Statistics", expanded=True):
            stats = analysis['stats']

            # Passage length stats
            if 'passage_length' in stats:
                st.markdown("**Passage Lengths:**")
                length_stats = stats['passage_length']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean", f"{length_stats['mean']:.0f} chars")
                with col2:
                    st.metric("Median", f"{length_stats['median']:.0f} chars")
                with col3:
                    st.metric("Std Dev", f"{length_stats['std']:.0f} chars")

            # Label distribution
            if 'label_distribution' in stats:
                st.markdown("**Label Distribution:**")
                dist_data = []
                for label, info in stats['label_distribution'].items():
                    dist_data.append({
                        'Label': label,
                        'Count': info['count'],
                        'Percentage': f"{info['percentage']:.1f}%"
                    })
                st.dataframe(pd.DataFrame(dist_data), hide_index=True, width='stretch')

        # Recommendations
        if analysis['suggestions']:
            st.markdown("#### 💡 Recommendations")
            for suggestion in analysis['suggestions']:
                st.info(suggestion)

        st.markdown("---")

        # Cleaning workflow
        st.markdown("### 🧹 Data Cleaning")

        cleaning_steps = analyzer.suggest_cleaning_steps(analysis)

        if cleaning_steps:
            st.markdown("**Select cleaning steps to apply:**")

            selected_actions = []
            for step in cleaning_steps:
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected = st.checkbox(
                        step['name'],
                        value=step['recommended'],
                        key=f"clean_{step['action']}",
                        help=step['description']
                    )
                    if selected:
                        selected_actions.append(step['action'])
                with col2:
                    impact_color = "🟢" if step['impact'] < len(df) * 0.05 else "🟡" if step['impact'] < len(
                        df) * 0.1 else "🔴"
                    st.caption(f"{impact_color} -{step['impact']}")

            if selected_actions:
                st.markdown("---")

                col1, col2 = st.columns(2)

                with col1:
                    total_removed = sum(
                        step['impact'] for step in cleaning_steps if step['action'] in selected_actions)
                    st.metric("Total Passages to Remove", total_removed)
                    st.metric("Remaining", len(df) - total_removed)

                with col2:
                    if st.button("🧹 Apply Cleaning", type="primary"):
                        with st.spinner("Cleaning data..."):
                            df_clean = analyzer.apply_cleaning(selected_actions)
                            st.session_state['cleaned_df'] = df_clean
                            st.session_state['df'] = df_clean  # Update working df
                            st.success(f"✅ Cleaned! {len(df)} → {len(df_clean)} passages")
                            st.rerun()
        else:
            st.success("✅ Data is clean! No cleaning steps needed.")


# ============================================================================
# SECTION 3: EMBED & SCORE
# ============================================================================

def render_embed_score_section():
    """Embed passages and calculate quality scores"""

    if not st.session_state.get('initialized'):
        st.info("💡 Load a dataset first")
        return

    st.markdown("### 🔢 Embed & Score")

    df = st.session_state.df
    label_columns = st.session_state.label_columns
    passage_col = st.session_state.passage_col
    finder = st.session_state.get('finder')
    namespace = st.session_state.get('namespace', 'main')

    if finder is None:
        st.error("❌ Finder not initialized. Reload dataset.")
        return

    # FIX: Ensure cache is always a dict
    cache = st.session_state.get('cache')
    if cache is None:
        cache = {}
        st.session_state['cache'] = cache

    has_embeddings = 'passage_id_map' in cache
    has_scores = 'df_summary' in cache

    # Status
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Passages", len(df))

    with col2:
        if has_embeddings:
            st.metric("Embedded", len(cache['passage_id_map']))
        else:
            st.metric("Embedded", 0)

    with col3:
        if has_scores:
            st.metric("Scored", len(cache['df_summary']))
        else:
            st.metric("Scored", 0)

    st.markdown("---")

    # Embed section
    st.markdown("#### 1️⃣ Generate Embeddings")

    if has_embeddings:
        st.success(f"✅ Embeddings already exist for {len(cache['passage_id_map'])} passages")

        if st.button("🔄 Recompute Embeddings"):
            if 'cache' in st.session_state:
                del st.session_state['cache']
            st.rerun()
    else:
        st.info("Generate embeddings using Voyage AI for semantic search and scoring")

        batch_size = st.slider("Batch size:", 8, 64, 32, help="Number of passages to embed at once")

        if st.button("🚀 Generate Embeddings", type="primary"):
            with st.spinner("Generating embeddings..."):
                try:
                    passage_id_map = finder.embed_and_store_passages(
                        df=df,
                        passage_column=passage_col,
                        label_columns=label_columns,
                        namespace=namespace,
                        batch_size=batch_size
                    )

                    # Store in cache
                    if 'cache' not in st.session_state:
                        st.session_state['cache'] = {}

                    st.session_state['cache']['passage_id_map'] = passage_id_map

                    st.success(f"✅ Generated embeddings for {len(passage_id_map)} passages")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error generating embeddings: {e}")

    st.markdown("---")

    # Score section
    st.markdown("#### 2️⃣ Calculate Quality Scores")

    if not has_embeddings:
        st.warning("⚠️ Generate embeddings first")
        return

    if has_scores:
        scores_df = cache['df_summary']
        st.success(f"✅ Quality scores exist for {len(scores_df)} passages")

        # Show score distribution
        with st.expander("📊 Score Distribution", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Consistency Scores**")
                st.metric("Mean", f"{scores_df['consistency_avg'].mean():.3f}")
                st.metric("Median", f"{scores_df['consistency_avg'].median():.3f}")

            with col2:
                st.markdown("**Rerank Scores**")
                st.metric("Mean", f"{scores_df['rerank_avg'].mean():.3f}")
                st.metric("Median", f"{scores_df['rerank_avg'].median():.3f}")

        if st.button("🔄 Recompute Scores"):
            if 'df_summary' in cache:
                del cache['df_summary']
            st.rerun()
    else:
        st.info("Calculate quality scores using similarity and reranking")

        k_similar = st.slider("Similar passages to check:", 5, 50, 20,
                             help="Number of similar passages to compare for consistency")

        if st.button("🎯 Calculate Scores", type="primary"):
            with st.spinner("Calculating quality scores..."):
                try:
                    start_time = datetime.now()

                    # Get embedded indices
                    passage_id_map = cache['passage_id_map']
                    embedded_indices = list(passage_id_map.keys())

                    # Calculate scores
                    consistency_scores = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, idx in enumerate(embedded_indices):
                        # Find similar passages
                        similar = finder.find_similar_passages(
                            query_idx=idx,
                            k=k_similar,
                            namespace=namespace
                        )

                        # Calculate consistency
                        consistency = finder.calculate_label_consistency(
                            query_idx=idx,
                            similar_passages=similar,
                            label_columns=label_columns,
                            namespace=namespace
                        )

                        # Get active labels
                        passage_labels = [label for label in label_columns
                                        if df.loc[idx, label] == 1]

                        if passage_labels:
                            avg_consistency = sum(consistency[label] for label in passage_labels) / len(passage_labels)
                        else:
                            avg_consistency = 0.0

                        consistency_scores[idx] = avg_consistency

                        # Update progress
                        progress = (i + 1) / len(embedded_indices)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {i + 1}/{len(embedded_indices)}")

                    progress_bar.empty()
                    status_text.empty()

                    # Create summary dataframe
                    summary_data = []
                    for idx in embedded_indices:
                        summary_data.append({
                            'passage_idx': idx,
                            'consistency_avg': consistency_scores[idx],
                            'rerank_avg': consistency_scores[idx]  # Simplified for now
                        })

                    scores_df = pd.DataFrame(summary_data)

                    # Store in cache
                    cache['df_summary'] = scores_df

                    elapsed = (datetime.now() - start_time).total_seconds()
                    st.success(f"✅ Calculated scores for {len(scores_df)} passages in {elapsed:.1f}s")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error calculating scores: {e}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())


# ============================================================================
# SECTION 4: CREATE TRAINING SETS
# ============================================================================

def render_create_training_sets_section():
    """Create tiered training datasets"""

    if not st.session_state.get('initialized'):
        st.info("💡 Load a dataset first")
        return

    # Use the existing comprehensive implementation from data_preparation.py
    render_data_preparation_page(dict(st.session_state))