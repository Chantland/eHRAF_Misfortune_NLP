"""
Data Page - REFACTORED with New Architecture + Interactive Preview

Complete implementation including:
- New DataObject pipeline workflow
- Original interactive data preview system
- Full integration with cache and object management
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import new architecture
from core.data_cache import CacheManager
from core.data_objects import (
    DataObject, DataObjectManager, DataPipeline, PipelineStage
)
from core.data_preparation import DataAnalyzer, DataSegmenter
from core.discovery_architecture import GoldenDatasetFinder


# ============================================================================
# MAIN RENDER FUNCTION
# ============================================================================

def render():
    """Main render function - NEW ARCHITECTURE"""

    st.markdown("# 📊 Data Pipeline")
    st.caption("RAW → CLEANED → EMBEDDED → SCORED → TIERED")

    # Initialize managers
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = DataPipeline()

    if 'cache_manager' not in st.session_state:
        st.session_state.cache_manager = CacheManager()

    # CRITICAL - Initialize finder on EVERY page load
    if 'finder' not in st.session_state or st.session_state['finder'] is None:
        from dotenv import load_dotenv
        import os
        load_dotenv()

        # Don't catch exceptions - let them show
        finder = GoldenDatasetFinder(
            voyage_api_key=os.getenv("VOYAGE_API_KEY"),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            index_name="hraf-misfortune-test",
            region="us-east-1"
        )
        st.session_state['finder'] = finder

    # Debug: Show finder status at top
    if 'finder' in st.session_state:
        st.success("✅ Finder is initialized")
    else:
        st.error("❌ Finder failed to initialize")

    # Current working object
    current_obj = st.session_state.get('current_data_object')

    # Show status bar
    render_status_bar(current_obj)

    st.markdown("---")

    # Main workflow
    if current_obj is None:
        render_start_workflow()
    else:
        render_pipeline_workflow(current_obj)


# ============================================================================
# STATUS BAR
# ============================================================================

def render_status_bar(current_obj: Optional[DataObject]):
    """Show current data object status"""

    if current_obj is None:
        st.info("💡 No data loaded. Start by loading or creating a new dataset.")
        return

    # Status card
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Current Object", current_obj.name)
        stage_emoji = {
            PipelineStage.RAW: "📦",
            PipelineStage.CLEANED: "🧹",
            PipelineStage.EMBEDDED: "🔢",
            PipelineStage.SCORED: "📊",
            PipelineStage.TIERED: "🎯"
        }
        st.caption(f"{stage_emoji.get(current_obj.stage, '📦')} Stage: **{current_obj.stage.value.upper()}**")

    with col2:
        st.metric("Passages", f"{len(current_obj.df):,}")
        st.caption(f"**{len(current_obj.label_columns)}** labels")

    with col3:
        if current_obj.has_embeddings:
            st.metric("Embeddings", "✅")
            st.caption(f"{len(current_obj.embeddings_cache):,} embedded")
        else:
            st.metric("Embeddings", "❌")
            st.caption("Not embedded")

    with col4:
        if current_obj.has_scores:
            st.metric("Scores", "✅")
            st.caption(f"{len(current_obj.scores_cache):,} scored")
        else:
            st.metric("Scores", "❌")
            st.caption("Not scored")

    with col5:
        if st.button("🔄 Switch Object"):
            st.session_state['show_object_browser'] = True
            st.rerun()


# ============================================================================
# START WORKFLOW
# ============================================================================

def render_start_workflow():
    """Initial workflow - load or create data"""

    st.markdown("### 🚀 Start New Pipeline")

    tab1, tab2 = st.tabs(["📂 Load Existing", "➕ Create New"])

    with tab1:
        render_object_browser()

    with tab2:
        render_new_data_loader()


def render_object_browser():
    """Browse and load existing data objects"""

    st.markdown("#### 📚 Saved Data Objects")

    manager = st.session_state.pipeline.manager

    # Group by stage
    all_objects = manager.list_objects()

    if not all_objects:
        st.info("💡 No saved objects found. Create a new one in the 'Create New' tab.")
        return

    # Filter by stage
    stage_filter = st.selectbox(
        "Filter by stage:",
        ["All"] + [s.value for s in PipelineStage],
        key="obj_browser_filter"
    )

    if stage_filter != "All":
        all_objects = [obj for obj in all_objects if obj['stage'] == stage_filter]

    st.caption(f"Showing {len(all_objects)} objects")

    # Display objects
    for obj_meta in all_objects:
        with st.expander(f"📦 {obj_meta['name']} ({obj_meta['stage'].upper()})", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Passages", f"{obj_meta.get('num_passages', 0):,}")
                st.caption(f"Labels: {obj_meta.get('num_labels', 0)}")

            with col2:
                emb_status = "✅" if obj_meta.get('has_embeddings') else "❌"
                st.metric("Embeddings", emb_status)
                score_status = "✅" if obj_meta.get('has_scores') else "❌"
                st.caption(f"Scores: {score_status}")

            with col3:
                st.caption(f"Created: {obj_meta.get('created_at', 'Unknown')[:10]}")
                if obj_meta.get('parent'):
                    st.caption(f"Parent: {obj_meta['parent']}")

            # Load button
            if st.button(f"📂 Load '{obj_meta['name']}'", key=f"load_{obj_meta['name']}_{obj_meta['stage']}"):
                load_data_object(obj_meta['name'], PipelineStage(obj_meta['stage']))


def render_new_data_loader():
    """Create new data object from file"""

    st.markdown("#### ➕ Create New Data Object")

    st.info("💡 Load a new dataset and configure it as a RAW data object")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose Excel file:",
        type=['xlsx', 'xls'],
        key="new_data_upload"
    )

    if uploaded_file:
        # Check if we already have a confirmed config
        if 'confirmed_config' not in st.session_state:
            st.session_state['confirmed_config'] = None
            st.session_state['confirmed_file_bytes'] = None

        # Show interactive preview if config not yet confirmed
        if st.session_state['confirmed_config'] is None:
            config = render_interactive_data_preview(uploaded_file)

            # Store config when confirmed
            if config:
                st.session_state['confirmed_config'] = config
                # Store the file bytes for later use
                uploaded_file.seek(0)  # Reset to beginning
                st.session_state['confirmed_file_bytes'] = uploaded_file.read()
                st.rerun()
        else:
            # Config already confirmed, show create object UI
            create_raw_data_object(
                st.session_state['confirmed_config'],
                st.session_state['confirmed_file_bytes']
            )


def load_data_object(name: str, stage: PipelineStage):
    """Load existing data object"""

    with st.spinner(f"Loading {name}..."):
        try:
            manager = st.session_state.pipeline.manager
            data_obj = manager.load(name, stage)

            if data_obj is None:
                st.error(f"❌ Could not load {name}")
                return

            # Set as current
            st.session_state['current_data_object'] = data_obj

            # ✅ POPULATE LEGACY SESSION STATE FOR TRAINING
            st.session_state['initialized'] = True
            st.session_state['df'] = data_obj.df
            st.session_state['label_columns'] = data_obj.label_columns
            st.session_state['passage_col'] = data_obj.passage_col
            st.session_state['namespace'] = data_obj.namespace

            # Populate cache if available
            if data_obj.has_embeddings or data_obj.has_scores:
                st.session_state['cache'] = {
                    'passage_id_map': data_obj.embeddings_cache if data_obj.has_embeddings else {},
                    'df_summary': data_obj.scores_cache if data_obj.has_scores else None
                }

            # Initialize finder if needed
            if 'finder' not in st.session_state:
                initialize_finder()

            st.success(f"✅ Loaded: {name}")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error loading: {e}")


def initialize_finder():
    """Initialize the GoldenDatasetFinder"""
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()

        voyage_key = os.getenv("VOYAGE_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")

        if not voyage_key:
            st.error("❌ VOYAGE_API_KEY not found in environment")
            st.info("💡 Create a .env file with: VOYAGE_API_KEY=your_key_here")
            return False

        if not pinecone_key:
            st.error("❌ PINECONE_API_KEY not found in environment")
            st.info("💡 Add to .env file: PINECONE_API_KEY=your_key_here")
            return False

        with st.spinner("Connecting to Voyage AI and Pinecone..."):
            finder = GoldenDatasetFinder(
                voyage_api_key=voyage_key,
                pinecone_api_key=pinecone_key,
                index_name="hraf-misfortune-test",
                region="us-east-1"
            )

            st.session_state['finder'] = finder
            return True

    except Exception as e:
        st.error(f"❌ Error initializing finder: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        return False


# ============================================================================
# INTERACTIVE DATA PREVIEW (from original data_page.py)
# ============================================================================

def make_df_display_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dataframe to display-safe format for Streamlit"""
    df_safe = df.copy()
    for col in df_safe.columns:
        if df_safe[col].dtype == 'object':
            df_safe[col] = df_safe[col].astype(str)
    return df_safe


def render_interactive_data_preview(uploaded_file):
    """
    Interactive data preview with complete column control
    Click the row to set header, check boxes to select labels

    Args:
        uploaded_file: Streamlit UploadedFile object
    """

    st.markdown("#### 📋 Data Preview & Configuration")
    st.caption("Configure exactly which columns to use - no assumptions")

    # Initialize preview state
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if 'preview_df' not in st.session_state or st.session_state.get('preview_file_id') != file_id:
        try:
            # Read directly from buffer for preview
            uploaded_file.seek(0)  # Reset to beginning
            df_raw = pd.read_excel(uploaded_file, header=None, nrows=10)

            # Convert to display-safe format
            df_raw_display = df_raw.copy()
            for col in df_raw_display.columns:
                df_raw_display[col] = df_raw_display[col].astype(str)

            st.session_state['preview_df'] = df_raw
            st.session_state['preview_df_display'] = df_raw_display
            st.session_state['preview_file_id'] = file_id
            st.session_state['preview_header_row'] = 0
            st.session_state['selected_columns'] = []
            st.session_state['selected_label_columns'] = []

        except Exception as e:
            st.error(f"❌ Error reading Excel file: {e}")
            st.info("💡 Make sure the file is a valid Excel file (.xlsx or .xls)")
            return None

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
            row_str = " | ".join([str(v)[:30] for v in row_data[:10]])
            if len(row_data) > 10:
                row_str += " | ..."

            if is_selected:
                st.success(f"**{row_str}**")
            else:
                st.text(row_str)

    header_row = current_header

    # Load with selected header row - READ FROM BUFFER AGAIN
    try:
        uploaded_file.seek(0)
        df_preview = pd.read_excel(uploaded_file, header=header_row, nrows=5)
    except Exception as e:
        st.error(f"❌ Error reading with header row {header_row}: {e}")
        return None

    # Create display-safe version
    df_preview_display = make_df_display_safe(df_preview)

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

    if "Passage" in all_columns:
        default_passage_col = "Passage"
    elif "passage" in all_columns:
        default_passage_col = "passage"
    else:
        for col in all_columns:
            if "passage" in str(col).lower():
                default_passage_col = col
                break

    if default_passage_col is None:
        text_keywords = ['text', 'content', 'body', 'description']
        for keyword in text_keywords:
            for col in all_columns:
                if keyword in str(col).lower():
                    default_passage_col = col
                    break
            if default_passage_col:
                break

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

    if default_passage_col is None:
        default_passage_col = all_columns[0]

    try:
        default_index = all_columns.index(default_passage_col)
    except:
        default_index = 0

    passage_col = st.selectbox(
        "Which column contains the passage text?",
        options=all_columns,
        index=default_index,
        key="interactive_passage_col",
        help="Select the column containing the full text passages"
    )

    if passage_col == default_passage_col and default_passage_col != all_columns[0]:
        st.caption(f"💡 Auto-detected '{passage_col}' as passage column")

    if passage_col:
        st.markdown("**Passage preview:**")
        sample_passage = str(df_preview[passage_col].iloc[0])

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
    # STEP 5: Review and Load
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
            # Read from uploaded_file buffer to get row count
            uploaded_file.seek(0)
            full_df = pd.read_excel(uploaded_file, header=header_row, usecols=[passage_col])
            st.metric("Total Rows", f"{len(full_df):,}")
        except:
            st.metric("Total Rows", "Unknown")

    # Warning for non-binary labels
    non_binary = [col for col in selected_label_columns if col not in potential_labels]
    if non_binary:
        st.warning(f"⚠️ Non-binary labels: {', '.join(non_binary[:3])}" +
                   ("..." if len(non_binary) > 3 else ""))

    # Load button
    if st.button("✅ Confirm Configuration", type="primary", width='stretch'):
        return {
            'filename': uploaded_file.name,
            'header_row': header_row,
            'passage_col': passage_col,
            'all_columns': selected_columns,
            'label_columns': selected_label_columns,
            'metadata_columns': metadata_columns
        }

    return None


def create_raw_data_object(config: dict, file_bytes: bytes):
    """Create RAW data object from configuration and file bytes"""

    st.markdown("---")
    st.markdown("#### 💾 Save as Data Object")

    # Name the object
    filename = config['filename']
    default_name = f"raw_{Path(filename).stem}_{datetime.now().strftime('%Y%m%d')}"

    object_name = st.text_input(
        "Data object name:",
        value=default_name,
        key="new_obj_name"
    )

    if st.button("✅ Create RAW Data Object", type="primary"):
        with st.spinner("Creating data object..."):
            try:
                # Load data from bytes with config
                import io
                df = pd.read_excel(io.BytesIO(file_bytes), header=config['header_row'])

                # Filter columns
                keep_cols = [config['passage_col']] + config['all_columns']
                df = df[keep_cols].copy()

                # Validate labels
                for label in config['label_columns']:
                    if df[label].dtype not in ['int64', 'float64', 'Int64']:
                        df[label] = pd.to_numeric(df[label], errors='coerce').fillna(0).astype(int)

                # Create RAW data object
                pipeline = st.session_state.pipeline

                data_obj = pipeline.create_raw(
                    name=object_name,
                    df=df,
                    passage_col=config['passage_col'],
                    label_columns=config['label_columns'],
                    metadata_columns=config['metadata_columns'],
                    source_file=filename,
                    header_row=config['header_row']
                )

                st.success(f"✅ Created RAW data object: '{object_name}'")

                # Load as current
                st.session_state['current_data_object'] = data_obj

                # Initialize finder
                if 'finder' not in st.session_state:
                    initialize_finder()

                # Clear ALL preview state including confirmed config
                for key in ['preview_df', 'preview_file_id', 'preview_header_row', 'selected_columns',
                            'selected_label_columns', 'column_selections', 'label_selections',
                            'confirmed_config', 'confirmed_file_bytes']:
                    if key in st.session_state:
                        del st.session_state[key]

                st.balloons()
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error creating data object: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

# ============================================================================
# PIPELINE WORKFLOW
# ============================================================================

def render_pipeline_workflow(current_obj: DataObject):
    """Main pipeline workflow based on current stage"""

    stage = current_obj.stage

    # Show next actions based on stage
    if stage == PipelineStage.RAW:
        render_raw_actions(current_obj)

    elif stage == PipelineStage.CLEANED:
        render_cleaned_actions(current_obj)

    elif stage == PipelineStage.EMBEDDED:
        render_embedded_actions(current_obj)

    elif stage == PipelineStage.SCORED:
        render_scored_actions(current_obj)

    elif stage == PipelineStage.TIERED:
        render_tiered_actions(current_obj)


# ============================================================================
# STAGE-SPECIFIC ACTIONS
# ============================================================================

def render_raw_actions(obj: DataObject):
    """Actions for RAW data"""

    st.markdown("### 🧹 Clean Data Quality")

    st.info("Analyze and clean your raw data before embedding")

    # Initialize analyzer
    analyzer = DataAnalyzer(obj.df, obj.label_columns, obj.passage_col)

    # Run analysis
    if st.button("🔎 Analyze Data Quality", type="primary"):
        with st.spinner("Analyzing..."):
            analysis = analyzer.analyze_quality()
            st.session_state['quality_analysis'] = analysis

    # Show results
    analysis = st.session_state.get('quality_analysis')

    if analysis:
        # Display issues
        if analysis['issues']:
            st.markdown("**⚠️ Issues Found:**")
            for issue in analysis['issues']:
                st.warning(issue)
        else:
            st.success("✅ No major issues detected!")

        # Show cleaning steps
        st.markdown("---")
        st.markdown("### Select Cleaning Steps")

        cleaning_steps = analyzer.suggest_cleaning_steps(analysis)

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
                st.caption(f"−{step['impact']}")

        st.markdown("---")

        # Name for cleaned object (used by both paths)
        default_name = f"cleaned_{obj.name}_{datetime.now().strftime('%H%M')}"
        cleaned_name = st.text_input("New object name:", value=default_name)

        # TWO PATHS: Apply cleaning OR Skip cleaning
        col1, col2 = st.columns(2)

        with col1:
            # Path 1: Apply selected cleaning steps
            if selected_actions:
                # Preview impact
                total_removed = sum(s['impact'] for s in cleaning_steps if s['action'] in selected_actions)
                st.info(f"Will remove {total_removed} passages, keeping {len(obj.df) - total_removed}")

                if st.button("🧹 Apply Cleaning", type="primary", key="apply_clean", width='stretch'):
                    with st.spinner("Cleaning..."):
                        df_cleaned = analyzer.apply_cleaning(selected_actions)

                        # Create CLEANED data object
                        pipeline = st.session_state.pipeline
                        cleaned_obj = pipeline.create_cleaned(
                            name=cleaned_name,
                            parent_obj=obj,
                            df_cleaned=df_cleaned,
                            cleaning_steps=selected_actions
                        )

                        st.session_state['current_data_object'] = cleaned_obj
                        st.success(f"✅ Created CLEANED object: {cleaned_name}")
                        st.rerun()
            else:
                st.info("💡 No cleaning steps selected")

        with col2:
            # Path 2: Skip cleaning entirely
            st.info("Skip all cleaning and proceed as-is")

            if st.button("⏭️ Skip Cleaning", type="secondary", key="skip_clean", width='stretch'):
                with st.spinner("Creating CLEANED object..."):
                    # Create CLEANED object with no changes
                    pipeline = st.session_state.pipeline
                    cleaned_obj = pipeline.create_cleaned(
                        name=cleaned_name,
                        parent_obj=obj,
                        df_cleaned=obj.df.copy(),  # No changes
                        cleaning_steps=["No cleaning applied"]
                    )

                    st.session_state['current_data_object'] = cleaned_obj
                    st.success(f"✅ Created CLEANED object (no changes): {cleaned_name}")
                    st.rerun()

    else:
        # Haven't run analysis yet - still offer skip option
        st.markdown("---")
        st.markdown("### Skip Analysis & Cleaning")

        st.info("💡 If your data is already clean, you can skip directly to embedding")

        default_name = f"cleaned_{obj.name}_{datetime.now().strftime('%H%M')}"
        cleaned_name = st.text_input("New object name:", value=default_name, key="skip_name")

        if st.button("⏭️ Skip to CLEANED Stage", type="secondary", width='stretch'):
            with st.spinner("Creating CLEANED object..."):
                pipeline = st.session_state.pipeline
                cleaned_obj = pipeline.create_cleaned(
                    name=cleaned_name,
                    parent_obj=obj,
                    df_cleaned=obj.df.copy(),
                    cleaning_steps=["Skipped cleaning - data already clean"]
                )

                st.session_state['current_data_object'] = cleaned_obj
                st.success(f"✅ Created CLEANED object (no changes): {cleaned_name}")
                st.rerun()


def render_cleaned_actions(obj: DataObject):
    """Actions for CLEANED data"""

    st.markdown("### 🔢 Generate Embeddings")

    st.info("Generate semantic embeddings using Voyage AI")

    # Check if finder is initialized
    if 'finder' not in st.session_state:
        st.warning("⚠️ Initializing finder...")
        if not initialize_finder():
            st.error("❌ Cannot generate embeddings without finder. Check API keys in .env file.")
            st.info("💡 Make sure you have VOYAGE_API_KEY and PINECONE_API_KEY set in your .env file")
            return

    # Check cache
    cache_manager = st.session_state.cache_manager
    has_cached = cache_manager.has_embeddings(obj.namespace)

    if has_cached and not obj.has_embeddings:
        st.warning("⚠️ Found cached embeddings! Load them instead of regenerating.")

        if st.button("📂 Load Cached Embeddings"):
            embeddings = cache_manager.load_embeddings(obj.namespace)

            # FILTER embeddings to only include passages in current dataframe
            valid_indices = set(obj.df.index.tolist())
            filtered_embeddings = {
                idx: pid for idx, pid in embeddings.items()
                if idx in valid_indices
            }

            if len(filtered_embeddings) < len(embeddings):
                removed = len(embeddings) - len(filtered_embeddings)
                st.info(f"ℹ️ Filtered out {removed} embeddings for removed passages")

            if len(filtered_embeddings) == 0:
                st.error("❌ No embeddings match current dataframe. Generate new embeddings instead.")
                return

            obj.embeddings_cache = filtered_embeddings
            st.success(f"✅ Loaded {len(filtered_embeddings)} cached embeddings")
            st.rerun()

    batch_size = st.slider("Batch size:", 8, 64, 32)

    # Name for embedded object
    default_name = f"embedded_{obj.name}_{datetime.now().strftime('%H%M')}"
    embedded_name = st.text_input("New object name:", value=default_name, key="embed_name_input")

    if st.button("🚀 Generate Embeddings", type="primary"):
        generate_embeddings(obj, batch_size, embedded_name)


def render_embedded_actions(obj: DataObject):
    """Actions for EMBEDDED data"""

    st.markdown("### 📊 Calculate Quality Scores")

    st.info("Calculate consistency and rerank scores")

    # Check if finder is initialized
    if 'finder' not in st.session_state:
        st.warning("⚠️ Initializing finder...")
        if not initialize_finder():
            st.error("❌ Cannot calculate scores without finder. Check API keys in .env file.")
            return

    # Check cache
    cache_manager = st.session_state.cache_manager
    has_cached = cache_manager.has_scores(obj.namespace)

    if has_cached and not obj.has_scores:
        st.warning("⚠️ Found cached scores! Load them instead of recalculating.")

        if st.button("📂 Load Cached Scores"):
            scores = cache_manager.load_scores(obj.namespace)
            obj.scores_cache = scores
            st.success(f"✅ Loaded {len(scores)} cached scores")
            st.rerun()

    k_similar = st.slider("Similar passages to check:", 5, 50, 20)

    # Name for scored object
    default_name = f"scored_{obj.name}_{datetime.now().strftime('%H%M')}"
    scored_name = st.text_input("New object name:", value=default_name, key="score_name_input")

    if st.button("🎯 Calculate Scores", type="primary"):
        calculate_scores(obj, k_similar, scored_name)


def render_scored_actions(obj: DataObject):
    """Actions for SCORED data"""

    st.markdown("### 🎯 Create Training Tiers")

    st.info("Create quality-stratified training sets with optional label targeting")

    # Show score distribution
    with st.expander("📊 Score Distribution", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Consistency**")
            st.metric("Mean", f"{obj.scores_cache['consistency_avg'].mean():.3f}")
            st.metric("Median", f"{obj.scores_cache['consistency_avg'].median():.3f}")

        with col2:
            st.markdown("**Rerank**")
            st.metric("Mean", f"{obj.scores_cache['rerank_avg'].mean():.3f}")
            st.metric("Median", f"{obj.scores_cache['rerank_avg'].median():.3f}")

    st.markdown("---")

    # Configuration method
    config_method = st.radio(
        "Configuration method:",
        ["Quick Presets", "Advanced + Label Targeting"],
        horizontal=True,
        key="tier_config_method"
    )

    if config_method == "Quick Presets":
        render_quick_tier_presets(obj)
    else:
        render_advanced_tier_config(obj)


def render_quick_tier_presets(obj: DataObject):
    """Simple preset-based tier creation"""

    st.markdown("#### Quick Tier Creation")

    preset = st.selectbox(
        "Configuration preset:",
        ["Balanced", "Conservative (High Quality)", "Aggressive (More Data)"]
    )

    # Name for tiered object
    default_name = f"tiered_{obj.name}_{datetime.now().strftime('%H%M')}"
    tiered_name = st.text_input("New object name:", value=default_name, key="tier_name_input")

    if st.button("🎯 Create Tiers", type="primary"):
        with st.spinner("Creating tiers..."):
            segmenter = DataSegmenter(obj.df, obj.scores_cache, obj.label_columns)

            # Configure based on preset
            if preset == "Balanced":
                tier1_config = {'min_consistency': 0.65, 'min_rerank': 0.45, 'target_size': int(len(obj.df) * 0.12)}
                tier2_config = {'min_consistency': 0.45, 'min_rerank': 0.30, 'target_size': int(len(obj.df) * 0.25)}
            elif preset == "Conservative (High Quality)":
                tier1_config = {'min_consistency': 0.70, 'min_rerank': 0.50, 'target_size': int(len(obj.df) * 0.10)}
                tier2_config = {'min_consistency': 0.50, 'min_rerank': 0.35, 'target_size': int(len(obj.df) * 0.22)}
            else:  # Aggressive
                tier1_config = {'min_consistency': 0.60, 'min_rerank': 0.40, 'target_size': int(len(obj.df) * 0.15)}
                tier2_config = {'min_consistency': 0.40, 'min_rerank': 0.25, 'target_size': int(len(obj.df) * 0.28)}

            tier1, tier2, inference, metadata = segmenter.create_quality_tiers(
                tier1_config, tier2_config
            )

            save_tiered_object(obj, tiered_name, tier1, tier2, inference, metadata)


def render_advanced_tier_config(obj: DataObject):
    """Advanced configuration with label targeting"""

    st.markdown("#### Advanced Configuration")

    # Analyze label distribution
    label_stats = []
    for label in obj.label_columns:
        count = int((obj.df[label] == 1).sum())
        pct = (count / len(obj.df)) * 100
        label_stats.append({
            'label': label,
            'count': count,
            'percentage': pct,
            'is_rare': pct < 10
        })

    label_stats_df = pd.DataFrame(label_stats).sort_values('percentage')

    # Show distribution
    with st.expander("📊 Label Distribution Analysis", expanded=True):
        st.dataframe(
            label_stats_df.assign(
                Percentage=label_stats_df['percentage'].apply(lambda x: f"{x:.1f}%"),
                Rare=label_stats_df['is_rare'].apply(lambda x: "⚠️ Rare" if x else "✓")
            )[['label', 'count', 'Percentage', 'Rare']],
            hide_index=True,
            width='stretch'
        )

    st.markdown("---")

    # Tier size targets
    st.markdown("##### Tier Sizes")

    col1, col2 = st.columns(2)

    with col1:
        tier1_size = st.number_input(
            "Tier 1 target size:",
            min_value=50,
            max_value=len(obj.df),
            value=int(len(obj.df) * 0.12),
            step=50,
            key="tier1_size"
        )

        tier1_min_cons = st.slider(
            "Tier 1 min consistency:",
            0.0, 1.0, 0.65, 0.05,
            key="tier1_cons"
        )

        tier1_min_rerank = st.slider(
            "Tier 1 min rerank:",
            0.0, 1.0, 0.45, 0.05,
            key="tier1_rerank"
        )

    with col2:
        tier2_size = st.number_input(
            "Tier 2 target size:",
            min_value=50,
            max_value=len(obj.df),
            value=int(len(obj.df) * 0.25),
            step=50,
            key="tier2_size"
        )

        tier2_min_cons = st.slider(
            "Tier 2 min consistency:",
            0.0, 1.0, 0.45, 0.05,
            key="tier2_cons"
        )

        tier2_min_rerank = st.slider(
            "Tier 2 min rerank:",
            0.0, 1.0, 0.30, 0.05,
            key="tier2_rerank"
        )

    st.markdown("---")

    # Label targeting
    st.markdown("##### 🎯 Label Targeting (Optional)")

    st.info("""
    💡 **Label Targeting** ensures rare or important labels are well-represented in your training tiers.

    Without targeting, tiers are selected purely by quality scores. With targeting, the system will 
    prioritize including passages with specific labels up to your target counts.
    """)

    use_targeting = st.checkbox(
        "Enable label targeting",
        value=False,
        help="Prioritize specific labels to ensure representation"
    )

    label_targets = None

    if use_targeting:
        st.markdown("**Set target counts for specific labels:**")

        # Quick actions
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🎯 Target All Rare Labels (<10%)", key="target_rare"):
                st.session_state['label_targeting_mode'] = 'rare'
                st.rerun()

        with col2:
            if st.button("🎯 Custom Selection", key="target_custom"):
                st.session_state['label_targeting_mode'] = 'custom'
                st.rerun()

        with col3:
            if st.button("🗑️ Clear Targets", key="clear_targets"):
                st.session_state['label_targeting_mode'] = None
                st.session_state['tier1_targets'] = {}
                st.session_state['tier2_targets'] = {}
                st.rerun()

        # Initialize targeting dictionaries
        if 'tier1_targets' not in st.session_state:
            st.session_state['tier1_targets'] = {}
        if 'tier2_targets' not in st.session_state:
            st.session_state['tier2_targets'] = {}

        # Auto-populate rare labels if mode is 'rare'
        if st.session_state.get('label_targeting_mode') == 'rare':
            rare_labels = [stat['label'] for stat in label_stats if stat['is_rare']]

            for label in rare_labels:
                count = next(s['count'] for s in label_stats if s['label'] == label)
                # Target: include most of the rare label passages
                st.session_state['tier1_targets'][label] = min(count, int(tier1_size * 0.3))
                st.session_state['tier2_targets'][label] = min(count, int(tier2_size * 0.3))

        # Display targeting UI
        st.markdown("**Tier 1 Targets:**")

        tier1_cols = st.columns(2)
        for i, stat in enumerate(label_stats_df.to_dict('records')):
            with tier1_cols[i % 2]:
                label = stat['label']
                count = stat['count']

                # Show current count
                current_target = st.session_state['tier1_targets'].get(label, 0)

                new_target = st.number_input(
                    f"{label} ({count} available)",
                    min_value=0,
                    max_value=count,
                    value=current_target,
                    step=5,
                    key=f"tier1_target_{label}",
                    help=f"{stat['percentage']:.1f}% of dataset"
                )

                if new_target > 0:
                    st.session_state['tier1_targets'][label] = new_target
                elif label in st.session_state['tier1_targets']:
                    del st.session_state['tier1_targets'][label]

        st.markdown("**Tier 2 Targets:**")

        tier2_cols = st.columns(2)
        for i, stat in enumerate(label_stats_df.to_dict('records')):
            with tier2_cols[i % 2]:
                label = stat['label']
                count = stat['count']

                current_target = st.session_state['tier2_targets'].get(label, 0)

                new_target = st.number_input(
                    f"{label} ({count} available)",
                    min_value=0,
                    max_value=count,
                    value=current_target,
                    step=5,
                    key=f"tier2_target_{label}",
                    help=f"{stat['percentage']:.1f}% of dataset"
                )

                if new_target > 0:
                    st.session_state['tier2_targets'][label] = new_target
                elif label in st.session_state['tier2_targets']:
                    del st.session_state['tier2_targets'][label]

        # Build label_targets dict
        if st.session_state['tier1_targets'] or st.session_state['tier2_targets']:
            label_targets = {}
            if st.session_state['tier1_targets']:
                label_targets['tier1'] = st.session_state['tier1_targets']
            if st.session_state['tier2_targets']:
                label_targets['tier2'] = st.session_state['tier2_targets']

            # Show summary
            with st.expander("📋 Targeting Summary", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Tier 1 Targets:**")
                    if st.session_state['tier1_targets']:
                        for label, count in st.session_state['tier1_targets'].items():
                            st.caption(f"• {label}: {count}")
                    else:
                        st.caption("No targets set")

                with col2:
                    st.markdown("**Tier 2 Targets:**")
                    if st.session_state['tier2_targets']:
                        for label, count in st.session_state['tier2_targets'].items():
                            st.caption(f"• {label}: {count}")
                    else:
                        st.caption("No targets set")

    st.markdown("---")

    # Name and create
    default_name = f"tiered_{obj.name}_{datetime.now().strftime('%H%M')}"
    if use_targeting and label_targets:
        default_name += "_targeted"

    tiered_name = st.text_input("New object name:", value=default_name, key="tier_name_advanced")

    if st.button("🎯 Create Tiers with Configuration", type="primary"):
        with st.spinner("Creating tiers with custom configuration..."):
            segmenter = DataSegmenter(obj.df, obj.scores_cache, obj.label_columns)

            tier1_config = {
                'min_consistency': tier1_min_cons,
                'min_rerank': tier1_min_rerank,
                'target_size': tier1_size
            }

            tier2_config = {
                'min_consistency': tier2_min_cons,
                'min_rerank': tier2_min_rerank,
                'target_size': tier2_size
            }

            tier1, tier2, inference, metadata = segmenter.create_quality_tiers(
                tier1_config,
                tier2_config,
                label_targets=label_targets  # ✅ NOW PASSING LABEL TARGETS
            )

            save_tiered_object(obj, tiered_name, tier1, tier2, inference, metadata)


def save_tiered_object(obj, tiered_name, tier1, tier2, inference, metadata):
    """Helper to save tiered object"""

    pipeline = st.session_state.pipeline
    tiered_obj = pipeline.create_tiered(
        name=tiered_name,
        parent_obj=obj,
        tier1_df=tier1,
        tier2_df=tier2,
        inference_df=inference,
        tier_config=metadata
    )

    st.session_state['current_data_object'] = tiered_obj

    # Show success with details
    st.success(f"✅ Created TIERED object: {tiered_name}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tier 1", len(tier1))
    with col2:
        st.metric("Tier 2", len(tier2))
    with col3:
        st.metric("Inference", len(inference))

    # Show label distribution if targeting was used
    if metadata.get('tiers', {}).get('tier1', {}).get('label_distribution'):
        with st.expander("📊 Achieved Label Distribution"):
            tier1_dist = metadata['tiers']['tier1']['label_distribution']

            st.markdown("**Tier 1 Label Counts:**")
            for label, info in tier1_dist.items():
                if info['count'] > 0:
                    st.caption(f"• {label}: {info['count']} ({info['percentage']:.1f}%)")

    st.balloons()
    st.rerun()




def render_tiered_actions(obj: DataObject):
    """Actions for TIERED data"""

    st.markdown("### ✅ Ready for Training")

    st.success("Your tiered datasets are ready!")

    # Show tier sizes
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Tier 1 (Elite)", obj.metadata.get('tier1_size', 'N/A'))

    with col2:
        st.metric("Tier 2 (Expansion)", obj.metadata.get('tier2_size', 'N/A'))

    with col3:
        st.metric("Inference (Test)", obj.metadata.get('inference_size', 'N/A'))

    st.markdown("---")

    st.markdown("**Next Steps:**")
    st.markdown("1. This data is already loaded and ready")
    st.markdown("2. Click **🤖 Models** in the sidebar")
    st.markdown("3. Choose training strategy (Tier 1 only, Combined, or Curriculum)")

    st.info("💡 Data is loaded. Use sidebar navigation to go to **🤖 Models** page")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_embeddings(obj: DataObject, batch_size: int, embedded_name: str):
    """Generate embeddings with checkpointing"""

    finder = st.session_state.get('finder')
    if finder is None:
        st.error("❌ Finder not initialized")
        return

    with st.spinner("Generating embeddings..."):
        try:
            # Embed with checkpointing
            passage_id_map = finder.embed_and_store_passages(
                df=obj.df,
                passage_column=obj.passage_col,
                label_columns=obj.label_columns,
                namespace=obj.namespace,
                batch_size=batch_size
            )

            # Save to cache
            cache_manager = st.session_state.cache_manager
            cache_manager.save_embeddings(obj.namespace, passage_id_map)

            # Create EMBEDDED object
            pipeline = st.session_state.pipeline
            embedded_obj = pipeline.create_embedded(
                name=embedded_name,
                parent_obj=obj,
                embeddings_cache=passage_id_map
            )

            st.session_state['current_data_object'] = embedded_obj
            st.success(f"✅ Created EMBEDDED object: {embedded_name}")
            st.balloons()
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


def calculate_scores(obj: DataObject, k_similar: int, scored_name: str):
    """Calculate quality scores with checkpointing"""

    finder = st.session_state.get('finder')
    if finder is None:
        st.error("❌ Finder not initialized")
        return

    with st.spinner("Calculating scores..."):
        try:
            # Get valid indices
            valid_df_indices = set(obj.df.index.tolist())
            embedded_indices = [
                idx for idx in obj.embeddings_cache.keys()
                if idx in valid_df_indices
            ]

            if not embedded_indices:
                st.error("❌ No valid embeddings found")
                return

            consistency_scores = {}
            progress = st.progress(0)
            status = st.empty()

            for i, df_idx in enumerate(embedded_indices):
                try:
                    # Pass DataFrame index directly - finder builds Pinecone ID internally
                    similar = finder.find_similar_passages(
                        query_idx=df_idx,  # ✅ Just the integer index
                        k=k_similar,
                        namespace=obj.namespace
                    )

                    # Filter similar passages to only those in current df
                    similar_filtered = [
                        s for s in similar
                        if s['passage_idx'] in valid_df_indices
                    ]

                    if not similar_filtered:
                        consistency_scores[df_idx] = 0.0
                        continue

                    # Calculate consistency - also pass DataFrame index
                    consistency = finder.calculate_label_consistency(
                        query_idx=df_idx,  # ✅ Just the integer index
                        similar_passages=similar_filtered,
                        label_columns=obj.label_columns,
                        namespace=obj.namespace
                    )

                    passage_labels = [l for l in obj.label_columns if obj.df.loc[df_idx, l] == 1]

                    if passage_labels:
                        avg = sum(consistency[l] for l in passage_labels) / len(passage_labels)
                    else:
                        avg = 0.0

                    consistency_scores[df_idx] = avg

                except Exception as e:
                    st.warning(f"⚠️ Error processing passage {df_idx}: {e}")
                    consistency_scores[df_idx] = 0.0

                # Update progress
                pct = (i + 1) / len(embedded_indices)
                progress.progress(pct)
                status.text(f"Processing: {i + 1}/{len(embedded_indices)}")

            progress.empty()
            status.empty()

            # Create scores DataFrame
            scores_df = pd.DataFrame([
                {
                    'passage_idx': idx,
                    'consistency_avg': consistency_scores[idx],
                    'rerank_avg': consistency_scores[idx]
                }
                for idx in embedded_indices
            ])

            # Save to cache
            cache_manager = st.session_state.cache_manager
            cache_manager.save_scores(obj.namespace, scores_df)

            # Create SCORED object
            pipeline = st.session_state.pipeline
            scored_obj = pipeline.create_scored(
                name=scored_name,
                parent_obj=obj,
                scores_df=scores_df
            )

            st.session_state['current_data_object'] = scored_obj
            st.success(f"✅ Created SCORED object: {scored_name}")
            st.balloons()
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
