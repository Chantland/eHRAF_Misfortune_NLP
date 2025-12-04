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
from core.quality_scoring import QualityScorer

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
        finder = QualityScorer(
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
        # Add source indicator
        source_icon = "📦" if obj_meta.get('source') == 'primary' else "📌"
        source_label = "" if obj_meta.get('source') == 'primary' else " (shared)"

        with st.expander(f"{source_icon} {obj_meta['name']}{source_label} ({obj_meta['stage'].upper()})",
                         expanded=False):
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
                # Show source
                if obj_meta.get('source') == 'fallback':
                    st.caption("📌 From _data (shared)")

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

            # ✅ FIX: Ensure stable IDs exist
            if 'passage_id' not in data_obj.df.columns:
                st.warning("⚠️ Adding stable IDs to loaded data...")
                from components.data_loader import SmartDataLoader
                data_obj.df = SmartDataLoader.add_stable_ids(
                    data_obj.df,
                    data_obj.passage_col
                )
                # Save the updated dataframe back
                manager.save(data_obj)
                st.success("✅ Added stable IDs and saved")

            # Set as current
            st.session_state['current_data_object'] = data_obj

            # ✅ POPULATE LEGACY SESSION STATE FOR TRAINING
            st.session_state['initialized'] = True
            st.session_state['df'] = data_obj.df
            st.session_state['label_columns'] = data_obj.label_columns
            st.session_state['passage_col'] = data_obj.passage_col
            st.session_state['namespace'] = data_obj.namespace

            # Populate cache if available
            # NOTE: embeddings_cache is now {stable_id: pinecone_id}, not {df.index: pinecone_id}
            if data_obj.has_embeddings or data_obj.has_scores:
                st.session_state['cache'] = {
                    'stable_id_to_pinecone': data_obj.embeddings_cache if data_obj.has_embeddings else {},
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
    """Initialize the QualityScorer"""
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
            scorer = QualityScorer(
                index_name="hraf-misfortune-test",
                region="us-east-1"
            )

            st.session_state['finder'] = scorer  # Keep key name for compatibility
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

                # ✅ FIX: Add stable IDs using SmartDataLoader
                from components.data_loader import SmartDataLoader
                df = SmartDataLoader.add_stable_ids(df, config['passage_col'])
                st.info(f"✅ Generated stable IDs for {len(df)} passages")

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

                        # ✅ FIX: Preserve stable IDs through cleaning
                        if 'passage_id' in obj.df.columns:
                            # Map stable IDs to cleaned data
                            passage_ids = obj.df.set_index(obj.passage_col)['passage_id'].to_dict()
                            df_cleaned['passage_id'] = df_cleaned[obj.passage_col].map(passage_ids)

                            # Generate new IDs for any new/modified passages
                            missing_ids = df_cleaned['passage_id'].isna()
                            if missing_ids.any():
                                from components.data_loader import SmartDataLoader
                                for idx in df_cleaned[missing_ids].index:
                                    text = df_cleaned.loc[idx, obj.passage_col]
                                    df_cleaned.loc[idx, 'passage_id'] = SmartDataLoader._generate_stable_id(text)

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

            # ✅ FIX: Rebuild mapping using stable IDs
            if 'passage_id' in obj.df.columns:
                # Rebuild the mapping based on stable IDs
                rebuilt_map = {}
                for idx in obj.df.index:
                    stable_id = obj.df.loc[idx, 'passage_id']
                    pinecone_id = f"passage_{stable_id}"

                    # Check if this exists in cached embeddings
                    if pinecone_id in [v for v in embeddings.values()]:
                        rebuilt_map[idx] = pinecone_id

                obj.embeddings_cache = rebuilt_map
                st.success(f"✅ Loaded {len(rebuilt_map)} cached embeddings (mapped by stable IDs)")
            else:
                # Fallback to original cached mapping
                obj.embeddings_cache = embeddings
                st.warning("⚠️ No stable IDs - using original index mapping")

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
    """Actions for SCORED data - ENHANCED with quality exploration"""

    st.markdown("### 🎯 Create Training Tiers")

    st.info("Create quality-stratified training sets with dynamic threshold control")

    # ========================================================================
    # QUALITY SCORE EXPLORATION
    # ========================================================================

    with st.expander("📊 Explore Quality Score Distribution", expanded=True):
        render_quality_score_explorer(obj)

    st.markdown("---")

    # ========================================================================
    # DYNAMIC TIER PREVIEW
    # ========================================================================

    st.markdown("### ⚙️ Configure Tiers with Live Preview")

    # Get recommended thresholds based on actual data
    recommendations = calculate_dynamic_thresholds(obj)

    st.info(f"""
    💡 **Dynamic Recommendations** (based on your data):

    - **Tier 1:** Top {recommendations['tier1_percentile']:.0f}th percentile quality
    - **Tier 2:** Top {recommendations['tier2_percentile']:.0f}th percentile quality
    - Targets {recommendations['target_tier1_pct'] * 100:.0f}% Tier 1, {recommendations['target_tier2_pct'] * 100:.0f}% Tier 2
    """)

    # Configuration tabs
    config_method = st.radio(
        "Configuration method:",
        ["🎯 Smart Presets (Data-Aware)", "🔧 Manual Thresholds + Preview", "🎲 Skip Quality Filtering"],
        horizontal=False,
        key="tier_config_method"
    )

    if config_method == "🎯 Smart Presets (Data-Aware)":
        render_smart_presets(obj, recommendations)
    elif config_method == "🔧 Manual Thresholds + Preview":
        render_manual_thresholds_with_preview(obj, recommendations)
    else:
        render_skip_quality_tiering(obj)


def render_quality_score_explorer(obj: DataObject):
    """Interactive quality score distribution explorer"""

    scores_df = obj.scores_cache

    if scores_df is None or len(scores_df) == 0:
        st.warning("No scores available")
        return

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Scored Passages", len(scores_df))

    with col2:
        mean_cons = scores_df['consistency_avg'].mean()
        st.metric("Avg Consistency", f"{mean_cons:.3f}")
        st.caption(f"Median: {scores_df['consistency_avg'].median():.3f}")

    with col3:
        mean_rerank = scores_df['rerank_avg'].mean()
        st.metric("Avg Rerank", f"{mean_rerank:.3f}")
        st.caption(f"Median: {scores_df['rerank_avg'].median():.3f}")

    with col4:
        composite = (scores_df['consistency_avg'] + scores_df['rerank_avg']) / 2
        st.metric("Avg Composite", f"{composite.mean():.3f}")
        st.caption(f"Median: {composite.median():.3f}")

    # Distribution plots
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Consistency distribution
    axes[0].hist(scores_df['consistency_avg'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0].axvline(scores_df['consistency_avg'].median(), color='red', linestyle='--',
                    label=f'Median: {scores_df["consistency_avg"].median():.3f}')
    axes[0].set_xlabel('Consistency Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Consistency Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Rerank distribution
    axes[1].hist(scores_df['rerank_avg'], bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
    axes[1].axvline(scores_df['rerank_avg'].median(), color='red', linestyle='--',
                    label=f'Median: {scores_df["rerank_avg"].median():.3f}')
    axes[1].set_xlabel('Rerank Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Rerank Distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Composite distribution
    axes[2].hist(composite, bins=30, color='#27AE60', alpha=0.7, edgecolor='black')
    axes[2].axvline(composite.median(), color='red', linestyle='--',
                    label=f'Median: {composite.median():.3f}')
    axes[2].set_xlabel('Composite Score')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Composite Distribution')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Percentile table
    st.markdown("**Score Percentiles:**")

    percentiles = [10, 25, 50, 75, 90, 95]
    percentile_data = []

    for p in percentiles:
        percentile_data.append({
            'Percentile': f'{p}th',
            'Consistency': f"{scores_df['consistency_avg'].quantile(p / 100):.3f}",
            'Rerank': f"{scores_df['rerank_avg'].quantile(p / 100):.3f}",
            'Composite': f"{composite.quantile(p / 100):.3f}"
        })

    st.dataframe(pd.DataFrame(percentile_data), hide_index=True, width='stretch')

    st.caption("💡 Use percentiles to set thresholds that capture the right amount of data")


def calculate_dynamic_thresholds(obj: DataObject) -> Dict:
    """Calculate recommended thresholds based on actual score distribution"""

    scores_df = obj.scores_cache
    total_passages = len(obj.df)

    # MORE AGGRESSIVE DEFAULTS
    target_tier1_pct = 0.18  # 18% for tier 1 (was 15%)
    target_tier2_pct = 0.30  # 30% for tier 2 (was 27%)
    target_total_pct = target_tier1_pct + target_tier2_pct  # 48% total training (was 42%)

    # Calculate what percentile these represent
    tier1_percentile = (1 - target_tier1_pct) * 100  # Top 18% = 82nd percentile
    tier2_percentile = (1 - target_total_pct) * 100  # Top 48% = 52nd percentile

    # Get actual score values at these percentiles
    composite = (scores_df['consistency_avg'] + scores_df['rerank_avg']) / 2

    tier1_composite_threshold = composite.quantile(tier1_percentile / 100)
    tier2_composite_threshold = composite.quantile(tier2_percentile / 100)

    tier1_cons = scores_df['consistency_avg'].quantile(tier1_percentile / 100)
    tier1_rerank = scores_df['rerank_avg'].quantile(tier1_percentile / 100)

    tier2_cons = scores_df['consistency_avg'].quantile(tier2_percentile / 100)
    tier2_rerank = scores_df['rerank_avg'].quantile(tier2_percentile / 100)

    # ✅ SANITY CHECK: If thresholds are absurdly high, use fallbacks
    if tier1_cons > 0.75 or tier1_rerank > 0.65:
        st.warning("⚠️ Calculated thresholds very high. Using relaxed fallbacks.")
        tier1_cons = min(tier1_cons, 0.60)
        tier1_rerank = min(tier1_rerank, 0.45)

    if tier2_cons > 0.65 or tier2_rerank > 0.55:
        tier2_cons = min(tier2_cons, 0.45)
        tier2_rerank = min(tier2_rerank, 0.30)

    return {
        'target_tier1_pct': target_tier1_pct,
        'target_tier2_pct': target_tier2_pct,
        'tier1_percentile': tier1_percentile,
        'tier2_percentile': tier2_percentile,

        'tier1_consistency': tier1_cons,
        'tier1_rerank': tier1_rerank,
        'tier1_composite': tier1_composite_threshold,

        'tier2_consistency': tier2_cons,
        'tier2_rerank': tier2_rerank,
        'tier2_composite': tier2_composite_threshold,
    }


def render_smart_presets(obj: DataObject, recommendations: Dict):
    """Data-aware presets that adapt to actual score distribution"""

    st.markdown("#### 🎯 Smart Presets")

    scores_df = obj.scores_cache
    total_passages = len(obj.df)

    # ✅ MUCH MORE AGGRESSIVE PRESETS
    presets = {
        "Aggressive (Recommended for Low Scores)": {
            'tier1_pct': 0.20,
            'tier2_pct': 0.32,
            'description': "Use more data with lower quality bars. Best when scores are generally low."
        },
        "Balanced": {
            'tier1_pct': 0.18,
            'tier2_pct': 0.30,
            'description': "Middle ground - 48% total training"
        },
        "Conservative (Only if Scores are High)": {
            'tier1_pct': 0.15,
            'tier2_pct': 0.27,
            'description': "Strict quality. Only use if most passages score >0.60"
        }
    }

    # ✅ ADD: Warn if scores are low
    median_composite = ((scores_df['consistency_avg'] + scores_df['rerank_avg']) / 2).median()

    if median_composite < 0.50:
        st.warning(f"""
        ⚠️ **Low quality scores detected** (median composite: {median_composite:.3f})

        Your quality scores are lower than typical. This usually means:
        - High inter-rater disagreement in the original labels
        - Confused/ambiguous labels (like Material_Physical)
        - Conservative reranking

        **Recommendation:** Use "Aggressive" preset or skip quality filtering entirely.
        """)
        default_preset = 0  # Aggressive
    else:
        default_preset = 1  # Balanced

    preset_choice = st.radio(
        "Select preset:",
        list(presets.keys()),
        index=default_preset,
        key="smart_preset_choice"
    )

    # Rest of function...

    preset = presets[preset_choice]

    # Calculate thresholds for this preset
    tier1_size_target = int(total_passages * preset['tier1_pct'])
    tier2_size_target = int(total_passages * preset['tier2_pct'])

    tier1_percentile = (1 - preset['tier1_pct']) * 100
    tier2_percentile = (1 - (preset['tier1_pct'] + preset['tier2_pct'])) * 100

    composite = (scores_df['consistency_avg'] + scores_df['rerank_avg']) / 2

    tier1_cons = scores_df['consistency_avg'].quantile(tier1_percentile / 100)
    tier1_rerank = scores_df['rerank_avg'].quantile(tier1_percentile / 100)

    tier2_cons = scores_df['consistency_avg'].quantile(tier2_percentile / 100)
    tier2_rerank = scores_df['rerank_avg'].quantile(tier2_percentile / 100)

    # Show what this preset will do
    st.info(f"""
    **{preset_choice}**

    {preset['description']}

    **Tier 1:** ~{tier1_size_target} passages ({preset['tier1_pct'] * 100:.0f}%)
    - Consistency ≥ {tier1_cons:.3f}
    - Rerank ≥ {tier1_rerank:.3f}

    **Tier 2:** ~{tier2_size_target} passages ({preset['tier2_pct'] * 100:.0f}%)
    - Consistency ≥ {tier2_cons:.3f}
    - Rerank ≥ {tier2_rerank:.3f}

    **Total Training:** ~{tier1_size_target + tier2_size_target} passages ({(preset['tier1_pct'] + preset['tier2_pct']) * 100:.0f}%)
    """)

    # No-label passage control
    st.markdown("---")
    st.markdown("**No-Label Passage Control**")

    no_label_count = (obj.df[obj.label_columns].sum(axis=1) == 0).sum()
    st.caption(f"Dataset contains {no_label_count} passages with no labels")

    no_label_strategy = st.radio(
        "Include passages with no labels:",
        ["Remove all", "Include limited number", "Include all"],
        index=0,
        key="smart_preset_no_label",
        help="Control how many passages with no active labels are included in training"
    )

    max_no_label = 0  # default: remove all

    if no_label_strategy == "Include limited number":
        # Calculate reasonable default (5-10% of tier1 size)
        default_limit = min(200, int(tier1_size_target * 0.1))

        max_no_label = st.number_input(
            "Max no-label passages per tier:",
            min_value=0,
            max_value=no_label_count,
            value=default_limit,
            step=50,
            key="smart_preset_no_label_limit",
            help="Number of unlabeled passages to include in each tier"
        )

        st.caption(f"💡 Will include up to {max_no_label} unlabeled passages in Tier 1 and Tier 2")

    elif no_label_strategy == "Include all":
        max_no_label = -1  # special value meaning "don't filter"
        st.info(f"ℹ️ All {no_label_count} unlabeled passages will be kept")

    # Label targeting option
    use_targeting = st.checkbox("Enable label targeting for rare labels", value=True)

    label_targets = None
    if use_targeting:
        label_targets = render_quick_label_targeting(obj, tier1_size_target, tier2_size_target)

    # Name and create
    default_name = f"tiered_{obj.name}_{preset_choice.split()[0].lower()}_{datetime.now().strftime('%H%M')}"
    tiered_name = st.text_input("New object name:", value=default_name, key="smart_preset_name")

    if st.button("🎯 Create Tiers with Smart Preset", type="primary"):
        tier1_config = {
            'min_consistency': tier1_cons,
            'min_rerank': tier1_rerank,
            'target_size': tier1_size_target
        }

        tier2_config = {
            'min_consistency': tier2_cons,
            'min_rerank': tier2_rerank,
            'target_size': tier2_size_target
        }

        create_tiers_and_save(
            obj,
            tier1_config,
            tier2_config,
            label_targets,
            tiered_name,
            max_no_label_passages=max_no_label
        )


def render_manual_thresholds_with_preview(obj: DataObject, recommendations: Dict):
    """Manual threshold control with real-time preview"""

    st.markdown("#### 🔧 Manual Threshold Control")

    scores_df = obj.scores_cache
    total_passages = len(obj.df)

    st.caption("Adjust thresholds and see live preview of tier sizes")

    # Use recommendations as defaults
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Tier 1 (Elite):**")

        tier1_cons = st.slider(
            "Min consistency:",
            0.0, 1.0,
            float(recommendations['tier1_consistency']),
            0.05,
            key="manual_tier1_cons"
        )

        tier1_rerank = st.slider(
            "Min rerank:",
            0.0, 1.0,
            float(recommendations['tier1_rerank']),
            0.05,
            key="manual_tier1_rerank"
        )

        tier1_size = st.number_input(
            "Target size:",
            min_value=100,
            max_value=total_passages,
            value=int(total_passages * recommendations['target_tier1_pct']),
            step=100,
            key="manual_tier1_size"
        )

    with col2:
        st.markdown("**Tier 2 (Expansion):**")

        tier2_cons = st.slider(
            "Min consistency:",
            0.0, 1.0,
            float(recommendations['tier2_consistency']),
            0.05,
            key="manual_tier2_cons"
        )

        tier2_rerank = st.slider(
            "Min rerank:",
            0.0, 1.0,
            float(recommendations['tier2_rerank']),
            0.05,
            key="manual_tier2_rerank"
        )

        tier2_size = st.number_input(
            "Target size:",
            min_value=100,
            max_value=total_passages,
            value=int(total_passages * recommendations['target_tier2_pct']),
            step=100,
            key="manual_tier2_size"
        )

    st.markdown("---")

    # REAL-TIME PREVIEW
    st.markdown("### 👁️ Live Preview")

    # Calculate how many passages meet each threshold
    tier1_mask = (
            (scores_df['consistency_avg'] >= tier1_cons) &
            (scores_df['rerank_avg'] >= tier1_rerank)
    )

    tier1_candidates = scores_df[tier1_mask]
    tier1_actual = min(len(tier1_candidates), tier1_size)

    # For tier 2, exclude tier1
    remaining_scores = scores_df[~tier1_mask]
    tier2_mask = (
            (remaining_scores['consistency_avg'] >= tier2_cons) &
            (remaining_scores['rerank_avg'] >= tier2_rerank)
    )

    tier2_candidates = remaining_scores[tier2_mask]
    tier2_actual = min(len(tier2_candidates), tier2_size)

    inference_actual = total_passages - tier1_actual - tier2_actual
    training_total = tier1_actual + tier2_actual

    # Show preview
    preview_col1, preview_col2, preview_col3, preview_col4 = st.columns(4)

    with preview_col1:
        st.metric("Tier 1", f"{tier1_actual:,}")
        pct = (tier1_actual / total_passages * 100)
        st.caption(f"{pct:.1f}% of data")
        if tier1_actual < tier1_size:
            st.warning(f"⚠️ Only {tier1_actual} meet criteria")

    with preview_col2:
        st.metric("Tier 2", f"{tier2_actual:,}")
        pct = (tier2_actual / total_passages * 100)
        st.caption(f"{pct:.1f}% of data")
        if tier2_actual < tier2_size:
            st.warning(f"⚠️ Only {tier2_actual} meet criteria")

    with preview_col3:
        st.metric("Training Total", f"{training_total:,}")
        pct = (training_total / total_passages * 100)
        st.caption(f"{pct:.1f}% of data")
        if pct < 35:
            st.error("❌ Too little training data!")
        elif pct < 40:
            st.warning("⚠️ Consider more data")
        else:
            st.success("✅ Good amount")

    with preview_col4:
        st.metric("Inference", f"{inference_actual:,}")
        pct = (inference_actual / total_passages * 100)
        st.caption(f"{pct:.1f}% of data")
        if pct > 65:
            st.warning("⚠️ Wasting data")

    # Rare label preview
    with st.expander("🔍 Rare Label Preview", expanded=False):
        rare_labels = [label for label in obj.label_columns
                       if (obj.df[label] == 1).sum() / len(obj.df) < 0.10]

        if rare_labels:
            st.markdown("**Expected rare label counts in each tier:**")

            for label in rare_labels:
                total_label = (obj.df[label] == 1).sum()
                expected_tier1 = int(tier1_actual * (total_label / total_passages))
                expected_tier2 = int(tier2_actual * (total_label / total_passages))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"**{label}**")
                with col2:
                    st.caption(f"Tier 1: ~{expected_tier1}")
                    if expected_tier1 < 30:
                        st.caption("⚠️ May be too few")
                with col3:
                    st.caption(f"Tier 2: ~{expected_tier2}")

    st.markdown("---")

    st.markdown("### 🧹 No-Label Passage Control")

    no_label_count = (obj.df[obj.label_columns].sum(axis=1) == 0).sum()
    st.info(f"ℹ️ Dataset contains **{no_label_count}** passages with no labels")

    col1, col2 = st.columns([1, 2])

    with col1:
        include_no_labels = st.checkbox(
            "Include unlabeled passages",
            value=False,
            key="manual_include_no_labels",
            help="Whether to include passages with no active labels"
        )

    with col2:
        if include_no_labels:
            max_no_label = st.number_input(
                "Max per tier:",
                min_value=1,
                max_value=no_label_count,
                value=min(100, int(tier1_actual * 0.05)),
                step=10,
                key="manual_no_label_limit"
            )
        else:
            max_no_label = 0  # Remove all

    if include_no_labels:
        st.caption(f"Will include up to {max_no_label} unlabeled passages in Tier 1 and Tier 2")
    else:
        st.caption("All unlabeled passages will be removed from training tiers")

    # Label targeting
    use_targeting = st.checkbox("Enable label targeting", value=True, key="manual_targeting")

    label_targets = None
    if use_targeting:
        label_targets = render_quick_label_targeting(obj, tier1_actual, tier2_actual)

    # Name and create
    default_name = f"tiered_{obj.name}_manual_{datetime.now().strftime('%H%M')}"
    tiered_name = st.text_input("New object name:", value=default_name, key="manual_name")

    if st.button("🎯 Create Tiers with Manual Settings", type="primary"):
        if training_total < total_passages * 0.30:
            st.error("❌ Training data too small! Relax thresholds or use more data.")
            return

        tier1_config = {
            'min_consistency': tier1_cons,
            'min_rerank': tier1_rerank,
            'target_size': tier1_size
        }

        tier2_config = {
            'min_consistency': tier2_cons,
            'min_rerank': tier2_rerank,
            'target_size': tier2_size
        }

        create_tiers_and_save(
            obj,
            tier1_config,
            tier2_config,
            label_targets,
            tiered_name,
            max_no_label_passages=max_no_label  # ADD THIS
        )


def render_quick_label_targeting(obj: DataObject, tier1_size: int, tier2_size: int) -> Dict:
    """Quick label targeting UI"""

    # Find rare labels
    rare_labels = []
    for label in obj.label_columns:
        count = (obj.df[label] == 1).sum()
        freq = count / len(obj.df)
        if freq < 0.15:  # Rare if <15%
            rare_labels.append({'label': label, 'count': count, 'freq': freq})

    if not rare_labels:
        st.info("No rare labels detected (<15% frequency)")
        return None

    rare_labels.sort(key=lambda x: x['freq'])  # Rarest first

    st.markdown("**Quick Label Targeting:**")
    st.caption(f"Setting minimums for {len(rare_labels)} rare labels")

    tier1_targets = {}
    tier2_targets = {}

    for label_info in rare_labels:
        label = label_info['label']
        count = label_info['count']
        freq = label_info['freq']

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.caption(f"**{label}** ({count} total, {freq * 100:.1f}%)")

        with col2:
            # Default: aim for 30-50 in tier1 for very rare, or 30% of total
            default_t1 = min(50, int(count * 0.4), int(tier1_size * 0.08))
            tier1_targets[label] = st.number_input(
                "T1",
                0, count,
                default_t1,
                5,
                key=f"quick_t1_{label}",
                label_visibility="collapsed"
            )

        with col3:
            default_t2 = min(40, int(count * 0.3), int(tier2_size * 0.05))
            tier2_targets[label] = st.number_input(
                "T2",
                0, count,
                default_t2,
                5,
                key=f"quick_t2_{label}",
                label_visibility="collapsed"
            )

    return {'tier1': tier1_targets, 'tier2': tier2_targets}


def render_skip_quality_tiering(obj: DataObject):
    """Skip quality-based tiering, use simple stratified split"""

    st.markdown("#### 🎲 Skip Quality Filtering")

    st.warning("""
    **This will ignore quality scores entirely.**

    Use stratified random sampling to ensure rare labels are represented.
    Good when quality scores are unreliable or you want maximum data usage.
    """)

    total = len(obj.df)

    col1, col2 = st.columns(2)

    with col1:
        tier1_pct = st.slider("Tier 1 %:", 10, 30, 15, 1)
        tier1_size = int(total * tier1_pct / 100)
        st.caption(f"~{tier1_size:,} passages")

    with col2:
        tier2_pct = st.slider("Tier 2 %:", 15, 40, 27, 1)
        tier2_size = int(total * tier2_pct / 100)
        st.caption(f"~{tier2_size:,} passages")

    training_pct = tier1_pct + tier2_pct
    st.info(f"**Total training: {training_pct}%** (~{tier1_size + tier2_size:,} passages)")

    st.markdown("---")
    st.markdown("**No-Label Passages**")

    no_label_count = (obj.df[obj.label_columns].sum(axis=1) == 0).sum()
    st.caption(f"Dataset contains {no_label_count} passages with no labels")

    keep_no_labels = st.checkbox(
        "Include unlabeled passages",
        value=False,
        key="random_keep_no_labels"
    )

    if not keep_no_labels and no_label_count > 0:
        st.info(f"ℹ️ Will remove all {no_label_count} unlabeled passages before splitting")

    # Stratification option
    rare_labels = [label for label in obj.label_columns
                   if (obj.df[label] == 1).sum() / len(obj.df) < 0.10]

    if rare_labels:
        stratify_by = st.selectbox(
            "Stratify by (ensures proportional representation):",
            ["None"] + rare_labels,
            index=1 if rare_labels else 0
        )
    else:
        stratify_by = "None"

    default_name = f"tiered_{obj.name}_random_{datetime.now().strftime('%H%M')}"
    tiered_name = st.text_input("New object name:", value=default_name, key="random_name")

    if st.button("🎲 Create Random Stratified Tiers", type="primary"):
        from sklearn.model_selection import train_test_split

        with st.spinner("Creating tiers..."):
            df = obj.df.copy()

            # Filter no-label passages if requested
            if not keep_no_labels:
                original_len = len(df)
                df = df[df[obj.label_columns].sum(axis=1) > 0].copy()
                removed = original_len - len(df)
                if removed > 0:
                    st.info(f"🧹 Removed {removed} passages with no labels")

            # Stratify if specified
            stratify_col = df[stratify_by] if stratify_by != "None" else None

            # Split: tier1 vs rest
            tier1_df, rest_df = train_test_split(
                df,
                test_size=(100 - tier1_pct) / 100,
                stratify=stratify_col,
                random_state=42
            )

            # Split rest: tier2 vs inference
            if stratify_by != "None":
                stratify_col2 = rest_df[stratify_by]
            else:
                stratify_col2 = None

            tier2_df, inference_df = train_test_split(
                rest_df,
                test_size=(100 - tier1_pct - tier2_pct) / (100 - tier1_pct),
                stratify=stratify_col2,
                random_state=42
            )

            # Create metadata
            metadata = {
                'method': 'random_stratified',
                'stratified_by': stratify_by,
                'tiers': {
                    'tier1': {'count': len(tier1_df), 'percentage': tier1_pct},
                    'tier2': {'count': len(tier2_df), 'percentage': tier2_pct},
                    'inference': {'count': len(inference_df), 'percentage': 100 - training_pct}
                }
            }

            save_tiered_object(obj, tiered_name, tier1_df, tier2_df, inference_df, metadata)


def create_tiers_and_save(
        obj: DataObject,
        tier1_config: Dict,
        tier2_config: Dict,
        label_targets: Optional[Dict],
        tiered_name: str,
        max_no_label_passages: int = 0  # NEW PARAMETER
):
    """Helper to create tiers with validation and retry options"""

    with st.spinner("Creating tiers..."):
        segmenter = DataSegmenter(obj.df, obj.scores_cache, obj.label_columns)

        # Use enhanced method if targeting enabled
        if label_targets:
            tier1, tier2, inference, metadata = segmenter.create_stratified_quality_tiers(
                tier1_config,
                tier2_config,
                label_targets=label_targets,
                max_no_label_passages=max_no_label_passages  # ADD THIS
            )
        else:
            tier1, tier2, inference, metadata = segmenter.create_quality_tiers(
                tier1_config,
                tier2_config,
                label_targets=None,
                max_no_label_passages=max_no_label_passages  # ADD THIS
            )

        # Validate before saving
        is_valid, error_msg = validate_tier_sizes_before_creation(
            len(tier1), len(tier2), len(obj.df)
        )

        if not is_valid:
            st.error("❌ Tier configuration rejected:")
            st.error(error_msg)

            # Show the problematic distribution
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tier 1", len(tier1),
                          delta=f"{len(tier1) - 600}",
                          delta_color="off" if len(tier1) < 600 else "normal")
            with col2:
                training_pct = (len(tier1) + len(tier2)) / len(obj.df) * 100
                st.metric("Training %", f"{training_pct:.1f}%",
                          delta=f"{training_pct - 35:.1f}%",
                          delta_color="off" if training_pct < 35 else "normal")
            with col3:
                wasted_pct = len(inference) / len(obj.df) * 100
                st.metric("Wasted %", f"{wasted_pct:.1f}%",
                          delta=f"{wasted_pct - 65:.1f}%",
                          delta_color="inverse")

            st.warning("⚠️ Tiers NOT created. Choose a solution below:")

            # ✅ PROVIDE IMMEDIATE SOLUTIONS
            st.markdown("---")
            st.markdown("### 🔧 Quick Fix Options")

            solution = st.radio(
                "How to fix this:",
                [
                    "🎲 Skip Quality Filtering (Use Random Stratified Split)",
                    "⚙️ Use Relaxed Thresholds (Auto-calculated)",
                    "✏️ Manually Adjust Settings Above"
                ],
                key="tier_fix_solution"
            )

            if solution.startswith("🎲"):
                st.info("""
                **Recommended:** Skip quality filtering entirely and use stratified random sampling.

                This will:
                - Use 18% for Tier 1 (~1,200 passages)
                - Use 30% for Tier 2 (~2,000 passages)  
                - Stratify by Material_Physical to ensure representation
                - Total: 48% for training
                """)

                if st.button("✅ Create Random Stratified Tiers", type="primary", key="quick_random_tiers"):
                    create_random_stratified_tiers_immediately(obj, tiered_name)

            elif solution.startswith("⚙️"):
                # Calculate what thresholds would give us 18% + 30%
                scores_df = obj.scores_cache

                # Target: top 18% for tier 1
                tier1_percentile = 82  # 100 - 18
                tier1_cons_relaxed = scores_df['consistency_avg'].quantile(tier1_percentile / 100)
                tier1_rerank_relaxed = scores_df['rerank_avg'].quantile(tier1_percentile / 100)

                # Target: top 48% total (18% + 30%)
                tier2_percentile = 52  # 100 - 48
                tier2_cons_relaxed = scores_df['consistency_avg'].quantile(tier2_percentile / 100)
                tier2_rerank_relaxed = scores_df['rerank_avg'].quantile(tier2_percentile / 100)

                # Cap at reasonable minimums
                tier1_cons_relaxed = max(0.40, min(tier1_cons_relaxed, 0.65))
                tier1_rerank_relaxed = max(0.25, min(tier1_rerank_relaxed, 0.50))
                tier2_cons_relaxed = max(0.25, min(tier2_cons_relaxed, 0.50))
                tier2_rerank_relaxed = max(0.15, min(tier2_rerank_relaxed, 0.35))

                target_tier1_size = int(len(obj.df) * 0.18)
                target_tier2_size = int(len(obj.df) * 0.30)

                st.info(f"""
                **Auto-calculated relaxed thresholds:**

                Tier 1 (~{target_tier1_size} passages):
                - Consistency ≥ {tier1_cons_relaxed:.3f}
                - Rerank ≥ {tier1_rerank_relaxed:.3f}

                Tier 2 (~{target_tier2_size} passages):
                - Consistency ≥ {tier2_cons_relaxed:.3f}
                - Rerank ≥ {tier2_rerank_relaxed:.3f}

                These are the 82nd and 52nd percentiles of your data.
                """)

                if st.button("✅ Create Tiers with Relaxed Thresholds", type="primary", key="quick_relaxed_tiers"):
                    relaxed_tier1_config = {
                        'min_consistency': tier1_cons_relaxed,
                        'min_rerank': tier1_rerank_relaxed,
                        'target_size': target_tier1_size
                    }

                    relaxed_tier2_config = {
                        'min_consistency': tier2_cons_relaxed,
                        'min_rerank': tier2_rerank_relaxed,
                        'target_size': target_tier2_size
                    }

                    # Recursively call with relaxed settings
                    create_tiers_and_save(
                        obj,
                        relaxed_tier1_config,
                        relaxed_tier2_config,
                        label_targets,  # Keep same label targets
                        f"{tiered_name}_relaxed"
                    )

            else:  # Manual adjustment
                st.info(
                    "👆 Scroll up and adjust the thresholds in the configuration section above, then click Create again.")

            return  # Don't save

        # If valid, save
        save_tiered_object(obj, tiered_name, tier1, tier2, inference, metadata)


def create_random_stratified_tiers_immediately(obj: DataObject, base_name: str):
    """Immediately create random stratified tiers without quality filtering"""

    from sklearn.model_selection import train_test_split

    with st.spinner("Creating random stratified tiers..."):
        df = obj.df.copy()

        # Use Material_Physical for stratification (most problematic rare label)
        rare_labels = [label for label in obj.label_columns
                       if (df[label] == 1).sum() / len(df) < 0.10]

        if 'Material_Physical' in rare_labels:
            stratify_col = df['Material_Physical']
            stratify_name = 'Material_Physical'
        elif rare_labels:
            stratify_col = df[rare_labels[0]]
            stratify_name = rare_labels[0]
        else:
            stratify_col = None
            stratify_name = 'None'

        # Split: 18% tier1, 30% tier2, 52% inference
        tier1_df, rest_df = train_test_split(
            df,
            test_size=0.82,  # Keep 82%
            stratify=stratify_col,
            random_state=42
        )

        # Split rest: 30% of original = 36.6% of remaining
        if stratify_name != 'None':
            stratify_col2 = rest_df[stratify_name]
        else:
            stratify_col2 = None

        tier2_df, inference_df = train_test_split(
            rest_df,
            test_size=0.634,  # 52/82 = 0.634
            stratify=stratify_col2,
            random_state=42
        )

        # Validate
        is_valid, error_msg = validate_tier_sizes_before_creation(
            len(tier1_df), len(tier2_df), len(df)
        )

        if not is_valid:
            st.error("❌ Even random stratified split failed! This shouldn't happen.")
            st.error(error_msg)
            return

        # Create metadata
        metadata = {
            'method': 'random_stratified',
            'stratified_by': stratify_name,
            'note': 'Quality scores ignored - random stratified split',
            'tiers': {
                'tier1': {'count': len(tier1_df), 'percentage': 18},
                'tier2': {'count': len(tier2_df), 'percentage': 30},
                'inference': {'count': len(inference_df), 'percentage': 52}
            }
        }

        save_tiered_object(obj, f"{base_name}_random", tier1_df, tier2_df, inference_df, metadata)


def save_tiered_object(obj, tiered_name, tier1, tier2, inference, metadata):
    """Helper to save tiered object with verification display"""

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
    st.success(f"✅ Created TIERED object: {tiered_name}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tier 1", len(tier1))
    with col2:
        st.metric("Tier 2", len(tier2))
    with col3:
        st.metric("Inference", len(inference))

    # ✅ SHOW VERIFICATION IF TARGETING WAS USED
    if metadata.get('label_targeting', {}).get('enabled'):
        st.markdown("---")
        st.markdown("### 🎯 Label Targeting Verification")

        verification = metadata['label_targeting']['verification']

        for tier_name, tier_verification in verification.items():
            st.markdown(f"**{tier_name.upper()}:**")

            for label, info in tier_verification.items():
                target = info['target']
                actual = info['actual']
                met = info['met']

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.caption(f"**{label}**")

                with col2:
                    status = "✅" if met else "⚠️"
                    st.caption(f"{status} {actual}/{target}")

                with col3:
                    st.caption(f"{info['percentage']:.1f}% of tier")

                with col4:
                    st.caption(f"Global: {info['global_frequency']:.1f}%")

            st.markdown("")

    st.balloons()
    st.rerun()


def render_tiered_actions(obj: DataObject):
    """Actions for TIERED data"""

    st.markdown("### ✅ Ready for Training")

    # ✅ ADD: Go back to parent
    if obj.parent:

        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("⬅️ Go Back to SCORED", type="secondary"):
                # Load parent object
                pipeline = st.session_state.pipeline
                parent_obj = pipeline.manager.load(obj.parent['name'], obj.parent['stage'])

                if parent_obj:
                    st.session_state['current_data_object'] = parent_obj
                    st.success(f"✅ Loaded parent: {obj.parent['name']}")
                    st.rerun()
                else:
                    st.error("❌ Could not load parent")

        with col2:
            st.caption("Delete this tiered object and try different settings")

        st.markdown("---")

    # Show current tiers
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


def validate_tier_sizes_before_creation(
        tier1_actual: int,
        tier2_actual: int,
        total: int,
        min_tier1: int = 600,
        min_training_pct: float = 0.35
) -> Tuple[bool, str]:
    """
    Validate tier sizes are reasonable before creating

    Returns: (is_valid, error_message)
    """

    training_total = tier1_actual + tier2_actual
    training_pct = training_total / total

    errors = []

    if tier1_actual < min_tier1:
        errors.append(f"❌ Tier 1 too small: {tier1_actual} < {min_tier1} minimum")

    if training_pct < min_training_pct:
        errors.append(f"❌ Training data too small: {training_pct * 100:.1f}% < {min_training_pct * 100:.0f}% minimum")

    if tier1_actual + tier2_actual < 1500:
        errors.append(f"❌ Total training too small: {tier1_actual + tier2_actual} < 1500 minimum")

    if errors:
        error_msg = "\n".join(errors)
        error_msg += f"\n\n**Current distribution:**\n"
        error_msg += f"- Tier 1: {tier1_actual} ({tier1_actual / total * 100:.1f}%)\n"
        error_msg += f"- Tier 2: {tier2_actual} ({tier2_actual / total * 100:.1f}%)\n"
        error_msg += f"- Training Total: {training_total} ({training_pct * 100:.1f}%)\n"
        error_msg += f"- Inference: {total - training_total} ({(total - training_total) / total * 100:.1f}%)\n\n"
        error_msg += "**Solutions:**\n"
        error_msg += "1. ⬇️ Lower quality thresholds (most common fix)\n"
        error_msg += "2. 🎲 Use 'Skip Quality Filtering' for random stratified split\n"
        error_msg += "3. ⚙️ Increase target tier sizes\n"

        return False, error_msg

    return True, ""

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
            # Returns {stable_id: pinecone_id} now
            stable_id_to_pinecone = finder.embed_and_store_passages(
                df=obj.df,
                passage_column=obj.passage_col,
                label_columns=obj.label_columns,
                namespace=obj.namespace,
                batch_size=batch_size
            )

            # Save to cache (now maps stable_id -> pinecone_id)
            cache_manager = st.session_state.cache_manager
            cache_manager.save_embeddings(obj.namespace, stable_id_to_pinecone)

            # Create EMBEDDED object
            pipeline = st.session_state.pipeline
            embedded_obj = pipeline.create_embedded(
                name=embedded_name,
                parent_obj=obj,
                embeddings_cache=stable_id_to_pinecone  # Changed mapping
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
    """Calculate quality scores with proper reranking and class imbalance handling"""

    finder = st.session_state.get('finder')
    if finder is None:
        st.error("❌ Finder not initialized")
        return

    if 'passage_id' not in obj.df.columns:
        st.error("❌ DataFrame missing stable passage_id column")
        return

    # Calculate label frequencies for weighting
    label_frequencies = {
        label: obj.df[label].mean()
        for label in obj.label_columns
    }

    st.info(f"""
    **Label Frequencies:**
    - Very Rare (<5%): {sum(1 for f in label_frequencies.values() if f < 0.05)} labels
    - Rare (5-15%): {sum(1 for f in label_frequencies.values() if 0.05 <= f < 0.15)} labels  
    - Common (>15%): {sum(1 for f in label_frequencies.values() if f >= 0.15)} labels
    """)

    with st.spinner("Step 1/3: Calculating consistency scores..."):
        try:
            consistency_scores = {}
            consistency_by_label = {}  # Track per-label scores

            progress = st.progress(0)
            status = st.empty()

            valid_rows = obj.df[obj.df['passage_id'].notna()]

            for i, (df_idx, row) in enumerate(valid_rows.iterrows()):
                stable_id = row['passage_id']

                try:
                    # Find similar passages using stable ID
                    similar = finder.search_similar_to_passage(
                        passage_idx=df_idx,
                        namespace=obj.namespace,
                        k=k_similar,
                        exclude_self=True,
                        df=obj.df
                    )

                    if not similar:
                        consistency_scores[df_idx] = 0.0
                        consistency_by_label[df_idx] = {l: 0.0 for l in obj.label_columns}
                        continue

                    # Calculate consistency with improved method
                    consistency = finder.calculate_label_consistency(
                        query_idx=df_idx,
                        similar_passages=similar,
                        label_columns=obj.label_columns,
                        df=obj.df,
                        namespace=obj.namespace,
                        label_frequencies=label_frequencies
                    )

                    # Store per-label scores
                    consistency_by_label[df_idx] = consistency

                    # Average across ACTIVE labels only
                    active_labels = [l for l in obj.label_columns if obj.df.loc[df_idx, l] == 1]

                    if active_labels:
                        # Weight by inverse frequency for averaging
                        weighted_scores = []
                        for label in active_labels:
                            freq = label_frequencies[label]
                            # Rare labels get higher weight
                            weight = 1.0 / (freq + 0.01)  # +0.01 to avoid division by zero
                            weighted_scores.append(consistency[label] * weight)

                        avg = sum(weighted_scores) / sum(1.0 / (label_frequencies[l] + 0.01) for l in active_labels)
                    else:
                        avg = 0.0

                    consistency_scores[df_idx] = avg

                except Exception as e:
                    st.warning(f"⚠️ Error processing {stable_id[:8]}: {e}")
                    consistency_scores[df_idx] = 0.0
                    consistency_by_label[df_idx] = {l: 0.0 for l in obj.label_columns}

                # Update progress
                pct = (i + 1) / len(valid_rows)
                progress.progress(pct)
                status.text(f"Consistency: {i + 1}/{len(valid_rows)}")

            progress.empty()
            status.empty()

            st.success(f"✅ Calculated consistency for {len(consistency_scores)} passages")

        except Exception as e:
            st.error(f"❌ Consistency calculation failed: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
            return

    # Step 2: Calculate rerank scores
    with st.spinner("Step 2/3: Calculating semantic relevance scores..."):
        try:
            # Prepare passages for reranking
            passages_to_rerank = [
                (idx, str(obj.df.loc[idx, obj.passage_col]))
                for idx in valid_rows.index
                if obj.df.loc[idx, obj.passage_col] is not None
            ]

            # ✅ FIX: Use correct method name
            rerank_scores = finder.calculate_rerank_scores(  # Changed from calculate_rerank_scores_batch
                passages=passages_to_rerank,
                label_columns=obj.label_columns,
                df=obj.df,
                batch_size=32
            )

            st.success(f"✅ Reranked {len(rerank_scores)} passages")

        except Exception as e:
            st.error(f"❌ Reranking failed: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
            return

    # Step 3: Combine scores and create DataFrame
    with st.spinner("Step 3/3: Creating scores dataframe..."):
        scores_data = []

        for df_idx in obj.df.index:
            if df_idx not in consistency_scores:
                continue

            # Get per-label scores
            cons_by_label = consistency_by_label.get(df_idx, {})
            rerank_by_label = rerank_scores.get(df_idx, {})

            # Calculate weighted averages for active labels
            active_labels = [l for l in obj.label_columns if obj.df.loc[df_idx, l] == 1]

            if active_labels:
                # Consistency avg (frequency-weighted)
                cons_weighted = []
                for label in active_labels:
                    freq = label_frequencies[label]
                    weight = 1.0 / (freq + 0.01)
                    cons_weighted.append(cons_by_label.get(label, 0.0) * weight)

                cons_avg = sum(cons_weighted) / sum(1.0 / (label_frequencies[l] + 0.01) for l in active_labels)

                # Rerank avg (frequency-weighted)
                rerank_weighted = []
                for label in active_labels:
                    freq = label_frequencies[label]
                    weight = 1.0 / (freq + 0.01)
                    rerank_weighted.append(rerank_by_label.get(label, 0.0) * weight)

                rerank_avg = sum(rerank_weighted) / sum(1.0 / (label_frequencies[l] + 0.01) for l in active_labels)
            else:
                cons_avg = 0.0
                rerank_avg = 0.0

            scores_data.append({
                'passage_idx': df_idx,
                'stable_id': obj.df.loc[df_idx, 'passage_id'],
                'consistency_avg': cons_avg,
                'rerank_avg': rerank_avg,
                'composite': 0.5 * cons_avg + 0.5 * rerank_avg,
                **{f'consistency_{l}': cons_by_label.get(l, 0.0) for l in obj.label_columns},
                **{f'rerank_{l}': rerank_by_label.get(l, 0.0) for l in obj.label_columns}
            })

        scores_df = pd.DataFrame(scores_data)

        # Show distribution
        st.markdown("**Score Distribution:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Consistency", f"{scores_df['consistency_avg'].mean():.3f}")
            st.caption(f"Median: {scores_df['consistency_avg'].median():.3f}")

        with col2:
            st.metric("Rerank", f"{scores_df['rerank_avg'].mean():.3f}")
            st.caption(f"Median: {scores_df['rerank_avg'].median():.3f}")

        with col3:
            st.metric("Composite", f"{scores_df['composite'].mean():.3f}")
            st.caption(f"Median: {scores_df['composite'].median():.3f}")

        # Save and continue
        cache_manager = st.session_state.cache_manager
        cache_manager.save_scores(obj.namespace, scores_df)

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
