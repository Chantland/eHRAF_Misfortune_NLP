"""
Simple HRAF Data Tool
No workflow, no steps, just functions
"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="HRAF Data Tool", layout="wide")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'passage_col' not in st.session_state:
    st.session_state.passage_col = None
if 'label_cols' not in st.session_state:
    st.session_state.label_cols = []
if 'quality_scores' not in st.session_state:
    st.session_state.quality_scores = None

st.title("🔬 HRAF Data Tool")
st.caption("Load data, filter by quality, train models")

# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

with st.expander("📂 1. Load Data", expanded=st.session_state.df is None):
    uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])

    if uploaded_file:
        col1, col2 = st.columns(2)

        with col1:
            header_row = st.number_input("Header row (0-indexed)", 0, 10, 0)

        if st.button("Load File"):
            with st.spinner("Loading..."):
                df = pd.read_excel(uploaded_file, header=header_row)

                # Fix mixed type columns that cause Arrow errors
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Convert all values to strings to avoid mixed types
                        df[col] = df[col].astype(str)

                st.session_state.df = df
                st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

                # Display preview
                preview_df = df.head().copy()
                st.dataframe(preview_df, width='stretch')

    # If data loaded, configure columns
    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown("**Configure Columns:**")

        col1, col2 = st.columns(2)

        with col1:
            passage_col = st.selectbox(
                "Passage column:",
                options=st.session_state.df.columns.tolist(),
                index=0
            )
            st.session_state.passage_col = passage_col

        with col2:
            # Auto-detect binary columns
            binary_cols = []
            for col in st.session_state.df.columns:
                try:
                    unique = st.session_state.df[col].dropna().unique()
                    if len(unique) <= 2 and all(v in [0, 1, 0.0, 1.0] for v in unique):
                        binary_cols.append(col)
                except:
                    pass

            label_cols = st.multiselect(
                "Label columns:",
                options=binary_cols,
                default=binary_cols
            )
            st.session_state.label_cols = label_cols

        if label_cols:
            st.success(f"✅ {len(label_cols)} labels configured")

            # Show label distribution
            dist_data = []
            for label in label_cols:
                count = int((st.session_state.df[label] == 1).sum())
                pct = count / len(st.session_state.df) * 100
                dist_data.append({'Label': label, 'Count': count, 'Percent': f"{pct:.1f}%"})

            st.dataframe(pd.DataFrame(dist_data), hide_index=True, width=800)

# ============================================================================
# SECTION 2: FILTERING
# ============================================================================

if st.session_state.df is not None and st.session_state.label_cols:
    with st.expander("🔍 2. Filter Data", expanded=True):
        df = st.session_state.df
        passage_col = st.session_state.passage_col
        label_cols = st.session_state.label_cols

        st.markdown("**Simple Filters:**")

        col1, col2, col3 = st.columns(3)

        with col1:
            min_length = st.number_input("Min passage length", 0, 10000, 0)

        with col2:
            max_length = st.number_input("Max passage length", 0, 20000, 20000)

        with col3:
            min_labels = st.number_input("Min labels per passage", 0, 10, 0)

        # Apply filters
        mask = pd.Series([True] * len(df))

        if min_length > 0:
            lengths = df[passage_col].astype(str).str.len()
            mask &= lengths >= min_length

        if max_length < 20000:
            lengths = df[passage_col].astype(str).str.len()
            mask &= lengths <= max_length

        if min_labels > 0:
            label_counts = df[label_cols].sum(axis=1)
            mask &= label_counts >= min_labels

        # Label-specific filtering
        st.markdown("**Filter by Labels:**")

        col1, col2 = st.columns(2)

        with col1:
            require_labels = st.multiselect(
                "Must have these labels:",
                options=label_cols
            )

        with col2:
            exclude_labels = st.multiselect(
                "Must NOT have these labels:",
                options=label_cols
            )

        for label in require_labels:
            mask &= df[label] == 1

        for label in exclude_labels:
            mask &= df[label] == 0

        # Show results
        filtered_df = df[mask]

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Passages", len(df))
        with col2:
            st.metric("After Filters", len(filtered_df))
        with col3:
            pct = len(filtered_df) / len(df) * 100 if len(df) > 0 else 0
            st.metric("Kept", f"{pct:.1f}%")

        # Show filtered distribution
        if len(filtered_df) > 0:
            st.markdown("**Filtered Label Distribution:**")
            dist_data = []
            for label in label_cols:
                count = int((filtered_df[label] == 1).sum())
                pct = count / len(filtered_df) * 100
                dist_data.append({'Label': label, 'Count': count, 'Percent': f"{pct:.1f}%"})

            st.dataframe(pd.DataFrame(dist_data), hide_index=True, width=800)

            # Export button
            if st.button("💾 Export Filtered Data"):
                output_path = Path("filtered_data.xlsx")
                filtered_df.to_excel(output_path, index=False)
                st.success(f"✅ Saved to {output_path}")

                # Offer download
                with open(output_path, 'rb') as f:
                    st.download_button(
                        "⬇️ Download Filtered Data",
                        f,
                        file_name="filtered_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

# ============================================================================
# SECTION 3: QUALITY ANALYSIS (OPTIONAL)
# ============================================================================

if st.session_state.df is not None and st.session_state.label_cols:
    with st.expander("🔬 3. Quality Analysis (Optional - requires API keys)", expanded=False):
        st.markdown("""
        **What this does:** Uses embeddings to find passages with inconsistent labels.
        
        **Requirements:**
        - VOYAGE_API_KEY in .env
        - PINECONE_API_KEY in .env
        
        **Skip this if:** You just want to filter and export data for training.
        """)

        import os
        has_keys = os.getenv("VOYAGE_API_KEY") and os.getenv("PINECONE_API_KEY")

        if not has_keys:
            st.warning("⚠️ API keys not found - quality analysis disabled")
        else:
            if st.button("Compute Quality Scores"):
                st.info("This would compute quality scores, but let's fix the basic filtering first")

# ============================================================================
# SECTION 4: TRAINING
# ============================================================================

if st.session_state.df is not None and st.session_state.label_cols:
    with st.expander("🎓 4. Train Model (Coming Soon)", expanded=False):
        st.markdown("""
        **Training interface will go here.**
        
        For now, export your filtered data and use your existing training scripts.
        """)

# ============================================================================
# SIDEBAR: QUICK STATS
# ============================================================================

with st.sidebar:
    st.markdown("### Current Data")

    if st.session_state.df is not None:
        st.metric("Passages", len(st.session_state.df))
        st.metric("Labels", len(st.session_state.label_cols))

        if st.session_state.passage_col:
            st.caption(f"Passage col: {st.session_state.passage_col}")
    else:
        st.info("No data loaded")

    st.markdown("---")

    if st.button("🔄 Reset All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()