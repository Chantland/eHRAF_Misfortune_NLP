"""
UI Helper Components
Common UI patterns, widgets, and utilities for consistent UX
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


def initialize_session_state():
    """Initialize all session state variables with defaults"""

    defaults = {
        # Data state
        'initialized': False,
        'df': None,
        'label_columns': None,
        'passage_col': None,
        'selected_file': None,
        'namespace': None,

        # Cache and scoring
        'cache': None,
        'finder': None,

        # Model state
        'model_manager': None,

        # Tier datasets
        'tier1_dataset': None,
        'tier2_dataset': None,
        'inference_dataset': None,
        'tier_metadata': None,

        # Chat state
        'show_global_chat': False,
        'chat_history': [],

        # UI state
        'current_page': '📊 Data',
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar_info():
    """Render dataset info in sidebar"""

    if not st.session_state.get('initialized'):
        st.info("💡 No dataset loaded")
        return

    st.markdown("### 📊 Current Dataset")

    df = st.session_state.get('df')
    cache = st.session_state.get('cache')
    label_columns = st.session_state.get('label_columns', [])

    if df is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Passages", len(df))

        with col2:
            if cache and cache.get('df_summary') is not None:  # ADDED: check for None
                scores_df = cache.get('df_summary')
                st.metric("Scored", len(scores_df))
            else:
                st.metric("Scored", "—")

def render_data_summary(
        df: pd.DataFrame,
        label_columns: List[str],
        passage_col: str,
        show_distribution: bool = True
) -> None:
    """
    Display comprehensive dataset summary

    Args:
        df: Dataset to summarize
        label_columns: List of label column names
        passage_col: Passage text column name
        show_distribution: Whether to show label distribution
    """

    st.markdown("### 📊 Dataset Summary")

    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Passages", f"{len(df):,}")

    with col2:
        valid = df[passage_col].notna().sum()
        st.metric("Valid Passages", f"{valid:,}")

    with col3:
        st.metric("Labels", len(label_columns))

    with col4:
        avg_length = df[passage_col].str.len().mean()
        st.metric("Avg Length", f"{avg_length:.0f}")

    # Label distribution
    if show_distribution:
        with st.expander("🏷️ Label Distribution"):
            dist_data = []

            for label in label_columns:
                count = int((df[label] == 1).sum())
                pct = (count / len(df) * 100)

                dist_data.append({
                    'Label': label,
                    'Count': count,
                    'Percentage': f"{pct:.1f}%",
                    'Visual': '█' * int(pct / 2)  # Simple bar chart
                })

            st.dataframe(
                pd.DataFrame(dist_data),
                hide_index=True,
                width='stretch'  # FIX: changed from width='stretch'
            )


def render_label_selector(
        label_columns: List[str],
        selected: Optional[List[str]] = None,
        key_suffix: str = "",
        help_text: str = "Select labels to include"
) -> List[str]:
    """
    Multi-select with search and organized by categories

    Args:
        label_columns: Available labels
        selected: Pre-selected labels
        key_suffix: Unique key suffix
        help_text: Help text for selector

    Returns:
        List of selected label names
    """

    if selected is None:
        selected = label_columns

    # Organize by category
    categories = {
        'EVENT': [],
        'CAUSE': [],
        'ACTION': [],
        'Other': []
    }

    event_keywords = ['Illness', 'Accident']
    cause_keywords = ['Material_Physical', 'Spirits_Gods', 'Witchcraft_Sorcery',
                      'Rule_Violation_Taboo', 'Just_Happens', 'Technical_Specialist']
    action_keywords = ['Physical_Material', 'Shaman_Medium_Healer',
                       'Priest_High_Religion', 'Divination']

    for label in label_columns:
        if any(kw in label for kw in event_keywords):
            categories['EVENT'].append(label)
        elif any(kw in label for kw in cause_keywords):
            categories['CAUSE'].append(label)
        elif any(kw in label for kw in action_keywords):
            categories['ACTION'].append(label)
        else:
            categories['Other'].append(label)

    # Quick selection buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Select All", key=f"select_all_{key_suffix}"):
            return label_columns

    with col2:
        if st.button("Select EVENT", key=f"select_event_{key_suffix}"):
            return categories['EVENT']

    with col3:
        if st.button("Select CAUSE", key=f"select_cause_{key_suffix}"):
            return categories['CAUSE']

    with col4:
        if st.button("Select ACTION", key=f"select_action_{key_suffix}"):
            return categories['ACTION']

    # Multi-select organized by category
    selected_labels = []

    for category, labels in categories.items():
        if not labels:
            continue

        with st.expander(f"{category} Labels", expanded=(category != 'Other')):
            for label in labels:
                checked = st.checkbox(
                    label,
                    value=(label in selected),
                    key=f"label_{label}_{key_suffix}"
                )
                if checked:
                    selected_labels.append(label)

    return selected_labels


def render_quality_threshold_controls(
        scores_df: pd.DataFrame,
        key_suffix: str = ""
) -> Dict[str, float]:
    """
    Interactive threshold controls with live preview

    Args:
        scores_df: DataFrame with quality scores
        key_suffix: Unique key suffix

    Returns:
        Dictionary with selected thresholds
    """

    st.markdown("### ⚙️ Quality Thresholds")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Consistency**")

        cons_median = scores_df['consistency_avg'].median()
        cons_mean = scores_df['consistency_avg'].mean()

        st.caption(f"Median: {cons_median:.3f} | Mean: {cons_mean:.3f}")

        min_consistency = st.slider(
            "Minimum:",
            0.0, 1.0,
            float(cons_median),
            0.05,
            key=f"cons_thresh_{key_suffix}"
        )

        # Preview
        meeting_cons = (scores_df['consistency_avg'] >= min_consistency).sum()
        st.caption(f"✓ {meeting_cons:,} passages meet threshold")

    with col2:
        st.markdown("**Rerank**")

        rerank_median = scores_df['rerank_avg'].median()
        rerank_mean = scores_df['rerank_avg'].mean()

        st.caption(f"Median: {rerank_median:.3f} | Mean: {rerank_mean:.3f}")

        min_rerank = st.slider(
            "Minimum:",
            0.0, 1.0,
            float(rerank_median),
            0.05,
            key=f"rerank_thresh_{key_suffix}"
        )

        # Preview
        meeting_rerank = (scores_df['rerank_avg'] >= min_rerank).sum()
        st.caption(f"✓ {meeting_rerank:,} passages meet threshold")

    # Combined preview
    meeting_both = (
            (scores_df['consistency_avg'] >= min_consistency) &
            (scores_df['rerank_avg'] >= min_rerank)
    ).sum()

    st.info(f"📊 **{meeting_both:,} passages** meet both thresholds ({meeting_both / len(scores_df) * 100:.1f}%)")

    return {
        'min_consistency': min_consistency,
        'min_rerank': min_rerank,
        'preview_count': meeting_both
    }


def render_progress_with_eta(
        current: int,
        total: int,
        start_time: datetime,
        task_name: str = "Processing"
) -> None:
    """
    Progress bar with time estimate

    Args:
        current: Current progress
        total: Total items
        start_time: When task started
        task_name: Name of task
    """

    progress = current / total if total > 0 else 0

    # Calculate ETA
    elapsed = (datetime.now() - start_time).total_seconds()

    if current > 0:
        time_per_item = elapsed / current
        remaining = total - current
        eta_seconds = time_per_item * remaining

        eta_minutes = int(eta_seconds // 60)
        eta_seconds = int(eta_seconds % 60)

        eta_str = f"{eta_minutes}m {eta_seconds}s remaining"
    else:
        eta_str = "Calculating..."

    st.progress(
        progress,
        text=f"{task_name}: {current}/{total} | {eta_str}"
    )


def show_success_with_action(
        message: str,
        action_label: str,
        action_callback,
        action_key: str
) -> None:
    """
    Success message with action button

    Args:
        message: Success message
        action_label: Button label
        action_callback: Function to call
        action_key: Unique key for button
    """

    col1, col2 = st.columns([3, 1])

    with col1:
        st.success(message)

    with col2:
        if st.button(action_label, key=action_key):
            action_callback()


def render_collapsible_config(
        title: str,
        config_dict: Dict[str, Any],
        show_json: bool = True
) -> None:
    """
    Render configuration in collapsible section

    Args:
        title: Section title
        config_dict: Configuration dictionary
        show_json: Whether to show raw JSON
    """

    with st.expander(f"⚙️ {title}"):
        # Formatted view
        for key, value in config_dict.items():
            if isinstance(value, (int, float)):
                st.metric(key.replace('_', ' ').title(), value)
            else:
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")

        # JSON view
        if show_json:
            st.markdown("---")
            st.json(config_dict)


def show_tooltip(text: str, icon: str = "ℹ️") -> None:
    """
    Inline tooltip

    Args:
        text: Tooltip text
        icon: Icon to display
    """
    st.markdown(
        f'<span title="{text}" style="cursor: help;">{icon}</span>',
        unsafe_allow_html=True
    )


def render_file_browser(
        base_dir: Path,
        file_types: List[str] = ['.xlsx'],
        key_suffix: str = ""
) -> Optional[Path]:
    """
    Simple file browser

    Args:
        base_dir: Base directory
        file_types: File extensions to show
        key_suffix: Unique key suffix

    Returns:
        Selected file path or None
    """

    if not base_dir.exists():
        st.warning(f"Directory not found: {base_dir}")
        return None

    # Get matching files
    files = []
    for file_type in file_types:
        files.extend(base_dir.glob(f"*{file_type}"))

    if not files:
        st.info(f"No {', '.join(file_types)} files found in {base_dir}")
        return None

    # Sort by modification time (newest first)
    files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)

    # File selector
    file_options = {f.name: f for f in files}

    selected_name = st.selectbox(
        "Select file:",
        options=list(file_options.keys()),
        key=f"file_browser_{key_suffix}"
    )

    selected_file = file_options[selected_name]

    # Show file info
    file_size = selected_file.stat().st_size / 1024  # KB
    file_modified = datetime.fromtimestamp(selected_file.stat().st_mtime)

    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Size: {file_size:.1f} KB")
    with col2:
        st.caption(f"Modified: {file_modified.strftime('%Y-%m-%d %H:%M')}")

    return selected_file


def render_confirmation_dialog(
        message: str,
        confirm_label: str = "Confirm",
        cancel_label: str = "Cancel",
        key_suffix: str = ""
) -> bool:
    """
    Confirmation dialog

    Args:
        message: Confirmation message
        confirm_label: Confirm button label
        cancel_label: Cancel button label
        key_suffix: Unique key suffix

    Returns:
        True if confirmed
    """

    st.warning(message)

    col1, col2 = st.columns(2)

    with col1:
        if st.button(confirm_label, type="primary", key=f"confirm_{key_suffix}"):
            return True

    with col2:
        if st.button(cancel_label, key=f"cancel_{key_suffix}"):
            return False

    return False