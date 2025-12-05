"""
Pipeline Progress Component - Visual progress indicator for data pipeline stages

Shows a clear, horizontal progress bar with:
- Stage icons and labels
- Current stage highlighting
- Completed stages marked
- Next steps guidance
"""

import streamlit as st
from enum import Enum
from typing import Optional

# Import PipelineStage from core
from core.data_objects import PipelineStage


# Stage configuration with visual styling
STAGE_CONFIG = {
    PipelineStage.RAW: {
        "icon": "📦",
        "label": "Raw",
        "description": "Original data loaded",
        "next_action": "Clean & validate data"
    },
    PipelineStage.CLEANED: {
        "icon": "🧹",
        "label": "Cleaned",
        "description": "Data validated & cleaned",
        "next_action": "Generate embeddings"
    },
    PipelineStage.EMBEDDED: {
        "icon": "🔢",
        "label": "Embedded",
        "description": "Semantic embeddings generated",
        "next_action": "Calculate quality scores"
    },
    PipelineStage.SCORED: {
        "icon": "📊",
        "label": "Scored",
        "description": "Quality scores calculated",
        "next_action": "Create training tiers"
    },
    PipelineStage.TIERED: {
        "icon": "🎯",
        "label": "Tiered",
        "description": "Ready for training",
        "next_action": "Go to Models page"
    }
}

# Ordered list of stages
STAGE_ORDER = [
    PipelineStage.RAW,
    PipelineStage.CLEANED,
    PipelineStage.EMBEDDED,
    PipelineStage.SCORED,
    PipelineStage.TIERED
]


def render_pipeline_progress(current_stage: Optional[PipelineStage] = None, compact: bool = False):
    """
    Render a visual pipeline progress indicator

    Args:
        current_stage: Current pipeline stage (None if no data loaded)
        compact: If True, show minimal version
    """
    if current_stage is None:
        if not compact:
            st.info("💡 **Start by loading data** to begin the pipeline")
        return

    current_idx = STAGE_ORDER.index(current_stage) if current_stage in STAGE_ORDER else -1

    if compact:
        # Compact single-line version
        _render_compact_progress(current_stage, current_idx)
    else:
        # Full progress bar
        _render_full_progress(current_stage, current_idx)


def _render_compact_progress(current_stage: PipelineStage, current_idx: int):
    """Render compact single-line progress indicator"""

    # Build progress string
    parts = []
    for i, stage in enumerate(STAGE_ORDER):
        config = STAGE_CONFIG[stage]

        if i < current_idx:
            # Completed - checkmark
            parts.append(f"✅")
        elif i == current_idx:
            # Current - highlighted
            parts.append(f"**{config['icon']}{config['label']}**")
        else:
            # Future - dimmed
            parts.append(f"○")

    # Join with arrows
    progress_str = " → ".join(parts)
    st.markdown(f"Pipeline: {progress_str}")


def _render_full_progress(current_stage: PipelineStage, current_idx: int):
    """Render full visual progress bar with details"""

    # Create columns for each stage
    cols = st.columns(len(STAGE_ORDER))

    for i, (col, stage) in enumerate(zip(cols, STAGE_ORDER)):
        config = STAGE_CONFIG[stage]

        with col:
            if i < current_idx:
                # Completed stage
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background-color: #d4edda; border-radius: 8px; border: 2px solid #28a745;">
                    <div style="font-size: 24px;">✅</div>
                    <div style="font-weight: bold; color: #28a745;">{config['label']}</div>
                </div>
                """, unsafe_allow_html=True)

            elif i == current_idx:
                # Current stage
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background-color: #cce5ff; border-radius: 8px; border: 2px solid #0066cc;">
                    <div style="font-size: 24px;">{config['icon']}</div>
                    <div style="font-weight: bold; color: #0066cc;">{config['label']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Future stage
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 8px; border: 2px dashed #dee2e6;">
                    <div style="font-size: 24px; opacity: 0.4;">{config['icon']}</div>
                    <div style="color: #6c757d;">{config['label']}</div>
                </div>
                """, unsafe_allow_html=True)

    # Show next action prompt
    if current_idx < len(STAGE_ORDER):
        config = STAGE_CONFIG[current_stage]
        if current_idx < len(STAGE_ORDER) - 1:
            st.markdown(f"**Next:** {config['next_action']} →")


def render_stage_badge(stage: PipelineStage):
    """Render a small badge showing the current stage"""
    config = STAGE_CONFIG.get(stage, {"icon": "❓", "label": "Unknown"})
    return f"{config['icon']} {config['label']}"


def get_stage_description(stage: PipelineStage) -> str:
    """Get the description for a stage"""
    config = STAGE_CONFIG.get(stage, {"description": "Unknown stage"})
    return config['description']


def render_quick_status(current_stage: Optional[PipelineStage], num_passages: int = 0, num_labels: int = 0):
    """
    Render a quick status bar showing current state

    Args:
        current_stage: Current pipeline stage
        num_passages: Number of passages in current data
        num_labels: Number of label columns
    """
    if current_stage is None:
        st.warning("📂 No data loaded - start by loading a dataset")
        return

    config = STAGE_CONFIG[current_stage]

    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

    with col1:
        st.markdown(f"**Stage:** {config['icon']} {config['label']}")

    with col2:
        st.markdown(f"**{num_passages:,}** passages")

    with col3:
        st.markdown(f"**{num_labels}** labels")

    with col4:
        if current_stage != PipelineStage.TIERED:
            st.markdown(f"**Next:** {config['next_action']}")
        else:
            st.markdown("✅ **Ready for training**")
