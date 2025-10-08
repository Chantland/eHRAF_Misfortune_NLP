"""
HRAF Quality Pipeline - Streamlined Version
Main Streamlit Application with Guided Workflow
"""

import streamlit as st
from pathlib import Path
import sys

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))

from workflows.pipeline import HRAFPipeline
from ui.workflow_steps import (
    render_step_1_load,
    render_step_2_quality,
    render_step_3_explore,
    render_step_4_train,
    render_step_5_iterate
)
from assistant.claude_agent import ProactiveAssistant

# Page config
st.set_page_config(
    page_title="HRAF Quality Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = HRAFPipeline()
    st.session_state.assistant = ProactiveAssistant()
    st.session_state.show_help = True

pipeline = st.session_state.pipeline
assistant = st.session_state.assistant

# Custom CSS for cleaner UI
st.markdown("""
<style>
    .step-container {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 10px 0;
    }
    .step-complete {
        border-left: 5px solid #28a745;
    }
    .step-current {
        border-left: 5px solid #007bff;
    }
    .step-pending {
        border-left: 5px solid #6c757d;
        opacity: 0.6;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🔬 HRAF Quality Pipeline")
st.markdown("*Data-first approach to NLP model training*")

# Sidebar: Progress tracker
with st.sidebar:
    st.markdown("### 📍 Workflow Progress")

    # Get current step
    current_step = pipeline.get_current_step()

    # Step indicators
    steps = [
        ("1. Load Data", "load", pipeline.has_data()),
        ("2. Compute Quality", "quality", pipeline.has_quality_scores()),
        ("3. Explore & Filter", "explore", pipeline.has_explored()),
        ("4. Train Model", "train", pipeline.has_model()),
        ("5. Analyze & Iterate", "iterate", pipeline.has_results())
    ]

    for name, step_id, is_complete in steps:
        if current_step == step_id:
            st.markdown(f"**→ {name}** 🔵")
        elif is_complete:
            st.markdown(f"✅ {name}")
        else:
            st.markdown(f"⚪ {name}")

    st.markdown("---")

    # Quick stats
    if pipeline.has_data():
        st.markdown("### 📊 Current Dataset")
        stats = pipeline.get_stats()
        st.metric("Passages", f"{stats['num_passages']:,}")
        st.metric("Labels", stats['num_labels'])

        if pipeline.has_quality_scores():
            st.metric("Avg Quality", f"{stats['avg_quality']:.3f}")

        if pipeline.has_model():
            st.metric("Best F1", f"{stats['best_f1']:.3f}")

    st.markdown("---")

    # Settings
    with st.expander("⚙️ Settings"):
        st.session_state.show_help = st.checkbox("Show help text", value=True)

        if st.button("🔄 Reset Pipeline"):
            st.session_state.pipeline = HRAFPipeline()
            st.rerun()

# Main content area
st.markdown("---")

# Proactive assistant suggestions
suggestion = assistant.get_suggestion(pipeline)
if suggestion:
    st.info(f"💡 **Suggestion:** {suggestion}")

# Render current step
if current_step == "load":
    render_step_1_load(pipeline, assistant)

elif current_step == "quality":
    render_step_2_quality(pipeline, assistant)

elif current_step == "explore":
    render_step_3_explore(pipeline, assistant)

elif current_step == "train":
    render_step_4_train(pipeline, assistant)

elif current_step == "iterate":
    render_step_5_iterate(pipeline, assistant)

else:
    st.error("Unknown workflow step. Please reset the pipeline.")

# Footer
st.markdown("---")
st.caption("HRAF Quality Pipeline v2.0 | Built with Streamlit + Claude")