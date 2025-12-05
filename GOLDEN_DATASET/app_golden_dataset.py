"""
HRAF Golden Dataset Discovery - Main Application

Simplified Architecture:
- Sidebar: Select dataset + model (one click each)
- Pages: Work with selected data/model
- No complex pipeline required for basic usage
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import page modules
from page_views import data_page, models_page, discover_page

# Import components
from components.sidebar_selectors import render_sidebar_selectors, get_dataset_info
from components.ui_helpers import initialize_session_state

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="HRAF Dataset Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

initialize_session_state()


# ============================================================================
# SIDEBAR - SIMPLE SELECTORS
# ============================================================================

with st.sidebar:
    st.markdown("## 🔍 HRAF Tool")

    # Dataset and Model selectors
    render_sidebar_selectors()

    st.markdown("---")

    # Navigation
    st.markdown("### 📍 Page")

    page = st.radio(
        "Go to:",
        ["📊 Data", "🤖 Models", "🔍 Discover"],
        key="main_navigation",
        label_visibility="collapsed"
    )

    # Show current data info
    info = get_dataset_info()
    if info:
        st.markdown("---")
        st.caption(f"**{info.get('name', 'Unknown')}**")
        st.caption(f"{info.get('rows', 0):,} rows • {len(info.get('label_columns', []))} labels")
        if info.get('prediction_mode'):
            st.caption("🔮 Prediction mode")


# ============================================================================
# PAGE ROUTING
# ============================================================================

# Route to appropriate page
if page == "📊 Data":
    data_page.render()

elif page == "🤖 Models":
    models_page.render()

elif page == "🔍 Discover":
    discover_page.render()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption(f"HRAF Golden Dataset Discovery • {datetime.now().strftime('%Y-%m-%d')}")