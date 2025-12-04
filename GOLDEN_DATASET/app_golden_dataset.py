"""
to run: streamlit run app_golden_dataset.py
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
from components.chat_assistant import GlobalChatAssistant
from components.ui_helpers import render_sidebar_info, initialize_session_state

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
# SIDEBAR - GLOBAL CONTROLS
# ============================================================================

with st.sidebar:
    st.markdown("## 🔍 HRAF Dataset Tool")
    st.caption("Explore • Prepare • Train • Discover")

    st.markdown("---")

    # Navigation - ADD ANALYSIS
    st.markdown("### 📍 Navigation")

    page = st.radio(
        "Go to:",
        ["📊 Data", "🔬 Analysis", "🤖 Models", "🔍 Discover"],  # ADDED ANALYSIS
        key="main_navigation",
        label_visibility="collapsed"
    )

    st.markdown("---")
    render_sidebar_info()

    # ✅ ADD: Collapsible AI Assistant at bottom
    st.markdown("---")

    with st.expander("💬 AI Assistant", expanded=False):
        if 'global_chat' not in st.session_state:
            st.session_state.global_chat = GlobalChatAssistant()

        st.session_state.global_chat.render(
            current_page=page,
            session_state=st.session_state
        )

# ============================================================================
# GLOBAL CHAT ASSISTANT
# ============================================================================

# Initialize chat assistant (singleton pattern)
if 'global_chat' not in st.session_state:
    st.session_state.global_chat = GlobalChatAssistant()

# Chat toggle in top right
chat_col1, chat_col2 = st.columns([6, 1])

with chat_col2:
    show_chat = st.toggle(
        "💬",
        value=st.session_state.get('show_global_chat', False),
        help="Toggle AI Assistant",
        key="chat_toggle"
    )
    st.session_state.show_global_chat = show_chat

# Render chat if enabled
if show_chat:
    with st.container():
        st.markdown("### 💬 AI Assistant")
        st.markdown("---")
        st.session_state.global_chat.render(
            current_page=page,
            session_state=st.session_state
        )
        st.markdown("---")

# ============================================================================
# PAGE ROUTING
# ============================================================================

# Update routing section - ADD ANALYSIS CASE
if page == "📊 Data":
    data_page.render()

elif page == "🔬 Analysis":  # NEW
    from page_views import analysis_page
    analysis_page.render()

elif page == "🤖 Models":
    models_page.render()

elif page == "🔍 Discover":
    discover_page.render()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption(f"HRAF Golden Dataset Discovery • {datetime.now().strftime('%Y-%m-%d')}")