"""
HRAF Golden Dataset Discovery - Main Application
Refactored for clarity, modularity, and better UX

Architecture:
- Main app handles navigation and global chat
- Pages are modular and self-contained
- Chat assistant available on all pages with full context
- Clean separation of concerns
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

    # Navigation
    st.markdown("### 📍 Navigation")

    page = st.radio(
        "Go to:",
        ["📊 Data", "🤖 Models", "🔍 Discover"],
        key="main_navigation",
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Dataset info (if loaded)
    render_sidebar_info()

    st.markdown("---")

    # Quick actions
    st.markdown("### ⚡ Quick Actions")

    if st.button("🔄 Refresh", width='stretch'):
        st.rerun()

    if st.button("🗑️ Clear Cache", width='stretch'):
        st.cache_data.clear()
        st.success("✅ Cache cleared")

    # Help
    with st.expander("❓ Help"):
        st.markdown("""
        **Data Page**: Load, clean, and prepare datasets for training
        
        **Models Page**: Train, evaluate, and compare classification models
        
        **Discover Page**: Explore data with semantic search and model inference
        
        **Chat**: Ask questions, get insights, automate tasks (available on all pages)
        """)

    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit • Voyage AI • Pinecone")

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