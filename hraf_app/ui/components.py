"""
Reusable UI components
"""

import streamlit as st
import pandas as pd
from typing import Dict, List


def render_search_interface(search_engine, title: str = "🔍 Search Passages"):
    """Render search interface"""
    st.markdown(f"### {title}")

    # Search input
    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "Search query:",
            placeholder="Enter keywords or description...",
            label_visibility="collapsed"
        )

    with col2:
        search_type = st.selectbox(
            "Method:",
            ["Hybrid", "Semantic", "Keyword"],
            label_visibility="collapsed"
        )

    # Advanced filters
    with st.expander("🔧 Advanced Filters"):
        col1, col2, col3 = st.columns(3)

        with col1:
            min_quality = st.slider("Min quality:", 0.0, 1.0, 0.0, 0.05)

        with col2:
            required_labels = st.multiselect(
                "Must have labels:",
                search_engine.label_columns
            )

        with col3:
            top_k = st.number_input("Results:", 5, 100, 20)

    # Search button
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching..."):
            filters = {}
            if required_labels:
                filters['required_labels'] = required_labels

            results = search_engine.search(
                query=query,
                search_type=search_type.lower(),
                filters=filters,
                top_k=top_k,
                min_quality=min_quality
            )

            if results:
                st.success(f"Found {len(results)} results")
                render_search_results(results, search_engine)
            else:
                st.warning("No results found")


def render_search_results(results: List[Dict], search_engine):
    """Render search results"""
    for i, result in enumerate(results):
        with st.expander(f"Result {i + 1} | Score: {result['score']:.3f}", expanded=i < 3):
            # Get full details
            details = search_engine.get_passage_details(result['idx'])

            # Show text
            st.markdown("**Text:**")
            st.write(details['text'])

            # Show labels
            st.markdown("**Labels:**")
            active_labels = [k for k, v in details['labels'].items() if v == 1]
            if active_labels:
                st.write(", ".join(active_labels))
            else:
                st.caption("No labels")

            # Show quality if available
            if 'quality' in details:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Quality", f"{details['quality']['overall']:.3f}")
                with col2:
                    st.metric("Tier", details['quality']['tier'])
                with col3:
                    st.metric("Consistency", f"{details['quality']['semantic_consistency']:.3f}")