"""
Discover Page - Semantic Search, Exploration, and Hypothesis Testing

Architecture:
- Self-contained page module
- Uses GoldenDatasetFinder from core/
- Uses ModelManager from components/
- Clean separation of concerns
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core functionality
from core.discovery_architecture import GoldenDatasetFinder


def render():
    """Main render function for Discover page"""

    st.markdown("# 🔍 Discover & Explore")
    st.caption("Semantic search, similarity analysis, and hypothesis testing")

    # Check if initialized
    if not st.session_state.get('initialized'):
        st.warning("⚠️ Load a dataset first")
        st.info("Go to **Data** page and load a dataset to begin exploring")
        return

    # Create tabs
    tabs = st.tabs([
        "🔍 Semantic Search",
        "🔗 Similar Passages",
        "🤖 Model Inference",
        "🧪 Hypothesis Testing"
    ])

    with tabs[0]:
        render_semantic_search_section()

    with tabs[1]:
        render_similar_passages_section()

    with tabs[2]:
        render_inference_section()

    with tabs[3]:
        render_hypothesis_testing_section()


# ============================================================================
# SEMANTIC SEARCH
# ============================================================================

def render_semantic_search_section():
    """Semantic search with filters and reranking"""

    st.markdown("### 🔍 Semantic Search")

    # Check if embeddings exist
    cache = st.session_state.get('cache', {})
    has_embeddings = 'passage_id_map' in cache

    if not has_embeddings:
        st.warning("⚠️ Generate embeddings first")
        st.info("Go to **Data** page → **Embed & Score** tab")
        return

    finder = st.session_state.get('finder')
    namespace = st.session_state.get('namespace', 'main')
    df = st.session_state.df
    label_columns = st.session_state.label_columns
    passage_col = st.session_state.passage_col

    st.markdown("#### Search Query")

    # Query input
    query = st.text_area(
        "Enter your search query:",
        placeholder="Example: passages about shamans healing illness through spiritual intervention",
        height=100,
        key="semantic_query"
    )

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)

        with col1:
            top_k_vector = st.slider(
                "Vector search candidates:",
                10, 200, 100,
                help="Number of passages from initial vector search"
            )

            label_filter = st.selectbox(
                "Filter by label:",
                ["None"] + label_columns,
                help="Only search passages with this label"
            )

        with col2:
            top_k_rerank = st.slider(
                "Final results:",
                5, 50, 10,
                help="Number of results after reranking"
            )

            use_rerank = st.checkbox(
                "Use reranking",
                value=True,
                help="Rerank with instruction-following model"
            )

    # Rerank instruction
    if use_rerank:
        rerank_instruction = st.text_input(
            "Rerank instruction (optional):",
            placeholder="Prioritize passages with clear, detailed descriptions",
            key="rerank_instruction"
        )
    else:
        rerank_instruction = None

    # Search button
    if st.button("🔍 Search", type="primary", disabled=not query):
        with st.spinner("Searching..."):
            try:
                # Prepare filter
                label_filter_value = None if label_filter == "None" else label_filter

                # Search
                results = finder.search_with_filters(
                    query=query,
                    namespace=namespace,
                    label_filter=label_filter_value,
                    top_k_vector=top_k_vector,
                    top_k_rerank=top_k_rerank if use_rerank else top_k_vector,
                    rerank_instruction=rerank_instruction if use_rerank else None
                )

                if not results:
                    st.warning("No results found. Try adjusting your query or filters.")
                    return

                # Store results
                st.session_state['search_results'] = results

                st.success(f"✅ Found {len(results)} results")

            except Exception as e:
                st.error(f"❌ Search error: {e}")

    # Display results
    if 'search_results' in st.session_state:
        results = st.session_state['search_results']

        st.markdown("---")
        st.markdown(f"#### 📊 Results ({len(results)})")

        for i, result in enumerate(results):
            passage_idx = result['passage_idx']

            # Get full passage
            if passage_idx in df.index:
                passage_text = df.loc[passage_idx, passage_col]

                # Get labels
                active_labels = [label for label in label_columns
                               if df.loc[passage_idx, label] == 1]

                with st.expander(f"**Result {i+1}** | Score: {result.get('combined_score', result.get('vector_score', 0)):.3f}", expanded=(i==0)):
                    # Metadata
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.caption(f"**Index:** {passage_idx}")

                    with col2:
                        if 'vector_score' in result:
                            st.caption(f"**Vector:** {result['vector_score']:.3f}")

                    with col3:
                        if 'rerank_score' in result:
                            st.caption(f"**Rerank:** {result['rerank_score']:.3f}")

                    # Labels
                    if active_labels:
                        st.markdown(f"**Labels:** {', '.join(active_labels)}")

                    # Passage text
                    st.markdown("**Passage:**")
                    st.markdown(f"> {passage_text}")

                    # Actions
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("🔗 Find Similar", key=f"similar_{passage_idx}_{i}"):
                            st.session_state['similar_query_idx'] = passage_idx
                            st.rerun()

                    with col2:
                        if st.button("🤖 Run Models", key=f"infer_{passage_idx}_{i}"):
                            st.session_state['inference_passage_idx'] = passage_idx
                            st.rerun()


# ============================================================================
# SIMILAR PASSAGES
# ============================================================================

def render_similar_passages_section():
    """Find passages similar to a given passage"""

    st.markdown("### 🔗 Similar Passages")

    # Check if embeddings exist
    cache = st.session_state.get('cache', {})
    has_embeddings = 'passage_id_map' in cache

    if not has_embeddings:
        st.warning("⚠️ Generate embeddings first")
        return

    finder = st.session_state.get('finder')
    namespace = st.session_state.get('namespace', 'main')
    df = st.session_state.df
    label_columns = st.session_state.label_columns
    passage_col = st.session_state.passage_col

    st.markdown("#### Select Query Passage")

    # Check if coming from search
    query_idx = st.session_state.get('similar_query_idx')

    if query_idx is None:
        # Manual selection
        query_idx = st.number_input(
            "Passage index:",
            min_value=0,
            max_value=len(df)-1,
            value=0,
            key="similar_passage_idx"
        )

    # Show query passage
    if query_idx in df.index:
        with st.expander("📄 Query Passage", expanded=True):
            passage_text = df.loc[query_idx, passage_col]
            active_labels = [label for label in label_columns
                           if df.loc[query_idx, label] == 1]

            st.caption(f"**Index:** {query_idx}")
            if active_labels:
                st.markdown(f"**Labels:** {', '.join(active_labels)}")
            st.markdown(f"> {passage_text}")

    # Options
    col1, col2 = st.columns(2)

    with col1:
        k = st.slider("Number of similar passages:", 5, 50, 20)

    with col2:
        label_filter = st.selectbox(
            "Filter by label:",
            ["None"] + label_columns,
            key="similar_label_filter"
        )

    # Find similar
    if st.button("🔗 Find Similar Passages", type="primary"):
        with st.spinner("Finding similar passages..."):
            try:
                label_filter_value = None if label_filter == "None" else label_filter

                similar = finder.search_similar_to_passage(
                    passage_idx=query_idx,
                    namespace=namespace,
                    k=k,
                    label_filter=label_filter_value
                )

                if not similar:
                    st.warning("No similar passages found")
                    return

                st.session_state['similar_results'] = similar
                st.success(f"✅ Found {len(similar)} similar passages")

            except Exception as e:
                st.error(f"❌ Error: {e}")

    # Display results
    if 'similar_results' in st.session_state:
        similar = st.session_state['similar_results']

        st.markdown("---")
        st.markdown(f"#### 📊 Similar Passages ({len(similar)})")

        for i, result in enumerate(similar):
            passage_idx = result['passage_idx']
            similarity = result['similarity']

            if passage_idx in df.index:
                passage_text = df.loc[passage_idx, passage_col]
                active_labels = [label for label in label_columns
                               if df.loc[passage_idx, label] == 1]

                with st.expander(f"**Passage {passage_idx}** | Similarity: {similarity:.3f}", expanded=(i<3)):
                    if active_labels:
                        st.markdown(f"**Labels:** {', '.join(active_labels)}")

                    st.markdown(f"> {passage_text}")


# ============================================================================
# MODEL INFERENCE
# ============================================================================

def render_inference_section():
    """Test model predictions on passages"""

    st.markdown("### 🤖 Model Inference")

    # Check if models loaded
    manager = st.session_state.get('model_manager')

    if manager is None or len(manager) == 0:
        st.info("💡 Load models first on the **Models** page")
        return

    df = st.session_state.df
    label_columns = st.session_state.label_columns
    passage_col = st.session_state.passage_col

    st.markdown("#### Select Passage")

    # Check if coming from search/similar
    passage_idx = st.session_state.get('inference_passage_idx')

    if passage_idx is None:
        passage_idx = st.number_input(
            "Passage index:",
            min_value=0,
            max_value=len(df)-1,
            value=0,
            key="inference_idx"
        )

    # Show passage
    if passage_idx in df.index:
        with st.expander("📄 Passage", expanded=True):
            passage_text = df.loc[passage_idx, passage_col]
            actual_labels = {label: int(df.loc[passage_idx, label])
                           for label in label_columns}
            active_labels = [label for label, val in actual_labels.items() if val == 1]

            st.caption(f"**Index:** {passage_idx}")
            if active_labels:
                st.markdown(f"**Actual Labels:** {', '.join(active_labels)}")
            st.markdown(f"> {passage_text}")

        # Model selection
        available_models = list(manager.models.keys())

        selected_models = st.multiselect(
            "Select models:",
            available_models,
            default=available_models,
            key="inference_models"
        )

        use_optimal = st.checkbox(
            "Use optimal thresholds",
            value=True,
            key="inference_optimal"
        )

        # Run inference
        if st.button("🤖 Run Inference", type="primary"):
            with st.spinner("Running models..."):
                st.markdown("---")
                st.markdown("#### 🎯 Predictions")

                for model_name in selected_models:
                    loader = manager.get_model(model_name)

                    if loader is None:
                        st.error(f"❌ Could not load {model_name}")
                        continue

                    try:
                        result = loader.predict_passage(
                            passage_text,
                            use_optimal_thresholds=use_optimal
                        )

                        predicted_labels = result['predicted_labels']
                        probabilities = result['probabilities']

                        # Display results
                        with st.expander(f"**{model_name}**", expanded=True):
                            # Predicted labels
                            if predicted_labels:
                                st.success(f"**Predicted:** {', '.join(predicted_labels)}")
                            else:
                                st.info("**Predicted:** None")

                            # Comparison with actual
                            st.markdown("**Comparison:**")

                            comparison_data = []
                            for label in label_columns:
                                actual = actual_labels.get(label, 0)
                                predicted = 1 if label in predicted_labels else 0

                                if actual == 1 and predicted == 1:
                                    status = "✅ TP"
                                elif actual == 0 and predicted == 1:
                                    status = "❌ FP"
                                elif actual == 1 and predicted == 0:
                                    status = "❌ FN"
                                else:
                                    status = "✓ TN"

                                prob = probabilities.get(label, 0)

                                if actual == 1 or predicted == 1:
                                    comparison_data.append({
                                        'Label': label,
                                        'Actual': actual,
                                        'Predicted': predicted,
                                        'Probability': f"{prob:.3f}",
                                        'Status': status
                                    })

                            st.dataframe(
                                pd.DataFrame(comparison_data),
                                hide_index=True,
                                width='stretch'
                            )

                    except Exception as e:
                        st.error(f"❌ Error with {model_name}: {e}")


# ============================================================================
# HYPOTHESIS TESTING
# ============================================================================

def render_hypothesis_testing_section():
    """Test hypotheses about label relationships"""

    st.markdown("### 🧪 Hypothesis Testing")

    df = st.session_state.df
    label_columns = st.session_state.label_columns

    st.markdown("#### Test Label Relationships")

    st.info("""
    Test hypotheses like:
    - "When EVENT_Illness occurs, is ACTION_Shaman_Medium_Healer more common?"
    - "Is CAUSE_Spirits_Gods associated with ACTION_Divination?"
    """)

    # Select labels
    col1, col2 = st.columns(2)

    with col1:
        label_a = st.selectbox(
            "When this label is present:",
            label_columns,
            key="hypothesis_label_a"
        )

    with col2:
        label_b = st.selectbox(
            "Is this label more common:",
            [l for l in label_columns if l != label_a],
            key="hypothesis_label_b"
        )

    # Test button
    if st.button("🧪 Test Hypothesis", type="primary"):
        with st.spinner("Running statistical test..."):
            try:
                # Calculate contingency table
                both = ((df[label_a] == 1) & (df[label_b] == 1)).sum()
                only_a = ((df[label_a] == 1) & (df[label_b] == 0)).sum()
                only_b = ((df[label_a] == 0) & (df[label_b] == 1)).sum()
                neither = ((df[label_a] == 0) & (df[label_b] == 0)).sum()

                contingency = np.array([[both, only_a], [only_b, neither]])

                # Chi-square test
                from scipy.stats import chi2_contingency

                chi2, p_value, dof, expected = chi2_contingency(contingency)

                # Display results
                st.markdown("---")
                st.markdown("#### 📊 Results")

                # Contingency table
                st.markdown("**Contingency Table:**")

                cont_df = pd.DataFrame({
                    '': [f'{label_b} = 1', f'{label_b} = 0'],
                    f'{label_a} = 1': [both, only_a],
                    f'{label_a} = 0': [only_b, neither]
                })

                st.dataframe(cont_df, hide_index=True, width='stretch')

                # Statistics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("χ² Statistic", f"{chi2:.3f}")

                with col2:
                    st.metric("p-value", f"{p_value:.4f}")

                with col3:
                    significant = "Yes" if p_value < 0.05 else "No"
                    st.metric("Significant (α=0.05)", significant)

                # Interpretation
                st.markdown("**Interpretation:**")

                if p_value < 0.05:
                    # Calculate rates
                    rate_with_a = both / (both + only_a) if (both + only_a) > 0 else 0
                    rate_without_a = only_b / (only_b + neither) if (only_b + neither) > 0 else 0

                    if rate_with_a > rate_without_a:
                        st.success(f"""
                        ✅ **Significant positive association** (p < 0.05)

                        When {label_a} is present:
                        - {label_b} occurs in {rate_with_a*100:.1f}% of cases

                        When {label_a} is absent:
                        - {label_b} occurs in {rate_without_a*100:.1f}% of cases

                        **Conclusion:** {label_b} is {rate_with_a/rate_without_a:.2f}x more common when {label_a} is present.
                        """)
                    else:
                        st.warning(f"""
                        ⚠️ **Significant negative association** (p < 0.05)

                        {label_b} is actually less common when {label_a} is present.
                        """)
                else:
                    st.info(f"""
                    ℹ️ **No significant association** (p ≥ 0.05)

                    There is no statistically significant relationship between {label_a} and {label_b}.
                    """)

            except Exception as e:
                st.error(f"❌ Error running test: {e}")