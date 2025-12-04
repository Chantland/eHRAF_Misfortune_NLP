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
    """Simple semantic search with diagnostics"""

    st.markdown("### 🔍 Semantic Search")

    # Check prerequisites
    cache = st.session_state.get('cache', {})
    has_embeddings = 'stable_id_to_pinecone' in cache

    if not has_embeddings:
        st.warning("⚠️ Generate embeddings first (Data page → Embed & Score)")
        return

    finder = st.session_state.get('finder')
    namespace = st.session_state.get('namespace', 'main')
    df = st.session_state.df
    label_columns = st.session_state.label_columns
    passage_col = st.session_state.passage_col

    # ========================================================================
    # DIAGNOSTIC SECTION - ADD THIS
    # ========================================================================

    with st.expander("🔧 Diagnostics", expanded=False):
        st.markdown("**Check if search is working correctly**")

        # In diagnostic test section:
        if st.button("Run Diagnostic Test"):
            with st.spinner("Testing search pipeline..."):
                # Pick a random passage
                test_idx = np.random.choice(df.index)

                # REQUIRE stable_id
                if 'passage_id' not in df.columns:
                    st.error("❌ DataFrame missing 'passage_id' column")
                    st.info("Re-load data with updated code to add stable IDs")
                    return

                stable_id = df.loc[test_idx, 'passage_id']
                if pd.isna(stable_id):
                    st.error(f"❌ Passage {test_idx} missing stable_id")
                    return

                passage_id = f"passage_{stable_id}"
                test_passage = str(df.loc[test_idx, passage_col])

                st.markdown(f"**Test passage {test_idx}:**")
                st.caption(f"Stable ID: {stable_id}")
                st.caption(test_passage[:200] + "...")

                try:
                    # ✅ FIX: Use stable ID for lookup
                    if 'passage_id' in df.columns:
                        stable_id = df.loc[test_idx, 'passage_id']
                        passage_id = f"passage_{stable_id}"
                        st.info(f"🔑 Using stable ID: {stable_id}")
                    else:
                        # Fallback to old method
                        passage_id = f"passage_{test_idx}"
                        st.warning("⚠️ No stable IDs - using index-based ID")

                    fetch_result = finder.index.fetch(ids=[passage_id], namespace=namespace)

                    if hasattr(fetch_result, 'vectors'):
                        vectors = fetch_result.vectors
                    else:
                        vectors = fetch_result.get('vectors', {})

                    if passage_id not in vectors:
                        st.error(f"❌ Passage {test_idx} NOT FOUND in Pinecone namespace '{namespace}'")
                        st.info("**Fix:** Re-run embedding generation on Data page")
                        return

                    st.success(f"✅ Found in Pinecone")

                    # Check 2: Does metadata match?
                    vector_data = vectors[passage_id]
                    if hasattr(vector_data, 'metadata'):
                        metadata = vector_data.metadata
                    else:
                        metadata = vector_data.get('metadata', {})

                    stored_text = metadata.get('text_preview', '')
                    stored_idx = metadata.get('passage_idx')

                    st.write(f"**Metadata passage_idx:** {stored_idx}")
                    st.write(f"**Stored text preview:** {stored_text[:100]}...")

                    if stored_text[:100] not in test_passage:
                        st.error("❌ MISMATCH: Stored text doesn't match actual passage!")
                        st.info("**Fix:** Re-run embedding generation")
                        return

                    st.success("✅ Metadata matches passage")

                    # Check 3: Self-similarity test
                    if hasattr(vector_data, 'values'):
                        embedding = vector_data.values
                    else:
                        embedding = vector_data['values']

                    search_results = finder.index.query(
                        vector=embedding,
                        top_k=5,
                        namespace=namespace,
                        include_metadata=True
                    )

                    if hasattr(search_results, 'matches'):
                        matches = search_results.matches
                    else:
                        matches = search_results.get('matches', [])

                    st.markdown("**Top 5 similar passages:**")

                    for i, match in enumerate(matches):
                        if hasattr(match, 'score'):
                            score = match.score
                            match_metadata = match.metadata
                        else:
                            score = match['score']
                            match_metadata = match['metadata']

                        match_idx = match_metadata['passage_idx']
                        match_text = match_metadata.get('text_preview', '')[:100]

                        st.write(f"{i + 1}. Index {match_idx} | Score: {score:.4f}")
                        st.caption(f"   {match_text}...")

                    # The first result should be itself with score ~1.0
                    top_match = matches[0]
                    if hasattr(top_match, 'score'):
                        top_score = top_match.score
                        top_metadata = top_match.metadata
                    else:
                        top_score = top_match['score']
                        top_metadata = top_match['metadata']

                    top_idx = top_metadata['passage_idx']

                    if top_idx == test_idx and top_score > 0.99:
                        st.success(f"✅ Self-similarity test PASSED (score: {top_score:.4f})")
                        st.success("🎉 **Search pipeline is working correctly!**")
                    else:
                        st.error(f"❌ Self-similarity test FAILED")
                        st.error(f"Expected: idx={test_idx}, score≈1.0")
                        st.error(f"Got: idx={top_idx}, score={top_score:.4f}")
                        st.info("**Fix:** Embeddings may be corrupted. Re-run embedding generation.")

                except Exception as e:
                    st.error(f"❌ Diagnostic failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        st.markdown("---")
        st.markdown("**Index Information:**")

        try:
            stats = finder.index.describe_index_stats()

            if hasattr(stats, 'total_vector_count'):
                total = stats.total_vector_count
                namespaces = stats.namespaces
            else:
                total = stats.get('total_vector_count', 0)
                namespaces = stats.get('namespaces', {})

            st.write(f"**Total vectors:** {total}")
            st.write(f"**Namespaces:** {list(namespaces.keys())}")
            st.write(f"**Current namespace:** '{namespace}'")

            if namespace in namespaces:
                ns_count = namespaces[namespace].get('vector_count', 0) if hasattr(namespaces[namespace],
                                                                                   'get') else getattr(
                    namespaces[namespace], 'vector_count', 0)
                st.write(f"**Vectors in '{namespace}':** {ns_count}")
            else:
                st.error(f"❌ Namespace '{namespace}' NOT FOUND in index!")
                st.info("**Available namespaces:** " + ", ".join(list(namespaces.keys())))
                st.info("**Fix:** Check namespace on Data page or re-run embeddings")

        except Exception as e:
            st.error(f"Could not get index stats: {e}")

    # ========================================================================
    # SIMPLE SEARCH INTERFACE
    # ========================================================================

    query = st.text_input(
        "What are you looking for?",
        placeholder="Example: pottery, shamans healing illness, spirit possession",
        key="search_query"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        num_results = st.slider("Number of results:", 5, 50, 10, 5)

    with col2:
        use_rerank = st.checkbox("Use AI reranking", value=True, help="Better results, slightly slower")

    # Advanced options (collapsed by default)
    with st.expander("⚙️ Advanced Options", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Filters:**")

            label_filter = st.selectbox(
                "Must have label:",
                ["None"] + label_columns,
                key="search_label_filter"
            )

            min_similarity = st.slider(
                "Min similarity:",
                0.0, 1.0, 0.0, 0.05,
                help="Filter out low-similarity results"
            )

        with col2:
            st.markdown("**Scoring:**")

            if use_rerank:
                vector_weight = st.slider(
                    "Vector weight:",
                    0.0, 1.0, 0.3, 0.1,
                    help="Balance between vector similarity and AI reranking"
                )
            else:
                vector_weight = 1.0

    # ========================================================================
    # SEARCH EXECUTION
    # ========================================================================

    col1, col2 = st.columns([3, 1])

    with col1:
        search_btn = st.button("🔍 Search", type="primary", disabled=not query, width='stretch')

    with col2:
        if st.button("Clear", width='stretch'):
            if 'search_results' in st.session_state:
                del st.session_state['search_results']
            st.rerun()

    if search_btn and query:
        with st.spinner("Searching..."):
            try:
                # Step 1: Get query embedding
                query_embedding = finder.voyage.embed(
                    texts=[query],
                    model="voyage-3-large",
                    input_type="query"
                ).embeddings[0]

                # Step 2: Build Pinecone filter
                pinecone_filter = None
                if label_filter and label_filter != "None":
                    pinecone_filter = {f"label_{label_filter}": {"$eq": 1}}

                # Step 3: Vector search (FIXED)
                search_results = finder.index.query(
                    vector=query_embedding,
                    top_k=num_results * 3 if use_rerank else num_results,
                    namespace=namespace,
                    include_metadata=True,
                    filter=pinecone_filter
                )

                # Extract matches
                if hasattr(search_results, 'matches'):
                    matches = search_results.matches
                else:
                    matches = search_results.get('matches', [])

                # Convert to standard format - CRITICAL FIX
                candidates = []
                for match in matches:
                    if hasattr(match, 'score'):
                        score = match.score
                        metadata = match.metadata
                        match_id = match.id
                    else:
                        score = match['score']
                        metadata = match['metadata']
                        match_id = match['id']

                    if score >= min_similarity:
                        # CRITICAL: Map back using stable_id, not stored passage_idx
                        stable_id = metadata.get('stable_id')

                        if not stable_id:
                            continue  # Skip if no stable_id

                        # Find current DataFrame row with this stable_id
                        matching_rows = df[df['passage_id'] == stable_id]

                        if matching_rows.empty:
                            continue  # Passage not in current dataset

                        passage_idx = matching_rows.index[0]

                        candidates.append({
                            'passage_idx': passage_idx,  # Current DataFrame index
                            'vector_score': score,
                            'text_preview': metadata.get('text_preview', ''),
                            'metadata': metadata,
                            'stable_id': stable_id
                        })

                if not candidates:
                    st.warning("No results found. Try:")
                    st.info("• Different search terms\n• Lower minimum similarity\n• Remove label filter")
                    return

                # Step 4: Optional reranking
                if use_rerank and len(candidates) > num_results:
                    st.caption(f"🔄 Reranking {len(candidates)} candidates...")

                    texts = [c['text_preview'] for c in candidates]

                    rerank_result = finder.voyage.rerank(
                        query=query,
                        documents=texts,
                        model="rerank-2.5",
                        top_k=num_results
                    )

                    # Combine scores
                    reranked = []
                    for result in rerank_result.results:
                        candidate = candidates[result.index].copy()
                        candidate['rerank_score'] = result.relevance_score
                        candidate['combined_score'] = (
                                vector_weight * candidate['vector_score'] +
                                (1 - vector_weight) * result.relevance_score
                        )
                        reranked.append(candidate)

                    results = sorted(reranked, key=lambda x: x['combined_score'], reverse=True)
                else:
                    # No reranking - use vector scores
                    for c in candidates:
                        c['combined_score'] = c['vector_score']
                    results = sorted(candidates, key=lambda x: x['combined_score'], reverse=True)[:num_results]

                # Store results
                st.session_state['search_results'] = results
                st.success(f"✅ Found {len(results)} results")

            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")
                import traceback
                with st.expander("Debug info"):
                    st.code(traceback.format_exc())
                    st.write("**Namespace:**", namespace)
                    st.write("**Index stats:**")
                    try:
                        stats = finder.index.describe_index_stats()
                        st.json(stats if isinstance(stats, dict) else stats.__dict__)
                    except Exception as e2:
                        st.write(f"Could not get stats: {e2}")

    # ========================================================================
    # RESULTS DISPLAY
    # ========================================================================

    if 'search_results' in st.session_state:
        results = st.session_state['search_results']

        st.markdown("---")

        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Results", len(results))
        with col2:
            avg_score = np.mean([r['combined_score'] for r in results])
            st.metric("Avg Score", f"{avg_score:.3f}")
        with col3:
            if results[0].get('rerank_score'):
                st.metric("Method", "Vector + Rerank")
            else:
                st.metric("Method", "Vector Only")

        # Export
        with st.expander("💾 Export"):
            export_data = []
            for i, result in enumerate(results):
                idx = result['passage_idx']
                if idx in df.index:
                    export_data.append({
                        'rank': i + 1,
                        'passage_idx': idx,
                        'score': result['combined_score'],
                        'passage': df.loc[idx, passage_col]
                    })

            if export_data:
                csv = pd.DataFrame(export_data).to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"search_{len(results)}.csv",
                    "text/csv"
                )

        st.markdown("---")

        # Display results
        for i, result in enumerate(results):
            idx = result['passage_idx']

            if idx not in df.index:
                continue

            passage = df.loc[idx, passage_col]
            labels = [lbl for lbl in label_columns if df.loc[idx, lbl] == 1]

            # Card title
            score = result['combined_score']
            title = f"#{i + 1} | Score: {score:.3f}"

            if result.get('rerank_score'):
                title += f" (V:{result['vector_score']:.2f} R:{result['rerank_score']:.2f})"

            with st.expander(title, expanded=(i < 3)):
                # Metadata
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.caption(f"**Index:** {idx}")
                with col2:
                    if labels:
                        st.caption(f"**Labels:** {', '.join(labels)}")

                # Passage text
                st.markdown(passage)

                # Actions
                st.markdown("---")
                act_col1, act_col2 = st.columns(2)

                with act_col1:
                    if st.button("🔗 Find Similar", key=f"sim_{idx}_{i}"):
                        st.session_state['similar_query_idx'] = idx
                        st.info("💡 Switch to 'Similar Passages' tab")

                with act_col2:
                    if st.button("🤖 Run Models", key=f"inf_{idx}_{i}"):
                        st.session_state['inference_passage_idx'] = idx
                        st.info("💡 Switch to 'Model Inference' tab")

# ============================================================================
# SIMILAR PASSAGES
# ============================================================================

def render_similar_passages_section():
    """Find passages similar to a given passage"""

    st.markdown("### 🔗 Similar Passages")

    # Check if embeddings exist
    cache = st.session_state.get('cache', {})
    has_embeddings = 'stable_id_to_pinecone' in cache

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

                # ✅ FIX: Pass dataframe for stable ID lookup
                similar = finder.search_similar_to_passage(
                    passage_idx=query_idx,
                    namespace=namespace,
                    k=k,
                    label_filter=label_filter_value,
                    df=df  # Add this!
                )

                if not similar:
                    st.warning("No similar passages found")
                    return

                st.session_state['similar_results'] = similar
                st.success(f"✅ Found {len(similar)} similar passages")

            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                with st.expander("Debug info"):
                    st.code(traceback.format_exc())

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
            actual_labels = {}
            for label in label_columns:
                val = df.loc[passage_idx, label]
                actual_labels[label] = int(val) if pd.notna(val) else 0

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