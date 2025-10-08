"""
Streamlined UI for workflow steps
Each function renders one step of the guided workflow
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def render_step_1_load(pipeline, assistant):
    """Step 1: Load and validate data"""
    st.markdown("## Step 1: Load Data")

    if st.session_state.get('show_help'):
        st.info("""
        **📖 What this does:** Load your labeled passages and validate data quality.

        **What you need:** Excel file with:
        - A column containing passage text
        - Binary label columns (0/1 values)
        """)

    # File upload
    uploaded_file = st.file_uploader(
        "Choose Excel file (.xlsx)",
        type=['xlsx'],
        help="File should contain passage text and binary labels"
    )

    if uploaded_file:
        # Save temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Configuration
        with st.expander("⚙️ Configuration (optional)"):
            col1, col2 = st.columns(2)

            with col1:
                passage_col = st.text_input(
                    "Passage column name:",
                    placeholder="Auto-detect if empty"
                )

            with col2:
                label_cols_input = st.text_input(
                    "Label columns (comma-separated):",
                    placeholder="Auto-detect if empty"
                )
                label_cols = [l.strip() for l in label_cols_input.split(',')] if label_cols_input else None

        # Load button
        if st.button("📂 Load Data", type="primary"):
            with st.spinner("Loading and validating data..."):
                try:
                    validation = pipeline.load_data(
                        tmp_path,
                        passage_col=passage_col or None,
                        label_columns=label_cols
                    )

                    # Show results
                    st.success("✅ Data loaded successfully!")

                    # Display validation results
                    st.markdown("### 📊 Validation Results")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Passages", validation['stats']['passage_lengths']['mean'])
                    with col2:
                        st.metric("Labels", len(pipeline.state.label_columns))
                    with col3:
                        st.metric("Avg Length", f"{validation['stats']['passage_lengths']['mean']:.0f}")

                    # Warnings
                    if validation['warnings']:
                        st.markdown("### ⚠️ Warnings")
                        for warning in validation['warnings']:
                            st.warning(warning)

                    # Label distribution
                    st.markdown("### 🏷️ Label Distribution")
                    label_dist = validation['stats']['label_distribution']
                    dist_df = pd.DataFrame([
                        {'Label': label, 'Count': info['count'], 'Percentage': f"{info['percentage']:.1f}%"}
                        for label, info in label_dist.items()
                    ])
                    st.dataframe(dist_df, hide_index=True, use_container_width=True)

                    # Next step prompt
                    st.markdown("---")
                    st.success("✅ Ready for next step: Compute Quality Scores")

                    if st.button("➡️ Continue to Quality Scoring", type="primary"):
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error loading data: {e}")
                    with st.expander("Error details"):
                        import traceback
                        st.code(traceback.format_exc())


def render_step_2_quality(pipeline, assistant):
    """Step 2: Compute quality scores"""
    st.markdown("## Step 2: Compute Quality Scores")

    if st.session_state.get('show_help'):
        st.info("""
        **📖 What this does:** Analyze each passage to determine how "learnable" it is.

        **Quality metrics:**
        - **Semantic consistency**: How similar to passages with same labels
        - **Label confidence**: How clearly the passage demonstrates labeled concepts

        This takes 2-5 minutes for 10,000 passages.
        """)

    # Configuration
    with st.expander("⚙️ Configuration"):
        use_embeddings = st.checkbox(
            "Use semantic embeddings (recommended)",
            value=True,
            help="Requires VoyageAI API key. More accurate but slower."
        )

        k_similar = st.slider(
            "Similarity neighborhood size:",
            5, 30, 15,
            help="Number of similar passages to compare"
        )

    # Compute button
    if st.button("🔬 Compute Quality Scores", type="primary"):
        with st.spinner("Computing quality scores... This may take a few minutes."):
            try:
                results = pipeline.compute_quality(
                    use_embeddings=use_embeddings,
                    k_similar=k_similar
                )

                # Show results
                st.success(f"✅ Computed scores for {results['num_scored']} passages!")

                # Distribution
                st.markdown("### 📊 Quality Distribution")
                dist = results['distribution']

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{dist['mean']:.3f}")
                with col2:
                    st.metric("Median", f"{dist['median']:.3f}")
                with col3:
                    st.metric("Elite %", f"{dist['tier_percentages']['elite']:.1f}%")
                with col4:
                    st.metric("Good %", f"{dist['tier_percentages']['good']:.1f}%")

                # Histogram
                fig, ax = plt.subplots(figsize=(10, 4))
                qualities = [
                    q.overall_quality
                    for q in pipeline.state.quality_scores.values()
                ]
                ax.hist(qualities, bins=50, edgecolor='black', alpha=0.7)
                ax.axvline(dist['median'], color='red', linestyle='--',
                           label=f"Median: {dist['median']:.3f}")
                ax.set_xlabel('Quality Score')
                ax.set_ylabel('Count')
                ax.set_title('Quality Score Distribution')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()

                # Recommendations
                st.markdown("### 💡 Recommendations")
                for rec in results['recommendations']:
                    st.info(rec)

                # Next step
                st.markdown("---")
                st.success("✅ Ready for next step: Explore & Filter Data")

                if st.button("➡️ Continue to Data Selection", type="primary"):
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error computing quality: {e}")
                with st.expander("Error details"):
                    import traceback
                    st.code(traceback.format_exc())


def render_step_3_explore(pipeline, assistant):
    """Step 3: Explore data and select training set"""
    st.markdown("## Step 3: Explore & Filter Data")

    if st.session_state.get('show_help'):
        st.info("""
        **📖 What this does:** Select which passages to use for training.

        **Strategy:** Higher quality = better learning, but you also need enough data.

        Use the filters below to balance quality and quantity.
        """)

    # Quality threshold selector
    st.markdown("### 🎯 Selection Criteria")

    col1, col2 = st.columns(2)

    with col1:
        # Show distribution
        qualities = [q.overall_quality for q in pipeline.state.quality_scores.values()]

        min_quality = st.slider(
            "Minimum quality threshold:",
            0.0, 1.0, 0.60, 0.05,
            help="Only use passages above this quality"
        )

        # Preview selection
        above_threshold = sum(1 for q in qualities if q >= min_quality)
        below_threshold = len(qualities) - above_threshold

        st.metric("Will use", f"{above_threshold:,} passages")
        st.caption(f"Excluding {below_threshold:,} low-quality passages")

    with col2:
        # Tier-based strategy (simplified)
        strategy = st.radio(
            "Selection strategy:",
            ["Conservative (Elite only)",
             "Balanced (Elite + Good)",
             "Aggressive (Include Fair)"],
            index=1,
            help="How aggressively to include lower-quality data"
        )

        # Adjust threshold based on strategy
        if "Conservative" in strategy:
            recommended_threshold = 0.75
        elif "Balanced" in strategy:
            recommended_threshold = 0.60
        else:
            recommended_threshold = 0.45

        if min_quality != recommended_threshold:
            st.info(f"💡 Recommended threshold: {recommended_threshold}")

    # Optional: Label targeting
    with st.expander("🏷️ Advanced: Label Targeting"):
        st.markdown("Ensure minimum counts for specific labels")

        use_targeting = st.checkbox("Enable label targeting")

        if use_targeting:
            label_targets = {}

            # Show only rare labels
            rare_labels = [
                label for label in pipeline.state.label_columns
                if (pipeline.state.df[label] == 1).sum() < len(pipeline.state.df) * 0.1
            ]

            for label in rare_labels[:5]:  # Limit to 5
                target = st.number_input(
                    f"{label} minimum:",
                    0, 500, 100,
                    key=f"target_{label}"
                )
                if target > 0:
                    label_targets[label] = target
        else:
            label_targets = None

    # Select button
    if st.button("✅ Select Training Data", type="primary"):
        with st.spinner("Selecting passages..."):
            try:
                results = pipeline.select_training_data(
                    min_quality=min_quality,
                    tier_strategy=strategy,
                    label_targets=label_targets
                )

                st.success(f"✅ Selected {results['num_selected']} passages!")

                # Show selection analysis
                st.markdown("### 📊 Selection Analysis")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Selected", f"{results['num_selected']:,}")
                with col2:
                    st.metric("% of Total", f"{results['percentage']:.1f}%")
                with col3:
                    st.metric("Avg Quality", f"{results['avg_quality']:.3f}")

                # Label coverage
                st.markdown("### 🏷️ Label Coverage")
                coverage_df = pd.DataFrame([
                    {
                        'Label': label,
                        'Count': info['count'],
                        'Coverage': f"{info['percentage']:.1f}%"
                    }
                    for label, info in results['label_coverage'].items()
                ])
                st.dataframe(coverage_df, hide_index=True, use_container_width=True)

                # Check for issues
                low_coverage = [
                    label for label, info in results['label_coverage'].items()
                    if info['count'] < 50
                ]

                if low_coverage:
                    st.warning(f"⚠️ Low coverage for: {', '.join(low_coverage)}")
                    st.info("💡 Consider lowering quality threshold or enabling label targeting")

                # Next step
                st.markdown("---")
                st.success("✅ Ready for next step: Train Model")

                if st.button("➡️ Continue to Training", type="primary"):
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error selecting data: {e}")


def render_step_4_train(pipeline, assistant):
    """Step 4: Train model"""
    st.markdown("## Step 4: Train Model")

    if st.session_state.get('show_help'):
        st.info("""
        **📖 What this does:** Train a hierarchical model on your selected data.

        **Model type:** Hierarchical multi-label classifier
        - Predicts main categories (EVENT, CAUSE, ACTION)
        - Then predicts specific subcategories based on main predictions

        Training takes 5-30 minutes depending on data size and hardware.
        """)

    # Simple configuration
    st.markdown("### ⚙️ Training Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        num_epochs = st.number_input("Epochs:", 1, 20, 10)
        batch_size = st.selectbox("Batch size:", [8, 16, 32], index=1)

    with col2:
        learning_rate = st.select_slider(
            "Learning rate:",
            options=[1e-5, 2e-5, 3e-5, 5e-5],
            value=2e-5,
            format_func=lambda x: f"{x:.0e}"
        )

    with col3:
        use_focal_loss = st.checkbox("Use focal loss", value=True,
                                     help="Helps with class imbalance")

    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        col1, col2 = st.columns(2)

        with col1:
            use_gating = st.checkbox("Use gated hierarchy", value=True)
            dropout = st.slider("Dropout:", 0.0, 0.5, 0.1, 0.05)

        with col2:
            teacher_forcing = st.slider("Teacher forcing:", 0.0, 1.0, 0.7, 0.1)

    # Experiment name
    experiment_name = st.text_input(
        "Experiment name:",
        value=f"training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    )

    # Train button
    if st.button("🎓 Start Training", type="primary"):
        # Prepare config
        config = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'use_focal_loss': use_focal_loss,
            'use_gating': use_gating,
            'dropout': dropout,
            'teacher_forcing_ratio': teacher_forcing
        }

        # Create progress placeholder
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        try:
            with st.spinner("Training model... This will take several minutes."):
                # This would need actual training implementation
                # For now, show structure
                status_placeholder.info("🎓 Epoch 1/10...")

                results = pipeline.train_model(
                    config=config,
                    experiment_name=experiment_name
                )

                # Show results
                st.success("✅ Training complete!")

                st.markdown("### 📊 Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test F1 (Micro)", f"{results['test_metrics']['f1_micro']:.3f}")
                with col2:
                    st.metric("Test F1 (Macro)", f"{results['test_metrics']['f1_macro']:.3f}")
                with col3:
                    best_epoch = max(results['history'], key=lambda x: x.get('eval_f1_micro', 0))
                    st.metric("Best Epoch", best_epoch['epoch'])

                # Next step
                st.markdown("---")
                st.success("✅ Ready for next step: Analyze & Iterate")

                if st.button("➡️ Continue to Iteration", type="primary"):
                    st.rerun()

        except Exception as e:
            st.error(f"❌ Training failed: {e}")


def render_step_5_iterate(pipeline, assistant):
    """Step 5: Analyze failures and iterate"""
    st.markdown("## Step 5: Analyze & Iterate")

    if st.session_state.get('show_help'):
        st.info("""
        **📖 What this does:** Find where the model struggles and how to fix it.

        **The key insight:** Model errors often point to data quality issues.

        This step suggests specific data improvements to boost performance.
        """)

    # Run analysis if not done
    if not pipeline.state.failure_analysis:
        if st.button("🔍 Analyze Model Failures", type="primary"):
            with st.spinner("Analyzing failures..."):
                results = pipeline.analyze_failures()
                st.rerun()
    else:
        # Show analysis results
        analysis = pipeline.state.failure_analysis
        suggestions = pipeline.state.improvement_suggestions

        st.markdown("### 📉 Failure Analysis")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Errors", analysis['summary']['total_errors'])
        with col2:
            st.metric("Over-predictions", analysis['summary']['over_prediction_rate'])
        with col3:
            st.metric("Under-predictions", analysis['summary']['under_prediction_rate'])

        # Worst labels
        st.markdown("### 🎯 Most Problematic Labels")
        worst = analysis['summary']['worst_labels'][:5]
        for label, count in worst:
            st.markdown(f"- **{label}**: {count} errors")

        # Suggestions
        st.markdown("### 💡 Improvement Suggestions")

        for suggestion in suggestions:
            priority_color = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }[suggestion['priority']]

            with st.expander(f"{priority_color} {suggestion['title']}", expanded=suggestion['priority'] == 'high'):
                st.markdown(suggestion['description'])
                st.markdown(f"**Action:** {suggestion['action']}")
                st.markdown(f"**Expected impact:** {suggestion['expected_impact']}")

                if st.button(f"✅ Apply This Improvement", key=f"apply_{suggestion['id']}"):
                    with st.spinner("Applying improvement..."):
                        result = pipeline.apply_improvement(suggestion['id'])
                        st.success("✅ Improvement applied!")
                        st.info(f"New training set: {result['new_selection_size']} passages")
                        st.info("💡 Return to Step 4 to retrain with improved data")

                        if st.button("➡️ Retrain Model", type="primary"):
                            # Go back to training step
                            st.rerun()