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
    """Step 1: Load and validate data - Simple visual selection"""
    st.markdown("## Step 1: Load Data")

    if st.session_state.get('show_help'):
        st.info("""
        **📖 What this does:** Load your Excel file and tell us which columns to use.

        Simply preview your file and click to select:
        - Which row has the headers
        - Which column contains passages
        - Which columns are labels
        """)

    # File upload
    uploaded_file = st.file_uploader(
        "Choose Excel file (.xlsx)",
        type=['xlsx'],
        help="Upload your labeled passages file"
    )

    if uploaded_file:
        # Save temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Load file for preview
        try:
            # Read without assuming headers first
            preview_df = pd.read_excel(tmp_path, header=None, nrows=10)

            st.markdown("---")
            st.markdown("### 📋 Step 1: Preview Your File")
            st.dataframe(preview_df, use_container_width=True)

            st.markdown("---")
            st.markdown("### 📍 Step 2: Select Header Row")

            header_row = st.number_input(
                "Which row contains the column headers? (0 = first row)",
                min_value=0,
                max_value=len(preview_df) - 1,
                value=0,
                help="Usually 0 (first row)"
            )

            # Now read with correct header
            df = pd.read_excel(tmp_path, header=header_row)

            st.success(f"✅ Using row {header_row} as headers")
            st.write("**Columns found:**", ", ".join([f"`{col}`" for col in df.columns]))

            st.markdown("---")
            st.markdown("### 📝 Step 3: Select Passage Column")

            passage_col = st.selectbox(
                "Which column contains the passage text?",
                options=df.columns,
                help="Select the column with your text passages"
            )

            # Show preview of selected column
            if passage_col:
                st.write("**Preview:**")
                sample_text = df[passage_col].dropna().iloc[0] if len(df[passage_col].dropna()) > 0 else "No text found"
                st.text_area("First passage:", str(sample_text)[:500], height=100, disabled=True)

            st.markdown("---")
            st.markdown("### 🏷️ Step 4: Select Label Columns")

            st.write("**Check all columns that are labels (should contain 0/1 values):**")

            # Show columns in a grid with checkboxes
            available_cols = [col for col in df.columns if col != passage_col]

            # Initialize session state for selections
            if 'selected_labels' not in st.session_state:
                st.session_state.selected_labels = []

            # Create columns for layout
            num_display_cols = 3
            cols = st.columns(num_display_cols)

            selected_labels = []

            for i, col in enumerate(available_cols):
                with cols[i % num_display_cols]:
                    # Show column info
                    unique_vals = df[col].dropna().unique()

                    # Check if it looks like a label
                    looks_like_label = False
                    label_info = ""

                    try:
                        if df[col].dtype in ['int64', 'float64', 'Int64', 'Float64']:
                            unique_set = set(float(v) for v in unique_vals if not pd.isna(v))
                            if unique_set.issubset({0.0, 1.0}):
                                looks_like_label = True
                                positive = int((df[col] == 1).sum())
                                label_info = f"({positive} positive)"
                    except:
                        pass

                    # Checkbox with suggestion
                    default_value = looks_like_label
                    is_selected = st.checkbox(
                        f"{col} {label_info}",
                        value=default_value,
                        key=f"label_{col}",
                        help=f"Values: {list(unique_vals)[:5]}"
                    )

                    if is_selected:
                        selected_labels.append(col)

            st.markdown("---")

            # Show summary
            if selected_labels:
                st.markdown("### 📊 Summary")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Passages", len(df))

                with col2:
                    st.metric("Passage Column", passage_col)

                with col3:
                    st.metric("Label Columns", len(selected_labels))

                # Show selected labels
                st.write("**Selected labels:**", ", ".join([f"`{label}`" for label in selected_labels]))

                # Load button
                if st.button("📂 Load Data", type="primary", use_container_width=True):
                    with st.spinner("Loading and validating data..."):
                        try:
                            # Load with specified configuration
                            validation = pipeline.load_data(
                                tmp_path,
                                passage_col=passage_col,
                                label_columns=selected_labels,
                                header_row=header_row
                            )

                            st.success("✅ Data loaded successfully!")

                            # Show validation results
                            st.markdown("### 📊 Validation Results")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Passages", len(pipeline.state.df))
                            with col2:
                                st.metric("Valid Passages", len(pipeline.state.df[pipeline.state.passage_col].dropna()))
                            with col3:
                                avg_length = validation['stats']['passage_lengths']['mean']
                                st.metric("Avg Length", f"{avg_length:.0f} chars")

                            # Warnings
                            if validation['warnings']:
                                st.markdown("### ⚠️ Warnings")
                                for warning in validation['warnings']:
                                    st.warning(warning)

                            # Label distribution
                            st.markdown("### 🏷️ Label Distribution")
                            label_dist = validation['stats']['label_distribution']
                            dist_df = pd.DataFrame([
                                {
                                    'Label': label,
                                    'Count': info['count'],
                                    'Percentage': f"{info['percentage']:.1f}%"
                                }
                                for label, info in label_dist.items()
                            ])
                            st.dataframe(dist_df, hide_index=True, use_container_width=True)

                            # Next step
                            st.markdown("---")
                            st.success("✅ Ready for next step: Compute Quality Scores")

                            if st.button("➡️ Continue to Quality Scoring", type="primary"):
                                st.rerun()

                        except Exception as e:
                            st.error(f"❌ Error loading data: {str(e)}")

                            with st.expander("🐛 Error Details"):
                                import traceback
                                st.code(traceback.format_exc())

            else:
                st.warning("⚠️ Please select at least one label column")

        except Exception as e:
            st.error(f"❌ Could not preview file: {str(e)}")
            with st.expander("🐛 Error Details"):
                import traceback
                st.code(traceback.format_exc())


def render_step_2_quality(pipeline, assistant):
    """Step 2: Compute quality scores with detailed progress"""
    st.markdown("## Step 2: Compute Quality Scores")

    if st.session_state.get('show_help'):
        st.info("""
        **📖 What this does:** Analyze each passage to determine how "learnable" it is.

        **Quality metrics:**
        - **Semantic consistency**: How similar to passages with same labels
        - **Label confidence**: How clearly the passage demonstrates labeled concepts

        **Time estimate:** ~2-5 minutes for 10,000 passages
        """)

    # Show data summary
    if pipeline.has_data():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Passages", len(pipeline.state.df))
        with col2:
            st.metric("Labels", len(pipeline.state.label_columns))
        with col3:
            # Estimate time
            num_passages = len(pipeline.state.df)
            est_minutes = (num_passages / 10000) * 3  # Rough estimate
            st.metric("Est. Time", f"~{est_minutes:.1f} min")

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

    st.markdown("---")

    # Compute button
    if st.button("🔬 Compute Quality Scores", type="primary", use_container_width=True):
        # Create placeholders for real-time updates
        status_container = st.container()
        progress_container = st.container()
        details_container = st.container()

        with status_container:
            status = st.empty()
            progress_bar = st.empty()

        with progress_container:
            metrics_cols = st.columns(4)
            metric_placeholders = [col.empty() for col in metrics_cols]

        with details_container:
            details_expander = st.expander("📊 Detailed Progress", expanded=True)
            with details_expander:
                log_placeholder = st.empty()

        log_messages = []

        def log(message):
            """Add a log message"""
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log_messages.append(f"[{timestamp}] {message}")
            # Keep last 20 messages
            display_messages = log_messages[-20:]
            log_placeholder.code("\n".join(display_messages))

        try:
            import time
            start_time = time.time()

            # Initialize
            status.info("🔧 Initializing quality scorer...")
            log("Starting quality computation...")
            log(f"Total passages: {len(pipeline.state.df)}")
            log(f"Embeddings enabled: {use_embeddings}")

            from core.quality import QualityScorer

            scorer = QualityScorer(
                df=pipeline.state.df,
                passage_col=pipeline.state.passage_col,
                label_columns=pipeline.state.label_columns,
                use_embeddings=use_embeddings
            )

            log("✓ Quality scorer initialized")

            # Phase 1: Compute embeddings if needed
            if use_embeddings:
                status.info("📊 Phase 1/3: Computing embeddings...")
                progress_bar.progress(0.0)

                valid_mask = pipeline.state.df[pipeline.state.passage_col].notna()
                valid_indices = pipeline.state.df[valid_mask].index.tolist()

                log(f"Computing embeddings for {len(valid_indices)} passages...")

                # This is a simplified version - we'll enhance the actual scorer
                scorer._compute_embeddings_with_progress(
                    valid_indices,
                    lambda current, total, msg: update_progress(
                        status, progress_bar, metric_placeholders, log,
                        current, total, "Computing embeddings", msg
                    )
                )

                log("✓ Embeddings computed")

            # Phase 2: Compute quality scores
            status.info("🎯 Phase 2/3: Computing quality scores...")
            progress_bar.progress(0.33)

            log("Computing quality metrics...")

            quality_scores = scorer.compute_all_with_progress(
                k_similar=k_similar,
                progress_callback=lambda current, total, msg: update_progress(
                    status, progress_bar, metric_placeholders, log,
                    current, total, "Computing quality", msg
                )
            )

            log(f"✓ Computed scores for {len(quality_scores)} passages")

            # Phase 3: Analyze distribution
            status.info("📈 Phase 3/3: Analyzing distribution...")
            progress_bar.progress(0.66)

            log("Analyzing quality distribution...")

            distribution = scorer.get_quality_report(quality_scores)

            log(f"✓ Mean quality: {distribution['mean']:.3f}")
            log(f"✓ Median quality: {distribution['median']:.3f}")

            # Store results
            pipeline.state.quality_scores = quality_scores
            pipeline.state.quality_distribution = distribution
            pipeline._save_state()

            # Complete
            elapsed = time.time() - start_time
            progress_bar.progress(1.0)
            status.success(f"✅ Quality computation complete! ({elapsed:.1f} seconds)")

            log(f"=== COMPLETE ===")
            log(f"Total time: {elapsed:.1f} seconds")
            log(f"Passages scored: {len(quality_scores)}")

            # Show results
            st.markdown("---")
            render_quality_results(distribution, quality_scores)

            # Next step
            st.markdown("---")
            if st.button("➡️ Continue to Data Selection", type="primary"):
                st.rerun()

        except Exception as e:
            status.error(f"❌ Error: {str(e)}")
            log(f"ERROR: {str(e)}")

            with st.expander("🐛 Full Error Details"):
                import traceback
                st.code(traceback.format_exc())


def update_progress(status, progress_bar, metric_placeholders, log, current, total, phase, message):
    """Update progress indicators"""
    progress = current / total if total > 0 else 0

    # Update status
    status.info(f"{phase}: {current}/{total} ({progress * 100:.1f}%)")

    # Update progress bar
    progress_bar.progress(progress)

    # Update metrics
    metric_placeholders[0].metric("Processed", f"{current:,}")
    metric_placeholders[1].metric("Total", f"{total:,}")
    metric_placeholders[2].metric("Progress", f"{progress * 100:.0f}%")
    metric_placeholders[3].metric("Remaining", f"{total - current:,}")

    # Log message if provided
    if message:
        log(message)


def render_quality_results(distribution, quality_scores):
    """Render quality results summary"""
    st.markdown("### 📊 Quality Score Results")

    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean Quality", f"{distribution['mean']:.3f}")
    with col2:
        st.metric("Median Quality", f"{distribution['median']:.3f}")
    with col3:
        st.metric("Std Dev", f"{distribution['std']:.3f}")
    with col4:
        st.metric("Passages Scored", f"{len(quality_scores):,}")

    # Tier distribution
    st.markdown("### 🎯 Quality Tiers")

    tier_cols = st.columns(4)
    tiers = [
        ("Elite", "elite", "🟢"),
        ("Good", "good", "🟡"),
        ("Fair", "fair", "🟠"),
        ("Low", "low", "🔴")
    ]

    for col, (name, key, emoji) in zip(tier_cols, tiers):
        with col:
            count = distribution['tier_distribution'].get(key, 0)
            pct = distribution['tier_percentages'].get(key, 0)
            st.metric(f"{emoji} {name}", f"{count:,}", f"{pct:.1f}%")

    # Histogram
    st.markdown("### 📈 Distribution")

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 4))

    qualities = [q.overall_quality for q in quality_scores.values()]

    ax.hist(qualities, bins=50, edgecolor='black', alpha=0.7, color='#1f77b4')
    ax.axvline(distribution['median'], color='red', linestyle='--',
               linewidth=2, label=f"Median: {distribution['median']:.3f}")
    ax.axvline(distribution['mean'], color='green', linestyle='--',
               linewidth=2, label=f"Mean: {distribution['mean']:.3f}")

    # Add tier boundaries
    ax.axvline(0.75, color='gray', linestyle=':', alpha=0.5, label='Elite (0.75)')
    ax.axvline(0.60, color='gray', linestyle=':', alpha=0.5, label='Good (0.60)')
    ax.axvline(0.45, color='gray', linestyle=':', alpha=0.5, label='Fair (0.45)')

    ax.set_xlabel('Quality Score')
    ax.set_ylabel('Count')
    ax.set_title('Quality Score Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close()

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