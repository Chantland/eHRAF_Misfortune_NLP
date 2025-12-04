"""
Models Page - Train, Evaluate, and Compare Classification Models

Architecture:
- Self-contained page module
- Uses ModelManager from components/
- Uses training code from core/
- Clean separation of concerns
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import components
from components.model_manager import (
    ModelManager,
    render_model_manager_ui,
    render_model_comparison_ui
)

# Import core functionality
from core.model_training import render_training_page
from core.model_inference import HRAFModelLoader, find_model_directories


def render():
    """Main render function for Models page"""

    st.markdown("# 🤖 Model Management")
    st.caption("Train, evaluate, and compare classification models")

    # Initialize model manager - FIXED: check for None too
    if 'model_manager' not in st.session_state or st.session_state.model_manager is None:
        st.session_state.model_manager = ModelManager()

    manager = st.session_state.model_manager

    # Create tabs
    tabs = st.tabs([
        "📚 Model Library",
        "🎓 Train New Model",
        "📊 Evaluate",
        "⚖️ Compare"
    ])

    with tabs[0]:
        render_model_library(manager)

    with tabs[1]:
        render_training_section()

    with tabs[2]:
        render_evaluation_section(manager)

    with tabs[3]:
        render_comparison_section(manager)


# ============================================================================
# MODEL LIBRARY
# ============================================================================

def render_model_library(manager: ModelManager):
    """Browse and manage loaded models"""

    st.markdown("### 📚 Model Library")

    # Show loaded models
    models = manager.list_models()

    if not models:
        st.info("💡 No models loaded. Load models below or train a new one.")
    else:
        st.markdown(f"**{len(models)} model(s) loaded**")

        for model_info in models:
            with st.expander(f"📦 {model_info['name']}", expanded=False):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.caption(f"**Architecture:** {model_info.get('architecture', 'Unknown')}")
                    st.caption(f"**Labels:** {model_info.get('labels', 'N/A')}")

                with col2:
                    test_f1 = model_info.get('test_f1')
                    if test_f1:
                        st.metric("Test F1", f"{test_f1:.3f}")
                    else:
                        st.caption("**Test F1:** N/A")

                with col3:
                    if st.button("🗑️ Unload", key=f"unload_{model_info['name']}"):
                        success = manager.unload_model(model_info['name'])
                        if success:
                            st.success(f"Unloaded {model_info['name']}")
                            st.rerun()

    st.markdown("---")

    # Load new model
    st.markdown("**Load Model**")

    # Find available models
    model_dirs = find_model_directories("./models")

    if not model_dirs:
        st.warning("No trained models found in ./models/")
        st.info("💡 Train a model first on the 'Train New Model' tab")
    else:
        model_options = {
            str(m.parent.name if m.name == "final_model" else m.name): str(m)
            for m in model_dirs
        }

        selected_model_name = st.selectbox(
            "Select model:",
            options=list(model_options.keys()),
            key="model_load_selector"
        )

        selected_model_path = model_options[selected_model_name]

        custom_name = st.text_input(
            "Custom name (optional):",
            value="",
            key="model_custom_name"
        )

        if st.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading..."):
                success = manager.load_model(
                    selected_model_path,
                    nickname=custom_name if custom_name else None
                )

                if success:
                    st.success(f"✅ Model loaded!")
                    st.rerun()


# ============================================================================
# TRAIN NEW MODEL
# ============================================================================

def render_training_section():
    """Train a new model"""

    st.markdown("### 🎓 Train New Model")

    # Check if data loaded
    if not st.session_state.get('initialized'):
        st.warning("⚠️ Load a dataset first")
        st.info("Go to **Data** page and load a dataset to begin training")
        return

    # ✅ RESPOND TO ASSISTANT ACTIONS
    if st.session_state.get('action_trigger') == 'start_training':
        st.info("🤖 **AI Assistant initiated training** - Review configuration below")
        st.session_state['action_trigger'] = None

    # ✅ FIX: Pass actual session_state, not a copy!
    render_training_page(st.session_state)

# ============================================================================
# EVALUATE
# ============================================================================

def render_evaluation_section(manager: ModelManager):
    """Evaluate models on test data"""

    st.markdown("### 📊 Model Evaluation")

    # Check if models loaded
    if len(manager) == 0:
        st.info("💡 Load a model first in the Model Library tab")
        return

    # Check if data loaded
    if not st.session_state.get('initialized'):
        st.warning("⚠️ Load a dataset first")
        st.info("Go to **Data** page to load test data")
        return

    df = st.session_state.df
    label_columns = st.session_state.label_columns
    passage_col = st.session_state.passage_col

    st.markdown("#### Select Model and Data")

    # Model selection
    available_models = list(manager.models.keys())
    selected_model = st.selectbox(
        "Model to evaluate:",
        available_models,
        key="eval_model_select"
    )

    # Data selection
    col1, col2 = st.columns(2)

    with col1:
        num_passages = st.slider(
            "Number of passages:",
            10, min(500, len(df)), 100,
            help="Random sample from dataset"
        )

    with col2:
        use_optimal = st.checkbox(
            "Use optimal thresholds",
            value=True,
            help="Use label-specific thresholds from training"
        )

    if st.button("🔍 Evaluate Model", type="primary"):
        with st.spinner("Running evaluation..."):
            # Sample passages
            sample_df = df.sample(n=num_passages, random_state=42)

            # Get passages and labels
            passages = sample_df[passage_col].tolist()

            # Ground truth
            actual_labels = {}
            for label in label_columns:
                actual_labels[label] = sample_df[label].tolist()

            # Run predictions
            loader = manager.get_model(selected_model)

            if loader is None:
                st.error("❌ Model not found")
                return

            results = []
            for passage in passages:
                result = loader.predict_passage(
                    passage,
                    use_optimal_thresholds=use_optimal
                )
                results.append(result)

            # Calculate metrics
            st.markdown("---")
            st.markdown("#### Results")

            # Aggregate predictions
            predicted = {label: [] for label in label_columns}
            for result in results:
                preds = result['predictions']
                for label in label_columns:
                    predicted[label].append(1 if preds.get(label, False) else 0)

            # Calculate per-label metrics
            from sklearn.metrics import f1_score, precision_score, recall_score
            import numpy as np

            metrics_data = []
            for label in label_columns:
                if label not in actual_labels:
                    continue

                actual = actual_labels[label]
                pred = predicted[label]

                f1 = f1_score(actual, pred, zero_division=0)
                precision = precision_score(actual, pred, zero_division=0)
                recall = recall_score(actual, pred, zero_division=0)

                metrics_data.append({
                    'Label': label,
                    'F1': f"{f1:.3f}",
                    'Precision': f"{precision:.3f}",
                    'Recall': f"{recall:.3f}",
                    'Support': sum(actual)
                })

            # Display metrics
            st.dataframe(
                pd.DataFrame(metrics_data),
                hide_index=True,
                use_container_width=True
            )

            # Overall metrics - FIXED FOR MULTI-LABEL
            col1, col2, col3 = st.columns(3)

            # Stack predictions into proper multi-label format (n_samples × n_labels)
            y_true = np.array([actual_labels[label] for label in label_columns if label in actual_labels]).T
            y_pred = np.array([predicted[label] for label in label_columns if label in actual_labels]).T

            # Calculate micro-averaged metrics (treats each label prediction equally)
            overall_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
            overall_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
            overall_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)

            with col1:
                st.metric("F1 Micro", f"{overall_f1:.3f}")

            with col2:
                st.metric("Precision Micro", f"{overall_precision:.3f}")

            with col3:
                st.metric("Recall Micro", f"{overall_recall:.3f}")

            # Also show macro for comparison
            st.caption("---")
            col1, col2, col3 = st.columns(3)

            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

            with col1:
                st.metric("F1 Macro", f"{macro_f1:.3f}")

            with col2:
                st.metric("Precision Macro", f"{macro_precision:.3f}")

            with col3:
                st.metric("Recall Macro", f"{macro_recall:.3f}")


# ============================================================================
# COMPARE
# ============================================================================

def render_comparison_section(manager: ModelManager):
    """Compare multiple models with fair sublabel-only metrics"""

    st.markdown("### ⚖️ Model Comparison")

    if len(manager) < 2:
        st.info("💡 Load at least 2 models to compare")
        return

    st.markdown("#### Select Models to Compare")

    available_models = list(manager.models.keys())

    col1, col2 = st.columns(2)

    with col1:
        model1_name = st.selectbox(
            "Model 1:",
            available_models,
            key="comparison_model1"
        )

    with col2:
        model2_options = [m for m in available_models if m != model1_name]
        model2_name = st.selectbox(
            "Model 2:",
            model2_options,
            key="comparison_model2"
        ) if model2_options else None

    if not model2_name:
        st.warning("Select two different models")
        return

    # Show model info
    st.markdown("---")
    st.markdown("#### 📋 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{model1_name}**")
        model1_info = next((m for m in manager.list_models() if m['name'] == model1_name), None)
        if model1_info:
            st.caption(f"Architecture: {model1_info.get('architecture', 'Unknown')}")
            st.caption(f"Labels: {model1_info.get('labels', 'N/A')}")
            test_f1 = model1_info.get('test_f1')
            if test_f1:
                st.caption(f"Test F1 (All): {test_f1:.3f}")

    with col2:
        st.markdown(f"**{model2_name}**")
        model2_info = next((m for m in manager.list_models() if m['name'] == model2_name), None)
        if model2_info:
            st.caption(f"Architecture: {model2_info.get('architecture', 'Unknown')}")
            st.caption(f"Labels: {model2_info.get('labels', 'N/A')}")
            test_f1 = model2_info.get('test_f1')
            if test_f1:
                st.caption(f"Test F1 (All): {test_f1:.3f}")

    st.markdown("---")

    # Comparison options
    st.markdown("#### ⚙️ Comparison Settings")

    col1, col2 = st.columns(2)

    with col1:
        comparison_type = st.radio(
            "Comparison type:",
            ["Fair Comparison (Sublabels Only)", "Prediction Samples"],
            help="Fair comparison uses only sublabels. Prediction samples shows actual predictions on passages."
        )

    with col2:
        if comparison_type == "Prediction Samples":
            # Check if data loaded
            if not st.session_state.get('initialized'):
                st.warning("⚠️ Load a dataset first (Data page)")
                return

            num_samples = st.slider(
                "Number of passages:",
                5, min(100, len(st.session_state.df)), 20
            )

    # Run comparison
    if st.button("🔍 Compare Models", type="primary"):

        if comparison_type == "Fair Comparison (Sublabels Only)":
            # Load test results from both models
            loader1 = manager.get_model(model1_name)
            loader2 = manager.get_model(model2_name)

            if not loader1 or not loader2:
                st.error("Could not load model information")
                return

            info1 = loader1.get_model_info()
            info2 = loader2.get_model_info()

            if not info1 or not info2:
                st.error("Models missing test results. Retrain models with updated code.")
                return

            results1 = info1.get('test_results', {})
            results2 = info2.get('test_results', {})

            if not results1 or not results2:
                st.error("Test results not found. Retrain models to generate test results.")
                return

            # Get label information
            # Check if label_names is in config (old format) or directly available
            if hasattr(loader1.model.config, 'label_names'):
                labels1 = loader1.model.config.label_names
            else:
                labels1 = loader1.label_names or []

            if hasattr(loader2.model.config, 'label_names'):
                labels2 = loader2.model.config.label_names
            else:
                labels2 = loader2.label_names or []

            # Identify main labels (hierarchical models)
            main1 = []
            main2 = []

            if info1.get('config', {}).get('use_hierarchy') and info1.get('config', {}).get('predict_main_labels'):
                # Try to extract from hierarchy config
                hierarchy1 = info1.get('config', {}).get('hierarchy_config', {})
                if hierarchy1:
                    main1 = [cat for cat, data in hierarchy1.get('categories', {}).items() if data.get('enabled')]
                else:
                    # Fallback: assume EVENT, CAUSE, ACTION
                    main1 = [l for l in labels1 if l in ['EVENT', 'CAUSE', 'ACTION']]

            if info2.get('config', {}).get('use_hierarchy') and info2.get('config', {}).get('predict_main_labels'):
                hierarchy2 = info2.get('config', {}).get('hierarchy_config', {})
                if hierarchy2:
                    main2 = [cat for cat, data in hierarchy2.get('categories', {}).items() if data.get('enabled')]
                else:
                    main2 = [l for l in labels2 if l in ['EVENT', 'CAUSE', 'ACTION']]

            # Import comparison function
            from core.model_training import compare_models_fairly

            with st.spinner("Comparing models..."):
                comparison_df, summary = compare_models_fairly(
                    results1, labels1, main1,
                    results2, labels2, main2,
                    model1_name, model2_name
                )

            # Display results
            st.markdown("---")
            st.markdown("### 📊 Comparison Results")

            # Summary cards
            col1, col2, col3 = st.columns(3)

            with col1:
                winner = summary['winner']
                if winner == 'Tie':
                    st.info("🤝 **Result: TIE**")
                else:
                    st.success(f"🏆 **Winner: {winner}**")

            with col2:
                diff = summary['difference']
                st.metric(
                    "F1 Difference",
                    f"{abs(diff):.3f}",
                    help=f"{model1_name if diff > 0 else model2_name} ahead"
                )

            with col3:
                st.metric(
                    "Common Sublabels",
                    summary['num_common_sublabels']
                )

            # Detailed comparison table
            st.markdown("#### Detailed Metrics")

            # Style the dataframe
            def highlight_winner(row):
                if row['Winner'] == model1_name:
                    return ['background-color: #d4edda'] * len(row)
                elif row['Winner'] == model2_name:
                    return ['background-color: #fff3cd'] * len(row)
                else:
                    return [''] * len(row)

            styled_df = comparison_df.style.apply(highlight_winner, axis=1)
            st.dataframe(styled_df, hide_index=True, width='stretch')

            # Win/loss breakdown
            st.markdown("---")
            st.markdown("#### 🎯 Label-Level Breakdown")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    f"{model1_name} Better",
                    summary['model1_better_count'],
                    help="Number of labels where Model 1 performs better"
                )

            with col2:
                st.metric(
                    f"{model2_name} Better",
                    summary['model2_better_count'],
                    help="Number of labels where Model 2 performs better"
                )

            with col3:
                st.metric(
                    "Ties",
                    summary['ties'],
                    help="Number of labels with similar performance"
                )

            # Visualization
            st.markdown("---")
            st.markdown("#### 📈 Visual Comparison")

            import matplotlib.pyplot as plt

            # Extract per-label data
            label_rows = comparison_df[comparison_df['Metric'].str.contains('---', na=False) == False]
            label_rows = label_rows[~label_rows['Metric'].str.contains('F1|⭐', na=False)]

            if len(label_rows) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # Plot 1: Side-by-side bar chart
                labels = label_rows['Metric'].tolist()
                m1_scores = [float(s) for s in label_rows[model1_name].tolist()]
                m2_scores = [float(s) for s in label_rows[model2_name].tolist()]

                x = np.arange(len(labels))
                width = 0.35

                bars1 = ax1.barh(x - width / 2, m1_scores, width, label=model1_name, alpha=0.8, color='#2E86AB')
                bars2 = ax1.barh(x + width / 2, m2_scores, width, label=model2_name, alpha=0.8, color='#A23B72')

                ax1.set_xlabel('F1 Score')
                ax1.set_title('Per-Label F1 Comparison')
                ax1.set_yticks(x)
                ax1.set_yticklabels(labels, fontsize=8)
                ax1.legend()
                ax1.grid(axis='x', alpha=0.3)
                ax1.set_xlim(0, 1)

                # Plot 2: Difference plot
                differences = [m1 - m2 for m1, m2 in zip(m1_scores, m2_scores)]
                colors = ['#27AE60' if d > 0 else '#E74C3C' if d < 0 else '#95a5a6' for d in differences]

                ax2.barh(x, differences, color=colors, alpha=0.7)
                ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                ax2.set_xlabel(f'F1 Difference ({model1_name} - {model2_name})')
                ax2.set_title('Performance Difference by Label')
                ax2.set_yticks(x)
                ax2.set_yticklabels(labels, fontsize=8)
                ax2.grid(axis='x', alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Export option
            st.markdown("---")

            csv = comparison_df.to_csv(index=False)
            st.download_button(
                "📥 Download Comparison CSV",
                csv,
                f"model_comparison_{model1_name}_vs_{model2_name}.csv",
                "text/csv"
            )

        else:
            # Prediction samples comparison
            df = st.session_state.df
            passage_col = st.session_state.passage_col
            label_columns = st.session_state.label_columns

            with st.spinner("Running predictions..."):
                # Sample passages
                sample_df = df.sample(n=num_samples, random_state=42)
                passages = sample_df[passage_col].tolist()

                # Run predictions from both models
                results = manager.predict_batch(
                    passages,
                    [model1_name, model2_name],
                    show_progress=True
                )

                # Build comparison
                st.markdown("---")
                st.markdown("### 🔍 Prediction Samples")

                for i, passage in enumerate(passages):
                    with st.expander(f"Passage {i + 1}: {passage[:100]}..."):
                        st.text(passage[:500] + ('...' if len(passage) > 500 else ''))

                        st.markdown("**Predictions:**")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(f"**{model1_name}**")
                            if i < len(results[model1_name]):
                                pred1 = results[model1_name][i]
                                labels1 = pred1['predicted_labels']
                                if labels1:
                                    for label in labels1:
                                        prob = pred1['probabilities'].get(label, 0)
                                        st.text(f"✓ {label} ({prob:.2f})")
                                else:
                                    st.text("(no predictions)")

                        with col2:
                            st.markdown(f"**{model2_name}**")
                            if i < len(results[model2_name]):
                                pred2 = results[model2_name][i]
                                labels2 = pred2['predicted_labels']
                                if labels2:
                                    for label in labels2:
                                        prob = pred2['probabilities'].get(label, 0)
                                        st.text(f"✓ {label} ({prob:.2f})")
                                else:
                                    st.text("(no predictions)")

                        # Show agreement/disagreement
                        if i < len(results[model1_name]) and i < len(results[model2_name]):
                            labels1 = set(results[model1_name][i]['predicted_labels'])
                            labels2 = set(results[model2_name][i]['predicted_labels'])

                            agreement = labels1 & labels2
                            only1 = labels1 - labels2
                            only2 = labels2 - labels1

                            if agreement:
                                st.success(f"✅ Agree on: {', '.join(agreement)}")
                            if only1:
                                st.info(f"🔵 Only {model1_name}: {', '.join(only1)}")
                            if only2:
                                st.warning(f"🟡 Only {model2_name}: {', '.join(only2)}")

