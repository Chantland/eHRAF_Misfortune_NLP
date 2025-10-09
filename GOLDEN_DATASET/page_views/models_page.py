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
        st.session_state['action_trigger'] = None  # Clear flag

    # Use the comprehensive training UI from core
    render_training_page(dict(st.session_state))

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
                width='stretch'
            )

            # Overall metrics
            col1, col2, col3 = st.columns(3)

            # Calculate overall
            all_actual = []
            all_pred = []
            for label in label_columns:
                if label in actual_labels:
                    all_actual.extend(actual_labels[label])
                    all_pred.extend(predicted[label])

            overall_f1 = f1_score(all_actual, all_pred, zero_division=0)
            overall_precision = precision_score(all_actual, all_pred, zero_division=0)
            overall_recall = recall_score(all_actual, all_pred, zero_division=0)

            with col1:
                st.metric("Overall F1", f"{overall_f1:.3f}")

            with col2:
                st.metric("Overall Precision", f"{overall_precision:.3f}")

            with col3:
                st.metric("Overall Recall", f"{overall_recall:.3f}")


# ============================================================================
# COMPARE
# ============================================================================

def render_comparison_section(manager: ModelManager):
    """Compare multiple models"""

    st.markdown("### ⚖️ Model Comparison")

    if len(manager) < 2:
        st.info("💡 Load at least 2 models to compare")
        return

    # Check if data loaded
    if not st.session_state.get('initialized'):
        st.warning("⚠️ Load a dataset first")
        return

    df = st.session_state.df
    passage_col = st.session_state.passage_col

    st.markdown("#### Select Models and Data")

    # Model selection
    available_models = list(manager.models.keys())

    selected_models = st.multiselect(
        "Models to compare:",
        available_models,
        default=available_models[:2] if len(available_models) >= 2 else available_models,
        key="comparison_model_select"
    )

    if len(selected_models) < 2:
        st.info("Select at least 2 models")
        return

    # Data selection
    num_passages = st.slider(
        "Number of passages:",
        5, min(100, len(df)), 20,
        help="Number of passages to compare"
    )

    if st.button("🔍 Compare Models", type="primary"):
        with st.spinner("Running comparison..."):
            # Sample passages
            sample_df = df.sample(n=num_passages, random_state=42)
            passages = sample_df[passage_col].tolist()

            # Run comparison
            comparison_df = manager.compare_models(
                passages,
                selected_models
            )

            # Display results
            st.markdown("---")
            st.markdown("#### Comparison Results")

            st.dataframe(
                comparison_df,
                hide_index=True,
                width='stretch'
            )

            # Agreement analysis
            st.markdown("#### Agreement Analysis")

            total_passages = len(passages)
            full_agreement = (comparison_df['agreement_count'] > 0).sum()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Full Agreement",
                    f"{full_agreement}/{total_passages}"
                )

            with col2:
                agreement_pct = (full_agreement / total_passages * 100) if total_passages > 0 else 0
                st.metric(
                    "Agreement %",
                    f"{agreement_pct:.1f}%"
                )

            with col3:
                disagreements = total_passages - full_agreement
                st.metric(
                    "Disagreements",
                    disagreements
                )

            # Model performance summary
            st.markdown("---")
            st.markdown("#### Model Performance Summary")

            summary_df = manager.get_model_performance_summary(selected_models)

            st.dataframe(
                summary_df,
                hide_index=True,
                width='stretch'
            )