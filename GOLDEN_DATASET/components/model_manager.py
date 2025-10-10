"""
Centralized Model Management
Handles loading, unloading, and managing multiple trained models
"""

import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Import from existing model_inference module
import sys

sys.path.append(str(Path(__file__).parent.parent))
from core.model_inference import HRAFModelLoader, find_model_directories


class ModelManager:
    """
    Centralized manager for multiple loaded models

    Handles:
    - Loading/unloading models
    - Batch inference across models
    - Model comparison
    - Performance tracking
    """

    def __init__(self):
        self.models: Dict[str, HRAFModelLoader] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.inference_cache: Dict[str, Any] = {}

    def load_model(
            self,
            model_path: str,
            nickname: Optional[str] = None
    ) -> bool:
        """
        Load a model with optional custom name

        Args:
            model_path: Path to model directory
            nickname: Optional custom name (uses directory name if None)

        Returns:
            True if successful
        """

        model_path = Path(model_path)

        if not model_path.exists():
            st.error(f"Model path not found: {model_path}")
            return False

        # Generate name
        if nickname:
            name = nickname
        else:
            # Use parent directory name if model is in "final_model" subdir
            if model_path.name == "final_model":
                name = model_path.parent.name
            else:
                name = model_path.name

        # Check if already loaded
        if name in self.models:
            st.warning(f"Model '{name}' already loaded")
            return False

        # Load model
        try:
            loader = HRAFModelLoader()
            success = loader.load_model(str(model_path))

            if not success:
                st.error(f"Failed to load model from {model_path}")
                return False

            # Store
            self.models[name] = loader

            # Get metadata
            info = loader.get_model_info()
            if info:
                self.model_metadata[name] = {
                    'path': str(model_path),
                    'loaded_at': datetime.now().isoformat(),
                    'config': info.get('config', {}),
                    'test_results': info.get('test_results', {}),
                    'label_names': loader.label_names
                }

            return True

        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False

    def unload_model(self, name: str) -> bool:
        """
        Unload a model from memory

        Args:
            name: Model name

        Returns:
            True if successful
        """

        if name not in self.models:
            st.warning(f"Model '{name}' not loaded")
            return False

        # Remove from memory
        del self.models[name]

        if name in self.model_metadata:
            del self.model_metadata[name]

        # Clear inference cache for this model
        self.inference_cache = {
            k: v for k, v in self.inference_cache.items()
            if not k.startswith(f"{name}:")
        }

        return True

    def get_model(self, name: str) -> Optional[HRAFModelLoader]:
        """Get a loaded model"""
        return self.models.get(name)

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Get all loaded models with metadata

        Returns:
            List of model info dictionaries
        """

        models_info = []

        for name, loader in self.models.items():
            info = {
                'name': name,
                'loaded': loader.is_loaded(),
            }

            # Add metadata if available
            if name in self.model_metadata:
                meta = self.model_metadata[name]
                info.update({
                    'path': meta.get('path'),
                    'loaded_at': meta.get('loaded_at'),
                    'architecture': 'Hierarchical' if meta.get('config', {}).get('use_hierarchy') else 'Flat',
                    'test_f1': meta.get('test_results', {}).get('eval_f1_micro'),
                    'labels': len(meta.get('label_names', []))
                })

            models_info.append(info)

        return models_info

    def predict_single(
            self,
            text: str,
            model_name: str,
            use_optimal_thresholds: bool = True,
            default_threshold: float = 0.5,
            use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Run inference with a single model

        Args:
            text: Input text
            model_name: Model to use
            use_optimal_thresholds: Use model's optimal thresholds
            default_threshold: Default threshold
            use_cache: Use cached results if available

        Returns:
            Prediction dictionary or None
        """

        if model_name not in self.models:
            st.error(f"Model '{model_name}' not loaded")
            return None

        # Check cache
        cache_key = f"{model_name}:{hash(text)}:{use_optimal_thresholds}:{default_threshold}"

        if use_cache and cache_key in self.inference_cache:
            return self.inference_cache[cache_key]

        # Run inference
        loader = self.models[model_name]

        try:
            result = loader.predict_passage(
                text,
                use_optimal_thresholds=use_optimal_thresholds,
                default_threshold=default_threshold
            )

            # Add model name to result
            result['model_name'] = model_name

            # Cache
            if use_cache:
                self.inference_cache[cache_key] = result

            return result

        except Exception as e:
            st.error(f"Inference error with {model_name}: {e}")
            return None

    def predict_batch(
            self,
            texts: List[str],
            model_names: Optional[List[str]] = None,
            use_optimal_thresholds: bool = True,
            default_threshold: float = 0.5,
            show_progress: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run batch inference with multiple models

        Args:
            texts: List of input texts
            model_names: Models to use (all if None)
            use_optimal_thresholds: Use optimal thresholds
            default_threshold: Default threshold
            show_progress: Show progress bar

        Returns:
            Dictionary mapping model names to results
        """

        if model_names is None:
            model_names = list(self.models.keys())

        # Validate models
        invalid_models = [m for m in model_names if m not in self.models]
        if invalid_models:
            st.error(f"Models not loaded: {invalid_models}")
            return {}

        results = {name: [] for name in model_names}

        # Progress bar
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()

        total_predictions = len(texts) * len(model_names)
        completed = 0

        # Run predictions
        for text in texts:
            for model_name in model_names:
                result = self.predict_single(
                    text,
                    model_name,
                    use_optimal_thresholds,
                    default_threshold
                )

                if result:
                    results[model_name].append(result)

                completed += 1

                if show_progress:
                    progress = completed / total_predictions
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Processing: {completed}/{total_predictions} "
                        f"({model_name})"
                    )

        if show_progress:
            progress_bar.empty()
            status_text.empty()

        return results

    def compare_models(
            self,
            texts: List[str],
            model_names: Optional[List[str]] = None,
            actual_labels: Optional[Dict[str, int]] = None
    ) -> pd.DataFrame:
        """
        Compare predictions across multiple models

        Args:
            texts: Input texts
            model_names: Models to compare (all if None)
            actual_labels: Ground truth labels for evaluation

        Returns:
            DataFrame with comparison results
        """

        if model_names is None:
            model_names = list(self.models.keys())

        # Run batch predictions
        predictions = self.predict_batch(
            texts,
            model_names,
            show_progress=True
        )

        # Build comparison data
        comparison_data = []

        for i, text in enumerate(texts):
            row = {
                'text_index': i,
                'text_preview': text[:100] + '...' if len(text) > 100 else text
            }

            # Add predictions from each model
            for model_name in model_names:
                if i < len(predictions[model_name]):
                    result = predictions[model_name][i]

                    # Predicted labels
                    row[f'{model_name}_predicted'] = ', '.join(result['predicted_labels'])

                    # Count of predictions
                    row[f'{model_name}_count'] = len(result['predicted_labels'])

            # Agreement analysis
            all_predictions = []
            for model_name in model_names:
                if i < len(predictions[model_name]):
                    all_predictions.append(
                        set(predictions[model_name][i]['predicted_labels'])
                    )

            if all_predictions:
                # Find intersection (all models agree)
                agreed = set.intersection(*all_predictions) if all_predictions else set()
                row['all_agree'] = ', '.join(agreed) if agreed else 'None'
                row['agreement_count'] = len(agreed)

                # Find union (any model predicted)
                any_predicted = set.union(*all_predictions) if all_predictions else set()
                row['any_predicted'] = ', '.join(any_predicted)

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def get_model_performance_summary(
            self,
            model_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get performance summary for loaded models

        Args:
            model_names: Models to include (all if None)

        Returns:
            DataFrame with performance metrics
        """

        if model_names is None:
            model_names = list(self.models.keys())

        summary_data = []

        for name in model_names:
            if name not in self.model_metadata:
                continue

            meta = self.model_metadata[name]
            config = meta.get('config', {})
            test_results = meta.get('test_results', {})

            summary_data.append({
                'Model': name,
                'Architecture': 'Hierarchical' if config.get('use_hierarchy') else 'Flat',
                'Gated': 'Yes' if config.get('gated_hierarchy') else 'No',
                'Focal Loss': 'Yes' if config.get('use_focal_loss') else 'No',
                'F1 Micro': f"{test_results.get('eval_f1_micro', 0):.3f}",
                'F1 Macro': f"{test_results.get('eval_f1_macro', 0):.3f}",
                'Labels': len(meta.get('label_names', []))
            })

        return pd.DataFrame(summary_data)

    def clear_cache(self):
        """Clear inference cache"""
        self.inference_cache.clear()

    def __len__(self):
        """Number of loaded models"""
        return len(self.models)

    def __contains__(self, name):
        """Check if model is loaded"""
        return name in self.models


# ============================================================================
# UI COMPONENTS FOR MODEL MANAGER
# ============================================================================

def render_model_manager_ui(manager: ModelManager) -> None:
    """
    Render UI for model manager

    Args:
        manager: ModelManager instance
    """

    st.markdown("### 🤖 Model Manager")

    # Current models
    models = manager.list_models()

    if not models:
        st.info("💡 No models loaded. Load models below.")
    else:
        st.markdown(f"**{len(models)} model(s) loaded**")

        # Display loaded models
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
    st.markdown("**Load New Model**")

    # Find available models
    model_dirs = find_model_directories("./models")

    if not model_dirs:
        st.warning("No trained models found in ./models/")
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


def render_model_comparison_ui(
        manager: ModelManager,
        texts: List[str],
        actual_labels: Optional[Dict[str, int]] = None
) -> None:
    """
    Render model comparison interface

    Args:
        manager: ModelManager instance
        texts: Texts to compare
        actual_labels: Optional ground truth
    """

    st.markdown("### 📊 Model Comparison")

    if len(manager) < 2:
        st.warning("Load at least 2 models to compare")
        return

    # Select models
    available_models = list(manager.models.keys())

    selected_models = st.multiselect(
        "Select models to compare:",
        available_models,
        default=available_models,
        key="comparison_model_select"
    )

    if len(selected_models) < 2:
        st.info("Select at least 2 models")
        return

    if st.button("🔍 Compare Models", type="primary"):
        with st.spinner("Running comparison..."):
            comparison_df = manager.compare_models(
                texts,
                selected_models,
                actual_labels
            )

            st.dataframe(
                comparison_df,
                hide_index=True,
                width='stretch'
            )

            # Agreement analysis
            st.markdown("#### Agreement Analysis")

            total_passages = len(texts)
            full_agreement = (
                        comparison_df['agreement_count'] == comparison_df[[f'{m}_count' for m in selected_models]].max(
                    axis=1)).sum()

            st.metric(
                "Full Agreement",
                f"{full_agreement}/{total_passages}",
                f"{full_agreement / total_passages * 100:.1f}%"
            )


def compare_models_fairly(
        hierarchical_results: Dict,
        flat_results: Dict,
        hierarchical_main_labels: List[str] = None
) -> pd.DataFrame:
    """
    Compare hierarchical and flat models using ONLY sublabel metrics

    Args:
        hierarchical_results: Test results from hierarchical model
        flat_results: Test results from flat model
        hierarchical_main_labels: Main category names to exclude

    Returns:
        Comparison dataframe
    """

    comparison = {
        'Metric': [],
        'Hierarchical': [],
        'Flat': [],
        'Difference': []
    }

    # Overall comparison (sublabels only)
    comparison['Metric'].append('F1 Micro (Sublabels)')
    hier_f1 = hierarchical_results.get('eval_f1_micro_sublabels', 0)
    flat_f1 = flat_results.get('eval_f1_micro', 0)  # Flat has no main labels
    comparison['Hierarchical'].append(f"{hier_f1:.3f}")
    comparison['Flat'].append(f"{flat_f1:.3f}")
    comparison['Difference'].append(f"{(hier_f1 - flat_f1):+.3f}")

    comparison['Metric'].append('F1 Macro (Sublabels)')
    hier_macro = hierarchical_results.get('eval_f1_macro_sublabels', 0)
    flat_macro = flat_results.get('eval_f1_macro', 0)
    comparison['Hierarchical'].append(f"{hier_macro:.3f}")
    comparison['Flat'].append(f"{flat_macro:.3f}")
    comparison['Difference'].append(f"{(hier_macro - flat_macro):+.3f}")

    # Per-sublabel comparison
    if hierarchical_main_labels:
        # Get sublabel names
        all_sublabels = set()

        for key in hierarchical_results.keys():
            if key.startswith('eval_f1_'):
                label_name = key.replace('eval_f1_', '')
                # Skip main categories and summary metrics
                if (label_name not in hierarchical_main_labels and
                        label_name not in ['micro', 'macro', 'micro_all', 'macro_all',
                                           'micro_sublabels', 'macro_sublabels']):
                    all_sublabels.add(label_name)

        for sublabel in sorted(all_sublabels):
            comparison['Metric'].append(sublabel)

            hier_val = hierarchical_results.get(f'eval_f1_{sublabel}', 0)
            flat_val = flat_results.get(f'eval_f1_{sublabel}', 0)

            comparison['Hierarchical'].append(f"{hier_val:.3f}")
            comparison['Flat'].append(f"{flat_val:.3f}")
            comparison['Difference'].append(f"{(hier_val - flat_val):+.3f}")

    return pd.DataFrame(comparison)
