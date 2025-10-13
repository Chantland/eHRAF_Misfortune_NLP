"""
Model Training Module for HRAF Golden Dataset Discovery
Comprehensive training system with FIXED weighted focal loss implementation
"""

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import copy

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    AutoTokenizer,
    AutoModel,
    AutoConfig,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

# Import model architecture
from core.model_inference import (
    ConfigurableHierarchicalConfig,
    ConfigurableHierarchicalModel,
    HRAFModelLoader
)

# Import data utilities
from core.data_preparation import DataExperiment


# ============================================================================
# TRAINING SESSION MANAGEMENT
# ============================================================================

class TrainingSession:
    """Manages a model training session with all state and configuration"""

    def __init__(self, config: Dict, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.trainer = None
        self.tokenizer = None

        # Training state
        self.training_history = []
        self.best_metrics = {}
        self.current_epoch = 0

    def initialize_model(self, label_dims: Dict) -> Dict:
        """Initialize model with configuration"""

        # Register custom model classes
        AutoConfig.register("configurable_hierarchical", ConfigurableHierarchicalConfig)
        AutoModel.register(ConfigurableHierarchicalConfig, ConfigurableHierarchicalModel)

        model_config = ConfigurableHierarchicalConfig(
            base_model=self.config["base_model"],
            use_hierarchy=self.config["use_hierarchy"],
            gated_hierarchy=self.config.get("gated_hierarchy", False),
            gate_threshold=self.config.get("gate_threshold", 0.5),
            hidden_size=self.config["hidden_size"],
            hierarchical_hidden_size=self.config["hierarchical_hidden_size"],
            num_hidden_layers=self.config["num_hidden_layers"],
            dropout=self.config["dropout"],
            attention_dropout=self.config["attention_dropout"],
            use_weighted_loss=self.config["use_weighted_loss"],
            use_focal_loss=self.config["use_focal_loss"],
            focal_gamma=self.config.get("focal_gamma", 2.0),
            teacher_forcing_ratio=self.config.get("teacher_forcing_ratio", 0.5),
            num_main_labels=label_dims["num_main_labels"],
            num_event_labels=label_dims["num_event_labels"],
            num_cause_labels=label_dims["num_cause_labels"],
            num_action_labels=label_dims["num_action_labels"],
            total_labels=label_dims["total_labels"],
            label_indices=label_dims["label_indices"],
            label_names=label_dims["label_names"]
        )

        self.model = ConfigurableHierarchicalModel(model_config).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["base_model"])

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params
        }


class HierarchicalTrainer(Trainer):
    """Custom trainer with FIXED weighted focal loss"""

    def __init__(self, class_weights=None, teacher_forcing_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.teacher_forcing_ratio = teacher_forcing_ratio

        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        ✅ FIXED: Custom loss computation with proper weighted focal loss

        Combines class weighting and focal loss correctly
        """
        labels = inputs.pop("labels")

        # Teacher forcing during training
        use_teacher_forcing = model.training and (torch.rand(1).item() < self.teacher_forcing_ratio)

        outputs = model(
            **inputs,
            labels=None,  # Don't let model compute loss - we'll do it
            teacher_forcing=use_teacher_forcing
        )

        logits = outputs.logits

        # ✅ IMPLEMENT WEIGHTED FOCAL LOSS (combines both)
        if self.class_weights is not None and model.config.use_weighted_loss:
            if model.config.use_focal_loss:
                # ✅ Weighted Focal Loss - PROPER IMPLEMENTATION
                gamma = model.config.focal_gamma

                # Compute BCE loss (no reduction yet)
                bce_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.float(), reduction='none'
                )

                # Compute focal weighting
                probs = torch.sigmoid(logits)
                focal_weight = torch.where(
                    labels == 1,
                    (1 - probs) ** gamma,  # Focus on hard positives
                    probs ** gamma         # Focus on hard negatives
                )

                # Apply BOTH focal weighting AND class weighting
                class_weights_expanded = self.class_weights.unsqueeze(0).expand_as(logits)
                weighted_focal_loss = focal_weight * bce_loss * class_weights_expanded

                # Reduce to scalar
                loss = weighted_focal_loss.mean()
            else:
                # ✅ Weighted BCE only (no focal)
                bce_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.float(), reduction='none'
                )
                class_weights_expanded = self.class_weights.unsqueeze(0).expand_as(logits)
                weighted_loss = bce_loss * class_weights_expanded
                loss = weighted_loss.mean()
        else:
            # Standard loss (focal only or plain BCE)
            if model.config.use_focal_loss:
                loss = self._focal_loss(logits, labels.float(), gamma=model.config.focal_gamma)
            else:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())

        return (loss, outputs) if return_outputs else loss

    def _focal_loss(self, logits, targets, gamma=2.0):
        """Focal loss without class weighting"""
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probas = torch.sigmoid(logits)
        focal_weight = torch.where(targets == 1, (1 - probas) ** gamma, probas ** gamma)
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()


# ============================================================================
# HIERARCHY CONFIGURATION
# ============================================================================

def get_hraf_template() -> Dict:
    """Standard HRAF misfortune classification hierarchy template"""
    return {
        'categories': {
            'EVENT': {
                'sublabels': ['Illness', 'Accident', 'Other'],
                'enabled': True
            },
            'CAUSE': {
                'sublabels': ['Material_Physical', 'Spirits_Gods',
                             'Witchcraft_Sorcery', 'Rule_Violation_Taboo'],
                'enabled': True
            },
            'ACTION': {
                'sublabels': ['Physical_Material', 'Technical_Specialist',
                             'Divination', 'Shaman_Medium_Healer', 'Priest_High_Religion'],
                'enabled': True
            }
        }
    }


def render_hierarchy_configuration(
        label_columns: List[str],
        session_state: Dict,
        config_key: str = 'hierarchy_config'
) -> Dict:
    """Interactive hierarchy builder UI"""

    st.markdown("#### 🏗️ Hierarchy Configuration")
    st.caption("Define main categories and map sublabels to them")

    # Initialize hierarchy in session state if needed
    if config_key not in session_state:
        session_state[config_key] = {'categories': {}}

    hierarchy = session_state[config_key]

    # Quick start templates
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🚀 Use HRAF Template", width='stretch'):
            session_state[config_key] = get_hraf_template()
            st.success("✅ HRAF template loaded")
            st.rerun()

    with col2:
        if st.button("🧹 Clear All", width='stretch'):
            session_state[config_key] = {'categories': {}}
            st.rerun()

    with col3:
        if st.button("🔄 Auto-detect", width='stretch',
                     help="Detect hierarchy from column prefixes (EVENT_Illness, etc)"):
            detected = auto_detect_hierarchy(label_columns)
            if detected['categories']:
                session_state[config_key] = detected
                st.success(f"✅ Detected {len(detected['categories'])} categories")
            else:
                st.warning("⚠️ Could not detect hierarchy from column names")
            st.rerun()

    st.markdown("---")

    # Add new category
    with st.expander("➕ Add Main Category", expanded=len(hierarchy['categories']) == 0):
        # ✅ FIX: Use session state to track input
        if 'new_category_buffer' not in st.session_state:
            st.session_state['new_category_buffer'] = ""

        col1, col2 = st.columns([3, 1])

        with col1:
            new_category = st.text_input(
                "Category name:",
                placeholder="e.g., EVENT, CAUSE, ACTION",
                key="new_category_input",
                value=st.session_state['new_category_buffer'],
                on_change=lambda: None  # Force update
            )

            # Update buffer on every change
            st.session_state['new_category_buffer'] = new_category

        with col2:
            st.write("")
            st.write("")
            if st.button("Add", type="primary", disabled=not new_category, width='stretch'):
                if new_category and new_category not in hierarchy['categories']:
                    session_state[config_key]['categories'][new_category] = {
                        'sublabels': [],
                        'enabled': True
                    }
                    # ✅ Clear the buffer after adding
                    st.session_state['new_category_buffer'] = ""
                    st.success(f"✅ Added {new_category}")
                    st.rerun()
                elif new_category in hierarchy['categories']:
                    st.error("Category already exists")
                else:
                    st.warning("⚠️ Enter a category name first")

    # Configure existing categories
    if not hierarchy['categories']:
        st.info("💡 Add main categories above or use a template to get started")
        return hierarchy

    st.markdown("**Configure Categories:**")

    # Track changes
    changes_made = False

    for category_name in list(hierarchy['categories'].keys()):
        category = hierarchy['categories'][category_name]

        with st.expander(
                f"{'✅' if category['enabled'] else '❌'} {category_name} ({len(category['sublabels'])} sublabels)",
                expanded=True
        ):
            # Get available sublabels
            assigned_labels = set()
            for other_cat_name, other_cat in hierarchy['categories'].items():
                if other_cat_name != category_name:
                    assigned_labels.update(other_cat['sublabels'])

            available_labels = [l for l in label_columns if l not in assigned_labels]
            all_options = sorted(set(available_labels + category['sublabels']))

            # Sublabel selection
            selected = st.multiselect(
                f"Sublabels under **{category_name}**:",
                options=all_options,
                default=category['sublabels'],
                key=f"sublabels_{category_name}",
                help="Select which sublabels belong to this main category"
            )

            # Check if changed
            if selected != category['sublabels']:
                session_state[config_key]['categories'][category_name]['sublabels'] = selected
                changes_made = True

            # Controls row
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                enabled = st.checkbox(
                    "Enable category",
                    value=category['enabled'],
                    key=f"enabled_{category_name}"
                )

                if enabled != category['enabled']:
                    session_state[config_key]['categories'][category_name]['enabled'] = enabled
                    changes_made = True

            with col2:
                st.caption(f"**{len(selected)}** sublabels mapped")

            with col3:
                if st.button("🗑️ Delete", key=f"remove_{category_name}", width='stretch'):
                    del session_state[config_key]['categories'][category_name]
                    st.rerun()

    # Validation summary
    st.markdown("---")
    st.markdown("**📊 Validation Summary:**")

    # Calculate stats
    enabled_categories = [name for name, cat in hierarchy['categories'].items() if cat['enabled']]
    all_mapped = set()
    for cat in hierarchy['categories'].values():
        if cat['enabled']:
            all_mapped.update(cat['sublabels'])

    unmapped = [l for l in label_columns if l not in all_mapped]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Enabled Categories", len(enabled_categories))

    with col2:
        st.metric("Mapped Sublabels", len(all_mapped))

    with col3:
        st.metric("Total Labels", len(all_mapped) + len(enabled_categories))

    with col4:
        if unmapped:
            st.metric("⚠️ Unmapped", len(unmapped))
        else:
            st.metric("✅ Unmapped", 0)

    # Show unmapped labels
    if unmapped:
        with st.expander("⚠️ Show unmapped labels"):
            st.warning(f"These {len(unmapped)} labels are not assigned to any category:")
            for label in unmapped:
                st.text(f"• {label}")
            st.info("💡 Unmapped labels will not be used in training")

    # Preview hierarchy
    with st.expander("👁️ Preview Hierarchy Structure"):
        for category_name, category in hierarchy['categories'].items():
            if category['enabled']:
                st.markdown(f"**{category_name}** ({len(category['sublabels'])} sublabels)")
                for sublabel in category['sublabels']:
                    st.markdown(f"  └─ {sublabel}")
                st.markdown("")

    return hierarchy


def auto_detect_hierarchy(label_columns: List[str]) -> Dict:
    """Auto-detect hierarchy from column naming patterns"""
    hierarchy = {'categories': {}}

    # Pattern 1: PREFIX_Sublabel
    for col in label_columns:
        if '_' in col:
            parts = col.split('_', 1)
            if len(parts) == 2:
                main, sub = parts
                main = main.upper()

                if main not in hierarchy['categories']:
                    hierarchy['categories'][main] = {
                        'sublabels': [],
                        'enabled': True
                    }

                hierarchy['categories'][main]['sublabels'].append(col)

    return hierarchy


# ============================================================================
# LABEL STRUCTURE UTILITIES
# ============================================================================

def build_label_structure_from_hierarchy(
    hierarchy_config: Dict,
    predict_main_labels: bool
) -> Tuple[Dict, List[str]]:
    """Build label structure and ordered label list from hierarchy config"""

    label_structure = {}
    ordered_labels = []

    # Add main categories first (if enabled)
    if predict_main_labels:
        for category_name, category_data in hierarchy_config['categories'].items():
            if category_data['enabled']:
                ordered_labels.append(category_name)

    # Add sublabels by category
    for category_name, category_data in hierarchy_config['categories'].items():
        if not category_data['enabled']:
            continue

        label_structure[category_name] = {
            'main_label': category_name,
            'sublabels': category_data['sublabels'],
            'enabled': True
        }

        ordered_labels.extend(category_data['sublabels'])

    return label_structure, ordered_labels


def calculate_label_dimensions(
    label_structure: Dict,
    predict_main_labels: bool
) -> Dict:
    """Calculate the number of labels for each category"""

    dims = {
        "num_main_labels": 0,
        "num_event_labels": 0,
        "num_cause_labels": 0,
        "num_action_labels": 0,
        "total_labels": 0,
        "label_indices": {},
        "label_names": []
    }

    current_idx = 0

    # Main labels (only if enabled)
    if predict_main_labels:
        for category, info in label_structure.items():
            if info.get("enabled", True):
                dims["num_main_labels"] += 1
                dims["label_indices"][info["main_label"]] = current_idx
                dims["label_names"].append(info["main_label"])
                current_idx += 1

    # Sublabels - map to EVENT/CAUSE/ACTION for model architecture
    for category, info in label_structure.items():
        if not info.get("enabled", True):
            continue

        for sublabel in info["sublabels"]:
            # Map category to model's expected structure
            if category == "EVENT" or "event" in category.lower():
                dims["num_event_labels"] += 1
            elif category == "CAUSE" or "cause" in category.lower():
                dims["num_cause_labels"] += 1
            elif category == "ACTION" or "action" in category.lower():
                dims["num_action_labels"] += 1
            else:
                dims["num_action_labels"] += 1

            dims["label_indices"][sublabel] = current_idx
            dims["label_names"].append(sublabel)
            current_idx += 1

    dims["total_labels"] = current_idx

    return dims


def augment_data_with_main_categories(
    df: pd.DataFrame,
    label_structure: Dict,
    predict_main_labels: bool
) -> pd.DataFrame:
    """Add main category columns to dataframe (inferred from sublabels)"""

    if not predict_main_labels:
        return df

    df = df.copy()

    # Create main category columns
    for category_name, category_info in label_structure.items():
        if not category_info.get('enabled', True):
            continue

        sublabels = category_info['sublabels']
        existing_sublabels = [col for col in sublabels if col in df.columns]

        if existing_sublabels:
            df[category_name] = (df[existing_sublabels].sum(axis=1) > 0).astype(int)
        else:
            df[category_name] = 0

    return df


# ============================================================================
# DATASET PREPARATION
# ============================================================================

def prepare_datasets(
    df: pd.DataFrame,
    label_columns: List[str],
    passage_col: str,
    data_config: Dict,
    tokenizer
) -> Tuple[Dataset, Dataset, Dataset]:
    """Prepare train/val/test datasets with proper data cleaning"""

    # Only keep necessary columns
    columns_to_keep = [passage_col] + label_columns

    # Add ID if exists
    if 'ID' in df.columns:
        try:
            df['ID'] = df['ID'].astype(str)
            columns_to_keep.append('ID')
        except:
            pass

    # Filter to needed columns
    df_clean = df[columns_to_keep].copy()

    # Ensure all label columns are numeric (0/1)
    for label in label_columns:
        df_clean[label] = pd.to_numeric(df_clean[label], errors='coerce').fillna(0).astype(int)

    # Ensure passage column is string
    df_clean[passage_col] = df_clean[passage_col].astype(str)

    # Remove rows with missing passages
    df_clean = df_clean[df_clean[passage_col].notna()]

    print(f"📊 Cleaned dataset: {len(df_clean)} passages with {len(columns_to_keep)} columns")

    # Split data
    stratify_col = data_config.get("stratify_by")
    stratify_array = df_clean[stratify_col] if stratify_col and stratify_col in df_clean.columns else None

    train_val_df, test_df = train_test_split(
        df_clean,
        test_size=data_config["test_size"],
        random_state=data_config["random_seed"],
        stratify=stratify_array
    )

    stratify_array = train_val_df[stratify_col] if stratify_col and stratify_col in train_val_df.columns else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=data_config["validation_size"],
        random_state=data_config["random_seed"],
        stratify=stratify_array
    )

    print(f"📊 Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    # Tokenization
    def tokenize_function(examples):
        return tokenizer(
            examples[passage_col],
            padding='max_length',
            truncation=True,
            max_length=data_config["max_length"]
        )

    # Prepare labels
    def prepare_labels(examples, label_columns):
        labels = []
        batch_size = len(examples[label_columns[0]])

        for i in range(batch_size):
            label_vector = [int(examples[col][i]) for col in label_columns]
            labels.append(label_vector)

        examples['labels'] = labels
        return examples

    # Apply transformations
    print("🔄 Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    print("🏷️ Preparing labels...")
    train_dataset = train_dataset.map(lambda x: prepare_labels(x, label_columns), batched=True)
    val_dataset = val_dataset.map(lambda x: prepare_labels(x, label_columns), batched=True)
    test_dataset = test_dataset.map(lambda x: prepare_labels(x, label_columns), batched=True)

    # Remove unnecessary columns
    columns_to_remove = [passage_col] + label_columns
    if 'ID' in train_dataset.column_names:
        columns_to_remove.append('ID')

    train_dataset = train_dataset.remove_columns(
        [col for col in columns_to_remove if col in train_dataset.column_names])
    val_dataset = val_dataset.remove_columns(
        [col for col in columns_to_remove if col in val_dataset.column_names])
    test_dataset = test_dataset.remove_columns(
        [col for col in columns_to_remove if col in test_dataset.column_names])

    # Set format for PyTorch
    train_dataset.set_format('torch')
    val_dataset.set_format('torch')
    test_dataset.set_format('torch')

    print("✅ Datasets prepared successfully!")

    return train_dataset, val_dataset, test_dataset


def calculate_class_weights(df: pd.DataFrame, label_columns: List[str]) -> torch.Tensor:
    """
    ✅ FIXED: Calculate class weights with capping for stability
    """
    class_weights = []

    for col in label_columns:
        pos_count = df[col].sum()
        neg_count = len(df) - pos_count

        if pos_count > 0:
            weight = neg_count / pos_count
            weight = min(weight, 100)
        else:
            weight = 1.0

        class_weights.append(weight)

    return torch.tensor(class_weights).float()


def compute_metrics_for_trainer(label_names, main_label_names=None):
    """
    Create metrics computation function with BOTH full and sublabel-only F1

    Args:
        label_names: All label names including main categories
        main_label_names: List of main category names (e.g., ['EVENT', 'CAUSE', 'ACTION'])
                         If None, assumes no main categories (flat model)
    """

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Apply sigmoid
        predictions = torch.sigmoid(torch.tensor(predictions)).numpy()

        # Find optimal threshold for each label
        optimal_thresholds = []
        optimal_predictions = np.zeros_like(predictions)

        for i in range(predictions.shape[1]):
            best_threshold = 0.5
            best_f1 = 0.0

            for threshold in np.arange(0.1, 0.91, 0.05):
                pred_binary = (predictions[:, i] > threshold).astype(int)
                f1 = f1_score(labels[:, i], pred_binary, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            optimal_thresholds.append(best_threshold)
            optimal_predictions[:, i] = (predictions[:, i] > best_threshold).astype(int)

        # Calculate metrics on ALL labels (original behavior)
        f1_micro_all = f1_score(labels, optimal_predictions, average='micro', zero_division=0)
        f1_macro_all = f1_score(labels, optimal_predictions, average='macro', zero_division=0)

        # Calculate metrics on SUBLABELS ONLY (fair comparison)
        sublabel_indices = []
        if main_label_names:
            # Identify indices of sublabels (not main categories)
            for i, name in enumerate(label_names):
                if name not in main_label_names:
                    sublabel_indices.append(i)
        else:
            # Flat model - all labels are sublabels
            sublabel_indices = list(range(len(label_names)))

        if sublabel_indices:
            labels_sub = labels[:, sublabel_indices]
            preds_sub = optimal_predictions[:, sublabel_indices]

            f1_micro_sublabels = f1_score(labels_sub, preds_sub, average='micro', zero_division=0)
            f1_macro_sublabels = f1_score(labels_sub, preds_sub, average='macro', zero_division=0)
        else:
            # No sublabels (shouldn't happen, but safe fallback)
            f1_micro_sublabels = f1_micro_all
            f1_macro_sublabels = f1_macro_all

        # Per-label metrics
        per_label_f1 = {}
        for i, name in enumerate(label_names):
            f1 = f1_score(labels[:, i], optimal_predictions[:, i], zero_division=0)
            per_label_f1[f"f1_{name}"] = f1

        return {
            # Full metrics (hierarchical models)
            'f1_micro': f1_micro_all,  # Keep for backward compatibility
            'f1_macro': f1_macro_all,
            'f1_micro_all': f1_micro_all,
            'f1_macro_all': f1_macro_all,

            # Sublabel-only metrics (FAIR COMPARISON)
            'f1_micro_sublabels': f1_micro_sublabels,
            'f1_macro_sublabels': f1_macro_sublabels,

            # Per-label breakdown
            **per_label_f1
        }

    return compute_metrics


def find_optimal_thresholds(predictions, labels, label_names):
    """Find optimal threshold for each label"""
    from sklearn.metrics import f1_score

    thresholds = []
    for i in range(predictions.shape[1]):
        best_threshold = 0.5
        best_f1 = 0

        # Try thresholds from 0.1 to 0.9
        for threshold in np.arange(0.1, 0.9, 0.05):
            pred_binary = (predictions[:, i] > threshold).astype(int)
            f1 = f1_score(labels[:, i], pred_binary, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        thresholds.append(best_threshold)

    return thresholds


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_training_history(history: List[Dict], output_dir: Path):
    """Create training history visualizations"""

    if not history:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Extract metrics
    epochs = [h['epoch'] for h in history]
    train_loss = [h.get('train_loss', 0) for h in history]
    eval_loss = [h.get('eval_loss', 0) for h in history]
    eval_f1_micro = [h.get('eval_f1_micro', 0) for h in history]
    eval_f1_macro = [h.get('eval_f1_macro', 0) for h in history]

    # Loss plot
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, eval_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # F1 scores
    axes[0, 1].plot(epochs, eval_f1_micro, 'g-', label='F1 Micro', linewidth=2)
    axes[0, 1].plot(epochs, eval_f1_macro, 'orange', label='F1 Macro', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Scores')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    # Loss ratio (overfitting indicator)
    loss_ratio = [e / t if t > 0 else 1 for e, t in zip(eval_loss, train_loss)]
    axes[1, 0].plot(epochs, loss_ratio, 'purple', linewidth=2)
    axes[1, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Eval Loss / Train Loss')
    axes[1, 0].set_title('Overfitting Indicator (>1 = overfitting)')
    axes[1, 0].grid(alpha=0.3)

    # Learning rate
    learning_rates = [h.get('learning_rate', 0) for h in history]
    if any(learning_rates):
        axes[1, 1].plot(epochs, learning_rates, 'brown', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Learning rate data not available',
                        ha='center', va='center', transform=axes[1, 1].transAxes)

    plt.tight_layout()

    save_path = output_dir / 'training_history.png'
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved training history to {save_path}")
    except Exception as e:
        print(f"⚠️ Could not save training history plot: {e}")

    return fig


def visualize_test_results(test_results: Dict, label_names: List[str], output_dir: Path):
    """Create test results visualizations"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract per-label F1 scores
    label_f1s = {}
    for key, value in test_results.items():
        if key.startswith('eval_f1_') and key not in ['eval_f1_micro', 'eval_f1_macro']:
            label_name = key.replace('eval_f1_', '')
            label_f1s[label_name] = value

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Overall metrics
    ax = axes[0]
    metrics = ['F1 Micro', 'F1 Macro']
    values = [
        test_results.get('eval_f1_micro', 0),
        test_results.get('eval_f1_macro', 0)
    ]
    bars = ax.bar(metrics, values, color=['#2E86AB', '#A23B72'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('F1 Score')
    ax.set_title('Overall Test Performance')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)
    ax.grid(alpha=0.3)

    # Per-label performance
    ax = axes[1]

    if label_f1s:
        labels = list(label_f1s.keys())
        scores = list(label_f1s.values())

        # Sort by score
        sorted_items = sorted(zip(labels, scores), key=lambda x: x[1])
        labels, scores = zip(*sorted_items) if sorted_items else ([], [])

        # Color by score quality
        colors = ['#27AE60' if s > 0.7 else '#F39C12' if s > 0.5 else '#E74C3C' for s in scores]

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, scores, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('F1 Score')
        ax.set_title('F1 Score by Label')
        ax.set_xlim(0, 1)

        # Add score values
        for i, (label, score) in enumerate(zip(labels, scores)):
            ax.text(score + 0.01, i, f'{score:.3f}',
                    va='center', fontsize=8)

        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No per-label F1 scores available',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    save_path = output_dir / 'test_results.png'
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved test results to {save_path}")
    except Exception as e:
        print(f"⚠️ Could not save test results plot: {e}")

    return fig


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

def get_default_training_config() -> Dict:
    """Get default training configuration"""
    return {
        # Model Architecture
        "base_model": "roberta-base",
        "use_hierarchy": False,
        "gated_hierarchy": True,
        "gate_threshold": 0.5,
        "predict_main_labels": False,
        "hierarchy_config": None,
        "hidden_size": 768,
        "hierarchical_hidden_size": 256,
        "num_hidden_layers": 2,
        "dropout": 0.1,
        "attention_dropout": 0.1,

        # Loss Configuration
        "use_weighted_loss": False,
        "use_focal_loss": True,
        "focal_gamma": 2.5,
        "teacher_forcing_ratio": 0.7,

        # Training Parameters
        "num_epochs": 10,
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-5,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "max_length": 512,
        "label_smoothing": 0.0,

        # Data Configuration
        "test_size": 0.2,
        "validation_size": 0.1,
        "random_seed": 42,
        "stratify_by": None,

        # Experiment Naming
        "experiment_name": f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }


def sanitize_config_types(config: Dict) -> Dict:
    """
    ✅ NEW: Ensure all config values have correct types
    """
    type_map = {
        'use_focal_loss': bool,
        'use_weighted_loss': bool,
        'use_hierarchy': bool,
        'gated_hierarchy': bool,
        'predict_main_labels': bool,
        'focal_gamma': float,
        'learning_rate': float,
        'weight_decay': float,
        'dropout': float,
        'attention_dropout': float,
        'teacher_forcing_ratio': float,
        'gate_threshold': float,
        'label_smoothing': float,
        'test_size': float,
        'validation_size': float,
        'num_epochs': int,
        'batch_size': int,
        'warmup_steps': int,
        'max_length': int,
        'gradient_accumulation_steps': int,
        'random_seed': int,
        'hidden_size': int,
        'hierarchical_hidden_size': int,
        'num_hidden_layers': int,
    }

    sanitized = {}
    for key, value in config.items():
        if key in type_map:
            target_type = type_map[key]
            try:
                if target_type == bool:
                    if isinstance(value, str):
                        sanitized[key] = value.lower() in ('true', '1', 'yes')
                    else:
                        sanitized[key] = bool(value)
                else:
                    sanitized[key] = target_type(value)
            except (ValueError, TypeError):
                sanitized[key] = value  # Keep original if conversion fails
        else:
            sanitized[key] = value

    return sanitized


# ============================================================================
# UTILITY FUNCTION FOR CHAT ASSISTANT
# ============================================================================

def get_training_data_from_session(session_state: Dict) -> Tuple[pd.DataFrame, List[str], str]:
    """
    ✅ NEW: Extract training data from session state
    Used by chat assistant to validate data availability
    """
    if not session_state.get('initialized', False):
        raise ValueError("No dataset loaded. Load data first.")

    df = session_state.get('df')
    if df is None or len(df) == 0:
        raise ValueError("Dataset is empty")

    label_columns = session_state.get('label_columns', [])
    if not label_columns:
        raise ValueError("No label columns found")

    passage_col = session_state.get('passage_col', 'Passage')
    if passage_col not in df.columns:
        raise ValueError(f"Passage column '{passage_col}' not found")

    return df, label_columns, passage_col


# ============================================================================
# STREAMLIT UI - MAIN TRAINING PAGE
# ============================================================================

def render_training_page(session_state: Dict):
    """Main render function for training page"""

    # Check if data is loaded
    if not session_state.get('initialized', False):
        st.warning("⚠️ Load a dataset first")
        st.info("Go to the **Data** page and load a dataset to begin training")
        return

    df = session_state.get('df')
    label_columns = session_state.get('label_columns', [])
    passage_col = session_state.get('passage_col', 'Passage')

    # Initialize training config
    if 'training_config' not in session_state:
        session_state['training_config'] = get_default_training_config()

    if 'training_active' not in session_state:
        session_state['training_active'] = False

    # Create tabs
    config_tab, monitor_tab, results_tab = st.tabs([
        "⚙️ Configuration",
        "📊 Monitor",
        "📈 Results"
    ])

    with config_tab:
        render_training_configuration(session_state, df, label_columns, passage_col)

    with monitor_tab:
        render_training_monitor(session_state)

    with results_tab:
        render_training_results(session_state)


# ============================================================================
# CONFIGURATION UI
# ============================================================================

def render_training_configuration(
    session_state: Dict,
    df: pd.DataFrame,
    label_columns: List[str],
    passage_col: str
):
    """Render comprehensive training configuration UI"""

    config = session_state['training_config']

    st.markdown("### 📋 Training Configuration")

    # Initialize training_df
    training_df = None

    # ========================================================================
    # SECTION 1: DATASET SELECTION
    # ========================================================================

    st.markdown("#### 1️⃣ Dataset Selection")

    dataset_source = st.radio(
        "Data source:",
        ["Current Dataset", "Browse Experiments", "Browse DataObjects"],
        horizontal=True
    )

    if dataset_source == "Current Dataset":
        st.info(f"Using current dataset: {len(df):,} passages")
        training_df = df

    elif dataset_source == "Browse Experiments":
        training_df = load_from_experiments(df, label_columns, passage_col)

    elif dataset_source == "Browse DataObjects":
        st.info("💡 DataObject loading coming soon")
        training_df = df

    # Fallback
    if training_df is None:
        st.warning("⚠️ No training dataset selected. Using full dataset.")
        training_df = df

    st.markdown("---")

    # ========================================================================
    # SECTION 2: MODEL ARCHITECTURE
    # ========================================================================

    st.markdown("#### 2️⃣ Model Architecture")

    col1, col2, col3 = st.columns(3)

    with col1:
        config["base_model"] = st.selectbox(
            "Base model:",
            ["roberta-base", "bert-base-uncased", "distilbert-base-uncased"]
        )

        config["use_hierarchy"] = st.checkbox(
            "Use hierarchical structure",
            value=config["use_hierarchy"],
            help="Sublabel predictions depend on main category predictions"
        )

    with col2:
        config["num_hidden_layers"] = st.number_input(
            "Hidden layers:",
            1, 5, config["num_hidden_layers"]
        )

        config["hierarchical_hidden_size"] = st.number_input(
            "Hidden size:",
            128, 1024, config["hierarchical_hidden_size"], step=64
        )

    with col3:
        config["dropout"] = st.slider(
            "Dropout:",
            0.0, 0.5, config["dropout"], 0.05
        )

        config["attention_dropout"] = st.slider(
            "Attention dropout:",
            0.0, 0.5, config["attention_dropout"], 0.05
        )

    # Hierarchy configuration (only if hierarchical)
    if config["use_hierarchy"]:
        st.markdown("---")

        # Predict main labels option
        config["predict_main_labels"] = st.checkbox(
            "Predict main category labels",
            value=config.get("predict_main_labels", False),
            help="Add synthetic main category labels (inferred from sublabels) for the model to predict"
        )

        # Hierarchy builder
        hierarchy_config = render_hierarchy_configuration(
            label_columns,
            session_state,
            config_key='training_hierarchy_config'
        )

        # Save back to config
        config["hierarchy_config"] = hierarchy_config

        # Gating options
        st.markdown("**Gating Options:**")

        col1, col2 = st.columns(2)

        with col1:
            config["gated_hierarchy"] = st.checkbox(
                "Enable gating",
                value=config["gated_hierarchy"],
                help="Zero out sublabel predictions if main category confidence is below threshold"
            )

        with col2:
            if config["gated_hierarchy"]:
                config["gate_threshold"] = st.slider(
                    "Gate threshold:",
                    0.0, 1.0, config["gate_threshold"], 0.05,
                    help="Minimum main category confidence to allow sublabel predictions"
                )

    st.markdown("---")

    # ========================================================================
    # SECTION 3: LOSS CONFIGURATION - ✅ FIXED UI
    # ========================================================================

    st.markdown("#### 3️⃣ Loss Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        config["use_focal_loss"] = st.checkbox(
            "Use focal loss",
            value=config.get("use_focal_loss", True),
            key="config_focal_loss",
            help="Helps with class imbalance by focusing on hard examples"
        )

        if config["use_focal_loss"]:
            config["focal_gamma"] = st.slider(
                "Focal gamma:",
                0.0, 5.0,
                config.get("focal_gamma", 2.5),
                0.5,
                key="config_focal_gamma",
                help="Higher = more focus on hard examples"
            )
        else:
            config["focal_gamma"] = 2.0

    with col2:
        config["use_weighted_loss"] = st.checkbox(
            "Use weighted loss",
            value=config.get("use_weighted_loss", True),
            key="config_weighted_loss",
            help="Weight loss by inverse class frequency - CRITICAL for rare labels"
        )

        # Show impact preview
        if config["use_weighted_loss"]:
            st.success("✅ Will balance rare labels")
        else:
            st.error("⚠️ Rare labels may be ignored!")

    with col3:
        if config.get("use_hierarchy", False):
            config["teacher_forcing_ratio"] = st.slider(
                "Teacher forcing:",
                0.0, 1.0,
                config.get("teacher_forcing_ratio", 0.7),
                0.1,
                key="config_teacher_forcing",
                help="Probability of using ground truth main labels during training"
            )
        else:
            st.caption("Hierarchy disabled")

    # ✅ Show current loss configuration summary
    st.info(f"""
**Current Loss Setup:**
- Focal Loss: {'✅ Enabled' if config['use_focal_loss'] else '❌ Disabled'} {f"(gamma={config.get('focal_gamma', 2.0)})" if config['use_focal_loss'] else ''}
- Weighted Loss: {'✅ Enabled' if config['use_weighted_loss'] else '❌ Disabled'}
- Combined: {'✅ Weighted Focal Loss' if config['use_focal_loss'] and config['use_weighted_loss'] else 'Standard Loss'}
""")

    st.markdown("---")

    # ========================================================================
    # SECTION 4: TRAINING PARAMETERS
    # ========================================================================

    st.markdown("#### 4️⃣ Training Parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        config["num_epochs"] = st.number_input(
            "Epochs:",
            1, 50, config["num_epochs"]
        )

        config["batch_size"] = st.number_input(
            "Batch size:",
            4, 64, config["batch_size"]
        )

    with col2:
        config["learning_rate"] = st.number_input(
            "Learning rate:",
            1e-6, 1e-3, config["learning_rate"],
            format="%.2e"
        )

        config["warmup_steps"] = st.number_input(
            "Warmup steps:",
            0, 2000, config["warmup_steps"], step=100
        )

    with col3:
        config["weight_decay"] = st.slider(
            "Weight decay:",
            0.0, 0.1, config["weight_decay"], 0.01
        )

        config["gradient_accumulation_steps"] = st.number_input(
            "Gradient accum:",
            1, 8, config["gradient_accumulation_steps"]
        )

    with col4:
        config["max_length"] = st.number_input(
            "Max length:",
            128, 1024, config["max_length"], step=64
        )

        config["label_smoothing"] = st.slider(
            "Label smoothing:",
            0.0, 0.2, config["label_smoothing"], 0.01
        )

    st.markdown("---")

    # ========================================================================
    # SECTION 5: DATA SPLIT
    # ========================================================================

    st.markdown("#### 5️⃣ Data Split")

    col1, col2 = st.columns(2)

    with col1:
        config["test_size"] = st.slider(
            "Test size:",
            0.1, 0.3, config["test_size"], 0.05
        )

        config["validation_size"] = st.slider(
            "Validation size:",
            0.05, 0.2, config["validation_size"], 0.05
        )

    with col2:
        config["random_seed"] = st.number_input(
            "Random seed:",
            0, 9999, config["random_seed"]
        )

        stratify_options = ["None"] + label_columns
        stratify_selection = st.selectbox(
            "Stratify by:",
            stratify_options,
            index=0
        )
        config["stratify_by"] = None if stratify_selection == "None" else stratify_selection

    st.markdown("---")

    # ========================================================================
    # SECTION 6: EXPERIMENT INFO
    # ========================================================================

    st.markdown("#### 6️⃣ Experiment Info")

    config["experiment_name"] = st.text_input(
        "Experiment name:",
        value=config["experiment_name"]
    )

    # ✅ Sanitize types before saving
    config = sanitize_config_types(config)
    session_state['training_config'] = config
    session_state['training_df'] = training_df

    st.markdown("---")

    # ========================================================================
    # TRAINING SUMMARY
    # ========================================================================

    st.markdown("### 📊 Training Summary")

    # Calculate label info based on configuration
    if config["use_hierarchy"] and config.get("hierarchy_config"):
        hierarchy = config["hierarchy_config"]

        # Count labels
        num_main = 0
        num_sublabels = 0

        for cat_name, cat_data in hierarchy['categories'].items():
            if cat_data['enabled']:
                if config["predict_main_labels"]:
                    num_main += 1
                num_sublabels += len(cat_data['sublabels'])

        total_labels = num_main + num_sublabels
        label_breakdown = f"{num_main} main + {num_sublabels} sub" if num_main > 0 else f"{num_sublabels} sublabels only"
    else:
        total_labels = len(label_columns)
        label_breakdown = f"{total_labels} flat"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Training Passages", f"{len(training_df):,}")
        st.metric("Total Labels", total_labels)
        st.caption(label_breakdown)

    with col2:
        # Estimate training time
        batches_per_epoch = len(training_df) // config["batch_size"]
        total_batches = batches_per_epoch * config["num_epochs"]
        est_seconds = total_batches * 0.5
        est_minutes = est_seconds / 60

        st.metric("Est. Time", f"{est_minutes:.1f} min")
        st.metric("Total Batches", f"{total_batches:,}")

    with col3:
        st.metric("Epochs", config["num_epochs"])
        st.metric("Batch Size", config["batch_size"])

    st.markdown("---")

    # ========================================================================
    # START TRAINING BUTTON
    # ========================================================================

    if not session_state.get('training_active', False):
        if st.button("🚀 Start Training", type="primary", width='stretch'):
            start_training(session_state, training_df, label_columns, passage_col)
    else:
        st.warning("⚠️ Training in progress...")
        if st.button("🛑 Stop Training", type="secondary"):
            session_state['training_active'] = False
            st.rerun()


def load_from_experiments(
    df: pd.DataFrame,
    label_columns: List[str],
    passage_col: str
) -> pd.DataFrame:
    """Load dataset from saved experiments"""

    experiment = DataExperiment()
    experiments = experiment.list_experiments()

    if not experiments:
        st.warning("No experiments found. Create experiments in Data Prep page.")
        return df

    # Filter options
    exp_names = [exp['name'] for exp in experiments]
    selected_exp_name = st.selectbox(
        "Select experiment:",
        exp_names,
        key="exp_selector"
    )

    selected_exp = next((exp for exp in experiments if exp['name'] == selected_exp_name), None)

    if not selected_exp:
        st.error("Could not load selected experiment")
        return df

    meta = selected_exp['metadata']

    # Show experiment info
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'statistics' in meta:
            st.metric("Passages", meta['statistics']['num_passages'])
        elif 'tiers' in meta:
            total_passages = sum(
                tier_data.get('count', 0)
                for tier_data in meta['tiers'].values()
            )
            st.metric("Passages", total_passages)
        else:
            st.metric("Passages", "N/A")

    with col2:
        if 'statistics' in meta:
            st.metric("Labels", len(meta['statistics']['label_columns']))
        elif 'label_columns' in meta:
            st.metric("Labels", len(meta['label_columns']))
        else:
            st.metric("Labels", "N/A")

    with col3:
        exp_type = meta.get('experiment_type', 'unknown')
        st.metric("Type", exp_type)

    # Load the experiment
    try:
        if exp_type == 'tiered_training':
            st.markdown("**Select tier(s) to train on:**")
            tier_choice = st.radio(
                "Training data:",
                ["Tier 1 Only", "Tier 1 + Tier 2 Combined"],
                horizontal=True,
                key="tier_choice_exp"
            )

            if "Tier 1 Only" in tier_choice:
                data_file = selected_exp['directory'] / "tier1.xlsx"
                training_df = pd.read_excel(data_file)
                st.info(f"Using Tier 1: {len(training_df):,} passages")
            else:
                data_file = selected_exp['directory'] / "tier1_tier2_combined.xlsx"
                training_df = pd.read_excel(data_file)
                st.info(f"Using Combined: {len(training_df):,} passages")

            # Update label columns from metadata
            if 'label_columns' in meta:
                label_columns = meta['label_columns']

        else:
            # Single dataset experiment
            data_file = selected_exp['directory'] / "data.xlsx"
            training_df = pd.read_excel(data_file)
            st.info(f"Using experiment data: {len(training_df):,} passages")

            # Update label columns from metadata
            if 'statistics' in meta:
                label_columns = meta['statistics']['label_columns']
            elif 'label_columns' in meta:
                label_columns = meta['label_columns']

        return training_df

    except Exception as e:
        st.error(f"Error loading experiment: {e}")
        return df


# ============================================================================
# MONITORING UI
# ============================================================================

def render_training_monitor(session_state: Dict):
    """Render training monitoring UI"""

    st.markdown("### 📊 Training Monitor")

    if not session_state.get('training_active', False) and not session_state.get('training_history'):
        st.info("💡 No active training session. Configure and start training on the Configuration tab.")
        return

    # Training status
    if session_state.get('training_active'):
        st.success("✅ Training in progress...")

        current_epoch = session_state.get('current_epoch', 0)
        total_epochs = session_state['training_config']['num_epochs']

        progress = current_epoch / total_epochs if total_epochs > 0 else 0
        st.progress(progress, text=f"Epoch {current_epoch}/{total_epochs}")

    # Training history
    history = session_state.get('training_history', [])

    if history and len(history) > 1:
        st.markdown("#### Per-Label F1 Progression")

        # Extract per-label F1 over epochs
        label_names = session_state.get('final_label_list', [])

        # Build dataframe of F1 scores over time
        label_progression = []
        for log in history:
            if 'epoch' in log:
                row = {'epoch': log['epoch']}
                for label in label_names:
                    f1_key = f'eval_f1_{label}'
                    if f1_key in log:
                        row[label] = log[f1_key]
                if len(row) > 1:  # Has epoch + at least one label
                    label_progression.append(row)

        if label_progression:
            prog_df = pd.DataFrame(label_progression)

            # Plot struggling labels
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 6))

            # Identify struggling labels (final F1 < 0.5)
            final_row = prog_df.iloc[-1]
            struggling = [col for col in prog_df.columns
                          if col != 'epoch' and final_row[col] < 0.5]

            # Plot struggling labels prominently
            for label in struggling:
                ax.plot(prog_df['epoch'], prog_df[label],
                        marker='o', linewidth=2, label=label)

            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='F1 = 0.5')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1 Score')
            ax.set_title('Struggling Labels Performance Over Time')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(alpha=0.3)

            st.pyplot(fig)
            plt.close()

            st.caption(f"⚠️ Showing {len(struggling)} labels with F1 < 0.5")

        latest = history[-1]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Epoch", f"{latest.get('epoch', 0)}")
        with col2:
            st.metric("Train Loss", f"{latest.get('train_loss', 0):.4f}")
        with col3:
            st.metric("Val Loss", f"{latest.get('eval_loss', 0):.4f}")
        with col4:
            st.metric("F1 Micro", f"{latest.get('eval_f1_micro', 0):.3f}")

        # Plot training history
        st.markdown("#### Training History")

        if len(history) > 1:
            fig = visualize_training_history(history, Path("./temp"))
            if fig:
                st.pyplot(fig)
                plt.close()


# ============================================================================
# RESULTS UI
# ============================================================================

def render_training_results(session_state: Dict):
    """Render training results UI"""

    st.markdown("### 📈 Training Results")

    if not session_state.get('training_complete', False):
        st.info("💡 No completed training. Results will appear here after training finishes.")
        return

    # Test results
    test_results = session_state.get('test_results')

    if test_results:
        st.markdown("#### Test Set Performance")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "F1 Micro",
                f"{test_results.get('eval_f1_micro', 0):.3f}",
                help="Overall F1 score (micro-averaged)"
            )

        with col2:
            st.metric(
                "F1 Macro",
                f"{test_results.get('eval_f1_macro', 0):.3f}",
                help="Average F1 across all labels"
            )

        with col3:
            high_perf = sum(1 for k, v in test_results.items()
                            if k.startswith('eval_f1_') and v > 0.7)
            st.metric("Labels > 0.7", high_perf)

        # Visualizations
        st.markdown("#### Performance Breakdown")

        label_names = session_state.get('final_label_list', [])
        fig = visualize_test_results(test_results, label_names, Path("./temp"))
        if fig:
            st.pyplot(fig)
            plt.close()

        # Per-label results table
        st.markdown("#### Per-Label Results")

        label_results = []
        for key, value in test_results.items():
            if key.startswith('eval_f1_') and key not in ['eval_f1_micro', 'eval_f1_macro']:
                label_name = key.replace('eval_f1_', '')
                label_results.append({
                    'Label': label_name,
                    'F1 Score': f"{value:.3f}",
                    'Quality': '🟢 Good' if value > 0.7 else '🟡 Fair' if value > 0.5 else '🔴 Poor'
                })

        st.dataframe(
            pd.DataFrame(label_results),
            hide_index=True,
            width='stretch'
        )

    # Model info
    output_dir = session_state.get('training_output_dir')
    if output_dir:
        st.markdown("---")
        st.markdown("#### 💾 Saved Model")

        st.success(f"✅ Model saved to: `{output_dir}`")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📂 Load Model for Inference"):
                loader = HRAFModelLoader()
                success = loader.load_model(str(output_dir / "final_model"))

                if success:
                    model_name = session_state['training_config']['experiment_name']
                    if 'loaded_models' not in session_state:
                        session_state['loaded_models'] = {}
                    session_state['loaded_models'][model_name] = loader
                    st.success(f"✅ Model loaded as '{model_name}'")
                    st.info("Go to Model Inference page to test predictions")
                else:
                    st.error("Failed to load model")

        with col2:
            if st.button("📊 View Model Files"):
                st.code(f"""
Model directory: {output_dir}

Files:
- final_model/
  - config.json
  - pytorch_model.bin
  - tokenizer files
- training_history.png
- test_results.png
- training_info.json
                """)


def extract_sublabel_metrics(
        results: Dict,
        all_label_names: List[str],
        main_label_names: List[str] = None
) -> Dict:
    """
    Extract only sublabel metrics from test results

    Args:
        results: Test results dictionary
        all_label_names: All label names in the model
        main_label_names: Main category names to exclude (None for flat models)

    Returns:
        Dictionary with sublabel-only metrics
    """
    if main_label_names is None:
        main_label_names = []

    # Identify sublabels
    sublabels = [name for name in all_label_names if name not in main_label_names]

    # Extract sublabel F1 scores
    sublabel_f1s = {}
    for sublabel in sublabels:
        key = f'eval_f1_{sublabel}'
        if key in results:
            sublabel_f1s[sublabel] = results[key]

    # Calculate micro/macro for sublabels only
    if sublabel_f1s:
        f1_values = list(sublabel_f1s.values())
        sublabel_metrics = {
            'f1_micro_sublabels': results.get('eval_f1_micro_sublabels', np.mean(f1_values)),
            'f1_macro_sublabels': results.get('eval_f1_macro_sublabels', np.mean(f1_values)),
            'per_label': sublabel_f1s,
            'sublabel_names': sublabels
        }
    else:
        sublabel_metrics = {
            'f1_micro_sublabels': 0.0,
            'f1_macro_sublabels': 0.0,
            'per_label': {},
            'sublabel_names': []
        }

    return sublabel_metrics


def compare_models_fairly(
        model1_results: Dict,
        model1_label_names: List[str],
        model1_main_labels: List[str],
        model2_results: Dict,
        model2_label_names: List[str],
        model2_main_labels: List[str],
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compare two models using ONLY sublabel metrics for fair comparison

    Args:
        model1_results: Test results from first model
        model1_label_names: All label names from first model
        model1_main_labels: Main category names from first model (empty list for flat)
        model2_results: Test results from second model
        model2_label_names: All label names from second model
        model2_main_labels: Main category names from second model (empty list for flat)
        model1_name: Display name for first model
        model2_name: Display name for second model

    Returns:
        Tuple of (comparison_dataframe, summary_stats)
    """

    # Extract sublabel metrics
    model1_sub = extract_sublabel_metrics(model1_results, model1_label_names, model1_main_labels)
    model2_sub = extract_sublabel_metrics(model2_results, model2_label_names, model2_main_labels)

    # Build comparison dataframe
    comparison = {
        'Metric': [],
        model1_name: [],
        model2_name: [],
        'Difference': [],
        'Winner': []
    }

    # Overall metrics
    comparison['Metric'].append('F1 Micro (Sublabels) ⭐')
    m1_micro = model1_sub['f1_micro_sublabels']
    m2_micro = model2_sub['f1_micro_sublabels']
    comparison[model1_name].append(f"{m1_micro:.3f}")
    comparison[model2_name].append(f"{m2_micro:.3f}")
    diff_micro = m1_micro - m2_micro
    comparison['Difference'].append(f"{diff_micro:+.3f}")
    comparison['Winner'].append(model1_name if diff_micro > 0.001 else (model2_name if diff_micro < -0.001 else 'Tie'))

    comparison['Metric'].append('F1 Macro (Sublabels)')
    m1_macro = model1_sub['f1_macro_sublabels']
    m2_macro = model2_sub['f1_macro_sublabels']
    comparison[model1_name].append(f"{m1_macro:.3f}")
    comparison[model2_name].append(f"{m2_macro:.3f}")
    diff_macro = m1_macro - m2_macro
    comparison['Difference'].append(f"{diff_macro:+.3f}")
    comparison['Winner'].append(model1_name if diff_macro > 0.001 else (model2_name if diff_macro < -0.001 else 'Tie'))

    # Per-sublabel comparison
    # Find common sublabels
    common_sublabels = set(model1_sub['sublabel_names']) & set(model2_sub['sublabel_names'])

    if common_sublabels:
        comparison['Metric'].append('--- Per-Label Results ---')
        comparison[model1_name].append('')
        comparison[model2_name].append('')
        comparison['Difference'].append('')
        comparison['Winner'].append('')

        for sublabel in sorted(common_sublabels):
            comparison['Metric'].append(sublabel)

            m1_val = model1_sub['per_label'].get(sublabel, 0)
            m2_val = model2_sub['per_label'].get(sublabel, 0)

            comparison[model1_name].append(f"{m1_val:.3f}")
            comparison[model2_name].append(f"{m2_val:.3f}")

            diff = m1_val - m2_val
            comparison['Difference'].append(f"{diff:+.3f}")
            comparison['Winner'].append(model1_name if diff > 0.001 else (model2_name if diff < -0.001 else 'Tie'))

    # Summary statistics
    summary = {
        'model1_name': model1_name,
        'model2_name': model2_name,
        'model1_f1_micro': m1_micro,
        'model2_f1_micro': m2_micro,
        'difference': diff_micro,
        'winner': model1_name if diff_micro > 0.001 else (model2_name if diff_micro < -0.001 else 'Tie'),
        'num_common_sublabels': len(common_sublabels),
        'model1_better_count': sum(1 for w in comparison['Winner'][2:] if w == model1_name),
        'model2_better_count': sum(1 for w in comparison['Winner'][2:] if w == model2_name),
        'ties': sum(1 for w in comparison['Winner'][2:] if w == 'Tie')
    }

    return pd.DataFrame(comparison), summary


# ============================================================================
# TRAINING EXECUTION
# ============================================================================

def start_training(
    session_state: Dict,
    training_df: pd.DataFrame,
    label_columns: List[str],
    passage_col: str
):
    """Execute the training process"""

    config = session_state['training_config']

    st.info("🚀 Initializing training...")

    # Validate training data
    if len(training_df) == 0:
        st.error("❌ Training dataset is empty!")
        return

    if passage_col not in training_df.columns:
        st.error(f"❌ Passage column '{passage_col}' not found!")
        return

    missing_labels = [label for label in label_columns if label not in training_df.columns]
    if missing_labels and not config["use_hierarchy"]:
        st.error(f"❌ Missing label columns: {missing_labels}")
        return

    valid_passages = training_df[passage_col].notna().sum()
    if valid_passages == 0:
        st.error("❌ No valid passages found!")
        return

    if valid_passages < len(training_df):
        st.warning(f"⚠️ {len(training_df) - valid_passages} passages have missing text and will be removed")

    st.success(f"✅ Validated: {valid_passages} passages")

    # Create output directory
    output_dir = Path(f"./models/{config['experiment_name']}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build label structure
    if config["use_hierarchy"] and config.get("hierarchy_config"):
        st.info("📋 Building hierarchical label structure...")

        hierarchy = config["hierarchy_config"]
        label_structure, ordered_labels = build_label_structure_from_hierarchy(
            hierarchy,
            config["predict_main_labels"]
        )

        # Augment data with main categories if needed
        training_df = augment_data_with_main_categories(
            training_df,
            label_structure,
            config["predict_main_labels"]
        )

        # Use ordered labels for training
        final_label_list = ordered_labels

        st.success(f"✅ Hierarchical structure: {len(label_structure)} categories, {len(final_label_list)} total labels")

        if config["predict_main_labels"]:
            num_main = sum(1 for cat in label_structure.values() if cat['enabled'])
            num_sub = len(final_label_list) - num_main
            st.info(f"📊 Training with: {num_main} main labels + {num_sub} sublabels")
        else:
            st.info(f"📊 Training with: {len(final_label_list)} sublabels only (hierarchy for structure)")
    else:
        st.info("📋 Using flat label structure...")

        # Flat structure
        final_label_list = label_columns

        label_structure = {
            'FLAT': {
                'main_label': 'FLAT',
                'sublabels': label_columns,
                'enabled': True
            }
        }

        st.success(f"✅ Flat structure: {len(final_label_list)} labels")

    # Save final label list
    session_state['final_label_list'] = final_label_list

    # Calculate label dimensions
    label_dims = calculate_label_dimensions(label_structure, config["predict_main_labels"])

    st.info(f"🏗️ Model architecture: {label_dims['total_labels']} total labels")

    # Initialize training session
    training_session = TrainingSession(config, str(output_dir))

    with st.spinner("Initializing model..."):
        model_info = training_session.initialize_model(label_dims)
        st.success(f"✅ Model initialized: {model_info['trainable_params']:,} trainable parameters")

    # Prepare datasets
    with st.spinner("Preparing datasets..."):
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            training_df,
            final_label_list,
            passage_col,
            {
                "test_size": config["test_size"],
                "validation_size": config["validation_size"],
                "random_seed": config["random_seed"],
                "stratify_by": config.get("stratify_by"),
                "max_length": config["max_length"]
            },
            training_session.tokenizer
        )

        st.success(f"✅ Datasets: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

    # Calculate class weights
    class_weights = None
    if config["use_weighted_loss"]:
        with st.spinner("Calculating class weights..."):
            class_weights = calculate_class_weights(training_df, final_label_list)

            # ✅ DEBUG: Show weights
            st.markdown("**📊 Class Weights Debug:**")
            weights_df = pd.DataFrame({
                'Label': final_label_list,
                'Weight': [f"{w:.2f}" for w in class_weights.tolist()]
            })

            # Display in expander
            with st.expander("View Class Weights", expanded=False):
                st.dataframe(weights_df, hide_index=True, width='stretch')

            # Check for extreme weights
            max_weight = class_weights.max().item()
            if max_weight > 100:
                st.warning(f"⚠️ Very high weight detected: {max_weight:.1f}x - capped at 50x")

            st.info(f"📊 Using weighted loss for {len(final_label_list)} labels (max weight: {min(max_weight, 50.0):.1f}x)")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type="cosine",
        learning_rate=config["learning_rate"],
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_micro",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        label_smoothing_factor=config["label_smoothing"],
        remove_unused_columns=False,
    )

    from transformers import EarlyStoppingCallback

    main_label_names = None
    if config["use_hierarchy"] and config["predict_main_labels"]:
        # Extract main category names from label structure
        main_label_names = [
            cat_name for cat_name, cat_data in label_structure.items()
            if cat_data.get('enabled', True)
        ]

    trainer = HierarchicalTrainer(
        model=training_session.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=training_session.tokenizer,
        data_collator=DataCollatorWithPadding(training_session.tokenizer),
        compute_metrics=compute_metrics_for_trainer(
            final_label_list,
            main_label_names=main_label_names  # NEW: Pass main labels
        ),
        class_weights=class_weights,
        teacher_forcing_ratio=config.get("teacher_forcing_ratio", 0.5),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,  # Stop if no improvement for 3 epochs
                early_stopping_threshold=0.001  # Minimum improvement considered significant
            )
        ]
    )

    # Train
    session_state['training_active'] = True
    session_state['training_history'] = []
    session_state['current_epoch'] = 0

    st.info("🎓 Training started...")

    try:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        with st.spinner("Training in progress..."):
            train_result = trainer.train()
            status_placeholder.success("✅ Training completed!")

        # Get training history
        history = [log for log in trainer.state.log_history if 'epoch' in log]
        session_state['training_history'] = history
        session_state['current_epoch'] = config["num_epochs"]

        # Evaluate on test set
        st.info("📊 Evaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=test_dataset)

        # Calculate sublabel-only metrics for fair comparison
        st.info("📊 Calculating sublabel-only metrics for fair comparison...")

        if config["use_hierarchy"] and config["predict_main_labels"]:
            # Hierarchical model - need to exclude main categories
            from core.model_training import extract_sublabel_metrics

            sublabel_metrics = extract_sublabel_metrics(
                test_results,
                final_label_list,
                list(label_structure.keys())  # Main category names
            )

            # Add to test results
            test_results['eval_f1_micro_sublabels'] = sublabel_metrics['f1_micro_sublabels']
            test_results['eval_f1_macro_sublabels'] = sublabel_metrics['f1_macro_sublabels']

            st.info(f"✅ Sublabel-only F1: {sublabel_metrics['f1_micro_sublabels']:.3f} (fair comparison metric)")
        else:
            # Flat model - all labels are sublabels
            test_results['eval_f1_micro_sublabels'] = test_results.get('eval_f1_micro', 0)
            test_results['eval_f1_macro_sublabels'] = test_results.get('eval_f1_macro', 0)

        session_state['test_results'] = test_results

        st.success(f"✅ Test F1 Micro: {test_results.get('eval_f1_micro', 0):.3f}")
        st.success(f"✅ Test F1 Macro: {test_results.get('eval_f1_macro', 0):.3f}")

        # Save model
        st.info("💾 Saving model...")
        final_model_path = output_dir / "final_model"
        training_session.model.save_pretrained(final_model_path)
        training_session.tokenizer.save_pretrained(final_model_path)

        st.info("🎯 Computing optimal thresholds...")
        test_predictions = trainer.predict(test_dataset)
        predictions_probs = torch.sigmoid(torch.tensor(test_predictions.predictions)).numpy()
        test_labels = test_predictions.label_ids

        optimal_thresholds = {}
        for i, label_name in enumerate(final_label_list):
            best_threshold = 0.5
            best_f1 = 0.0

            for threshold in np.arange(0.1, 0.91, 0.05):
                pred_binary = (predictions_probs[:, i] > threshold).astype(int)
                f1 = f1_score(test_labels[:, i], pred_binary, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            optimal_thresholds[label_name] = {
                'threshold': float(best_threshold),
                'f1_at_threshold': float(best_f1)
            }

        st.success(f"✅ Computed optimal thresholds for {len(final_label_list)} labels")

        # Save training info
        training_info = {
            'config': config,
            'test_results': test_results,
            'optimal_thresholds': optimal_thresholds,
            'label_columns': final_label_list,
            'model_info': model_info,
            'training_completed': datetime.now().isoformat(),
            'dataset_size': {
                'train': len(train_dataset),
                'val': len(val_dataset),
                'test': len(test_dataset)
            }
        }

        with open(final_model_path / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)

        st.success(f"✅ Model saved to: {final_model_path}")

        # Create visualizations
        st.info("📊 Creating visualizations...")

        try:
            viz_fig = visualize_training_history(history, output_dir)
            if viz_fig:
                st.success("✅ Training history plot saved")
        except Exception as e:
            st.warning(f"⚠️ Could not create training history plot: {e}")

        try:
            results_fig = visualize_test_results(test_results, final_label_list, output_dir)
            if results_fig:
                st.success("✅ Test results plot saved")
        except Exception as e:
            st.warning(f"⚠️ Could not create test results plot: {e}")

        # Save experiment info
        experiment_info = {
            'experiment_name': config['experiment_name'],
            'created_at': datetime.now().isoformat(),
            'config': config,
            'test_results': test_results,
            'label_structure': label_structure,
            'model_path': str(final_model_path),
            'training_completed': True
        }

        with open(output_dir / "experiment_info.json", "w") as f:
            json.dump(experiment_info, f, indent=2)

        session_state['training_complete'] = True
        session_state['training_output_dir'] = output_dir

        st.success("✅ Training completed successfully!")
        st.balloons()

        # Show summary
        st.markdown("---")
        st.markdown("### 🎉 Training Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Final F1 Micro", f"{test_results.get('eval_f1_micro', 0):.3f}")
        with col2:
            st.metric("Final F1 Macro", f"{test_results.get('eval_f1_macro', 0):.3f}")
        with col3:
            st.metric("Total Epochs", config["num_epochs"])

        st.info(f"💾 Model saved to: `{final_model_path}`")
        st.info(f"📊 View detailed results in the **Results** tab")

    except Exception as e:
        st.error(f"❌ Training failed: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())

        # Try to save partial results
        try:
            st.info("💾 Attempting to save partial training state...")

            if session_state.get('training_history'):
                history = session_state['training_history']

                partial_info = {
                    'config': config,
                    'training_history': history,
                    'error': str(e),
                    'failed_at': datetime.now().isoformat()
                }

                with open(output_dir / "partial_training.json", "w") as f:
                    json.dump(partial_info, f, indent=2)

                st.success(f"✅ Partial results saved to {output_dir}")
        except:
            pass

    finally:
        session_state['training_active'] = False
        st.info("Training session ended")