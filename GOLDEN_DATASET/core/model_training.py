"""
Model Training Module for HRAF Golden Dataset Discovery
Comprehensive training system with FIXED weighted focal loss implementation
"""
import hashlib

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
    TrainerCallback
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

class CurriculumLearningCallback(TrainerCallback):
    """Switches dataset mid-training for curriculum learning"""

    def __init__(self, switch_epoch, tier1_dataset, combined_dataset):
        self.switch_epoch = switch_epoch
        self.tier1_dataset = tier1_dataset
        self.combined_dataset = combined_dataset
        self.switched = False
        self.trainer = None  # ✅ Will be set after trainer creation

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Check if we should switch datasets"""
        current_epoch = int(state.epoch) if state.epoch else 0

        if current_epoch >= self.switch_epoch and not self.switched:
            print(f"\n🎓 CURRICULUM SWITCH: Moving from Tier 1 to Combined dataset (epoch {current_epoch})")
            self.switched = True

            # ✅ FIX: Access trainer from callback's stored reference
            if self.trainer is not None:
                self.trainer.train_dataset = self.combined_dataset
                print(f"✅ Switched to combined dataset: {len(self.combined_dataset)} examples")
            else:
                print("⚠️ Warning: Trainer reference not set in callback")

            return control


class CurriculumEarlyStoppingCallback(TrainerCallback):
    """Early stopping that pauses during curriculum transitions"""

    def __init__(self, patience=3, threshold=0.001, curriculum_callback=None):
        self.patience = patience
        self.threshold = threshold
        self.curriculum_callback = curriculum_callback
        self.best_metric = None
        self.epochs_without_improvement = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_metric = metrics.get("eval_f1_micro", 0)

        # Don't apply early stopping right after curriculum switch
        if self.curriculum_callback and self.curriculum_callback.switched:
            switch_epoch = self.curriculum_callback.switch_epoch
            current_epoch = int(state.epoch) if state.epoch else 0

            # Give model 3 epochs to adjust after switch
            if current_epoch < switch_epoch + 3:
                return control

        # Standard early stopping logic
        if self.best_metric is None:
            self.best_metric = current_metric
        elif current_metric > self.best_metric + self.threshold:
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            control.should_training_stop = True

        return control

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
    """Standard HRAF misfortune classification hierarchy template - COMPLETE"""
    return {
        'categories': {
            'EVENT': {
                'sublabels': ['Illness', 'Accident', 'Other'],
                'enabled': True
            },
            'CAUSE': {
                'sublabels': [
                    'Material_Physical',
                    'Spirits_Gods',
                    'Witchcraft_Sorcery',
                    'Rule_Violation_Taboo'
                ],
                'enabled': True
            },
            'ACTION': {
                'sublabels': [
                    'Physical_Material',
                    'Technical_Specialist',
                    'Divination',
                    'Shaman_Medium_Healer',
                    'Priest_High_Religion',
                ],
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
    """
    Calculate the number of labels for each category

    For FLAT models, infers category from label names
    """

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

    # ✅ FIXED: Sublabel mapping
    for category, info in label_structure.items():
        if not info.get("enabled", True):
            continue

        sublabels = info.get("sublabels", [])

        if len(sublabels) == 0:
            raise ValueError(
                f"❌ Category '{category}' is enabled but has NO sublabels!"
            )

        for sublabel in sublabels:
            # ✅ FIX: Determine category from sublabel NAME, not category name
            target_category = _infer_label_category(sublabel, category)

            if target_category == "EVENT":
                dims["num_event_labels"] += 1
            elif target_category == "CAUSE":
                dims["num_cause_labels"] += 1
            elif target_category == "ACTION":
                dims["num_action_labels"] += 1

            dims["label_indices"][sublabel] = current_idx
            dims["label_names"].append(sublabel)
            current_idx += 1

    dims["total_labels"] = current_idx

    if dims["total_labels"] == 0:
        raise ValueError("No labels configured! Check hierarchy configuration.")

    return dims


def _infer_label_category(label_name: str, category_name: str) -> str:
    """
    Infer whether a label belongs to EVENT, CAUSE, or ACTION

    Checks category name first, then falls back to label name patterns
    """

    label_lower = label_name.lower()
    category_lower = category_name.lower()

    # Priority 1: Explicit category name
    if category_name in ["EVENT", "CAUSE", "ACTION"]:
        return category_name

    if "event" in category_lower:
        return "EVENT"
    elif "cause" in category_lower:
        return "CAUSE"
    elif "action" in category_lower:
        return "ACTION"

    # Priority 2: Known EVENT sublabels
    event_labels = [
        'illness', 'accident', 'disease', 'sickness', 'injury', 'death'
    ]
    if any(pattern in label_lower for pattern in event_labels):
        return "EVENT"

    # Priority 3: Known CAUSE sublabels
    cause_labels = [
        'material_physical', 'spirits_gods', 'witchcraft', 'sorcery',
        'rule_violation', 'taboo', 'just_happens', 'physical', 'spirit',
        'god', 'deity'
    ]
    if any(pattern in label_lower for pattern in cause_labels):
        return "CAUSE"

    # Priority 4: Known ACTION sublabels
    action_labels = [
        'physical_material', 'technical_specialist', 'divination',
        'shaman', 'medium', 'healer', 'priest', 'ritual', 'ceremony',
        'medicine', 'treatment'
    ]
    if any(pattern in label_lower for pattern in action_labels):
        return "ACTION"

    # Priority 5: Pattern matching
    # If label starts with known prefixes
    if label_lower.startswith(('event_', 'illness', 'accident')):
        return "EVENT"
    elif label_lower.startswith(('cause_', 'material', 'spirits', 'witch')):
        return "CAUSE"
    elif label_lower.startswith(('action_', 'physical', 'shaman', 'priest')):
        return "ACTION"

    # Default: ACTION (most common category)
    return "ACTION"

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

    # ✅ FIX: Filter label_columns to only those that exist in df
    existing_label_columns = [col for col in label_columns if col in df.columns]

    missing_labels = [col for col in label_columns if col not in df.columns]
    if missing_labels:
        print(f"⚠️  Warning: {len(missing_labels)} labels not in DataFrame: {missing_labels[:5]}")
        print(f"    This is normal if labels were removed during cleaning")

    # Only keep necessary columns
    columns_to_keep = [passage_col] + existing_label_columns

    # Add optional columns if they exist
    for optional_col in ['ID', 'passage_id']:
        if optional_col in df.columns:
            try:
                if optional_col == 'ID':
                    df[optional_col] = df[optional_col].astype(str)
                columns_to_keep.append(optional_col)
            except:
                pass

    # Filter to needed columns
    df_clean = df[columns_to_keep].copy()

    # ✅ Ensure all label columns are numeric (0/1)
    for label in existing_label_columns:
        df_clean[label] = pd.to_numeric(df_clean[label], errors='coerce').fillna(0).astype(int)

    # Ensure passage column is string
    df_clean[passage_col] = df_clean[passage_col].astype(str)

    # Remove rows with missing passages
    df_clean = df_clean[df_clean[passage_col].notna()]

    print(f"📊 Cleaned dataset: {len(df_clean)} passages with {len(existing_label_columns)} labels")

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

    # Prepare labels - ✅ USE EXISTING LABELS ONLY
    def prepare_labels(examples, label_cols):
        labels = []
        batch_size = len(examples[label_cols[0]])

        for i in range(batch_size):
            label_vector = [int(examples[col][i]) for col in label_cols]
            labels.append(label_vector)

        examples['labels'] = labels
        return examples

    # Apply transformations
    print("🔄 Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    print("🏷️ Preparing labels...")
    train_dataset = train_dataset.map(lambda x: prepare_labels(x, existing_label_columns), batched=True)
    val_dataset = val_dataset.map(lambda x: prepare_labels(x, existing_label_columns), batched=True)
    test_dataset = test_dataset.map(lambda x: prepare_labels(x, existing_label_columns), batched=True)

    # Remove unnecessary columns
    columns_to_remove = [passage_col] + existing_label_columns
    for col in ['ID', 'passage_id']:
        if col in train_dataset.column_names:
            columns_to_remove.append(col)

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
    """Create training history visualizations with FIXED train/eval loss tracking"""

    if not history:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate step-level logs (training) from epoch-level logs (evaluation)
    step_logs = [h for h in history if 'loss' in h and 'eval_loss' not in h]
    epoch_logs = [h for h in history if 'eval_loss' in h]

    if not epoch_logs:
        print("⚠️ No evaluation logs found")
        return None

    # Extract evaluation metrics (these are per-epoch)
    eval_epochs = [h['epoch'] for h in epoch_logs]
    eval_loss = [h['eval_loss'] for h in epoch_logs]
    eval_f1_micro = [h.get('eval_f1_micro', 0) for h in epoch_logs]
    eval_f1_macro = [h.get('eval_f1_macro', 0) for h in epoch_logs]

    # Calculate average training loss per epoch
    train_loss_per_epoch = []
    for eval_epoch in eval_epochs:
        # Get all training step losses for this epoch
        epoch_step_losses = [
            h['loss'] for h in step_logs
            if 'epoch' in h and abs(h['epoch'] - eval_epoch) < 0.5  # Within same epoch
        ]

        if epoch_step_losses:
            avg_loss = np.mean(epoch_step_losses)
            train_loss_per_epoch.append(avg_loss)
        else:
            # Fallback: use last known train loss from epoch log if available
            train_loss_per_epoch.append(epoch_logs[len(train_loss_per_epoch)].get('train_loss', 0))

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ========================================================================
    # PLOT 1: Training and Validation Loss
    # ========================================================================
    ax = axes[0, 0]
    ax.plot(eval_epochs, train_loss_per_epoch, 'b-', label='Train Loss', linewidth=2, marker='o')
    ax.plot(eval_epochs, eval_loss, 'r-', label='Val Loss', linewidth=2, marker='s')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Highlight divergence point (where val loss starts increasing)
    if len(eval_loss) > 1:
        min_val_idx = np.argmin(eval_loss)
        ax.axvline(x=eval_epochs[min_val_idx], color='green', linestyle='--',
                   alpha=0.5, label=f'Best Val Loss (Epoch {eval_epochs[min_val_idx]:.1f})')
        ax.legend(fontsize=9)

    # ========================================================================
    # PLOT 2: F1 Scores
    # ========================================================================
    ax = axes[0, 1]
    ax.plot(eval_epochs, eval_f1_micro, 'g-', label='F1 Micro', linewidth=2, marker='o')
    ax.plot(eval_epochs, eval_f1_macro, 'orange', label='F1 Macro', linewidth=2, marker='s')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('F1 Scores', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])

    # Highlight best F1
    max_f1_idx = np.argmax(eval_f1_micro)
    ax.axvline(x=eval_epochs[max_f1_idx], color='purple', linestyle='--',
               alpha=0.5, label=f'Best F1 (Epoch {eval_epochs[max_f1_idx]:.1f})')
    ax.legend(fontsize=9)

    # ========================================================================
    # PLOT 3: Overfitting Indicator (FIXED)
    # ========================================================================
    ax = axes[1, 0]

    # Calculate loss ratio correctly
    loss_ratio = []
    for train_l, eval_l in zip(train_loss_per_epoch, eval_loss):
        if train_l > 0.001:  # Avoid division by zero
            ratio = eval_l / train_l
        else:
            ratio = 1.0
        loss_ratio.append(ratio)

    ax.plot(eval_epochs, loss_ratio, color='purple', linewidth=2, marker='d')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5,
               label='No overfitting (ratio=1)')
    ax.fill_between(eval_epochs, 1.0, max(loss_ratio) + 0.1, alpha=0.1, color='red',
                    label='Overfitting zone')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Eval Loss / Train Loss', fontsize=11)
    ax.set_title('Overfitting Indicator (>1 = overfitting)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Add annotations for severe overfitting
    for i, (epoch, ratio) in enumerate(zip(eval_epochs, loss_ratio)):
        if ratio > 1.5:  # Significant overfitting
            ax.annotate(f'{ratio:.2f}',
                        xy=(epoch, ratio),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    # ========================================================================
    # PLOT 4: Learning Rate Schedule
    # ========================================================================
    ax = axes[1, 1]

    # Extract learning rates from step logs
    lr_epochs = []
    learning_rates = []

    for h in step_logs:
        if 'learning_rate' in h and 'epoch' in h:
            lr_epochs.append(h['epoch'])
            learning_rates.append(h['learning_rate'])

    if learning_rates:
        ax.plot(lr_epochs, learning_rates, color='brown', linewidth=1, alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Learning Rate', fontsize=11)
        ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    else:
        ax.text(0.5, 0.5, 'Learning rate data not available',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

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

        "use_early_stopping": True,
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.001,

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
    # SECTION 1: DATASET SELECTION - FIXED WITH TIER DETECTION
    # ========================================================================

    st.markdown("#### 1️⃣ Dataset Selection")

    dataset_source = st.radio(
        "Data source:",
        ["Current Dataset", "Browse Tiered Data", "Browse Experiments"],
        horizontal=True
    )

    if dataset_source == "Current Dataset":
        st.info(f"Using current dataset: {len(df):,} passages")
        training_df = df

    elif dataset_source == "Browse Tiered Data":
        training_df = load_from_tiered_objects(df, label_columns, passage_col)

    # Fallback
    if training_df is None:
        st.warning("⚠️ No training dataset selected. Using full dataset.")
        training_df = df

    st.markdown("---")

    # ========================================================================
    # VISUAL CONFIRMATION OF TRAINING DATA
    # ========================================================================

    st.markdown("---")
    st.markdown("#### ✅ Training Data Confirmation")

    with st.expander("📊 Verify Training Data", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Passages", f"{len(training_df):,}")

        with col2:
            st.metric("Total Labels", len(label_columns))

        with col3:
            # Check which columns exist
            has_tier_cols = 'confidence_composite' in training_df.columns
            if has_tier_cols:
                st.metric("Data Type", "✅ Tiered")
                avg_confidence = training_df['confidence_composite'].mean()
                st.caption(f"Avg confidence: {avg_confidence:.3f}")
            else:
                st.metric("Data Type", "⚠️ Full Dataset")
                st.caption("No quality scores")

        with col4:
            # Count passages per label
            avg_labels = training_df[label_columns].sum(axis=1).mean()
            st.metric("Avg Labels/Passage", f"{avg_labels:.1f}")

        # Show label distribution in training data
        st.markdown("**Label Distribution in Training Data:**")

        label_dist = []
        for label in label_columns[:10]:  # Show first 10
            if label in training_df.columns:
                count = int((training_df[label] == 1).sum())
                pct = (count / len(training_df)) * 100
                label_dist.append({
                    'Label': label,
                    'Count': count,
                    'Percentage': f"{pct:.1f}%"
                })

        if label_dist:
            st.dataframe(
                pd.DataFrame(label_dist),
                hide_index=True,
                width='stretch'
            )

        # Sample passage preview
        st.markdown("**Sample Passage from Training Data:**")
        sample_idx = training_df.index[0]
        sample_text = str(training_df.loc[sample_idx, passage_col])[:200]
        st.caption(f"{sample_text}...")

        # Show which file was loaded (if from tiered data)
        if dataset_source == "Browse Tiered Data":
            st.success(f"✅ **Loaded from tiered dataset**")
            if has_tier_cols:
                st.caption("Contains quality scores - this is tiered data")
        elif dataset_source == "Browse Experiments":
            st.info(f"ℹ️ **Loaded from experiment**")
        else:
            st.warning(f"⚠️ **Using full dataset** - not quality-stratified")

    st.markdown("---")

    # Store for start_training to use
    session_state['training_working_data'] = {
        'df': training_df,
        'label_columns': label_columns,
        'passage_col': passage_col
    }

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
            "Classifier hidden layers:",
            1, 5, config["num_hidden_layers"]
        )

        config["hierarchical_hidden_size"] = st.number_input(
            "Classifier Hidden size:",
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
    # SECTION 5: EARLY STOPPING (NEW)
    # ========================================================================

    st.markdown("#### 5️⃣ Early Stopping")

    st.info("🛑 Stop training automatically when validation performance stops improving")

    col1, col2, col3 = st.columns(3)

    with col1:
        config["use_early_stopping"] = st.checkbox(
            "Enable early stopping",
            value=config.get("use_early_stopping", True),
            help="Automatically stop training when validation F1 stops improving"
        )

    with col2:
        if config["use_early_stopping"]:
            config["early_stopping_patience"] = st.number_input(
                "Patience (epochs):",
                1, 10,
                config.get("early_stopping_patience", 3),
                help="Stop after N epochs without improvement"
            )
        else:
            st.caption("Early stopping disabled")
            config["early_stopping_patience"] = 3

    with col3:
        if config["use_early_stopping"]:
            config["early_stopping_threshold"] = st.number_input(
                "Min improvement:",
                0.0001, 0.01,
                config.get("early_stopping_threshold", 0.001),
                format="%.4f",
                help="Minimum F1 improvement to count as better"
            )
        else:
            st.caption("Early stopping disabled")
            config["early_stopping_threshold"] = 0.001

    # Show what will happen
    if config["use_early_stopping"]:
        st.success(
            f"✅ Will stop training after {config['early_stopping_patience']} epochs "
            f"without ≥{config['early_stopping_threshold']:.4f} improvement in validation F1"
        )
    else:
        st.warning(
            f"⚠️ Will train for full {config['num_epochs']} epochs "
            "(may lead to overfitting)"
        )

    st.markdown("---")

    # ========================================================================
    # SECTION 6: DATA SPLIT
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
    # SECTION :7 EXPERIMENT INFO
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
    # OPTIONAL DIAGNOSTICS - JSON Config Dump
    # ========================================================================

    with st.expander("🔍 View Configuration (JSON)", expanded=False):

        diagnostic_data = {
            'training_config': config,
            'data_info': {
                'num_passages': len(training_df),
                'num_labels': len(label_columns),
                'label_columns': label_columns,
                'passage_column': passage_col,
                'has_tier_data': 'confidence_composite' in training_df.columns,
                'data_shape': list(training_df.shape)
            },
            'label_distribution': {
                label: {
                    'count': int((training_df[label] == 1).sum()) if label in training_df.columns else 0,
                    'percentage': float(
                        (training_df[label] == 1).sum() / len(training_df) * 100) if label in training_df.columns else 0
                }
                for label in label_columns[:20]  # First 20 labels
            }
        }

        st.json(diagnostic_data)

    st.markdown("---")

    # ========================================================================
    # START TRAINING BUTTON
    # ========================================================================

    if not session_state.get('training_active', False):
        if st.button("🚀 Start Training", type="primary", width='stretch'):
            working = session_state['training_working_data']
            start_training(
                session_state,
                working['df'],
                working['label_columns'],
                working['passage_col']
            )
    else:
        st.warning("⚠️ Training in progress...")
        if st.button("🛑 Stop Training", type="secondary", width='stretch'):
            # Set the flag that the monitor callback checks
            st.session_state['request_training_stop'] = True
            st.session_state['training_active'] = False
            st.warning("⚠️ Stop requested - finishing current epoch...")
            st.rerun()


# core/model_training.py - REPLACE existing function

def load_from_tiered_objects(
        df: pd.DataFrame,
        label_columns: List[str],
        passage_col: str
) -> pd.DataFrame:
    """
    Load tiered training data from saved DataObjects with full tracking

    Returns:
        DataFrame with training data (or original df if loading fails)
    """

    from core.data_objects import DataObjectManager, PipelineStage

    manager = DataObjectManager()
    tiered_objects = manager.list_objects(stage=PipelineStage.TIERED)

    if not tiered_objects:
        st.warning("⚠️ No tiered datasets found in ./data/objects/tiered/")
        st.info("💡 Create tiered datasets first on the Data page")
        return df

    # ========================================================================
    # TIER SELECTION
    # ========================================================================

    st.markdown("**📂 Select Tiered Dataset**")

    tiered_names = [obj['name'] for obj in tiered_objects]
    selected_name = st.selectbox(
        "Tiered dataset:",
        tiered_names,
        key="tiered_dataset_selector"
    )

    selected_obj = next(obj for obj in tiered_objects if obj['name'] == selected_name)
    obj_dir = Path(selected_obj['directory'])

    # ========================================================================
    # SHOW TIER STATISTICS
    # ========================================================================

    st.markdown("**📊 Tier Statistics**")

    metadata = selected_obj.get('metadata', {})

    col1, col2, col3 = st.columns(3)

    with col1:
        tier1_size = metadata.get('tier1_size', 'N/A')
        st.metric("Tier 1 (Elite)", tier1_size)
        if tier1_size != 'N/A':
            st.caption("Highest quality")

    with col2:
        tier2_size = metadata.get('tier2_size', 'N/A')
        st.metric("Tier 2 (Expansion)", tier2_size)
        if tier2_size != 'N/A':
            st.caption("Good quality")

    with col3:
        inference_size = metadata.get('inference_size', 'N/A')
        st.metric("Inference (Test)", inference_size)
        if inference_size != 'N/A':
            st.caption("Eval/test set")

    # Show quality scores if available
    if 'tier_configuration' in metadata:
        with st.expander("🎯 Quality Thresholds Used", expanded=False):
            tier_config = metadata['tier_configuration']

            if 'tiers' in tier_config:
                for tier_name, tier_data in tier_config['tiers'].items():
                    if 'quality' in tier_data:
                        st.markdown(f"**{tier_name.title()}**")
                        q = tier_data['quality']
                        st.caption(f"Consistency: {q.get('consistency_mean', 0):.3f}")
                        st.caption(f"Rerank: {q.get('rerank_mean', 0):.3f}")

    st.markdown("---")

    # ========================================================================
    # TRAINING STRATEGY SELECTION
    # ========================================================================

    st.markdown("**🎯 Training Strategy**")

    tier_choice = st.radio(
        "Select data to use:",
        [
            "Tier 1 Only (Highest quality)",
            "Tier 1 + Tier 2 Combined (All quality data)",
            "🎓 Curriculum Learning (Tier 1 → Combined)",
            "Inference Set (Test/eval only)"
        ],
        key="tier_training_strategy",
        help="""
        - **Tier 1 Only**: Train on only the highest quality passages (conservative)
        - **Combined**: Use all quality-filtered data (recommended for most cases)
        - **Curriculum**: Start with Tier 1, then switch to combined mid-training
        - **Inference**: Use this only for final evaluation, not training
        """
    )

    # ========================================================================
    # CURRICULUM LEARNING SETUP
    # ========================================================================

    if "Curriculum" in tier_choice:
        st.info("📚 **Curriculum Learning Strategy**")
        st.caption("Train on elite data first, then expand to all quality data")

        col1, col2 = st.columns(2)

        with col1:
            curriculum_split_epoch = st.number_input(
                "Switch to combined after epoch:",
                min_value=1,
                max_value=30,
                value=5,
                key="curriculum_split_epoch_input",
                help="Train on Tier 1 for N epochs, then switch to Tier 1 + Tier 2"
            )

        with col2:
            total_epochs = st.session_state.get('training_config', {}).get('num_epochs', 10)
            remaining = max(0, total_epochs - curriculum_split_epoch)
            st.metric(
                "Remaining epochs on combined",
                remaining,
                help="After the switch, train on combined for this many epochs"
            )

            if remaining < 5:
                st.warning("⚠️ Consider increasing total epochs for better results")

        # ✅ LOAD BOTH DATASETS
        tier1_path = obj_dir / "tier1.xlsx"
        combined_path = obj_dir / "data.xlsx"  # This is tier1 + tier2

        # Validate files exist
        if not tier1_path.exists():
            st.error(f"❌ Tier 1 file not found: {tier1_path}")
            return df

        if not combined_path.exists():
            st.error(f"❌ Combined file not found: {combined_path}")
            return df

        # Load DataFrames
        try:
            tier1_df = pd.read_excel(tier1_path)
            combined_df = pd.read_excel(combined_path)

            # Validate data
            if len(tier1_df) == 0 or len(combined_df) == 0:
                st.error("❌ Loaded datasets are empty")
                return df

            if passage_col not in tier1_df.columns or passage_col not in combined_df.columns:
                st.error(f"❌ Passage column '{passage_col}' not found in tier data")
                return df

            # ✅ STORE CURRICULUM CONFIG IN SESSION STATE
            st.session_state['curriculum_config'] = {
                'enabled': True,
                'switch_epoch': curriculum_split_epoch,
                'tier1_df': tier1_df,
                'combined_df': combined_df
            }

            # ✅ STORE DATASET METADATA
            st.session_state['last_loaded_dataset'] = {
                'source': 'tiered_object',
                'source_type': 'curriculum',
                'name': selected_name,
                'tier_choice': tier_choice,
                'path': str(obj_dir),
                'tier1_path': str(tier1_path),
                'combined_path': str(combined_path),
                'tier1_size': len(tier1_df),
                'combined_size': len(combined_df),
                'switch_epoch': curriculum_split_epoch,
                'loaded_at': datetime.now().isoformat(),
                'has_quality_scores': 'confidence_composite' in tier1_df.columns
            }

            st.success(f"✅ Loaded Tier 1: {len(tier1_df):,} passages")
            st.success(f"✅ Loaded Combined: {len(combined_df):,} passages")
            st.info(f"🎓 Will switch from Tier 1 to Combined at epoch {curriculum_split_epoch}")

            # Return tier1 as initial training data
            return tier1_df

        except Exception as e:
            st.error(f"❌ Error loading curriculum datasets: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
            return df

    # ========================================================================
    # STANDARD (NON-CURRICULUM) LOADING
    # ========================================================================

    else:
        # Clear any existing curriculum config
        if 'curriculum_config' in st.session_state:
            del st.session_state['curriculum_config']

        # Determine which file to load
        if "Tier 1 Only" in tier_choice:
            file_path = obj_dir / "tier1.xlsx"
            tier_type = "tier1"
        elif "Combined" in tier_choice:
            file_path = obj_dir / "data.xlsx"  # Combined tier1 + tier2
            tier_type = "combined"
        else:  # Inference
            file_path = obj_dir / "inference.xlsx"
            tier_type = "inference"

        # Validate file exists
        if not file_path.exists():
            st.error(f"❌ File not found: {file_path}")
            st.info(f"Expected path: {file_path}")
            return df

        # Load DataFrame
        try:
            training_df = pd.read_excel(file_path)

            # Validate data
            if len(training_df) == 0:
                st.error("❌ Loaded dataset is empty")
                return df

            if passage_col not in training_df.columns:
                st.error(f"❌ Passage column '{passage_col}' not found")
                st.info(f"Available columns: {', '.join(training_df.columns.tolist()[:10])}")
                return df

            # Check for quality scores
            has_quality_scores = 'confidence_composite' in training_df.columns

            # ✅ STORE DATASET METADATA
            st.session_state['last_loaded_dataset'] = {
                'source': 'tiered_object',
                'source_type': tier_type,
                'name': selected_name,
                'tier_choice': tier_choice,
                'path': str(obj_dir),
                'file_path': str(file_path),
                'num_passages': len(training_df),
                'loaded_at': datetime.now().isoformat(),
                'has_quality_scores': has_quality_scores
            }

            # Show success message with details
            st.success(f"✅ Loaded: {len(training_df):,} passages from {file_path.name}")

            if has_quality_scores:
                avg_quality = training_df['confidence_composite'].mean()
                st.info(f"📊 Quality scores present (avg: {avg_quality:.3f})")

            return training_df

        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
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


def get_improved_config_for_rare_labels(base_config: Dict) -> Dict:
    """
    Enhanced config for better rare label performance

    Key changes:
    - Higher focal gamma (more focus on hard examples)
    - Increased dropout (prevent overfitting on common labels)
    - Longer warmup (stabilize rare label learning)
    - Label smoothing (reduce overconfidence)
    """

    improved = base_config.copy()

    # Increase focal loss intensity
    improved['focal_gamma'] = 5.0  # Up from 4.5 - even more focus on hard examples

    # Increase regularization
    improved['dropout'] = 0.20  # Up from 0.15
    improved['attention_dropout'] = 0.15  # Up from 0.1
    improved['weight_decay'] = 0.02  # Up from 0.01

    # Add label smoothing to reduce overconfidence
    improved['label_smoothing'] = 0.05  # Smooth labels slightly

    # Longer warmup for stability
    improved['warmup_steps'] = 800  # Up from 500

    # Smaller batch size for better rare label gradients
    improved['batch_size'] = 8  # Down from 12
    improved['gradient_accumulation_steps'] = 2  # Effective batch = 16

    # More hidden capacity for complex patterns
    improved['num_hidden_layers'] = 4  # Up from 3

    return improved

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

    # ========================================================================
    # VALIDATION
    # ========================================================================

    if len(training_df) == 0:
        st.error("❌ Training dataset is empty!")
        return

    if passage_col not in training_df.columns:
        st.error(f"❌ Passage column '{passage_col}' not found!")
        return

    valid_passages = training_df[passage_col].notna().sum()
    if valid_passages == 0:
        st.error("❌ No valid passages found!")
        return

    st.success(f"✅ Validated: {valid_passages} passages")

    # ✅ CAPTURE DATASET METADATA
    import hashlib

    dataset_metadata = {
        'num_passages': len(training_df),
        'num_labels': len(label_columns),
        'label_columns': label_columns,
        'passage_column': passage_col,

        # Source tracking
        'source_info': session_state.get('last_loaded_dataset', {
            'source': 'unknown',
            'source_type': 'current'
        }),

        # Data characteristics
        'has_quality_scores': 'confidence_composite' in training_df.columns,
        'passage_stats': {
            'mean_length': float(training_df[passage_col].str.len().mean()),
            'median_length': float(training_df[passage_col].str.len().median()),
            'min_length': int(training_df[passage_col].str.len().min()),
            'max_length': int(training_df[passage_col].str.len().max())
        },

        # Label distribution
        'label_distribution': {
            label: {
                'count': int((training_df[label] == 1).sum()),
                'frequency': float((training_df[label] == 1).sum() / len(training_df))
            }
            for label in label_columns
        },

        # Reproducibility
        'dataset_hash': hashlib.md5(
            training_df[passage_col].astype(str).str.cat().encode()
        ).hexdigest()[:16]
    }

    # Add quality score stats if available
    if dataset_metadata['has_quality_scores']:
        dataset_metadata['quality_stats'] = {
            'consistency_mean': float(training_df['confidence_consistency'].mean())
            if 'confidence_consistency' in training_df.columns else None,
            'rerank_mean': float(training_df['confidence_rerank'].mean())
            if 'confidence_rerank' in training_df.columns else None,
            'composite_mean': float(training_df['confidence_composite'].mean())
        }

    # Store for use during training
    session_state['dataset_metadata'] = dataset_metadata

    # ========================================================================
    # SETUP OUTPUT DIRECTORY
    # ========================================================================

    output_dir = Path(f"./models/{config['experiment_name']}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # BUILD LABEL STRUCTURE
    # ========================================================================

    st.info("📋 Building label structure...")

    if config["use_hierarchy"] and config.get("hierarchy_config"):
        hierarchy = config["hierarchy_config"]
        label_structure, ordered_labels = build_label_structure_from_hierarchy(
            hierarchy,
            config["predict_main_labels"]
        )

        training_df = augment_data_with_main_categories(
            training_df,
            label_structure,
            config["predict_main_labels"]
        )

        final_label_list = ordered_labels
        st.success(f"✅ Hierarchical structure: {len(label_structure)} categories, {len(final_label_list)} labels")
    else:
        final_label_list = label_columns

        label_structure = {
            'FLAT': {
                'main_label': 'FLAT',
                'sublabels': label_columns,
                'enabled': True
            }
        }
        st.success(f"✅ Flat structure: {len(final_label_list)} labels")

    session_state['final_label_list'] = final_label_list

    # Calculate label dimensions
    label_dims = calculate_label_dimensions(label_structure, config["predict_main_labels"])
    st.info(f"🏗️ Model architecture: {label_dims['total_labels']} total labels")

    # ========================================================================
    # INITIALIZE MODEL
    # ========================================================================

    training_session = TrainingSession(config, str(output_dir))

    with st.spinner("Initializing model..."):
        model_info = training_session.initialize_model(label_dims)
        st.success(f"✅ Model initialized: {model_info['trainable_params']:,} trainable parameters")

    # ========================================================================
    # PREPARE DATASETS
    # ========================================================================

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

        dataset_metadata['splits'] = {
            'train': len(train_dataset),
            'validation': len(val_dataset),
            'test': len(test_dataset),
            'train_pct': len(train_dataset) / len(training_df) * 100,
            'val_pct': len(val_dataset) / len(training_df) * 100,
            'test_pct': len(test_dataset) / len(training_df) * 100
        }

    # ========================================================================
    # CALCULATE CLASS WEIGHTS
    # ========================================================================

    class_weights = None
    if config["use_weighted_loss"]:
        with st.spinner("Calculating class weights..."):
            class_weights = calculate_class_weights(training_df, final_label_list)
            max_weight = class_weights.max().item()
            st.info(f"📊 Using weighted loss (max weight: {min(max_weight, 100.0):.1f}x)")

    # ========================================================================
    # CREATE TRAINING MONITOR (BEFORE TRAINER!)
    # ========================================================================

    from components.training_monitor import create_training_monitor

    st.markdown("---")
    st.markdown("### 🎯 Training Monitor")

    monitor_callback, monitor_placeholder = create_training_monitor()

    # ========================================================================
    # CREATE TRAINING ARGUMENTS
    # ========================================================================

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
        max_grad_norm=1.0,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        label_smoothing_factor=config["label_smoothing"],
        remove_unused_columns=False,
    )

    # ========================================================================
    # IDENTIFY MAIN LABELS (for metrics)
    # ========================================================================

    main_label_names = None
    if config["use_hierarchy"] and config["predict_main_labels"]:
        main_label_names = [
            cat_name for cat_name, cat_data in label_structure.items()
            if cat_data.get('enabled', True)
        ]

    # ========================================================================
    # CREATE ADDITIONAL CALLBACKS
    # ========================================================================

    # Curriculum learning callback (if enabled)
    curriculum_config = session_state.get('curriculum_config')
    curriculum_callback = None

    if curriculum_config and curriculum_config.get('enabled'):
        st.info("📚 Curriculum Learning Active")

        combined_df = curriculum_config['combined_df']
        necessary_cols = [passage_col] + final_label_list
        existing_necessary = [col for col in necessary_cols if col in combined_df.columns]
        combined_df_clean = combined_df[existing_necessary].copy()
        combined_df_clean[passage_col] = combined_df_clean[passage_col].astype(str)

        for label in final_label_list:
            if label in combined_df_clean.columns:
                combined_df_clean[label] = pd.to_numeric(
                    combined_df_clean[label],
                    errors='coerce'
                ).fillna(0).astype(int)

        def tokenize_function_combined(examples):
            return training_session.tokenizer(
                examples[passage_col],
                padding='max_length',
                truncation=True,
                max_length=config["max_length"]
            )

        def prepare_labels_combined(examples):
            labels = []
            first_label = next(l for l in final_label_list if l in existing_necessary)
            batch_size = len(examples[first_label])

            for i in range(batch_size):
                label_vector = []
                for col in final_label_list:
                    if col in existing_necessary:
                        label_vector.append(int(examples[col][i]))
                    else:
                        label_vector.append(0)
                labels.append(label_vector)

            examples['labels'] = labels
            return examples

        combined_dataset = Dataset.from_pandas(combined_df_clean.reset_index(drop=True))
        combined_dataset = combined_dataset.map(tokenize_function_combined, batched=True)
        combined_dataset = combined_dataset.map(prepare_labels_combined, batched=True)

        cols_to_remove = [col for col in combined_dataset.column_names
                          if col not in ['input_ids', 'attention_mask', 'labels']]

        if cols_to_remove:
            combined_dataset = combined_dataset.remove_columns(cols_to_remove)

        combined_dataset.set_format('torch')

        curriculum_callback = CurriculumLearningCallback(
            switch_epoch=curriculum_config['switch_epoch'],
            tier1_dataset=train_dataset,
            combined_dataset=combined_dataset
        )

        st.success(f"✅ Curriculum: Tier 1 → Combined at epoch {curriculum_config['switch_epoch']}")

    # Early stopping callback
    if config.get("use_early_stopping", True):
        early_stop_callback = CurriculumEarlyStoppingCallback(
            patience=config.get("early_stopping_patience", 3),
            threshold=config.get("early_stopping_threshold", 0.001),
            curriculum_callback=curriculum_callback
        )
    else:
        early_stop_callback = None

    # ========================================================================
    # COMBINE ALL CALLBACKS
    # ========================================================================

    all_callbacks = [monitor_callback]  # Monitor first

    if early_stop_callback:  # Only add if enabled
        all_callbacks.append(early_stop_callback)

    if curriculum_callback:
        all_callbacks.append(curriculum_callback)

    # ========================================================================
    # CREATE TRAINER
    # ========================================================================

    trainer = HierarchicalTrainer(
        model=training_session.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=training_session.tokenizer,
        data_collator=DataCollatorWithPadding(training_session.tokenizer),
        compute_metrics=compute_metrics_for_trainer(
            final_label_list,
            main_label_names=main_label_names
        ),
        class_weights=class_weights,
        teacher_forcing_ratio=config.get("teacher_forcing_ratio", 0.5),
        callbacks=all_callbacks
    )

    # Set trainer reference for curriculum callback
    if curriculum_callback is not None:
        curriculum_callback.trainer = trainer

    # ========================================================================
    # TRAIN WITH MONITORING
    # ========================================================================

    session_state['training_active'] = True
    session_state['training_history'] = []
    session_state['current_epoch'] = 0

    st.info("🎓 Training started with real-time monitoring...")

    try:
        # Training happens here - monitor updates automatically
        train_result = trainer.train()

        # Check if stopped by user
        if session_state.get('training_stopped_by_user'):
            st.warning("⚠️ Training stopped by user")
        else:
            st.success("✅ Training completed!")

        # Get training history
        history = [log for log in trainer.state.log_history if 'epoch' in log]
        session_state['training_history'] = history
        session_state['current_epoch'] = config["num_epochs"]

        # ====================================================================
        # EVALUATE ON TEST SET
        # ====================================================================

        st.info("📊 Evaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=test_dataset)

        # Calculate sublabel-only metrics
        if config["use_hierarchy"] and config["predict_main_labels"]:
            from core.model_training import extract_sublabel_metrics

            sublabel_metrics = extract_sublabel_metrics(
                test_results,
                final_label_list,
                list(label_structure.keys())
            )

            test_results['eval_f1_micro_sublabels'] = sublabel_metrics['f1_micro_sublabels']
            test_results['eval_f1_macro_sublabels'] = sublabel_metrics['f1_macro_sublabels']
        else:
            test_results['eval_f1_micro_sublabels'] = test_results.get('eval_f1_micro', 0)
            test_results['eval_f1_macro_sublabels'] = test_results.get('eval_f1_macro', 0)

        session_state['test_results'] = test_results

        st.success(f"✅ Test F1 Micro: {test_results.get('eval_f1_micro', 0):.3f}")

        # ====================================================================
        # SAVE MODEL
        # ====================================================================

        st.info("💾 Saving model...")
        final_model_path = output_dir / "final_model"
        training_session.model.save_pretrained(final_model_path)
        training_session.tokenizer.save_pretrained(final_model_path)

        # Compute optimal thresholds
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
            visualize_training_history(history, output_dir)
            visualize_test_results(test_results, final_label_list, output_dir)
            st.success("✅ Visualizations saved")
        except Exception as e:
            st.warning(f"⚠️ Could not create visualizations: {e}")

        # Save experiment info
        experiment_info = {
            'experiment_name': config['experiment_name'],
            'created_at': datetime.now().isoformat(),
            'config': config,
            'test_results': test_results,
            'label_structure': label_structure,
            'model_path': str(final_model_path),
            'training_completed': True,

            # ✅ COMPLETE DATASET TRACKING
            'dataset': dataset_metadata,

            # ✅ CURRICULUM INFO
            'curriculum_learning': session_state.get('curriculum_config', {
                'enabled': False
            }) if session_state.get('curriculum_config') else {'enabled': False}
        }

        with open(output_dir / "experiment_info.json", "w") as f:
            json.dump(experiment_info, f, indent=2)

        session_state['training_complete'] = True
        session_state['training_output_dir'] = output_dir
        session_state['training_just_completed'] = True  # ← ADD THIS LINE
        st.success("✅ Training completed successfully!")
        st.balloons()

    except Exception as e:
        st.error(f"❌ Training failed: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())

    finally:
        session_state['training_active'] = False
        st.info("Training session ended")