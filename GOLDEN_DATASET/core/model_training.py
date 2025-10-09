"""
Model Training Module for HRAF Golden Dataset Discovery
Provides UI and backend for training hierarchical multi-label models
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
from typing import Dict, List, Tuple

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
from sklearn.metrics import f1_score

# Import model architecture from model_inference
from core.model_inference import (
    ConfigurableHierarchicalConfig,
    ConfigurableHierarchicalModel,
    HRAFModelLoader
)

# Import DataExperiment from data_preparation
from core.data_preparation import DataExperiment


class TrainingSession:
    """Manages a model training session"""

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

    def initialize_model(self, label_dims: Dict):
        """Initialize model with configuration"""

        # Register custom model classes
        AutoConfig.register("configurable_hierarchical", ConfigurableHierarchicalConfig)
        AutoModel.register(ConfigurableHierarchicalConfig, ConfigurableHierarchicalModel)

        model_config = ConfigurableHierarchicalConfig(
            base_model=self.config["base_model"],
            use_hierarchy=self.config["use_hierarchy"],
            gated_hierarchy=self.config["gated_hierarchy"],
            gate_threshold=self.config["gate_threshold"],
            hidden_size=self.config["hidden_size"],
            hierarchical_hidden_size=self.config["hierarchical_hidden_size"],
            num_hidden_layers=self.config["num_hidden_layers"],
            dropout=self.config["dropout"],
            attention_dropout=self.config["attention_dropout"],
            use_weighted_loss=self.config["use_weighted_loss"],
            use_focal_loss=self.config["use_focal_loss"],
            focal_gamma=self.config["focal_gamma"],
            teacher_forcing_ratio=self.config["teacher_forcing_ratio"],
            predict_main_labels=self.config["predict_main_labels"],
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
    """Custom trainer with teacher forcing and weighted loss"""

    def __init__(self, class_weights=None, teacher_forcing_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.teacher_forcing_ratio = teacher_forcing_ratio

        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """Custom loss computation with optional weighted loss"""
        labels = inputs.pop("labels")

        # Teacher forcing during training
        use_teacher_forcing = model.training and (torch.rand(1).item() < self.teacher_forcing_ratio)

        outputs = model(
            **inputs,
            labels=labels,
            teacher_forcing=use_teacher_forcing
        )

        # Custom weighted loss if class weights provided
        if self.class_weights is not None and model.config.use_weighted_loss:
            logits = outputs.logits
            weighted_bce = nn.BCEWithLogitsLoss(weight=self.class_weights)
            loss = weighted_bce(logits, labels.float())
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def calculate_label_dimensions(label_structure: Dict, predict_main_labels: bool = True) -> Dict:
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
            if not info.get("enabled", True):
                continue
            dims["num_main_labels"] += 1
            dims["label_indices"][info["main_label"]] = current_idx
            dims["label_names"].append(info["main_label"])
            current_idx += 1

    # Sublabels
    for category, info in label_structure.items():
        if not info.get("enabled", True):
            continue

        for sublabel in info["sublabels"]:
            if category == "EVENT":
                dims["num_event_labels"] += 1
            elif category == "CAUSE":
                dims["num_cause_labels"] += 1
            elif category == "ACTION":
                dims["num_action_labels"] += 1

            dims["label_indices"][sublabel] = current_idx
            dims["label_names"].append(sublabel)
            current_idx += 1

    dims["total_labels"] = current_idx

    return dims

def prepare_datasets(
        df: pd.DataFrame,
        label_columns: List[str],
        passage_col: str,
        data_config: Dict,
        tokenizer
) -> Tuple[Dataset, Dataset, Dataset]:
    """Prepare train/val/test datasets"""

    # IMPORTANT: Only keep columns we need to avoid type conversion issues
    # Keep: passage column, label columns, and optionally ID
    columns_to_keep = [passage_col] + label_columns

    # Add ID if it exists and is useful
    if 'ID' in df.columns:
        try:
            # Try to convert ID to string to avoid type issues
            df['ID'] = df['ID'].astype(str)
            columns_to_keep.append('ID')
        except:
            pass  # Skip ID if conversion fails

    # Filter to only needed columns
    df_clean = df[columns_to_keep].copy()

    # Ensure all label columns are numeric (0/1)
    for label in label_columns:
        df_clean[label] = pd.to_numeric(df_clean[label], errors='coerce').fillna(0).astype(int)

    # Ensure passage column is string
    df_clean[passage_col] = df_clean[passage_col].astype(str)

    # Remove any rows with NaN in passage column
    df_clean = df_clean[df_clean[passage_col].notna()]

    print(f"📊 Cleaned dataset: {len(df_clean)} passages with {len(columns_to_keep)} columns")

    # First split: train+val vs test
    stratify_col = data_config.get("stratify_by")
    stratify_array = df_clean[stratify_col] if stratify_col and stratify_col in df_clean.columns else None

    train_val_df, test_df = train_test_split(
        df_clean,
        test_size=data_config["test_size"],
        random_state=data_config["random_seed"],
        stratify=stratify_array
    )

    # Second split: train vs val
    stratify_array = train_val_df[stratify_col] if stratify_col and stratify_col in train_val_df.columns else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=data_config["validation_size"],
        random_state=data_config["random_seed"],
        stratify=stratify_array
    )

    print(f"📊 Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    # Convert to HuggingFace datasets
    # Reset index to avoid index-related issues
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    # Tokenization function
    def tokenize_function(examples):
        """Tokenize passages"""
        return tokenizer(
            examples[passage_col],
            padding='max_length',
            truncation=True,
            max_length=data_config["max_length"]
        )

    # Prepare labels function
    def prepare_labels(examples, label_columns):
        """Prepare label vectors"""
        labels = []
        batch_size = len(examples[label_columns[0]])

        for i in range(batch_size):
            label_vector = []
            for col in label_columns:
                label_vector.append(int(examples[col][i]))
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

    # Remove unnecessary columns (keep only tokenizer outputs and labels)
    columns_to_remove = [passage_col] + label_columns
    if 'ID' in train_dataset.column_names:
        columns_to_remove.append('ID')

    train_dataset = train_dataset.remove_columns(
        [col for col in columns_to_remove if col in train_dataset.column_names])
    val_dataset = val_dataset.remove_columns([col for col in columns_to_remove if col in val_dataset.column_names])
    test_dataset = test_dataset.remove_columns([col for col in columns_to_remove if col in test_dataset.column_names])

    # Set format for PyTorch
    train_dataset.set_format('torch')
    val_dataset.set_format('torch')
    test_dataset.set_format('torch')

    print("✅ Datasets prepared successfully!")

    return train_dataset, val_dataset, test_dataset


def compute_metrics_for_trainer(label_names):
    """Create metrics computation function for trainer"""

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Apply sigmoid and threshold
        predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
        predictions = np.where(predictions > 0.5, 1, 0)

        # Overall metrics
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)

        # Per-label metrics
        per_label_f1 = {}
        for i, name in enumerate(label_names):
            f1 = f1_score(labels[:, i], predictions[:, i], zero_division=0)
            per_label_f1[f"f1_{name}"] = f1

        return {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            **per_label_f1
        }

    return compute_metrics


def calculate_class_weights(df: pd.DataFrame, label_columns: List[str]) -> torch.Tensor:
    """Calculate class weights for handling imbalance"""
    class_weights = []

    for col in label_columns:
        pos_count = df[col].sum()
        neg_count = len(df) - pos_count

        if pos_count > 0:
            weight = neg_count / pos_count
        else:
            weight = 1.0

        class_weights.append(weight)

    return torch.tensor(class_weights).float()


def visualize_training_history(history: List[Dict], output_dir: Path):
    """Create training history visualizations"""

    if not history:
        return None

    # Ensure output directory exists
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

    # Learning rate (if available)
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

    # Save
    save_path = output_dir / 'training_history.png'
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved training history to {save_path}")
    except Exception as e:
        print(f"⚠️ Could not save training history plot: {e}")

    return fig


def visualize_test_results(test_results: Dict, label_names: List[str], output_dir: Path):
    """Create test results visualizations"""

    # Ensure output directory exists
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

    if label_f1s:  # Only plot if we have per-label scores
        labels = list(label_f1s.keys())
        scores = list(label_f1s.values())

        # Sort by score
        sorted_items = sorted(zip(labels, scores), key=lambda x: x[1])
        labels, scores = zip(*sorted_items) if sorted_items else ([], [])

        # Color based on category
        colors = []
        for label in labels:
            if 'EVENT' in label or label in ['Illness', 'Accident']:
                colors.append('#FF6B6B')
            elif 'CAUSE' in label or label in ['Just_Happens', 'Material_Physical', 'Spirits_Gods',
                                               'Witchcraft_Sorcery', 'Rule_Violation_Taboo']:
                colors.append('#4ECDC4')
            elif 'ACTION' in label or label in ['Physical_Material', 'Technical_Specialist', 'Divination',
                                                'Shaman_Medium_Healer', 'Priest_High_Religion']:
                colors.append('#45B7D1')
            else:
                colors.append('#95A5A6')

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

    # Save
    save_path = output_dir / 'test_results.png'
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved test results to {save_path}")
    except Exception as e:
        print(f"⚠️ Could not save test results plot: {e}")

    return fig

# ============================================================================
# STREAMLIT UI COMPONENTS
# ============================================================================

def render_training_page(session_state: Dict):
    """Render the training page"""

    st.markdown("## 🎓 Train Model")

    # Check if data is loaded
    if not session_state.get('initialized', False):
        st.warning("⚠️ Load a dataset first")
        st.info("Go to the Overview page and load a dataset to begin training")
        return

    df = session_state.get('df')
    label_columns = session_state.get('label_columns', [])
    passage_col = session_state.get('passage_col', 'Passage')

    # Initialize training state
    if 'training_config' not in session_state:
        session_state['training_config'] = get_default_training_config()

    if 'training_active' not in session_state:
        session_state['training_active'] = False

    # Create tabs for configuration and monitoring
    config_tab, monitor_tab, results_tab = st.tabs(["⚙️ Configuration", "📊 Monitor", "📈 Results"])

    with config_tab:
        render_training_configuration(session_state, df, label_columns, passage_col)

    with monitor_tab:
        render_training_monitor(session_state)

    with results_tab:
        render_training_results(session_state)


def get_default_training_config() -> Dict:
    """Get default training configuration"""
    return {
        # Model Architecture
        "base_model": "roberta-base",
        "use_hierarchy": True,
        "gated_hierarchy": True,
        "gate_threshold": 0.5,
        "predict_main_labels": True,
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

        # K-Fold Configuration
        "use_kfold": False,
        "n_splits": 5,

        # Experiment Naming
        "experiment_name": f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }


def render_training_configuration(session_state: Dict, df: pd.DataFrame, label_columns: List[str], passage_col: str):
    """Render training configuration UI"""

    config = session_state['training_config']

    st.markdown("### 📋 Training Configuration")

    # Initialize training_df to avoid UnboundLocalError
    training_df = None

    # Dataset selection
    st.markdown("#### 1️⃣ Dataset Selection")

    dataset_source = st.radio(
        "Data source:",
        ["Full Dataset", "Browse Experiments", "Tiered Datasets"],
        horizontal=True
    )

    if dataset_source == "Full Dataset":
        st.info(f"Using full dataset: {len(df)} passages")
        training_df = df

    elif dataset_source == "Browse Experiments":
        # Browse saved experiments
        experiment = DataExperiment()
        experiments = experiment.list_experiments()

        if not experiments:
            st.warning("No experiments found. Create experiments in Data Prep page.")
            training_df = df  # Fallback to full dataset
            st.info("💡 Using full dataset as fallback")
        else:
            # Filter options
            exp_names = [exp['name'] for exp in experiments]
            selected_exp_name = st.selectbox(
                "Select experiment:",
                exp_names,
                key="exp_selector"
            )

            selected_exp = next((exp for exp in experiments if exp['name'] == selected_exp_name), None)

            if selected_exp:
                meta = selected_exp['metadata']

                # Show experiment info
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Handle both regular and tiered experiment metadata structures
                    if 'statistics' in meta:
                        # Regular experiment
                        st.metric("Passages", meta['statistics']['num_passages'])
                    elif 'tiers' in meta:
                        # Tiered experiment - calculate total from tiers
                        total_passages = sum(
                            tier_data.get('count', 0)
                            for tier_data in meta['tiers'].values()
                        )
                        st.metric("Passages", total_passages)
                    else:
                        st.metric("Passages", "N/A")

                with col2:
                    # Handle labels
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
                            ["Tier 1 Only", "Tier 1 + Tier 2 Combined", "Tier 1 then Tier 2 (Curriculum)"],
                            horizontal=True,
                            key="tier_choice_exp"
                        )

                        if "Tier 1 Only" in tier_choice:
                            data_file = selected_exp['directory'] / "tier1.xlsx"
                            training_df = pd.read_excel(data_file)
                            st.info(f"Using Tier 1: {len(training_df)} passages")
                        elif "Combined" in tier_choice:
                            data_file = selected_exp['directory'] / "tier1_tier2_combined.xlsx"
                            training_df = pd.read_excel(data_file)
                            st.info(f"Using Combined: {len(training_df)} passages")
                        else:
                            st.warning("Curriculum learning not yet implemented")
                            training_df = df  # Fallback
                            st.info("💡 Using full dataset as fallback")

                        # Update label columns from metadata
                        if 'label_columns' in meta:
                            label_columns = meta['label_columns']
                        if 'passage_column' in meta:
                            passage_col = meta['passage_column']

                    else:
                        # Single dataset experiment
                        data_file = selected_exp['directory'] / "data.xlsx"
                        training_df = pd.read_excel(data_file)
                        st.info(f"Using experiment data: {len(training_df)} passages")

                        # Update label columns from metadata
                        if 'statistics' in meta:
                            label_columns = meta['statistics']['label_columns']
                            passage_col = meta['statistics']['passage_column']
                        elif 'label_columns' in meta:
                            label_columns = meta['label_columns']
                            if 'passage_column' in meta:
                                passage_col = meta['passage_column']

                except Exception as e:
                    st.error(f"Error loading experiment: {e}")
                    training_df = df  # Fallback
                    st.info("💡 Using full dataset as fallback")
            else:
                st.error("Could not load selected experiment")
                training_df = df  # Fallback
                st.info("💡 Using full dataset as fallback")

    elif dataset_source == "Tiered Datasets":
        tier1 = session_state.get('tier1_dataset')
        tier2 = session_state.get('tier2_dataset')

        if tier1 is None or tier2 is None:
            st.warning("⚠️ No tiered datasets available. Create tiers first on the Data Prep page.")
            training_df = df  # Fallback
            st.info("💡 Using full dataset as fallback")
        else:
            tier_strategy = st.selectbox(
                "Training strategy:",
                [
                    "Tier 1 Only (High Quality)",
                    "Tier 1 + Tier 2 (Balanced)",
                    "Sequential: Tier 1 then Tier 1+2 (Curriculum)"
                ],
                key="tier_strategy_select"
            )

            if "Tier 1 Only" in tier_strategy:
                training_df = tier1
                st.info(f"Using Tier 1: {len(tier1)} passages")
            elif "Tier 1 + Tier 2" in tier_strategy:
                training_df = pd.concat([tier1, tier2])
                st.info(f"Using Tier 1+2: {len(training_df)} passages")
            else:
                st.warning("Curriculum learning not yet implemented")
                training_df = df  # Fallback
                st.info("💡 Using full dataset as fallback")

    # If training_df is still None (shouldn't happen but safety check)
    if training_df is None:
        st.warning("⚠️ No training dataset selected. Using full dataset.")
        training_df = df

    st.markdown("---")

    # Model Architecture
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

        if config["use_hierarchy"]:
            config["gated_hierarchy"] = st.checkbox(
                "Enable gating",
                value=config["gated_hierarchy"],
                help="Zero out sublabel predictions if main category not predicted"
            )

            if config["gated_hierarchy"]:
                config["gate_threshold"] = st.slider(
                    "Gate threshold:",
                    0.0, 1.0, config["gate_threshold"], 0.05
                )

    with col2:
        config["predict_main_labels"] = st.checkbox(
            "Predict main labels",
            value=config["predict_main_labels"],
            help="Predict EVENT, CAUSE, ACTION in addition to sublabels"
        )

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

    st.markdown("---")

    # Loss Configuration
    st.markdown("#### 3️⃣ Loss Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        config["use_focal_loss"] = st.checkbox(
            "Use focal loss",
            value=config["use_focal_loss"],
            help="Helps with class imbalance by focusing on hard examples"
        )

        if config["use_focal_loss"]:
            config["focal_gamma"] = st.slider(
                "Focal gamma:",
                0.0, 5.0, config["focal_gamma"], 0.5,
                help="Higher = more focus on hard examples"
            )

    with col2:
        config["use_weighted_loss"] = st.checkbox(
            "Use weighted loss",
            value=config["use_weighted_loss"],
            help="Weight loss by inverse class frequency"
        )

    with col3:
        if config["use_hierarchy"]:
            config["teacher_forcing_ratio"] = st.slider(
                "Teacher forcing:",
                0.0, 1.0, config["teacher_forcing_ratio"], 0.1,
                help="Use ground truth main labels during training"
            )

    st.markdown("---")

    # Training Parameters
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

    # Data Split Configuration
    st.markdown("#### 5️⃣ Data Split")

    col1, col2 = st.columns(2)

    with col1:
        config["use_kfold"] = st.checkbox(
            "Use K-fold cross-validation",
            value=config["use_kfold"]
        )

        if config["use_kfold"]:
            config["n_splits"] = st.number_input(
                "Number of folds:",
                2, 10, config["n_splits"]
            )
        else:
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

    # Experiment Name
    st.markdown("#### 6️⃣ Experiment Info")

    config["experiment_name"] = st.text_input(
        "Experiment name:",
        value=config["experiment_name"]
    )

    # Save configuration
    session_state['training_config'] = config
    session_state['training_df'] = training_df

    st.markdown("---")

    # Training summary
    st.markdown("### 📊 Training Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Training Passages", len(training_df))
        st.metric("Labels", len(label_columns))

    with col2:
        # Estimate training time
        batches_per_epoch = len(training_df) // config["batch_size"]
        total_batches = batches_per_epoch * config["num_epochs"]
        est_seconds = total_batches * 0.5  # Rough estimate
        est_minutes = est_seconds / 60

        st.metric("Est. Time", f"{est_minutes:.1f} min")
        st.metric("Total Batches", total_batches)

    with col3:
        st.metric("Epochs", config["num_epochs"])
        st.metric("Batch Size", config["batch_size"])

    # Start training button
    st.markdown("---")

    if not session_state.get('training_active', False):
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            start_training(session_state, training_df, label_columns, passage_col)
    else:
        st.warning("⚠️ Training in progress...")
        if st.button("🛑 Stop Training", type="secondary"):
            session_state['training_active'] = False
            st.rerun()


def render_training_monitor(session_state: Dict):
    """Render training monitoring UI"""

    st.markdown("### 📊 Training Monitor")

    if not session_state.get('training_active', False) and not session_state.get('training_history'):
        st.info("💡 No active training session. Configure and start training on the Configuration tab.")
        return

    # Training status
    if session_state.get('training_active'):
        st.success("✅ Training in progress...")

        # Progress bar
        current_epoch = session_state.get('current_epoch', 0)
        total_epochs = session_state['training_config']['num_epochs']

        progress = current_epoch / total_epochs if total_epochs > 0 else 0
        st.progress(progress, text=f"Epoch {current_epoch}/{total_epochs}")

    # Training history
    history = session_state.get('training_history', [])

    if history:
        st.markdown("#### Recent Metrics")

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
            st.pyplot(fig)
            plt.close()


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
            # Count high-performing labels
            high_perf = sum(1 for k, v in test_results.items()
                            if k.startswith('eval_f1_') and v > 0.7)
            st.metric("Labels > 0.7", high_perf)

        # Visualizations
        st.markdown("#### Performance Breakdown")

        label_names = session_state.get('label_columns', [])
        fig = visualize_test_results(test_results, label_names, Path("./temp"))
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
            use_container_width=True
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
                # Load the trained model
                loader = HRAFModelLoader()
                success = loader.load_model(str(output_dir / "final_model"))

                if success:
                    # Add to loaded models
                    model_name = session_state['training_config']['experiment_name']
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


def augment_labels_with_main_categories(
        df: pd.DataFrame,
        label_columns: List[str],
        predict_main_labels: bool
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add synthetic main category labels if predict_main_labels=True

    Main categories are inferred from sublabels:
    - EVENT = 1 if any of [Illness, Accident, Other] = 1
    - CAUSE = 1 if any of [Just_Happens, Material_Physical, etc.] = 1
    - ACTION = 1 if any of [Physical_Material, Technical_Specialist, etc.] = 1

    Args:
        df: DataFrame with sublabel columns
        label_columns: List of sublabel column names
        predict_main_labels: Whether to add main category labels

    Returns:
        Tuple of (augmented_df, augmented_label_columns)
    """
    if not predict_main_labels:
        return df, label_columns

    df = df.copy()

    # Define sublabel mappings
    event_sublabels = ['Illness', 'Accident', 'Other']
    cause_sublabels = ['Material_Physical', 'Spirits_Gods',
                       'Witchcraft_Sorcery', 'Rule_Violation_Taboo']
    action_sublabels = ['Physical_Material', 'Technical_Specialist', 'Divination',
                        'Shaman_Medium_Healer', 'Priest_High_Religion']

    # Create main category columns (inferred from sublabels)
    event_cols = [col for col in label_columns if col in event_sublabels]
    if event_cols:
        df['EVENT'] = (df[event_cols].sum(axis=1) > 0).astype(int)
    else:
        df['EVENT'] = 0

    cause_cols = [col for col in label_columns if col in cause_sublabels]
    if cause_cols:
        df['CAUSE'] = (df[cause_cols].sum(axis=1) > 0).astype(int)
    else:
        df['CAUSE'] = 0

    action_cols = [col for col in label_columns if col in action_sublabels]
    if action_cols:
        df['ACTION'] = (df[action_cols].sum(axis=1) > 0).astype(int)
    else:
        df['ACTION'] = 0

    # Prepend main categories to label_columns
    main_labels = ['EVENT', 'CAUSE', 'ACTION']
    augmented_label_columns = main_labels + label_columns

    return df, augmented_label_columns

def start_training(session_state: Dict, training_df: pd.DataFrame, label_columns: List[str], passage_col: str):
    """Start the training process"""

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
    if missing_labels:
        st.error(f"❌ Missing label columns: {missing_labels}")
        return

    # Check for valid passages
    valid_passages = training_df[passage_col].notna().sum()
    if valid_passages == 0:
        st.error("❌ No valid passages found!")
        return

    if valid_passages < len(training_df):
        st.warning(f"⚠️ {len(training_df) - valid_passages} passages have missing text and will be removed")

    st.success(f"✅ Validated: {valid_passages} passages, {len(label_columns)} labels")

    # Create output directory
    output_dir = Path(f"./models/{config['experiment_name']}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # AUGMENT LABELS WITH MAIN CATEGORIES IF NEEDED
    st.info("📋 Preparing label structure...")
    training_df, augmented_label_columns = augment_labels_with_main_categories(
        training_df,
        label_columns,
        config["predict_main_labels"]
    )

    if config["predict_main_labels"]:
        st.success(f"✅ Added main category labels: EVENT, CAUSE, ACTION")
        st.info(f"📊 Total labels for training: {len(augmented_label_columns)} (3 main + {len(label_columns)} sub)")
    else:
        st.info(f"📊 Training with {len(augmented_label_columns)} sublabels only")

    # Calculate label dimensions
    label_structure = {
        "EVENT": {
            "main_label": "EVENT",
            "sublabels": [l for l in label_columns if l in ['Illness', 'Accident', 'Other']],
            "enabled": True
        },
        "CAUSE": {
            "main_label": "CAUSE",
            "sublabels": [l for l in label_columns if
                          l in ['Just_Happens', 'Material_Physical', 'Spirits_Gods',
                                'Witchcraft_Sorcery', 'Rule_Violation_Taboo', 'Other.1']],
            "enabled": True
        },
        "ACTION": {
            "main_label": "ACTION",
            "sublabels": [l for l in label_columns if
                          l in ['Physical_Material', 'Technical_Specialist', 'Divination',
                                'Shaman_Medium_Healer', 'Priest_High_Religion', 'Other.2']],
            "enabled": True
        }
    }

    label_dims = calculate_label_dimensions(label_structure, config["predict_main_labels"])

    st.info(f"🏗️ Model will predict {label_dims['total_labels']} labels: "
            f"{label_dims['num_main_labels']} main + "
            f"{label_dims['num_event_labels']} EVENT + "
            f"{label_dims['num_cause_labels']} CAUSE + "
            f"{label_dims['num_action_labels']} ACTION")

    # Initialize training session
    training_session = TrainingSession(config, str(output_dir))

    with st.spinner("Initializing model..."):
        model_info = training_session.initialize_model(label_dims)
        st.success(f"✅ Model initialized: {model_info['trainable_params']:,} trainable parameters")

    # Prepare datasets - USE AUGMENTED COLUMNS
    with st.spinner("Preparing datasets..."):
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            training_df,
            augmented_label_columns,  # USE AUGMENTED
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

        st.success(f"✅ Datasets prepared: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

    # Calculate class weights - USE AUGMENTED COLUMNS
    class_weights = None
    if config["use_weighted_loss"]:
        with st.spinner("Calculating class weights..."):
            class_weights = calculate_class_weights(training_df, augmented_label_columns)
            st.info(f"📊 Using weighted loss for {len(augmented_label_columns)} labels")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        learning_rate=config["learning_rate"],
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_micro",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        label_smoothing_factor=config["label_smoothing"],
        remove_unused_columns=False,
    )

    # Initialize trainer - USE AUGMENTED COLUMNS
    trainer = HierarchicalTrainer(
        model=training_session.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=training_session.tokenizer,
        data_collator=DataCollatorWithPadding(training_session.tokenizer),
        compute_metrics=compute_metrics_for_trainer(augmented_label_columns),  # USE AUGMENTED
        class_weights=class_weights,
        teacher_forcing_ratio=config["teacher_forcing_ratio"],
    )

    # Train
    session_state['training_active'] = True
    session_state['training_history'] = []
    session_state['current_epoch'] = 0

    st.info("🎓 Training started...")

    try:
        # Create progress placeholder
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        with st.spinner("Training in progress..."):
            # Train the model
            train_result = trainer.train()

            status_placeholder.success("✅ Training completed!")

        # Get training history
        history = []
        for log in trainer.state.log_history:
            if 'epoch' in log:
                history.append(log)

        session_state['training_history'] = history
        session_state['current_epoch'] = config["num_epochs"]

        # Evaluate on test set
        st.info("📊 Evaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        session_state['test_results'] = test_results

        # Display test results
        st.success(f"✅ Test F1 Micro: {test_results.get('eval_f1_micro', 0):.3f}")
        st.success(f"✅ Test F1 Macro: {test_results.get('eval_f1_macro', 0):.3f}")

        # Save model
        st.info("💾 Saving model...")
        final_model_path = output_dir / "final_model"
        training_session.model.save_pretrained(final_model_path)
        training_session.tokenizer.save_pretrained(final_model_path)

        # Save training info with augmented label information
        training_info = {
            'config': config,
            'test_results': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                             for k, v in test_results.items()},
            'label_structure': label_structure,
            'label_columns': augmented_label_columns,  # Save augmented columns
            'original_label_columns': label_columns,  # Keep original for reference
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
            results_fig = visualize_test_results(test_results, augmented_label_columns, output_dir)
            if results_fig:
                st.success("✅ Test results plot saved")
        except Exception as e:
            st.warning(f"⚠️ Could not create test results plot: {e}")

        # Save experiment info to parent directory
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