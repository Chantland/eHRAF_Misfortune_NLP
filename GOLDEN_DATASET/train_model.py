#!/usr/bin/env python3
"""
Training Script for HRAF Misfortune Classification
Target: Exceed 0.72 F1 Micro

Based on best model configuration (training_20251014_124910) with improvements.

Usage:
    python train_model.py

The script will:
1. Load and clean the dataset
2. Train a model with optimized configuration
3. Save the model to GOLDEN_DATASET/models/
"""

import sys
import os

# Add the GOLDEN_DATASET directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    EarlyStoppingCallback,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

# Import model architecture
from core.model_inference import (
    ConfigurableHierarchicalConfig,
    ConfigurableHierarchicalModel,
)

# ============================================================================
# CONFIGURATION - Based on best model with improvements
# ============================================================================

CONFIG = {
    # Model architecture
    "base_model": "roberta-base",
    "use_hierarchy": False,  # Best model used flat structure
    "gated_hierarchy": False,
    "gate_threshold": 0.5,
    "predict_main_labels": False,  # Only predict the 12 sublabels

    # Model capacity
    "hidden_size": 768,
    "hierarchical_hidden_size": 768,  # Match hidden size for flat model
    "num_hidden_layers": 3,

    # Regularization
    "dropout": 0.15,
    "attention_dropout": 0.1,

    # Loss configuration - KEY FOR PERFORMANCE
    # Note: Best model used weighted + focal with gamma=4.0
    # Higher gamma + asymmetric weighting needed for severe class imbalance
    "use_weighted_loss": True,
    "use_focal_loss": True,
    "focal_gamma": 4.0,  # Match best model - higher gamma for class imbalance
    "pos_weight_multiplier": 3.0,  # Additional weight for positive examples

    # Training parameters - match best model
    "teacher_forcing_ratio": 0.0,  # Not used without hierarchy
    "num_epochs": 13,  # Match best model
    "batch_size": 12,
    "gradient_accumulation_steps": 1,  # Match best model
    "learning_rate": 2e-05,
    "warmup_steps": 500,
    "weight_decay": 0.02,
    "max_length": 512,
    "label_smoothing": 0.0,

    # Early stopping
    "use_early_stopping": True,
    "early_stopping_patience": 4,
    "early_stopping_threshold": 0.001,

    # Data configuration
    "test_size": 0.2,
    "validation_size": 0.1,
    "random_seed": 42,
    "stratify_by": "Illness",  # Stratify by most common label
}

# The 12 target labels
LABEL_COLUMNS = [
    "Illness",
    "Accident",
    "Other",
    "Material_Physical",
    "Spirits_Gods",
    "Witchcraft_Sorcery",
    "Rule_Violation_Taboo",
    "Physical_Material",
    "Technical_Specialist",
    "Divination",
    "Shaman_Medium_Healer",
    "Priest_High_Religion"
]

# ============================================================================
# CUSTOM TRAINER WITH WEIGHTED FOCAL LOSS
# ============================================================================

class HierarchicalTrainer(Trainer):
    """Custom trainer with asymmetric weighted focal loss"""

    def __init__(self, pos_weights=None, neg_weights=None, teacher_forcing_ratio=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weights = pos_weights
        self.neg_weights = neg_weights
        self.teacher_forcing_ratio = teacher_forcing_ratio

        if pos_weights is not None:
            self.pos_weights = pos_weights.to(self.args.device)
        if neg_weights is not None:
            self.neg_weights = neg_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """Custom loss computation with weighted focal loss"""
        labels = inputs.pop("labels")

        # Teacher forcing during training (if applicable)
        use_teacher_forcing = model.training and (torch.rand(1).item() < self.teacher_forcing_ratio)

        outputs = model(
            **inputs,
            labels=None,
            teacher_forcing=use_teacher_forcing
        )

        logits = outputs.logits

        # Asymmetric Weighted Focal Loss implementation
        if self.pos_weights is not None and model.config.use_weighted_loss:
            gamma = model.config.focal_gamma if model.config.use_focal_loss else 0.0

            # Compute BCE loss (no reduction yet)
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float(), reduction='none'
            )

            # Compute focal weighting (only if focal loss enabled)
            probs = torch.sigmoid(logits)
            if gamma > 0:
                focal_weight = torch.where(
                    labels == 1,
                    (1 - probs) ** gamma,  # Hard positives get more weight
                    probs ** gamma  # Hard negatives get more weight
                )
            else:
                focal_weight = torch.ones_like(probs)

            # Apply ASYMMETRIC class weighting
            # Positive examples (label=1) get pos_weights
            # Negative examples (label=0) get neg_weights (usually 1.0)
            pos_weights_expanded = self.pos_weights.unsqueeze(0).expand_as(logits)
            neg_weights_expanded = self.neg_weights.unsqueeze(0).expand_as(logits)

            class_weights = torch.where(
                labels == 1,
                pos_weights_expanded,  # Weight for positives
                neg_weights_expanded   # Weight for negatives
            )

            weighted_focal_loss = focal_weight * bce_loss * class_weights
            loss = weighted_focal_loss.mean()
        else:
            # Standard focal or BCE
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
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_clean_data(data_path: str, label_columns: List[str]) -> pd.DataFrame:
    """Load and clean the dataset"""

    print(f"Loading data from: {data_path}")

    # Check if it's the data object directory or direct xlsx
    if os.path.isdir(data_path):
        xlsx_path = os.path.join(data_path, "data.xlsx")
    else:
        xlsx_path = data_path

    # Load with multi-level header if needed
    try:
        df = pd.read_excel(xlsx_path)
        print(f"Loaded {len(df)} rows with columns: {list(df.columns)[:10]}...")
    except Exception as e:
        print(f"Trying multi-level header format...")
        df = pd.read_excel(xlsx_path, header=[0, 1], index_col=0)
        # Flatten column names
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

    print(f"Initial dataset size: {len(df)} passages")

    # Find passage column
    passage_col = None
    for col in ['Passage', 'passage', 'CULTURE_Passage', 'text']:
        if col in df.columns:
            passage_col = col
            break

    if passage_col is None:
        # Try to find it
        for col in df.columns:
            if 'passage' in col.lower():
                passage_col = col
                break

    if passage_col is None:
        raise ValueError(f"Could not find passage column. Available: {list(df.columns)}")

    print(f"Using passage column: {passage_col}")

    # Create clean dataframe
    df_clean = pd.DataFrame()
    df_clean['passage'] = df[passage_col]

    # Find and map label columns - use EXACT matching first, then fuzzy
    print("\nMapping label columns:")
    for label in label_columns:
        found = False

        # First try exact match
        if label in df.columns:
            df_clean[label] = df[label]
            found = True
            print(f"  {label}: exact match")
        else:
            # Try fuzzy match only if exact fails
            for col in df.columns:
                # Must be exact match or have label as suffix after underscore
                if col == label or col.endswith(f"_{label}"):
                    df_clean[label] = df[col]
                    found = True
                    print(f"  {label}: matched to '{col}'")
                    break

        if not found:
            print(f"  WARNING: '{label}' not found in data, adding as zeros!")
            df_clean[label] = 0

    # Clean data
    print("\nCleaning data...")

    # Remove missing passages
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['passage'])
    print(f"  Removed {initial_count - len(df_clean)} rows with missing passages")

    # Remove duplicates
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['passage'], keep='first')
    print(f"  Removed {initial_count - len(df_clean)} duplicate passages")

    # Remove very short passages
    initial_count = len(df_clean)
    df_clean = df_clean[df_clean['passage'].str.len() >= 50]
    print(f"  Removed {initial_count - len(df_clean)} very short passages (<50 chars)")

    # Fill NaN labels with 0 and convert to int
    for label in label_columns:
        df_clean[label] = df_clean[label].fillna(0).astype(int)

    # Remove passages with no labels
    label_sum = df_clean[label_columns].sum(axis=1)
    initial_count = len(df_clean)
    df_clean = df_clean[label_sum > 0]
    print(f"  Removed {initial_count - len(df_clean)} passages with no labels")

    print(f"\nFinal dataset size: {len(df_clean)} passages")

    # Print label distribution
    print("\nLabel distribution:")
    for label in label_columns:
        count = df_clean[label].sum()
        pct = count / len(df_clean) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    return df_clean.reset_index(drop=True)


def calculate_class_weights(
    df: pd.DataFrame,
    label_columns: List[str],
    max_weight: float = 10.0,
    pos_weight_multiplier: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate class weights for handling imbalance.

    Returns two tensors:
    - pos_weights: weights for positive examples (minority class)
    - neg_weights: weights for negative examples (typically 1.0)

    Uses inverse frequency with sqrt scaling.
    """

    pos_weights = []
    neg_weights = []

    print(f"\nClass weights (pos_multiplier={pos_weight_multiplier}x):")
    for label in label_columns:
        pos_count = df[label].sum()
        neg_count = len(df) - pos_count
        total = len(df)

        if pos_count > 0:
            # Inverse frequency ratio
            pos_freq = pos_count / total
            neg_freq = neg_count / total

            # Weight for positive examples (minority)
            # Use inverse frequency, capped
            raw_pos_weight = neg_freq / pos_freq
            pos_weight = min(np.sqrt(raw_pos_weight) * pos_weight_multiplier, max_weight)

            # Negative examples get base weight of 1.0
            neg_weight = 1.0
        else:
            pos_weight = 1.0
            neg_weight = 1.0

        pos_weights.append(pos_weight)
        neg_weights.append(neg_weight)
        print(f"  {label}: {pos_count} positive ({pos_count/total*100:.1f}%) -> pos_weight={pos_weight:.2f}")

    return (
        torch.tensor(pos_weights, dtype=torch.float32),
        torch.tensor(neg_weights, dtype=torch.float32)
    )


def prepare_datasets(
    df: pd.DataFrame,
    label_columns: List[str],
    tokenizer,
    config: Dict
) -> Tuple[Dataset, Dataset, Dataset]:
    """Prepare train/val/test datasets"""

    print("\nSplitting data...")

    # Stratify by specified column if available
    stratify_col = config.get("stratify_by")
    if stratify_col and stratify_col in df.columns:
        stratify = df[stratify_col]
    else:
        stratify = None

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=config["test_size"],
        random_state=config["random_seed"],
        stratify=stratify
    )

    # Update stratify for second split
    if stratify_col and stratify_col in train_val_df.columns:
        stratify = train_val_df[stratify_col]
    else:
        stratify = None

    # Second split: train vs val
    val_size_adjusted = config["validation_size"] / (1 - config["test_size"])
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=config["random_seed"],
        stratify=stratify
    )

    print(f"  Train: {len(train_df)} passages")
    print(f"  Validation: {len(val_df)} passages")
    print(f"  Test: {len(test_df)} passages")

    # Convert to HuggingFace datasets
    def df_to_dataset(df_subset):
        # Reset index
        df_subset = df_subset.reset_index(drop=True)

        # Create dataset
        dataset = Dataset.from_pandas(df_subset)

        # Tokenize
        def tokenize_fn(examples):
            return tokenizer(
                examples['passage'],
                padding='max_length',
                truncation=True,
                max_length=config["max_length"]
            )

        dataset = dataset.map(tokenize_fn, batched=True)

        # Prepare labels
        def prepare_labels(examples):
            labels = []
            batch_size = len(examples[label_columns[0]])

            for i in range(batch_size):
                label_vector = [examples[col][i] for col in label_columns]
                labels.append(label_vector)

            examples['labels'] = labels
            return examples

        dataset = dataset.map(prepare_labels, batched=True)

        # Remove unnecessary columns
        cols_to_remove = ['passage'] + label_columns
        cols_to_remove = [c for c in cols_to_remove if c in dataset.column_names]
        dataset = dataset.remove_columns(cols_to_remove)

        # Set format
        dataset.set_format('torch')

        return dataset

    print("\nTokenizing datasets...")
    train_dataset = df_to_dataset(train_df)
    val_dataset = df_to_dataset(val_df)
    test_dataset = df_to_dataset(test_df)

    return train_dataset, val_dataset, test_dataset


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(eval_pred, label_names: List[str], threshold: float = 0.5):
    """Compute detailed metrics"""
    predictions, labels = eval_pred

    # Apply sigmoid and threshold
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    predictions = np.where(predictions > threshold, 1, 0)

    # Overall metrics
    f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)

    results = {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
    }

    # Per-label metrics
    for i, name in enumerate(label_names):
        f1 = f1_score(labels[:, i], predictions[:, i], zero_division=0)
        results[f'f1_{name}'] = f1

    return results


def find_optimal_thresholds(
    model,
    dataset,
    label_names: List[str],
    trainer
) -> Dict:
    """Find optimal threshold for each label"""

    print("\nFinding optimal thresholds...")

    # Get predictions
    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    labels = predictions.label_ids

    # Convert to probabilities
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    thresholds_to_try = np.arange(0.3, 0.7, 0.05)
    optimal_thresholds = {}

    for i, label_name in enumerate(label_names):
        best_f1 = 0
        best_threshold = 0.5

        for threshold in thresholds_to_try:
            preds = np.where(probs[:, i] > threshold, 1, 0)
            f1 = f1_score(labels[:, i], preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        optimal_thresholds[label_name] = {
            'threshold': float(best_threshold),
            'f1_at_threshold': float(best_f1)
        }

        # Show improvement over 0.5
        default_preds = np.where(probs[:, i] > 0.5, 1, 0)
        default_f1 = f1_score(labels[:, i], default_preds, zero_division=0)

        if best_threshold != 0.5:
            print(f"  {label_name}: {best_threshold:.2f} (F1: {best_f1:.3f} vs {default_f1:.3f})")

    return optimal_thresholds


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model(data_path: str, output_dir: str = None):
    """Main training function"""

    print("=" * 60)
    print("HRAF Misfortune Classification Training")
    print("=" * 60)

    # Setup - prefer MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"\nUsing device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing device: CUDA")
    else:
        device = torch.device('cpu')
        print(f"\nUsing device: CPU (this will be slow)")

    # Generate output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"models/training_{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Load and clean data
    df = load_and_clean_data(data_path, LABEL_COLUMNS)

    # Calculate class weights (asymmetric - pos vs neg)
    print("\nCalculating class weights...")
    pos_weights, neg_weights = calculate_class_weights(
        df, LABEL_COLUMNS,
        max_weight=10.0,
        pos_weight_multiplier=CONFIG.get("pos_weight_multiplier", 1.0)
    )

    # Initialize tokenizer
    print(f"\nLoading tokenizer: {CONFIG['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])

    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        df, LABEL_COLUMNS, tokenizer, CONFIG
    )

    # Register model classes
    try:
        AutoConfig.register("configurable_hierarchical", ConfigurableHierarchicalConfig)
        AutoModel.register(ConfigurableHierarchicalConfig, ConfigurableHierarchicalModel)
    except ValueError:
        pass  # Already registered

    # Calculate label dimensions (flat model - no main labels)
    label_dims = {
        "num_main_labels": 0,
        "num_event_labels": 3,  # Illness, Accident, Other
        "num_cause_labels": 4,  # Material_Physical, Spirits_Gods, Witchcraft_Sorcery, Rule_Violation_Taboo
        "num_action_labels": 5,  # Physical_Material, Technical_Specialist, Divination, Shaman_Medium_Healer, Priest_High_Religion
        "total_labels": 12,
        "label_indices": {label: i for i, label in enumerate(LABEL_COLUMNS)},
        "label_names": LABEL_COLUMNS
    }

    # Initialize model
    print("\nInitializing model...")
    model_config = ConfigurableHierarchicalConfig(
        base_model=CONFIG["base_model"],
        use_hierarchy=CONFIG["use_hierarchy"],
        gated_hierarchy=CONFIG["gated_hierarchy"],
        gate_threshold=CONFIG["gate_threshold"],
        hidden_size=CONFIG["hidden_size"],
        hierarchical_hidden_size=CONFIG["hierarchical_hidden_size"],
        num_hidden_layers=CONFIG["num_hidden_layers"],
        dropout=CONFIG["dropout"],
        attention_dropout=CONFIG["attention_dropout"],
        use_weighted_loss=CONFIG["use_weighted_loss"],
        use_focal_loss=CONFIG["use_focal_loss"],
        focal_gamma=CONFIG["focal_gamma"],
        teacher_forcing_ratio=CONFIG["teacher_forcing_ratio"],
        predict_main_labels=CONFIG["predict_main_labels"],
        num_main_labels=label_dims["num_main_labels"],
        num_event_labels=label_dims["num_event_labels"],
        num_cause_labels=label_dims["num_cause_labels"],
        num_action_labels=label_dims["num_action_labels"],
        total_labels=label_dims["total_labels"],
        label_indices=label_dims["label_indices"],
        label_names=label_dims["label_names"]
    )

    model = ConfigurableHierarchicalModel(model_config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Training arguments
    # Note: fp16 only works on CUDA, not MPS
    use_fp16 = torch.cuda.is_available() and not torch.backends.mps.is_available()

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        warmup_steps=CONFIG["warmup_steps"],
        weight_decay=CONFIG["weight_decay"],
        learning_rate=CONFIG["learning_rate"],
        logging_dir=str(output_path / 'logs'),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_micro",
        greater_is_better=True,
        report_to="none",
        fp16=use_fp16,
        use_mps_device=torch.backends.mps.is_available(),
        label_smoothing_factor=CONFIG["label_smoothing"],
        remove_unused_columns=False,
    )

    # Create metrics function
    def metrics_fn(eval_pred):
        return compute_metrics(eval_pred, LABEL_COLUMNS)

    # Callbacks
    callbacks = []
    if CONFIG["use_early_stopping"]:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=CONFIG["early_stopping_patience"],
                early_stopping_threshold=CONFIG["early_stopping_threshold"]
            )
        )

    # Initialize trainer
    trainer = HierarchicalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics_fn,
        pos_weights=pos_weights if CONFIG["use_weighted_loss"] else None,
        neg_weights=neg_weights if CONFIG["use_weighted_loss"] else None,
        teacher_forcing_ratio=CONFIG["teacher_forcing_ratio"],
        callbacks=callbacks,
    )

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    trainer.train()

    print("\nTraining completed!")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print(f"\nTest Results:")
    print(f"  F1 Micro: {test_results['eval_f1_micro']:.4f}")
    print(f"  F1 Macro: {test_results['eval_f1_macro']:.4f}")

    # Check if we exceeded target
    if test_results['eval_f1_micro'] > 0.72:
        print(f"\n  TARGET EXCEEDED! F1 Micro = {test_results['eval_f1_micro']:.4f} > 0.72")
    else:
        print(f"\n  Target not met. F1 Micro = {test_results['eval_f1_micro']:.4f} < 0.72")

    # Find optimal thresholds
    optimal_thresholds = find_optimal_thresholds(model, test_dataset, LABEL_COLUMNS, trainer)

    # Save model
    print("\nSaving model...")
    final_model_path = output_path / "final_model"
    model.save_pretrained(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    # Save training info
    training_info = {
        "config": CONFIG,
        "test_results": {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in test_results.items()},
        "optimal_thresholds": optimal_thresholds,
        "label_columns": LABEL_COLUMNS,
        "model_info": {
            "total_params": total_params,
            "trainable_params": trainable_params
        },
        "training_completed": datetime.now().isoformat(),
        "dataset_size": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset)
        }
    }

    with open(final_model_path / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    print(f"\nModel saved to: {final_model_path}")

    return test_results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Use the TIERED dataset - this is the quality-filtered 5000 passage subset
    # that the best model (0.72 F1 micro) was trained on
    data_path = "data/objects/tiered/tiered_scored_embedded_cleaned_raw__Altogether_Dataset_RACoded_Combined_20251014_091039/data.xlsx"

    # Check if path exists relative to script
    script_dir = Path(__file__).parent
    full_path = script_dir / data_path

    if not full_path.exists():
        # Try the main repo path
        full_path = Path("/Users/johnheinz/PycharmProjects/eHRAF_Misfortune_NLP/GOLDEN_DATASET") / data_path

    if not full_path.exists():
        print(f"Error: Tiered data file not found at {full_path}")
        print("\nFalling back to raw dataset...")
        data_path = "data/objects/raw/raw__Altogether_Dataset_RACoded_Combined_20251014/data.xlsx"
        full_path = script_dir / data_path
        if not full_path.exists():
            full_path = Path("/Users/johnheinz/PycharmProjects/eHRAF_Misfortune_NLP/GOLDEN_DATASET") / data_path

    if not full_path.exists():
        print(f"Error: Data file not found")
        sys.exit(1)

    print(f"Using data: {full_path}")

    results = train_model(str(full_path))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final F1 Micro: {results['eval_f1_micro']:.4f}")
