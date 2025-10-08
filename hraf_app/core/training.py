"""
Model training infrastructure
"""

import torch
import torch.nn as nn
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    DataCollatorWithPadding
)
from datasets import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import json

from core.models import HierarchicalModel, HierarchicalConfig


class ModelTrainer:
    """
    Handles model training with hierarchical architecture
    """

    def __init__(
            self,
            label_columns: List[str],
            passage_col: str,
            config: Dict = None
    ):
        self.label_columns = label_columns
        self.passage_col = passage_col
        self.config = config or self._default_config()

        self.model = None
        self.tokenizer = None
        self.trainer = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _default_config(self) -> Dict:
        """Default training configuration"""
        return {
            'base_model': 'roberta-base',
            'num_epochs': 10,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'max_length': 512,
            'test_size': 0.2,
            'val_size': 0.1,
            'random_seed': 42,
            'use_gating': True,
            'use_focal_loss': True,
            'focal_gamma': 2.5,
            'teacher_forcing_ratio': 0.7,
            'dropout': 0.1,
            'num_hidden_layers': 2,
            'hierarchical_hidden_size': 256
        }

    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train model on provided data

        Returns:
            Dict with training results
        """
        print("🎓 Starting training...")

        # Prepare datasets
        train_dataset, val_dataset, test_dataset = self._prepare_datasets(df)

        # Initialize model
        self._initialize_model()

        # Setup trainer
        training_args = self._get_training_args()

        self.trainer = HierarchicalTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self._compute_metrics,
            teacher_forcing_ratio=self.config['teacher_forcing_ratio']
        )

        # Train
        train_result = self.trainer.train()

        # Evaluate on test set
        test_metrics = self.trainer.evaluate(eval_dataset=test_dataset)

        print(f"✅ Training complete! Test F1: {test_metrics['eval_f1_micro']:.3f}")

        return {
            'train_result': train_result,
            'test_metrics': test_metrics,
            'history': self.trainer.state.log_history
        }

    def _prepare_datasets(
            self,
            df: pd.DataFrame
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Prepare train/val/test datasets"""

        # Clean data
        df_clean = df[[self.passage_col] + self.label_columns].copy()
        df_clean = df_clean[df_clean[self.passage_col].notna()]

        for label in self.label_columns:
            df_clean[label] = pd.to_numeric(df_clean[label], errors='coerce').fillna(0).astype(int)

        # Split
        train_val_df, test_df = train_test_split(
            df_clean,
            test_size=self.config['test_size'],
            random_state=self.config['random_seed']
        )

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.config['val_size'],
            random_state=self.config['random_seed']
        )

        print(f"📊 Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

        # Convert to HF datasets
        train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
        val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
        test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

        # Tokenize
        train_dataset = self._tokenize_dataset(train_dataset)
        val_dataset = self._tokenize_dataset(val_dataset)
        test_dataset = self._tokenize_dataset(test_dataset)

        return train_dataset, val_dataset, test_dataset

    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize and prepare dataset"""

        def tokenize_function(examples):
            return self.tokenizer(
                examples[self.passage_col],
                padding='max_length',
                truncation=True,
                max_length=self.config['max_length']
            )

        def prepare_labels(examples):
            labels = []
            batch_size = len(examples[self.label_columns[0]])

            for i in range(batch_size):
                label_vector = [int(examples[col][i]) for col in self.label_columns]
                labels.append(label_vector)

            examples['labels'] = labels
            return examples

        dataset = dataset.map(tokenize_function, batched=True)
        dataset = dataset.map(prepare_labels, batched=True)

        # Remove unnecessary columns
        columns_to_remove = [self.passage_col] + self.label_columns
        dataset = dataset.remove_columns(
            [col for col in columns_to_remove if col in dataset.column_names]
        )

        dataset.set_format('torch')

        return dataset

    def _initialize_model(self):
        """Initialize model and tokenizer"""

        # Infer label structure
        label_structure = self._infer_label_structure()

        # Create config
        model_config = HierarchicalConfig(
            base_model=self.config['base_model'],
            num_main_labels=label_structure['num_main'],
            num_event_labels=label_structure['num_event'],
            num_cause_labels=label_structure['num_cause'],
            num_action_labels=label_structure['num_action'],
            hidden_size=768,  # Standard for base models
            hierarchical_hidden_size=self.config['hierarchical_hidden_size'],
            num_hidden_layers=self.config['num_hidden_layers'],
            dropout=self.config['dropout'],
            use_gating=self.config['use_gating'],
            use_focal_loss=self.config['use_focal_loss'],
            focal_gamma=self.config['focal_gamma']
        )

        # Initialize
        self.model = HierarchicalModel(model_config).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'])

        print(
            f"✅ Model initialized: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable params")

    def _infer_label_structure(self) -> Dict:
        """Infer label structure from column names"""
        structure = {
            'num_main': 0,
            'num_event': 0,
            'num_cause': 0,
            'num_action': 0
        }

        # Count main categories
        main_labels = ['EVENT', 'CAUSE', 'ACTION']
        structure['num_main'] = sum(1 for label in self.label_columns if label in main_labels)

        # Count subcategories
        event_keywords = ['Illness', 'Accident']
        cause_keywords = ['Material_Physical', 'Spirits_Gods', 'Witchcraft', 'Rule_Violation', 'Just_Happens']
        action_keywords = ['Physical_Material', 'Technical_Specialist', 'Divination', 'Shaman', 'Priest']

        for label in self.label_columns:
            if any(kw in label for kw in event_keywords):
                structure['num_event'] += 1
            elif any(kw in label for kw in cause_keywords):
                structure['num_cause'] += 1
            elif any(kw in label for kw in action_keywords):
                structure['num_action'] += 1

        return structure

    def _get_training_args(self) -> TrainingArguments:
        """Get training arguments"""
        return TrainingArguments(
            output_dir='./models/training_temp',
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            learning_rate=self.config['learning_rate'],
            warmup_steps=self.config['warmup_steps'],
            weight_decay=self.config['weight_decay'],
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_micro",
            greater_is_better=True,
            report_to="none",
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False
        )

    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred

        # Apply sigmoid and threshold
        predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
        predictions = np.where(predictions > 0.5, 1, 0)

        # Compute metrics
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)

        # Per-label F1
        per_label_f1 = {}
        for i, label in enumerate(self.label_columns):
            f1 = f1_score(labels[:, i], predictions[:, i], zero_division=0)
            per_label_f1[f"f1_{label}"] = f1

        return {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            **per_label_f1
        }

    def predict_passage(self, text: str, threshold: float = 0.5) -> Dict:
        """Predict labels for a single passage"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config['max_length']
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        predictions = {}
        for i, label in enumerate(self.label_columns):
            predictions[label] = probs[i] > threshold

        return {
            'predictions': predictions,
            'probabilities': {label: float(probs[i]) for i, label in enumerate(self.label_columns)},
            'predicted_labels': [k for k, v in predictions.items() if v]
        }


class HierarchicalTrainer(Trainer):
    """Custom trainer with teacher forcing"""

    def __init__(self, teacher_forcing_ratio=0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss with teacher forcing"""
        labels = inputs.pop("labels")

        # Use teacher forcing during training
        use_teacher_forcing = model.training and (torch.rand(1).item() < self.teacher_forcing_ratio)

        outputs = model(
            **inputs,
            labels=labels,
            teacher_forcing=use_teacher_forcing
        )

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss