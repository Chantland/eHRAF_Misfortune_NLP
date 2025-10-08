"""
Model inference utilities
Load trained models and make predictions on new data
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoConfig, AutoModel
from tqdm import tqdm

from core.models import HierarchicalModel, HierarchicalConfig


class ModelInference:
    """
    Load and run inference with trained models
    """

    def __init__(self, model_path: str):
        """
        Initialize inference

        Args:
            model_path: Path to saved model directory
        """
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.tokenizer = None
        self.config = None
        self.label_names = None
        self.optimal_thresholds = None

        self._load_model()

    def _load_model(self):
        """Load model, tokenizer, and metadata"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        print(f"Loading model from {self.model_path}...")

        # Register custom model if needed
        try:
            AutoConfig.register("hierarchical_multilabel", HierarchicalConfig)
            AutoModel.register(HierarchicalConfig, HierarchicalModel)
        except:
            pass  # Already registered

        # Load model
        try:
            self.model = HierarchicalModel.from_pretrained(str(self.model_path))
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        # Load metadata
        self._load_metadata()

    def _load_metadata(self):
        """Load training metadata and optimal thresholds"""
        # Try training_info.json
        info_file = self.model_path / "training_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                info = json.load(f)

                self.optimal_thresholds = info.get('optimal_thresholds', {})

                # Extract label names from label_structure
                if 'label_structure' in info:
                    self.label_names = self._extract_label_names(info['label_structure'])

        # Try to get from model config
        if self.label_names is None and hasattr(self.model.config, 'label_names'):
            self.label_names = self.model.config.label_names

        # Fallback: generate generic names
        if self.label_names is None:
            num_labels = self.model.config.num_main_labels
            num_labels += self.model.config.num_event_labels
            num_labels += self.model.config.num_cause_labels
            num_labels += self.model.config.num_action_labels

            self.label_names = [f"Label_{i}" for i in range(num_labels)]

        print(f"✅ Loaded metadata: {len(self.label_names)} labels")

    def _extract_label_names(self, label_structure: Dict) -> List[str]:
        """Extract ordered label names from structure"""
        names = []

        # Main labels first
        for category in ['EVENT', 'CAUSE', 'ACTION']:
            if category in label_structure:
                info = label_structure[category]
                if info.get('enabled', True):
                    names.append(info.get('main_label', category))

        # Then sublabels
        for category in ['EVENT', 'CAUSE', 'ACTION']:
            if category in label_structure:
                info = label_structure[category]
                if info.get('enabled', True):
                    names.extend(info.get('sublabels', []))

        return names

    def predict_single(
            self,
            text: str,
            use_optimal_thresholds: bool = True,
            default_threshold: float = 0.5,
            return_probabilities: bool = True
    ) -> Dict:
        """
        Predict labels for a single passage

        Args:
            text: Passage text
            use_optimal_thresholds: Use per-label optimal thresholds
            default_threshold: Default threshold if optimal not available
            return_probabilities: Include probability scores

        Returns:
            Dict with predictions and optionally probabilities
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Apply thresholds
        predictions = {}
        probabilities = {}

        for i, label in enumerate(self.label_names):
            if i >= len(probs):
                break

            prob = float(probs[i])

            # Get threshold
            if use_optimal_thresholds and self.optimal_thresholds and label in self.optimal_thresholds:
                threshold = self.optimal_thresholds[label].get('threshold', default_threshold)
            else:
                threshold = default_threshold

            predictions[label] = prob > threshold

            if return_probabilities:
                probabilities[label] = prob

        result = {
            'predictions': predictions,
            'predicted_labels': [k for k, v in predictions.items() if v]
        }

        if return_probabilities:
            result['probabilities'] = probabilities

        return result

    def predict_batch(
            self,
            texts: List[str],
            batch_size: int = 16,
            use_optimal_thresholds: bool = True,
            default_threshold: float = 0.5,
            show_progress: bool = True
    ) -> List[Dict]:
        """
        Predict labels for multiple passages

        Args:
            texts: List of passage texts
            batch_size: Batch size for processing
            use_optimal_thresholds: Use per-label thresholds
            default_threshold: Default threshold
            show_progress: Show progress bar

        Returns:
            List of prediction dictionaries
        """
        results = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting")

        for i in iterator:
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy()

            # Process each prediction
            for j in range(len(batch_texts)):
                predictions = {}
                probabilities = {}

                for k, label in enumerate(self.label_names):
                    if k >= probs.shape[1]:
                        break

                    prob = float(probs[j, k])

                    # Get threshold
                    if use_optimal_thresholds and self.optimal_thresholds and label in self.optimal_thresholds:
                        threshold = self.optimal_thresholds[label].get('threshold', default_threshold)
                    else:
                        threshold = default_threshold

                    predictions[label] = prob > threshold
                    probabilities[label] = prob

                results.append({
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'predicted_labels': [k for k, v in predictions.items() if v]
                })

        return results

    def predict_dataframe(
            self,
            df: pd.DataFrame,
            text_column: str,
            batch_size: int = 16,
            add_to_df: bool = True
    ) -> pd.DataFrame:
        """
        Predict labels for all passages in a DataFrame

        Args:
            df: DataFrame with passages
            text_column: Name of text column
            batch_size: Batch size
            add_to_df: Add predictions as new columns

        Returns:
            DataFrame with predictions (original + predictions if add_to_df)
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found")

        print(f"Predicting labels for {len(df)} passages...")

        texts = df[text_column].tolist()
        results = self.predict_batch(texts, batch_size=batch_size)

        if add_to_df:
            df_result = df.copy()

            # Add prediction columns
            for label in self.label_names:
                pred_col = f"pred_{label}"
                prob_col = f"prob_{label}"

                df_result[pred_col] = [
                    int(r['predictions'].get(label, False))
                    for r in results
                ]

                df_result[prob_col] = [
                    r['probabilities'].get(label, 0.0)
                    for r in results
                ]
        else:
            # Create new DataFrame with just predictions
            df_result = pd.DataFrame(results)

        print("✅ Predictions complete!")

        return df_result

    def evaluate_on_labeled_data(
            self,
            df: pd.DataFrame,
            text_column: str,
            label_columns: List[str],
            batch_size: int = 16
    ) -> Dict:
        """
        Evaluate model on labeled data

        Args:
            df: DataFrame with passages and labels
            text_column: Text column name
            label_columns: Label column names
            batch_size: Batch size

        Returns:
            Dict with evaluation metrics
        """
        print(f"Evaluating on {len(df)} labeled passages...")

        # Get predictions
        texts = df[text_column].tolist()
        predictions = self.predict_batch(texts, batch_size=batch_size, show_progress=True)

        # Extract prediction arrays
        y_true = []
        y_pred = []

        for i, row in df.iterrows():
            true_labels = [int(row[label]) for label in label_columns]
            pred_labels = [
                int(predictions[i]['predictions'].get(label, False))
                for label in label_columns
            ]

            y_true.append(true_labels)
            y_pred.append(pred_labels)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Compute metrics
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

        metrics = {
            'f1_micro': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'precision_micro': float(precision_score(y_true, y_pred, average='micro', zero_division=0)),
            'recall_micro': float(recall_score(y_true, y_pred, average='micro', zero_division=0)),
            'accuracy': float(accuracy_score(y_true, y_pred))
        }

        # Per-label metrics
        per_label = {}
        for i, label in enumerate(label_columns):
            per_label[label] = {
                'f1': float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                'precision': float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                'recall': float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0))
            }

        metrics['per_label'] = per_label

        print(f"✅ Evaluation complete! F1 Micro: {metrics['f1_micro']:.3f}")

        return metrics

    def get_model_info(self) -> Dict:
        """Get model information"""
        info = {
            'model_path': str(self.model_path),
            'device': str(self.device),
            'num_labels': len(self.label_names),
            'label_names': self.label_names,
            'has_optimal_thresholds': self.optimal_thresholds is not None
        }

        if hasattr(self.model.config, 'base_model'):
            info['base_model'] = self.model.config.base_model
            info['use_gating'] = self.model.config.use_gating
            info['use_focal_loss'] = self.model.config.use_focal_loss

        return info


class ModelComparer:
    """Compare predictions from multiple models"""

    def __init__(self, models: Dict[str, ModelInference]):
        """
        Initialize comparer

        Args:
            models: Dict mapping model names to ModelInference objects
        """
        self.models = models

    def compare_predictions(
            self,
            text: str
    ) -> pd.DataFrame:
        """
        Get predictions from all models for a single text

        Returns:
            DataFrame comparing predictions
        """
        results = []

        for name, model in self.models.items():
            pred = model.predict_single(text)

            for label in pred['predictions']:
                results.append({
                    'Model': name,
                    'Label': label,
                    'Predicted': pred['predictions'][label],
                    'Probability': pred['probabilities'].get(label, 0.0)
                })

        return pd.DataFrame(results)

    def compare_on_dataset(
            self,
            df: pd.DataFrame,
            text_column: str,
            label_columns: List[str]
    ) -> Dict:
        """
        Compare all models on a labeled dataset

        Returns:
            Dict with comparison metrics
        """
        comparison = {}

        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            metrics = model.evaluate_on_labeled_data(
                df, text_column, label_columns
            )
            comparison[name] = metrics

        # Create summary
        summary_df = pd.DataFrame([
            {
                'Model': name,
                'F1 Micro': metrics['f1_micro'],
                'F1 Macro': metrics['f1_macro'],
                'Precision': metrics['precision_micro'],
                'Recall': metrics['recall_micro']
            }
            for name, metrics in comparison.items()
        ])

        comparison['summary'] = summary_df

        return comparison