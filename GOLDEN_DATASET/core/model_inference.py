"""
Model Inference Module for HRAF Golden Dataset Discovery
Loads trained hierarchical models and provides inference capabilities
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn

# ============================================================================
# MODEL ARCHITECTURE DEFINITIONS (copied from training notebook)
# ============================================================================

class ConfigurableHierarchicalConfig(PretrainedConfig):
    """Configuration for configurable hierarchical model"""
    model_type = "configurable_hierarchical"

    def __init__(
        self,
        base_model="roberta-base",
        use_hierarchy=True,
        gated_hierarchy=True,
        gate_threshold=0.5,
        hidden_size=768,
        hierarchical_hidden_size=256,
        num_hidden_layers=2,
        dropout=0.2,
        attention_dropout=0.1,
        use_weighted_loss=False,
        use_focal_loss=True,
        focal_gamma=2.5,
        teacher_forcing_ratio=0.7,
        num_main_labels=3,
        num_event_labels=2,
        num_cause_labels=4,
        num_action_labels=3,
        total_labels=12,
        label_indices=None,
        label_names=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.use_hierarchy = use_hierarchy
        self.gated_hierarchy = gated_hierarchy
        self.gate_threshold = gate_threshold
        self.hidden_size = hidden_size
        self.hierarchical_hidden_size = hierarchical_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.use_weighted_loss = use_weighted_loss
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.num_main_labels = num_main_labels
        self.num_event_labels = num_event_labels
        self.num_cause_labels = num_cause_labels
        self.num_action_labels = num_action_labels
        self.total_labels = total_labels
        self.label_indices = label_indices or {}
        self.label_names = label_names or []


class ConfigurableHierarchicalModel(PreTrainedModel):
    """Highly configurable hierarchical multi-label classifier"""
    config_class = ConfigurableHierarchicalConfig
    base_model_prefix = "configurable_hierarchical"
    supports_gradient_checkpointing = True

    def __init__(self, config: ConfigurableHierarchicalConfig):
        super().__init__(config)
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.base_model)

        if hasattr(config, 'attention_dropout') and config.attention_dropout > 0:
            self.encoder.config.attention_probs_dropout_prob = config.attention_dropout

        self.main_classifier = nn.Linear(config.hidden_size, config.num_main_labels)

        if config.use_hierarchy:
            hierarchical_input_size = config.hidden_size + config.num_main_labels
        else:
            hierarchical_input_size = config.hidden_size

        self.event_classifier = self._build_sublabel_classifier(
            hierarchical_input_size, config.num_event_labels,
            config.hierarchical_hidden_size, config.num_hidden_layers, config.dropout
        )
        self.cause_classifier = self._build_sublabel_classifier(
            hierarchical_input_size, config.num_cause_labels,
            config.hierarchical_hidden_size, config.num_hidden_layers, config.dropout
        )
        self.action_classifier = self._build_sublabel_classifier(
            hierarchical_input_size, config.num_action_labels,
            config.hierarchical_hidden_size, config.num_hidden_layers, config.dropout
        )

        self.use_hierarchy = config.use_hierarchy
        self.gated_hierarchy = config.gated_hierarchy
        self.gate_threshold = config.gate_threshold
        self.post_init()

    def _build_sublabel_classifier(self, input_size, output_size, hidden_size, num_layers, dropout):
        if output_size == 0:
            return None
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, output_size))
        return nn.Sequential(*layers)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            teacher_forcing=False,
            return_dict=None,
            **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_output = encoder_outputs.last_hidden_state[:, 0]
        main_logits = self.main_classifier(pooled_output)

        if self.use_hierarchy:
            if teacher_forcing and labels is not None:
                main_probs = labels[:, :self.config.num_main_labels].float()
            else:
                main_probs = torch.sigmoid(main_logits)
            hierarchical_input = torch.cat([pooled_output, main_probs], dim=1)
        else:
            hierarchical_input = pooled_output

        event_logits = self.event_classifier(hierarchical_input) if self.event_classifier else torch.zeros(
            main_logits.shape[0], 0).to(main_logits.device)
        cause_logits = self.cause_classifier(hierarchical_input) if self.cause_classifier else torch.zeros(
            main_logits.shape[0], 0).to(main_logits.device)
        action_logits = self.action_classifier(hierarchical_input) if self.action_classifier else torch.zeros(
            main_logits.shape[0], 0).to(main_logits.device)

        # ✅ FIX: Only gate if we have main labels AND sublabels
        if self.gated_hierarchy and self.use_hierarchy and self.config.num_main_labels > 0:
            main_probs = torch.sigmoid(main_logits)

            # Gate EVENT sublabels (only if we have both main and sublabels)
            if event_logits.shape[1] > 0 and main_probs.shape[1] > 0:
                event_gate = torch.where(
                    main_probs[:, 0:1] > self.gate_threshold,
                    torch.ones_like(main_probs[:, 0:1]),
                    torch.zeros_like(main_probs[:, 0:1])
                )
                # Broadcast gate to match sublabel dimensions
                event_gate = event_gate.expand(-1, event_logits.shape[1])
                event_logits = event_logits * event_gate

            # Gate CAUSE sublabels
            if cause_logits.shape[1] > 0 and main_probs.shape[1] > 1:
                cause_gate = torch.where(
                    main_probs[:, 1:2] > self.gate_threshold,
                    torch.ones_like(main_probs[:, 1:2]),
                    torch.zeros_like(main_probs[:, 1:2])
                )
                cause_gate = cause_gate.expand(-1, cause_logits.shape[1])
                cause_logits = cause_logits * cause_gate

            # Gate ACTION sublabels
            if action_logits.shape[1] > 0 and main_probs.shape[1] > 2:
                action_gate = torch.where(
                    main_probs[:, 2:3] > self.gate_threshold,
                    torch.ones_like(main_probs[:, 2:3]),
                    torch.zeros_like(main_probs[:, 2:3])
                )
                action_gate = action_gate.expand(-1, action_logits.shape[1])
                action_logits = action_logits * action_gate

        logits = torch.cat([main_logits, event_logits, cause_logits, action_logits], dim=1)

        loss = None
        if labels is not None:
            if self.config.use_focal_loss and hasattr(self.config, 'focal_gamma'):
                loss = self._focal_loss(logits, labels.float(), gamma=self.config.focal_gamma)
            else:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _focal_loss(self, logits, targets, gamma=2.0):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probas = torch.sigmoid(logits)
        focal_weight = torch.where(targets == 1, (1 - probas) ** gamma, probas ** gamma)
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()


# ============================================================================
# MODEL LOADER CLASS
# ============================================================================

class HRAFModelLoader:
    """Load and manage trained HRAF models for inference"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_names = None
        self.optimal_thresholds = None
        self.model_info = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Register custom model classes
        AutoConfig.register("configurable_hierarchical", ConfigurableHierarchicalConfig)
        AutoModel.register(ConfigurableHierarchicalConfig, ConfigurableHierarchicalModel)

    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model from directory

        Args:
            model_path: Path to saved model directory

        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = Path(model_path)

            if not model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")

            # Load model and tokenizer
            self.model = AutoModel.from_pretrained(str(model_path))
            self.model.to(self.device)
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            # Build model_info from model config
            self.model_info = {
                'config': {
                    'base_model': getattr(self.model.config, 'base_model', 'unknown'),
                    'use_hierarchy': getattr(self.model.config, 'use_hierarchy', False),
                    'gated_hierarchy': getattr(self.model.config, 'gated_hierarchy', False),
                    'gate_threshold': getattr(self.model.config, 'gate_threshold', 0.5),
                    'use_focal_loss': getattr(self.model.config, 'use_focal_loss', False),
                    'focal_gamma': getattr(self.model.config, 'focal_gamma', 2.0),
                }
            }

            # Try to load training_info.json (has test results and optimal thresholds)
            info_path = model_path / "training_info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    training_info = json.load(f)
                    self.model_info['test_results'] = training_info.get('test_results', {})
                    self.optimal_thresholds = training_info.get('optimal_thresholds', {})
                    # Override config with training_info if available
                    if 'config' in training_info:
                        self.model_info['config'].update(training_info['config'])

            # Get label names from model config
            if hasattr(self.model.config, 'label_names'):
                self.label_names = self.model.config.label_names

            # Try experiment_info.json in parent directory
            experiment_info_path = model_path.parent / "experiment_info.json"
            if experiment_info_path.exists():
                with open(experiment_info_path, 'r') as f:
                    experiment_info = json.load(f)
                    if 'test_results' not in self.model_info:
                        self.model_info['test_results'] = {}
                    if 'training_config' in experiment_info:
                        self.model_info['training_config'] = experiment_info['training_config']

            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict_passage(
        self,
        text: str,
        use_optimal_thresholds: bool = True,
        default_threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Predict labels for a single passage

        Args:
            text: Passage text
            use_optimal_thresholds: Whether to use label-specific thresholds
            default_threshold: Default threshold if optimal not available

        Returns:
            Dictionary with predictions and probabilities
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

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

        # Get label names
        if isinstance(self.label_names, dict):
            # Extract from label_structure
            label_list = []
            for category, info in self.label_names.items():
                if info.get('enabled', True):
                    label_list.append(info.get('main_label', category))
                    label_list.extend(info.get('sublabels', []))
        else:
            label_list = self.label_names or [f"Label_{i}" for i in range(len(probs))]

        # Apply thresholds
        predictions = {}
        probabilities = {}

        for i, label in enumerate(label_list):
            if i >= len(probs):
                break

            prob = float(probs[i])
            probabilities[label] = prob

            # Get threshold
            if use_optimal_thresholds and self.optimal_thresholds:
                threshold = self.optimal_thresholds.get(label, {}).get('threshold', default_threshold)
            else:
                threshold = default_threshold

            predictions[label] = prob > threshold  # Changed from >= to avoid 0.50 false positives

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'predicted_labels': [k for k, v in predictions.items() if v]
        }

    def predict_batch(
        self,
        texts: List[str],
        use_optimal_thresholds: bool = True,
        default_threshold: float = 0.5,
        batch_size: int = 16
    ) -> List[Dict[str, any]]:
        """
        Predict labels for multiple passages

        Args:
            texts: List of passage texts
            use_optimal_thresholds: Whether to use label-specific thresholds
            default_threshold: Default threshold
            batch_size: Batch size for processing

        Returns:
            List of prediction dictionaries
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            for text in batch_texts:
                result = self.predict_passage(text, use_optimal_thresholds, default_threshold)
                results.append(result)

        return results

    def get_model_info(self) -> Optional[Dict]:
        """Get loaded model information"""
        return self.model_info

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_model_directories(base_path: str = "./results") -> List[Path]:
    """
    Find all model directories in results folder

    Args:
        base_path: Base path to search

    Returns:
        List of paths to model directories
    """
    base_path = Path(base_path)

    if not base_path.exists():
        return []

    model_dirs = []

    # Look for directories with final_model subdirectory
    for item in base_path.rglob("final_model"):
        if item.is_dir():
            model_dirs.append(item)

    # Also look for directories with config.json (direct model saves)
    for item in base_path.rglob("config.json"):
        if item.parent not in model_dirs:
            model_dirs.append(item.parent)

    return sorted(model_dirs)


def compare_predictions_to_labels(
    predictions: Dict[str, bool],
    actual_labels: Dict[str, int]
) -> Dict[str, str]:
    """
    Compare model predictions to actual labels

    Args:
        predictions: Dictionary of predicted labels (with prefixes like EVENT_Illness)
        actual_labels: Dictionary of actual labels (may be without prefixes like Illness)

    Returns:
        Dictionary with comparison results
    """
    comparison = {}

    # First, infer main categories from sublabels in actual_labels
    # If any EVENT sublabel is present, EVENT should be considered present
    inferred_main_categories = set()
    for label in actual_labels:
        if actual_labels[label] == 1:
            # Check if this looks like a sublabel
            if label in ['Illness', 'Accident', 'Other']:
                inferred_main_categories.add('EVENT')
            elif label in ['Just_Happens', 'Material_Physical', 'Spirits_Gods',
                          'Witchcraft_Sorcery', 'Rule_Violation_Taboo']:
                inferred_main_categories.add('CAUSE')
            elif label in ['Physical_Material', 'Technical_Specialist', 'Divination',
                          'Shaman_Medium_Healer', 'Priest_High_Religion', 'Other.2']:
                inferred_main_categories.add('ACTION')

    for label in predictions:
        pred = predictions[label]

        # For main categories, check if inferred from sublabels
        if label in ['EVENT', 'CAUSE', 'ACTION']:
            actual = 1 if label in inferred_main_categories else actual_labels.get(label, 0)
        else:
            # Try exact match first
            actual = actual_labels.get(label, None)

            # If no exact match, try without prefix (EVENT_Illness -> Illness)
            if actual is None and '_' in label:
                parts = label.split('_', 1)  # Split on first underscore only
                suffix = parts[1] if len(parts) > 1 else label
                actual = actual_labels.get(suffix, 0)

            if actual is None:
                actual = 0

        if pred and actual == 1:
            comparison[label] = "✓ Correct (True Positive)"
        elif pred and actual == 0:
            comparison[label] = "✗ False Positive"
        elif not pred and actual == 1:
            comparison[label] = "✗ False Negative"
        else:
            comparison[label] = "✓ Correct (True Negative)"

    return comparison