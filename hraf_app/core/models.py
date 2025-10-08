"""
Hierarchical model architectures
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple


class HierarchicalConfig(PretrainedConfig):
    """Configuration for hierarchical model"""
    model_type = "hierarchical_multilabel"

    def __init__(
            self,
            base_model: str = "roberta-base",
            num_main_labels: int = 3,
            num_event_labels: int = 3,
            num_cause_labels: int = 6,
            num_action_labels: int = 6,
            hidden_size: int = 768,
            hierarchical_hidden_size: int = 256,
            num_hidden_layers: int = 2,
            dropout: float = 0.1,
            use_gating: bool = True,
            gate_threshold: float = 0.5,
            use_focal_loss: bool = True,
            focal_gamma: float = 2.5,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.num_main_labels = num_main_labels
        self.num_event_labels = num_event_labels
        self.num_cause_labels = num_cause_labels
        self.num_action_labels = num_action_labels
        self.hidden_size = hidden_size
        self.hierarchical_hidden_size = hierarchical_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_gating = use_gating
        self.gate_threshold = gate_threshold
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma


class HierarchicalModel(PreTrainedModel):
    """
    Hierarchical multi-label classifier

    Architecture:
    1. Encoder (BERT/RoBERTa) generates embeddings
    2. Main classifier predicts EVENT, CAUSE, ACTION
    3. Sublabel classifiers predict specific categories
    4. Optional gating: zero out sublabels if main not predicted
    """

    config_class = HierarchicalConfig
    base_model_prefix = "hierarchical"

    def __init__(self, config: HierarchicalConfig):
        super().__init__(config)
        self.config = config

        # Load encoder
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(config.base_model)

        # Main category classifier
        self.main_classifier = nn.Linear(
            config.hidden_size,
            config.num_main_labels
        )

        # Sublabel classifiers
        # Input = encoder output + main predictions
        hierarchical_input_size = config.hidden_size + config.num_main_labels

        self.event_classifier = self._build_sublabel_classifier(
            hierarchical_input_size,
            config.num_event_labels
        )

        self.cause_classifier = self._build_sublabel_classifier(
            hierarchical_input_size,
            config.num_cause_labels
        )

        self.action_classifier = self._build_sublabel_classifier(
            hierarchical_input_size,
            config.num_action_labels
        )

        self.post_init()

    def _build_sublabel_classifier(
            self,
            input_size: int,
            output_size: int
    ) -> nn.Module:
        """Build multi-layer sublabel classifier"""
        layers = []

        current_size = input_size
        for _ in range(self.config.num_hidden_layers):
            layers.extend([
                nn.Linear(current_size, self.config.hierarchical_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.config.dropout)
            ])
            current_size = self.config.hierarchical_hidden_size

        layers.append(nn.Linear(current_size, output_size))

        return nn.Sequential(*layers)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            teacher_forcing: bool = False,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> SequenceClassifierOutput:
        """Forward pass with hierarchical structure"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled_output = encoder_outputs.last_hidden_state[:, 0]

        # Main predictions
        main_logits = self.main_classifier(pooled_output)

        # Get main probs for hierarchical input
        if teacher_forcing and labels is not None:
            # Use ground truth during training
            main_probs = labels[:, :self.config.num_main_labels].float()
        else:
            main_probs = torch.sigmoid(main_logits)

        # Concatenate encoder output with main predictions
        hierarchical_input = torch.cat([pooled_output, main_probs], dim=1)

        # Sublabel predictions
        event_logits = self.event_classifier(hierarchical_input)
        cause_logits = self.cause_classifier(hierarchical_input)
        action_logits = self.action_classifier(hierarchical_input)

        # Optional gating
        if self.config.use_gating and not teacher_forcing:
            main_probs_for_gating = torch.sigmoid(main_logits)

            # Gate EVENT sublabels
            event_gate = (main_probs_for_gating[:, 0:1] > self.config.gate_threshold).float()
            event_logits = event_logits * event_gate

            # Gate CAUSE sublabels
            cause_gate = (main_probs_for_gating[:, 1:2] > self.config.gate_threshold).float()
            cause_logits = cause_logits * cause_gate

            # Gate ACTION sublabels
            action_gate = (main_probs_for_gating[:, 2:3] > self.config.gate_threshold).float()
            action_logits = action_logits * action_gate

        # Combine all logits
        logits = torch.cat([
            main_logits,
            event_logits,
            cause_logits,
            action_logits
        ], dim=1)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if self.config.use_focal_loss:
                loss = self._focal_loss(logits, labels.float())
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

    def _focal_loss(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss for handling class imbalance"""
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        probs = torch.sigmoid(logits)
        focal_weight = torch.where(
            targets == 1,
            (1 - probs) ** self.config.focal_gamma,
            probs ** self.config.focal_gamma
        )

        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()