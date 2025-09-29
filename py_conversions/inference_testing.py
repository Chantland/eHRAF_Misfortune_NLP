#%% md
# # HIERARCHICAL MODEL INFERENCE TESTING NOTEBOOK
# 
#%% md
# Purpose: Test and evaluate trained hierarchical classification models
#%%
# ============================================================================
# CELL 1: SETUP AND IMPORTS
# ============================================================================

import torch
import torch.nn as nn
import json
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    AutoModel,
    AutoModelForSequenceClassification  # Added this import
)
from pathlib import Path
import os
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

# For loading safetensors if needed
try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    print("Warning: safetensors not installed. Install with: pip install safetensors")
    load_safetensors = None
#%%
# ============================================================================
# CELL 2: DEFINE CUSTOM MODEL ARCHITECTURE BASED ON PRE-TRAIN CONFIG
# ============================================================================

class ConfigurableHierarchicalConfig(PretrainedConfig):
    """Configuration for configurable hierarchical model"""
    model_type = "configurable_hierarchical"

    def __init__(self, config_dict=None, label_dims=None, **kwargs):
        super().__init__(**kwargs)

        if config_dict:
            for key, value in config_dict.items():
                setattr(self, key, value)

        if label_dims:
            for key, value in label_dims.items():
                setattr(self, key, value)


class ConfigurableHierarchicalModel(PreTrainedModel):
    """Highly configurable hierarchical multi-label classifier"""
    config_class = ConfigurableHierarchicalConfig

    def __init__(self, config: ConfigurableHierarchicalConfig):
        super().__init__(config)

        # Load base encoder
        self.encoder = AutoModel.from_pretrained(config.base_model)

        # Apply additional dropout to encoder if specified
        if hasattr(config, 'attention_dropout') and config.attention_dropout > 0:
            self.encoder.config.attention_probs_dropout_prob = config.attention_dropout

        # Main classifiers (always present)
        self.main_classifier = nn.Linear(config.hidden_size, config.num_main_labels)

        # Build sublabel classifiers based on configuration
        if config.use_hierarchy:
            # Hierarchical: sublabels depend on main labels
            hierarchical_input_size = config.hidden_size + config.num_main_labels
        else:
            # Non-hierarchical: sublabels independent
            hierarchical_input_size = config.hidden_size

        # Create sublabel classifiers with configurable depth
        self.event_classifier = self._build_sublabel_classifier(
            hierarchical_input_size,
            config.num_event_labels,
            config.hierarchical_hidden_size,
            config.num_hidden_layers,
            config.dropout
        )

        self.cause_classifier = self._build_sublabel_classifier(
            hierarchical_input_size,
            config.num_cause_labels,
            config.hierarchical_hidden_size,
            config.num_hidden_layers,
            config.dropout
        )

        self.action_classifier = self._build_sublabel_classifier(
            hierarchical_input_size,
            config.num_action_labels,
            config.hierarchical_hidden_size,
            config.num_hidden_layers,
            config.dropout
        )

        # Store config for forward pass
        self.use_hierarchy = config.use_hierarchy
        self.gated_hierarchy = config.gated_hierarchy if hasattr(config, 'gated_hierarchy') else False
        self.gate_threshold = config.gate_threshold if hasattr(config, 'gate_threshold') else 0.5

        self.init_weights()

    def _build_sublabel_classifier(self, input_size, output_size, hidden_size, num_layers, dropout):
        """Build a sublabel classifier with configurable depth"""
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

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        return nn.Sequential(*layers)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        teacher_forcing=False,
        **kwargs
    ):
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Pool the outputs
        pooled_output = encoder_outputs.last_hidden_state[:, 0]

        # Get main predictions
        main_logits = self.main_classifier(pooled_output)

        if self.use_hierarchy:
            # Use main predictions for sublabel input
            main_probs = torch.sigmoid(main_logits)
            hierarchical_input = torch.cat([pooled_output, main_probs], dim=1)
        else:
            # Non-hierarchical: use only pooled output
            hierarchical_input = pooled_output

        # Get sublabel predictions
        event_logits = self.event_classifier(hierarchical_input) if self.event_classifier else torch.zeros(main_logits.shape[0], 0).to(main_logits.device)
        cause_logits = self.cause_classifier(hierarchical_input) if self.cause_classifier else torch.zeros(main_logits.shape[0], 0).to(main_logits.device)
        action_logits = self.action_classifier(hierarchical_input) if self.action_classifier else torch.zeros(main_logits.shape[0], 0).to(main_logits.device)

        # Apply gating if configured
        if self.gated_hierarchy and self.use_hierarchy:
            main_probs = torch.sigmoid(main_logits)

            # Gate EVENT sublabels
            if event_logits.shape[1] > 0:
                event_gate = torch.where(
                    main_probs[:, 0:1] > self.gate_threshold,
                    torch.ones_like(main_probs[:, 0:1]),
                    torch.zeros_like(main_probs[:, 0:1])
                )
                event_logits = event_logits * event_gate

            # Gate CAUSE sublabels
            if cause_logits.shape[1] > 0:
                cause_gate = torch.where(
                    main_probs[:, 1:2] > self.gate_threshold,
                    torch.ones_like(main_probs[:, 1:2]),
                    torch.zeros_like(main_probs[:, 1:2])
                )
                cause_logits = cause_logits * cause_gate

            # Gate ACTION sublabels
            if action_logits.shape[1] > 0:
                action_gate = torch.where(
                    main_probs[:, 2:3] > self.gate_threshold,
                    torch.ones_like(main_probs[:, 2:3]),
                    torch.zeros_like(main_probs[:, 2:3])
                )
                action_logits = action_logits * action_gate

        # Concatenate all logits
        logits = torch.cat([
            main_logits, event_logits, cause_logits, action_logits
        ], dim=1)

        # Return simple output for inference
        return type('Output', (), {'logits': logits})()

#%%
# ============================================================================
# CELL 3: LIST AVAILABLE EXPERIMENTS
# ============================================================================
print("Available trained models:")
results_dir = Path("./results")
experiments = [d.name for d in results_dir.iterdir() if d.is_dir()]
for i, exp in enumerate(experiments):
    print(f"  {i+1}. {exp}")

#%%

# ============================================================================
# FIXED CELL 4: PROPER MODEL LOADING
# ============================================================================

# Change this to test different models
EXPERIMENT_NAME = "hierarchical_no_others_or_rare_labels"
MODEL_PATH = f"./results/{EXPERIMENT_NAME}/final_model"

print(f"\nLoading model: {EXPERIMENT_NAME}")

# Load configuration
with open(f"{MODEL_PATH}/experiment_config.json", "r") as f:
    experiment_config = json.load(f)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Create the model configuration
model_config = ConfigurableHierarchicalConfig(
    config_dict=experiment_config['model_config'],
    label_dims=experiment_config['label_dimensions']
)

# Initialize model architecture manually (avoiding from_pretrained issues)
print("Initializing model architecture...")
model = ConfigurableHierarchicalModel(model_config)

# Load the saved state dict
model_state_path = f"{MODEL_PATH}/model.safetensors"
if not os.path.exists(model_state_path):
    # Try pytorch format if safetensors doesn't exist
    model_state_path = f"{MODEL_PATH}/pytorch_model.bin"

if os.path.exists(model_state_path):
    print(f"Loading model weights from: {model_state_path}")

    if model_state_path.endswith('.safetensors'):
        if load_safetensors:
            state_dict = load_safetensors(model_state_path)
        else:
            raise ImportError("safetensors is required to load this model. Install with: pip install safetensors")
    else:
        state_dict = torch.load(model_state_path, map_location='cpu')

    # Load the state dict
    model.load_state_dict(state_dict, strict=False)
    print("Model weights loaded successfully!")
else:
    print(f"Warning: Could not find model weights at {model_state_path}")
    print("The model will use randomly initialized weights.")

model.eval()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

label_names = experiment_config['label_dimensions']['label_names']
print(f"Model loaded! Labels: {label_names}")
print(f"Device: {device}")

#%%
# ============================================================================
# CELL 5: LOAD TEST DATA
# ============================================================================

# Option 1: Load your test Excel file
TEST_FILE = "_Altogether_Dataset_RACoded_Combined.xlsx"  # <-- Change this to test different data collections.

# Simple loading for testing (adjust based on your needs)
if TEST_FILE.endswith('.xlsx'):
    df_test = pd.read_excel(TEST_FILE, header=[0,1], index_col=0)
    # Extract just the passage text (adjust column names as needed)
    test_passages = df_test[('CULTURE', 'Passage')].head(100).tolist()  # Testing first 100
elif TEST_FILE.endswith('.csv'):
    df_test = pd.read_csv(TEST_FILE)
    test_passages = df_test['passage'].head(100).tolist()
else:
    # Manual test cases
    test_passages = [
        "The child fell ill with a mysterious fever that wouldn't break.",
        "After breaking the taboo, the village elder performed a cleansing ritual.",
        "The accident happened suddenly when the cart overturned.",
        # Add your test passages here
    ]

print(f"Loaded {len(test_passages)} test passages")

#%%
# ============================================================================
# CELL 6: RUN INFERENCE (UPDATED)
# ============================================================================

def predict_batch(passages, model, tokenizer, batch_size=16):
    """Simple batch prediction function"""
    all_predictions = []
    all_probabilities = []

    device = next(model.parameters()).device  # Get model's device

    for i in range(0, len(passages), batch_size):
        batch = passages[i:i+batch_size]

        # Tokenize
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)  # Move to device

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_predictions.extend(preds)
            all_probabilities.extend(probs)

    return np.array(all_predictions), np.array(all_probabilities)

# Run predictions
print("Running inference...")
predictions, probabilities = predict_batch(test_passages, model, tokenizer)
print(f"Predictions shape: {predictions.shape}")
#%%
# ============================================================================
# CELL 7: ANALYZE RESULTS
# ============================================================================
# Convert to DataFrame for easy analysis
results_df = pd.DataFrame(predictions, columns=label_names)
probs_df = pd.DataFrame(probabilities, columns=label_names)

# Basic statistics
print("\n=== PREDICTION STATISTICS ===")
print("\nLabel frequencies (% positive):")
for label in label_names:
    pct = results_df[label].mean() * 100
    print(f"  {label:30s}: {pct:5.1f}%")

# Check hierarchical consistency (if applicable)
if 'hierarchical' in EXPERIMENT_NAME.lower():
    print("\n=== HIERARCHICAL CONSISTENCY CHECK ===")
    inconsistencies = 0

    # Check EVENT sublabels without EVENT
    event_subs = [col for col in label_names if col.startswith('EVENT_')]
    if event_subs and 'EVENT' in label_names:
        mask = (results_df[event_subs].any(axis=1)) & (~results_df['EVENT'])
        inconsistencies += mask.sum()
        if mask.sum() > 0:
            print(f"⚠️  {mask.sum()} passages have EVENT sublabels without EVENT")

    # Similar checks for CAUSE and ACTION
    for main in ['CAUSE', 'ACTION']:
        subs = [col for col in label_names if col.startswith(f'{main}_')]
        if subs and main in label_names:
            mask = (results_df[subs].any(axis=1)) & (~results_df[main])
            inconsistencies += mask.sum()
            if mask.sum() > 0:
                print(f"⚠️  {mask.sum()} passages have {main} sublabels without {main}")

    if inconsistencies == 0:
        print("✅ All predictions are hierarchically consistent!")
#%%
# ============================================================================
# CELL 8: COMPARE MULTIPLE MODELS
# ============================================================================
def compare_models(experiment_names, test_passages):
    """Compare predictions across different trained models"""
    comparison = {}

    for exp_name in experiment_names:
        print(f"\nTesting: {exp_name}")
        model_path = f"./results/{exp_name}/final_model"

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Predict
        preds, probs = predict_batch(test_passages, model, tokenizer)
        comparison[exp_name] = {
            'predictions': preds,
            'probabilities': probs,
            'avg_confidence': probs.mean(),
            'positive_rate': preds.mean()
        }

    return comparison

# Example: Compare different experiments
experiments_to_compare = [
    f"{EXPERIMENT_NAME}",
    # "flat_model_baseline",  # Add other experiments here
    # "hierarchical_with_focal_loss",
]

# Uncomment to run comparison:
# comparison_results = compare_models(experiments_to_compare, test_passages[:10])

#%%
# ============================================================================
# CELL 9: EXPORT RESULTS
# ============================================================================

# Save predictions for further analysis
output_file = f"./predictions_{EXPERIMENT_NAME}.csv"
results_with_text = pd.DataFrame({
    'passage': test_passages,
    **{f'pred_{label}': results_df[label] for label in label_names},
    **{f'prob_{label}': probs_df[label] for label in label_names}
})
results_with_text.to_csv(output_file, index=False)
print(f"\nPredictions saved to: {output_file}")
#%%
# ============================================================================
# CELL 10: INTERACTIVE SINGLE PASSAGE TEST (UPDATED)
# ============================================================================

def test_single_passage(text, model=model, tokenizer=tokenizer):
    """Test a single passage and display detailed results"""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    print(f"\nPassage: '{text[:100]}...'\n")
    print("Predictions:")
    print("-" * 40)

    # Group predictions by category for better readability
    main_labels = ['EVENT', 'CAUSE', 'ACTION']

    # Print main labels first
    print("\nMAIN CATEGORIES:")
    for label in main_labels:
        if label in label_names:
            i = label_names.index(label)
            prob = probs[i]
            pred = "✓" if prob > 0.5 else "✗"
            print(f"{pred} {label:30s}: {prob:.3f}")

    # Print sublabels grouped by category
    for main in main_labels:
        sublabels = [l for l in label_names if l.startswith(f"{main}_")]
        if sublabels:
            print(f"\n{main} SUBCATEGORIES:")
            for label in sublabels:
                i = label_names.index(label)
                prob = probs[i]
                pred = "✓" if prob > 0.5 else "✗"
                # Check if gated (only show if main category is positive)
                main_idx = label_names.index(main) if main in label_names else -1
                if main_idx >= 0 and probs[main_idx] < 0.5:
                    print(f"  (gated) {label[len(main)+1:]:28s}: {prob:.3f}")
                else:
                    print(f"  {pred} {label[len(main)+1:]:28s}: {prob:.3f}")

    return dict(zip(label_names, probs))

# Test examples
print("="*50)
print("TEST 1: Illness with spiritual cause and shaman action")
test_single_passage("The shaman performed a ritual to heal the sick child who had angered the forest spirits.")

print("\n" + "="*50)
print("TEST 2: Physical accident")
test_single_passage("He broke his leg when he fell from the tree while gathering fruit.")

print("\n" + "="*50)
print("TEST 3: Taboo violation")
test_single_passage("After entering the sacred grove without permission, the man fell ill and sought forgiveness through ritual cleansing.")