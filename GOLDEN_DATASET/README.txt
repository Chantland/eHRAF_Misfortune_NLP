HRAF Golden Dataset Discovery Tool
A comprehensive NLP application for classifying and analyzing anthropological passages about misfortune from the eHRAF (electronic Human Relations Area Files) database. This tool provides an end-to-end pipeline for data preparation, quality assessment, model training, and semantic exploration.
📋 Project Overview
This research tool enables classification of passages into hierarchical categories describing:

EVENT: Types of misfortune (Illness, Accident, Other)
CAUSE: Explanations for misfortune (Material/Physical, Spirits/Gods, Witchcraft/Sorcery, Rule Violation/Taboo, Just Happens)
ACTION: Responses to misfortune (Physical/Material, Technical Specialist, Divination, Shaman/Medium/Healer, Priest/High Religion)

Principal Investigator: Dr. Pascal Boyer (Washington University in St. Louis)
Funding: Templeton Religion Trust ($2M grant)
✨ Key Features
🔄 Data Pipeline

RAW → CLEANED → EMBEDDED → SCORED → TIERED workflow
Interactive data loading with flexible column configuration
Stable ID system for tracking passages across transformations
Automated quality assessment using semantic similarity
Curriculum learning through quality-based data tiering

🤖 Model Training

Hierarchical multi-label classification
Configurable architecture (flat or hierarchical)
Focal loss and weighted loss for class imbalance
Curriculum learning support (Tier 1 → Combined)
Real-time training monitoring
Early stopping with patience control

🔍 Discovery & Analysis

Semantic search using VoyageAI embeddings
Similar passage finder
Multi-model inference comparison
Hypothesis testing (chi-square analysis)
Label relationship exploration

💬 AI Assistant

Built-in Claude integration
Context-aware help on every page
Can execute actions (load models, configure training, run searches)
Understands project data and current state

🚀 Installation
Prerequisites

Python 3.8+
CUDA-capable GPU (recommended for training)

Setup

Clone the repository

bashgit clone <repository-url>
cd GOLDEN_DATASET

Create virtual environment

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bashpip install -r requirements.txt

Configure environment variables

Create a .env file in the project root:
env# Required for embeddings and search
VOYAGE_API_KEY=your_voyage_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Optional: For AI chat assistant
ANTHROPIC_API_KEY=your_anthropic_api_key_here
Get API keys from:

VoyageAI: https://www.voyageai.com/
Pinecone: https://www.pinecone.io/
Anthropic: https://www.anthropic.com/

📊 Usage
Start the Application
bashstreamlit run app_golden_dataset.py
The app will open in your browser at http://localhost:8501
Workflow Guide
1. Data Page - Prepare Your Dataset
Load Data:

Upload Excel file (.xlsx)
Interactive preview with header row selection
Mark label columns vs metadata columns
Auto-detection of binary labels

Clean Data:

Analyze quality issues
Remove duplicates, missing values, short passages
Or skip cleaning if data is ready

Generate Embeddings:

Uses VoyageAI voyage-3-large model
Stores in Pinecone vector database
Cached for reuse

Calculate Quality Scores:

Consistency: Agreement with similar passages
Rerank: Semantic relevance to label definitions
Handles class imbalance intelligently

Create Training Tiers:

Smart Presets: Data-aware recommendations
Manual Control: Fine-tune thresholds with live preview
Label Targeting: Ensure rare labels are represented
Produces: Tier 1 (elite), Tier 2 (expansion), Inference (test)

2. Analysis Page - Explore Your Data

Quick Stats: Dataset overview and label distribution
Label Diagnostics: Deep dive into specific labels

Co-occurrence analysis
Class imbalance assessment
Sample passage inspection


Code Playground: Execute custom Python analysis

3. Models Page - Train and Evaluate
Model Library:

Load trained models from disk
Manage multiple models
View architecture and performance

Train New Model:

Select training data (Tier 1, Combined, or Curriculum)
Configure architecture (flat or hierarchical)
Set loss functions (focal, weighted)
Monitor training in real-time
Automatic early stopping

Configuration Options:
python# Architecture
- Base model: roberta-base, bert-base-uncased, distilbert-base-uncased
- Hierarchical: Enable dependency between main → sublabels
- Gating: Zero sublabels if main category < threshold
- Hidden layers: 1-5
- Dropout: 0-0.5

# Loss Configuration
- Focal loss: Focus on hard examples (gamma: 0-5)
- Weighted loss: Balance class frequencies
- Teacher forcing: Use ground truth during training

# Training Parameters
- Epochs: 1-50
- Batch size: 4-64
- Learning rate: 1e-6 to 1e-3
- Warmup steps: 0-2000
- Early stopping: Patience 1-10 epochs
Evaluate & Compare:

Test models on held-out data
Fair comparison (sublabels only)
Side-by-side prediction samples

4. Discover Page - Semantic Exploration
Semantic Search:

Natural language queries
AI reranking for better results
Label filtering
Diagnostics to verify search pipeline

Similar Passages:

Find semantically similar passages
Understand label consistency

Model Inference:

Test predictions on individual passages
Compare with ground truth
Multi-model comparison

Hypothesis Testing:

Chi-square tests for label relationships
Example: "Is Shaman_Medium_Healer more common when EVENT_Illness occurs?"

🏗️ Architecture
Data Objects
The application uses an immutable data pipeline:
pythonDataObject:
  - name: Unique identifier
  - stage: RAW | CLEANED | EMBEDDED | SCORED | TIERED
  - df: pandas DataFrame
  - passage_col: Column with text
  - label_columns: Classification labels
  - embeddings_cache: stable_id → pinecone_id mapping
  - scores_cache: Quality scores DataFrame
  - metadata: Lineage and configuration
```

### Key Components
```
GOLDEN_DATASET/
├── app_golden_dataset.py          # Main application entry
├── requirements.txt               # Requirements file
├── README.txt                     # README.txt
├── components/
│   ├── chat_assistant.py          # AI assistant with action execution
│   ├── data_loader.py             # Smart data loading
│   ├── model_manager.py           # Model lifecycle management
│   └── training_monitor.py        # Real-time training feedback
├── core/
│   ├── data_cache.py              # Persistent caching
│   ├── data_objects.py            # Data pipeline abstraction
│   ├── data_preparation.py        # Cleaning and tiering
│   ├── quality_scoring.py         # Consistency & rerank scoring
│   ├── model_training.py          # Training orchestration
│   └── model_inference.py         # Model loading and prediction
├── page_views/
│   ├── data_page.py               # Data preparation UI
│   ├── analysis_page.py           # Interactive analysis
│   ├── models_page.py             # Training and evaluation UI
│   └── discover_page.py           # Search and exploration
├── models/
└── data/
    ├── cache/                     # Embeddings and scores
    └── objects/                   # Saved DataObjects
```

## 🔬 Model Architecture

### Hierarchical Classification
```
Input Text → RoBERTa Encoder → [CLS] token
                                    ↓
                     ┌──────────────┴──────────────┐
                     ↓                             ↓
              Main Classifier              (Optional) Freeze until trained
              [EVENT, CAUSE, ACTION]
                     ↓
              Concatenate [CLS + Main Predictions]
                     ↓
         ┌───────────┼───────────┐
         ↓           ↓           ↓
    Event MLP    Cause MLP    Action MLP
    [Illness,    [Material,   [Physical,
     Accident]    Spirits,     Shaman,
                  Witch...]    Priest...]
Key Features:

Gated Hierarchy: Sublabels zeroed if main category confidence < threshold
Teacher Forcing: Uses ground truth main labels during training with probability p
Focal Loss: FL(p) = -(1-p)^γ * log(p) focuses on hard examples
Weighted Loss: Class weights = neg_count / pos_count (capped at 100x)

📈 Performance
Current best model (RoBERTa):

Overall F1 Micro: 0.664
Best Labels: EVENT_Illness (0.876), CAUSE_Spirits_Gods (0.728)
Challenging Labels: ACTION_Priest_High_Religion (0.375), ACTION_Divination (0.406)

🐛 Troubleshooting
Embeddings Not Found

Check .env has valid VOYAGE_API_KEY and PINECONE_API_KEY
Run diagnostics in Discover → Semantic Search
Re-generate embeddings on Data page

Training Fails

Ensure data has at least 1,000 passages
Check label columns are binary (0/1)
Verify passage column contains text
Check CUDA is available: torch.cuda.is_available()

Search Returns No Results

Verify namespace matches between embedding and search
Check index stats in diagnostic section
Try lowering minimum similarity threshold

📚 Citation
If you use this tool in your research, please cite:
bibtex@software{hraf_golden_dataset_2024,
  title={HRAF Golden Dataset Discovery Tool},
  author={Boyer, Pascal and Chantland, Eric},
  year={2024},
  organization={Washington University in St. Louis},
  funding={Templeton Religion Trust}
}

Contributing
This is a research project. For questions or collaboration:

Eric Chantland: eric.c@wustl.edu
Pascal Boyer: pboyer@wustl.edu
John Heinz: john.heinz@bettergood.net

License
[Specify license here]

Acknowledgments

Funding: Templeton Religion Trust
Research Program: "Wild Religions and Misfortune"
Institution: Washington University in St. Louis


Built with: Python • Streamlit • PyTorch • Transformers • VoyageAI • Pinecone • Claude