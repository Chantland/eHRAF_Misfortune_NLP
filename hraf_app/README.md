# HRAF Quality Pipeline v2.0

> A streamlined, data-first approach to NLP model training for anthropological text classification

## What This Does

The HRAF Quality Pipeline helps you build better NLP models by **finding and fixing data quality issues before they hurt performance**.

Instead of just training models and hoping for the best, it:

1. **Analyzes your data quality** - Identifies which passages are good training examples
2. **Guides data selection** - Helps you choose the right balance of quality and quantity
3. **Trains hierarchical models** - Builds models that understand label relationships
4. **Closes the loop** - Analyzes failures and suggests specific data improvements
5. **Tracks everything** - Records what worked and what didn't

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd hraf_pipeline

# Install dependencies
pip install -r requirements.txt

# Set up API keys (create .env file)
cp .env.example .env
# Edit .env and add your keys
```

### Requirements

```txt
# requirements.txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.0
voyageai>=0.2.0
pinecone>=7.0.0
anthropic>=0.40.0
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
python-dotenv>=1.0.0
```

### Environment Setup

Create `.env` file:

```bash
# VoyageAI (for embeddings and reranking)
VOYAGE_API_KEY=your_voyage_key_here

# Pinecone (for vector storage)
PINECONE_API_KEY=your_pinecone_key_here

# Anthropic Claude (for AI assistant)
ANTHROPIC_API_KEY=your_claude_key_here
```

### Launch

```bash
streamlit run app.py
```

Then open `http://localhost:8501`

## Workflow

The tool guides you through 5 steps:

### Step 1: Load Data
- Upload Excel file with passages and labels
- Auto-detects passage text and label columns
- Validates data quality

**What you need:** Excel file with:
- One column containing passage text (any name)
- Binary label columns (0/1 values)

### Step 2: Compute Quality
- Analyzes semantic consistency
- Scores label confidence
- Identifies elite, good, fair, and low-quality passages

**Takes:** 2-5 minutes for 10,000 passages

### Step 3: Explore & Filter
- Select training data based on quality
- Balance quality vs. quantity
- Optional: Target specific labels

**Strategies:**
- **Conservative:** Elite only (>0.75 quality)
- **Balanced:** Elite + Good (>0.60 quality)
- **Aggressive:** Include Fair (>0.45 quality)

### Step 4: Train Model
- Trains hierarchical multi-label classifier
- Predicts main categories then subcategories
- Tracks performance metrics

**Takes:** 5-30 minutes depending on data size

### Step 5: Analyze & Iterate
- Identifies failure patterns
- Suggests specific data improvements
- Apply improvements and retrain

**The Key Innovation:** Model errors point to data quality issues

## Architecture

```
hraf_pipeline/
├── app.py                    # Main Streamlit app
├── core/
│   ├── quality.py           # Unified quality scoring
│   ├── training.py          # Model training
│   ├── models.py            # Model architectures  
│   └── experiments.py       # Experiment tracking
├── workflows/
│   ├── pipeline.py          # Workflow orchestration
│   └── active_learning.py   # Iteration loop
├── ui/
│   ├── workflow_steps.py    # Step-by-step UI
│   └── components.py        # Reusable components
├── assistant/
│   └── claude_agent.py      # AI assistant
└── utils/
    ├── data.py              # Data utilities
    └── search.py            # Smart search
```

## Key Concepts

### Unified Quality Score

Every passage gets **one quality score** (0-1) based on:

- **Semantic consistency** (50%): How similar to passages with same labels
- **Label confidence** (50%): How clearly passage demonstrates labeled concepts
- **Model agreement** (optional): If predictions match labels after training

### Quality Tiers

Passages are automatically assigned to tiers:

- **Elite** (≥0.75): Unambiguous, perfect training examples
- **Good** (0.60-0.75): Clear examples with minor ambiguity
- **Fair** (0.45-0.60): Useful but some confusion
- **Low** (<0.45): Ambiguous or potentially mislabeled

### Active Learning Loop

The innovation that closes the gap:

```
Train Model → Analyze Failures → Identify Data Issues → Fix Data → Retrain
```

Instead of just reporting metrics, the tool tells you **exactly what to fix**.

## Example Use Case

### Starting Data
- 10,000 labeled passages
- 12 labels (EVENT, CAUSE, ACTION subcategories)
- Low inter-rater agreement (Kappa ~0.4)

### After Quality Analysis
- Median quality: 0.52
- 15% elite, 25% good, 35% fair, 25% low
- Recommendation: Use balanced strategy (elite + good)

### First Training
- Selected 4,000 passages (elite + good)
- Model F1: 0.67
- Issue: False negatives on "Witchcraft_Sorcery"

### After Iteration
- Added 50 high-quality Witchcraft examples
- Removed 200 low-quality passages causing errors
- Model F1: 0.74 ✅

## Tips & Best Practices

### Data Preparation

✅ **Do:**
- Clean passage text (remove artifacts, formatting)
- Ensure labels are truly binary (0 or 1)
- Review label definitions for clarity
- Check for duplicate passages

❌ **Don't:**
- Use ambiguous or overlapping label definitions
- Include passages with no labels
- Mix different annotation styles

### Quality Thresholds

**Conservative (0.75+):**
- Use when you have >2,000 elite passages
- Best for initial model development
- Reduces noise, improves learning

**Balanced (0.60+):**
- Default choice for most cases
- Good balance of quality and quantity
- Works with 5,000-10,000 passages

**Aggressive (0.45+):**
- Only if you have <5,000 passages total
- Or if targeting specific rare labels
- Requires more epochs to overcome noise

### Training

**Start simple:**
- Default settings work for 90% of cases
- Only tune if you see specific issues
- More epochs ≠ better results (watch for overfitting)

**When to tune:**
- Small dataset (<2,000): Lower learning rate (1e-5)
- Large dataset (>10,000): Can use 32 batch size
- Rare labels: Enable focal loss
- Fast iteration: Reduce epochs to 5

### Iteration

**Priority order:**
1. Remove low-quality passages causing errors (biggest impact)
2. Add examples for underrepresented labels (if needed)
3. Review and relabel ambiguous passages (time-intensive)
4. Adjust prediction thresholds (quick fix, modest impact)

## Troubleshooting

### Low Quality Scores (median <0.5)

**Cause:** High labeling disagreement or ambiguous definitions

**Fix:**
- Review label definitions
- Check inter-rater agreement
- Consider combining similar labels
- Focus on clearest examples only

### Model F1 <0.60

**Causes:**
- Insufficient high-quality data
- Ambiguous label definitions
- Severe class imbalance

**Fix:**
1. Run failure analysis
2. Check which labels are problematic
3. Review training data for those labels
4. Consider training separate models per label

### Model F1 >0.80

**You're done!** 🎉

That's excellent performance for multi-label text classification.

### "Out of Memory" Errors

**Fix:**
- Reduce batch size (try 8)
- Reduce max sequence length (try 256)
- Use gradient accumulation
- Train on GPU if possible

## Advanced Features

### Smart Search
- Search by text, label, or quality
- Find similar passages
- Identify confusing examples

### Experiment Tracking
- Compare different training runs
- Track data selection criteria
- Identify what worked best

### Multi-Model Comparison
- Train multiple architectures
- Compare predictions side-by-side
- Ensemble for better performance

## API Keys & Costs

### VoyageAI
- Free tier: 5M tokens/month
- Embeddings: ~$0.12 per 1M tokens
- Reranking: ~$0.02 per 1K queries

**Cost for 10K passages:** ~$2-3

### Pinecone
- Free tier: 1 index, 100K vectors
- Sufficient for most research projects

### Anthropic Claude
- Optional (for AI assistant)
- ~$3 per 1M input tokens
- Pay as you go

**Total cost for typical project:** $5-10

## Support

- **Issues:** Open GitHub issue
- **Questions:** Check discussions
- **Email:** [your-email]

## Citation

If you use this tool in your research:

```bibtex
@software{hraf_quality_pipeline,
  title={HRAF Quality Pipeline: Data-First NLP for Anthropology},
  author={[Your Name]},
  year={2025},
  url={https://github.com/your-repo}
}
```

## License

MIT License - See LICENSE file

## Acknowledgments

Built for the HRAF Misfortune Classification project (PI: Dr. Pascal Boyer)
Funded by Templeton Religion Trust

---

**Questions?** The tool includes an AI assistant that can answer questions about your specific data and models.