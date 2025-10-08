# Tiered Training Experiment: tiers_20251007_191858
        
        **Created:** 2025-10-07T19:19:04.343553  
        **Type:** Quality-Based Tiered Training Data
        
        ## Overview
        
        This experiment contains quality-stratified training data for curriculum learning.
        
        ### Tier Statistics
        
        | Tier | Count | Percentage | Purpose |
        |------|-------|------------|---------|
        | Tier 1 (Elite) | 726 | 9.5% | Initial training |
        | Tier 2 (Expansion) | 1,879 | 24.6% | Generalization |
        | Inference (Test) | 5,029 | 65.9% | Evaluation |
        | **Combined** | 2,605 | - | Full training |
        
        ## Files
        
        - **`tier1.xlsx`** - 726 highest quality passages
        - **`tier2.xlsx`** - 1879 good quality passages  
        - **`tier1_tier2_combined.xlsx`** - 2605 combined training data
        - **`inference.xlsx`** - 5029 test/validation data
        - **`metadata.json`** - Complete experiment metadata
        - **`README.md`** - This file
        
        ## Provenance
        
        **Source:** `data/_Altogether_Dataset_RACoded_Combined.xlsx`  
        **Dataset Type:** original  
        **Quality Scores:** Yes
        
        ## Training Strategies
        
        ### Strategy 1: Curriculum Learning (Recommended)Stage 1 (Epochs 1-5): Train on tier1.xlsx
        └─ Learn from highest quality examplesStage 2 (Epochs 6-10): Fine-tune on tier1_tier2_combined.xlsx
        └─ Generalize to broader patternsStage 3: Evaluate on inference.xlsx
        └─ Final model testing
        
        ### Strategy 2: Single-Pass TrainingTrain on tier1_tier2_combined.xlsx for full epochs
        └─ Use all training data from start
        
        ### Strategy 3: Elite-Only TrainingTrain on tier1.xlsx only
        └─ Maximum quality, smaller dataset
        
        ## Label Distribution
        
        
See `metadata.json` for detailed label distribution per tier.

            ## Usage in HRAF Tool
            
            ### Loading for Training
            
            1. Navigate to **Train Model** page
            2. Under "Dataset Selection", choose **Tiered Datasets**
            3. Select training strategy:
               - **Tier 1 Only** → Use `tier1.xlsx`
               - **Tier 1 + Tier 2** → Use `tier1_tier2_combined.xlsx`
               - **Curriculum** → Train on tier1 first, then combined
            
            ### Configuration
            
            Tier configuration used to create this dataset is in `metadata.json` under `tier_configuration`.
            
            ## Quality Thresholds
            
            This experiment was created with the following quality criteria:
            
            
### Tier1
- Consistency: 0.783
- Rerank: 0.502

### Tier2
- Consistency: 0.543
- Rerank: 0.391

### Inference
- Consistency: 0.432
- Rerank: 0.413
