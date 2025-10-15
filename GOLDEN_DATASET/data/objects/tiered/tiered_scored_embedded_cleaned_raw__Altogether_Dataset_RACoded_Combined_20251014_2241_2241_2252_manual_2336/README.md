# tiered_scored_embedded_cleaned_raw__Altogether_Dataset_RACoded_Combined_20251014_2241_2241_2252_manual_2336

**Stage:** TIERED  
**Created:** 2025-10-14T23:37:54.363233  
**Parent:** scored_embedded_cleaned_raw__Altogether_Dataset_RACoded_Combined_20251014_2241_2241_2252

## Overview

- **Passages:** 5,000
- **Labels:** 12
- **Embeddings:** ✅ Yes
- **Quality Scores:** ✅ Yes

## Configuration

- **Passage Column:** `Passage`
- **Namespace:** `raw__altogether_dataset_racoded_combined_20251014`

## Labels

- **Illness**: 3,511 (70.2%)
- **Accident**: 484 (9.7%)
- **Other**: 1,484 (29.7%)
- **Material_Physical**: 1,035 (20.7%)
- **Spirits_Gods**: 1,598 (32.0%)
- **Witchcraft_Sorcery**: 580 (11.6%)
- **Rule_Violation_Taboo**: 916 (18.3%)
- **Physical_Material**: 2,275 (45.5%)
- **Technical_Specialist**: 566 (11.3%)
- **Divination**: 201 (4.0%)
- **Shaman_Medium_Healer**: 749 (15.0%)
- **Priest_High_Religion**: 217 (4.3%)


## Files

- `data.xlsx` - Dataset with 5,000 passages
- `metadata.json` - Complete metadata
- `README.md` - This file

- Embeddings cached in: `data/cache/raw__altogether_dataset_racoded_combined_20251014_embeddings.json`
- Scores cached in: `data/cache/raw__altogether_dataset_racoded_combined_20251014_scores.parquet`


## Usage

Load this data object in the application:
1. Go to **Data** page
2. Click **Browse Saved Objects**
3. Select this object from the list
4. Click **Load**

Or use programmatically:
```python
from core.data_objects import DataObjectManager, PipelineStage

manager = DataObjectManager()
data_obj = manager.load("tiered_scored_embedded_cleaned_raw__Altogether_Dataset_RACoded_Combined_20251014_2241_2241_2252_manual_2336", PipelineStage.TIERED)
```
