# tiered_scored_embedded_cleaned_raw__Altogether_Dataset_RACoded_Combined_20251011_0904_0904_0932_manual_1039

**Stage:** TIERED  
**Created:** 2025-10-11T11:30:47.183344  
**Parent:** scored_embedded_cleaned_raw__Altogether_Dataset_RACoded_Combined_20251011_0904_0904_0932

## Overview

- **Passages:** 5,000
- **Labels:** 12
- **Embeddings:** ✅ Yes
- **Quality Scores:** ✅ Yes

## Configuration

- **Passage Column:** `Passage`
- **Namespace:** `raw__altogether_dataset_racoded_combined_20251011`

## Labels

- **Illness**: 3,379 (67.6%)
- **Accident**: 359 (7.2%)
- **Other**: 1,636 (32.7%)
- **Material_Physical**: 997 (19.9%)
- **Spirits_Gods**: 1,593 (31.9%)
- **Witchcraft_Sorcery**: 559 (11.2%)
- **Rule_Violation_Taboo**: 726 (14.5%)
- **Physical_Material**: 2,347 (46.9%)
- **Technical_Specialist**: 381 (7.6%)
- **Divination**: 233 (4.7%)
- **Shaman_Medium_Healer**: 716 (14.3%)
- **Priest_High_Religion**: 135 (2.7%)


## Files

- `data.xlsx` - Dataset with 5,000 passages
- `metadata.json` - Complete metadata
- `README.md` - This file

- Embeddings cached in: `data/cache/raw__altogether_dataset_racoded_combined_20251011_embeddings.json`
- Scores cached in: `data/cache/raw__altogether_dataset_racoded_combined_20251011_scores.parquet`


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
data_obj = manager.load("tiered_scored_embedded_cleaned_raw__Altogether_Dataset_RACoded_Combined_20251011_0904_0904_0932_manual_1039", PipelineStage.TIERED)
```
