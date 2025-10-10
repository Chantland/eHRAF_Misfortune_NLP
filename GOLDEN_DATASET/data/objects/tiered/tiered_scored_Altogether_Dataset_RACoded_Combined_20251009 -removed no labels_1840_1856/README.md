# tiered_scored_Altogether_Dataset_RACoded_Combined_20251009 -removed no labels_1840_1856

**Stage:** TIERED  
**Created:** 2025-10-09T19:08:48.598987  
**Parent:** scored_Altogether_Dataset_RACoded_Combined_20251009 -removed no labels_1840

## Overview

- **Passages:** 2,621
- **Labels:** 12
- **Embeddings:** ✅ Yes
- **Quality Scores:** ✅ Yes

## Configuration

- **Passage Column:** `Passage`
- **Namespace:** `raw_altogether_dataset_racoded_combined_20251009`

## Labels

- **Illness**: 1,924 (73.4%)
- **Accident**: 237 (9.0%)
- **Other**: 624 (23.8%)
- **Material_Physical**: 536 (20.5%)
- **Spirits_Gods**: 964 (36.8%)
- **Witchcraft_Sorcery**: 303 (11.6%)
- **Rule_Violation_Taboo**: 320 (12.2%)
- **Physical_Material**: 1,236 (47.2%)
- **Technical_Specialist**: 292 (11.1%)
- **Divination**: 175 (6.7%)
- **Shaman_Medium_Healer**: 435 (16.6%)
- **Priest_High_Religion**: 171 (6.5%)


## Files

- `data.xlsx` - Dataset with 2,621 passages
- `metadata.json` - Complete metadata
- `README.md` - This file

- Embeddings cached in: `data/cache/raw_altogether_dataset_racoded_combined_20251009_embeddings.json`
- Scores cached in: `data/cache/raw_altogether_dataset_racoded_combined_20251009_scores.parquet`


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
data_obj = manager.load("tiered_scored_Altogether_Dataset_RACoded_Combined_20251009 -removed no labels_1840_1856", PipelineStage.TIERED)
```
