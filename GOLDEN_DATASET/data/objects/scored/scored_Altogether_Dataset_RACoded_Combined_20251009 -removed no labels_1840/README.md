# scored_Altogether_Dataset_RACoded_Combined_20251009 -removed no labels_1840

**Stage:** SCORED  
**Created:** 2025-10-09T18:55:12.208260  
**Parent:** Altogether_Dataset_RACoded_Combined_20251009 -removed no labels

## Overview

- **Passages:** 7,086
- **Labels:** 12
- **Embeddings:** ✅ Yes
- **Quality Scores:** ✅ Yes

## Configuration

- **Passage Column:** `Passage`
- **Namespace:** `raw_altogether_dataset_racoded_combined_20251009`

## Labels

- **Illness**: 4,285 (60.5%)
- **Accident**: 667 (9.4%)
- **Other**: 2,705 (38.2%)
- **Material_Physical**: 1,773 (25.0%)
- **Spirits_Gods**: 2,010 (28.4%)
- **Witchcraft_Sorcery**: 675 (9.5%)
- **Rule_Violation_Taboo**: 1,077 (15.2%)
- **Physical_Material**: 3,360 (47.4%)
- **Technical_Specialist**: 714 (10.1%)
- **Divination**: 275 (3.9%)
- **Shaman_Medium_Healer**: 871 (12.3%)
- **Priest_High_Religion**: 363 (5.1%)


## Files

- `data.xlsx` - Dataset with 7,086 passages
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
data_obj = manager.load("scored_Altogether_Dataset_RACoded_Combined_20251009 -removed no labels_1840", PipelineStage.SCORED)
```
