# tiered_scored_cleaned-misfortune-data-10092025_1501_1526

**Stage:** TIERED  
**Created:** 2025-10-09T15:26:55.959790  
**Parent:** scored_cleaned-misfortune-data-10092025_1501

## Overview

- **Passages:** 4,071
- **Labels:** 12
- **Embeddings:** ✅ Yes
- **Quality Scores:** ✅ Yes

## Configuration

- **Passage Column:** `Passage`
- **Namespace:** `cleaned-misfortune-data-10092025`

## Labels

- **Illness**: 3,199 (78.6%)
- **Accident**: 246 (6.0%)
- **Other**: 979 (24.0%)
- **Material_Physical**: 793 (19.5%)
- **Spirits_Gods**: 1,440 (35.4%)
- **Witchcraft_Sorcery**: 407 (10.0%)
- **Rule_Violation_Taboo**: 481 (11.8%)
- **Physical_Material**: 1,930 (47.4%)
- **Technical_Specialist**: 322 (7.9%)
- **Divination**: 117 (2.9%)
- **Shaman_Medium_Healer**: 524 (12.9%)
- **Priest_High_Religion**: 62 (1.5%)


## Files

- `data.xlsx` - Dataset with 4,071 passages
- `metadata.json` - Complete metadata
- `README.md` - This file

- Embeddings cached in: `data/cache/cleaned-misfortune-data-10092025_embeddings.json`
- Scores cached in: `data/cache/cleaned-misfortune-data-10092025_scores.parquet`


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
data_obj = manager.load("tiered_scored_cleaned-misfortune-data-10092025_1501_1526", PipelineStage.TIERED)
```
