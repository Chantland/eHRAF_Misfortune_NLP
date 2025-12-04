# cleaned_raw__Altogether_Dataset_RACoded_Combined_20251014_091039

**Stage:** CLEANED  
**Created:** 2025-10-14T22:41:27.802564  
**Parent:** raw__Altogether_Dataset_RACoded_Combined_20251014_091039

## Overview

- **Passages:** 10,388
- **Labels:** 12
- **Embeddings:** ❌ No
- **Quality Scores:** ❌ No

## Configuration

- **Passage Column:** `Passage`
- **Namespace:** `raw__altogether_dataset_racoded_combined_20251014`

## Labels

- **Illness**: 4,285 (41.2%)
- **Accident**: 667 (6.4%)
- **Other**: 2,705 (26.0%)
- **Material_Physical**: 1,773 (17.1%)
- **Spirits_Gods**: 2,010 (19.3%)
- **Witchcraft_Sorcery**: 675 (6.5%)
- **Rule_Violation_Taboo**: 1,077 (10.4%)
- **Physical_Material**: 3,360 (32.3%)
- **Technical_Specialist**: 714 (6.9%)
- **Divination**: 275 (2.6%)
- **Shaman_Medium_Healer**: 871 (8.4%)
- **Priest_High_Religion**: 363 (3.5%)


## Files

- `data.xlsx` - Dataset with 10,388 passages
- `metadata.json` - Complete metadata
- `README.md` - This file



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
data_obj = manager.load("cleaned_raw__Altogether_Dataset_RACoded_Combined_20251014_091039", PipelineStage.CLEANED)
```
