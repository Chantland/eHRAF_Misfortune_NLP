# cleaned_cleaned-misfortune-data-10092025_1442

**Stage:** CLEANED  
**Created:** 2025-10-09T14:42:54.469154  
**Parent:** cleaned-misfortune-data-10092025

## Overview

- **Passages:** 11,005
- **Labels:** 12
- **Embeddings:** ❌ No
- **Quality Scores:** ❌ No

## Configuration

- **Passage Column:** `Passage`
- **Namespace:** `cleaned-misfortune-data-10092025`

## Labels

- **Illness**: 4,668 (42.4%)
- **Accident**: 708 (6.4%)
- **Other**: 2,868 (26.1%)
- **Material_Physical**: 1,898 (17.2%)
- **Spirits_Gods**: 2,250 (20.4%)
- **Witchcraft_Sorcery**: 736 (6.7%)
- **Rule_Violation_Taboo**: 1,116 (10.1%)
- **Physical_Material**: 3,574 (32.5%)
- **Technical_Specialist**: 768 (7.0%)
- **Divination**: 285 (2.6%)
- **Shaman_Medium_Healer**: 976 (8.9%)
- **Priest_High_Religion**: 374 (3.4%)


## Files

- `data.xlsx` - Dataset with 11,005 passages
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
data_obj = manager.load("cleaned_cleaned-misfortune-data-10092025_1442", PipelineStage.CLEANED)
```
