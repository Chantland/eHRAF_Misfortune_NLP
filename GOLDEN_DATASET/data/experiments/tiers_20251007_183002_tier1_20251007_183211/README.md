# Data Experiment: tiers_20251007_183002_tier1

**Created:** 2025-10-07T18:32:12.151502  
**Type:** custom

## Overview

This dataset was created using the HRAF Data Preparation tool.

### Dataset Statistics
- **Passages:** 726
- **Labels:** 15
- **Columns:** 48

## Provenance

**Source File:** `data/_Altogether_Dataset_RACoded_Combined.xlsx`  
**Working Dataset:** original


## Label Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| Illness | 654 | 90.1% |
| Accident | 9 | 1.2% |
| Other | 56 | 7.7% |
| Just_Happens | 3 | 0.4% |
| Material_Physical | 32 | 4.4% |
| Spirits_Gods | 396 | 54.5% |
| Witchcraft_Sorcery | 96 | 13.2% |
| Rule_Violation_Taboo | 73 | 10.1% |
| Other.1 | 3 | 0.4% |
| Physical_Material | 179 | 24.7% |
| Technical_Specialist | 13 | 1.8% |
| Divination | 16 | 2.2% |
| Shaman_Medium_Healer | 158 | 21.8% |
| Priest_High_Religion | 2 | 0.3% |
| Other.2 | 8 | 1.1% |

## Quality Metrics

- **Consistency Mean:** 0.783
- **Consistency Median:** 0.767
- **Rerank Mean:** 0.502
- **Rerank Median:** 0.490
- **Scored Passages:** 726

## Usage

### Loading in Python

```python
import pandas as pd

df = pd.read_excel('data.xlsx')
```

### Using in HRAF Tool

1. Go to **Train Model** page
2. Select this experiment directory
3. File: `data.xlsx`

### Metadata

Full metadata available in `metadata.json`
