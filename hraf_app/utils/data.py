"""
Data utility functions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def load_excel_smart(
        filepath: str,
        passage_col: Optional[str] = None,
        label_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Smart Excel loader with auto-detection

    Returns:
        (dataframe, passage_column, label_columns)
    """
    df = pd.read_excel(filepath)

    # Auto-detect passage column
    if passage_col is None:
        passage_col = detect_passage_column(df)

    # Auto-detect labels
    if label_columns is None:
        label_columns = detect_label_columns(df)

    return df, passage_col, label_columns


def detect_passage_column(df: pd.DataFrame) -> str:
    """Auto-detect passage column"""
    candidates = ['Passage', 'passage', 'Text', 'text', 'Content', 'content']

    for col in candidates:
        if col in df.columns:
            return col

    # Look for long text columns
    for col in df.columns:
        if df[col].dtype == 'object':
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if avg_len > 100:
                return col

    raise ValueError("Could not detect passage column")


def detect_label_columns(df: pd.DataFrame) -> List[str]:
    """Auto-detect binary label columns"""
    exclude = {
        'ID', 'Passage', 'Text', 'Description', 'Culture',
        'Region', 'Author', 'Year', 'Page', 'Section'
    }

    label_cols = []

    for col in df.columns:
        if col in exclude:
            continue

        if df[col].dtype in ['int64', 'float64', 'Int64']:
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                if (df[col] == 1).sum() > 0:
                    label_cols.append(col)

    return label_cols


def inspect_excel_columns(filepath: str) -> pd.DataFrame:
    """
    Inspect Excel file columns in detail

    Returns:
        DataFrame with column information
    """
    df = pd.read_excel(filepath)

    inspection = []

    for col in df.columns:
        col_data = df[col]

        # Basic info
        info = {
            'Column': col,
            'Type': str(col_data.dtype),
            'Non-Null': col_data.notna().sum(),
            'Null': col_data.isna().sum(),
        }

        # Unique values
        unique_vals = col_data.dropna().unique()
        info['Unique Values'] = len(unique_vals)

        # Show sample values
        if len(unique_vals) <= 10:
            info['Sample Values'] = str(list(unique_vals))
        else:
            info['Sample Values'] = str(list(unique_vals[:5])) + "..."

        # Check if could be label
        if col_data.dtype in ['int64', 'float64', 'Int64', 'Float64']:
            is_binary = set(unique_vals).issubset({0, 1, 0.0, 1.0})
            info['Could Be Label'] = '✓' if is_binary else '✗'
        else:
            # Try converting
            try:
                converted = pd.to_numeric(col_data, errors='coerce')
                unique_converted = converted.dropna().unique()
                is_binary = set(unique_converted).issubset({0, 1, 0.0, 1.0})
                info['Could Be Label'] = '✓ (after conversion)' if is_binary else '✗'
            except:
                info['Could Be Label'] = '✗'

        inspection.append(info)

    return pd.DataFrame(inspection)


def validate_data(
        df: pd.DataFrame,
        passage_col: str,
        label_columns: List[str]
) -> Dict:
    """Validate data quality"""
    validation = {
        'valid': True,
        'warnings': [],
        'stats': {}
    }

    # Check missing passages
    missing = df[passage_col].isna().sum()
    if missing > 0:
        pct = (missing / len(df)) * 100
        validation['warnings'].append(
            f"{missing} passages missing ({pct:.1f}%)"
        )

    # Check passage lengths
    lengths = df[passage_col].dropna().str.len()
    validation['stats']['passage_lengths'] = {
        'mean': float(lengths.mean()),
        'median': float(lengths.median()),
        'min': int(lengths.min()),
        'max': int(lengths.max())
    }

    # Check duplicates
    duplicates = df[passage_col].duplicated().sum()
    if duplicates > 0:
        validation['warnings'].append(
            f"{duplicates} duplicate passages"
        )

    # Label distribution
    label_stats = {}
    for label in label_columns:
        count = int((df[label] == 1).sum())
        pct = (count / len(df)) * 100
        label_stats[label] = {
            'count': count,
            'percentage': pct
        }

        if pct < 2:
            validation['warnings'].append(
                f"Label '{label}' very rare ({count}, {pct:.1f}%)"
            )

    validation['stats']['label_distribution'] = label_stats

    return validation


def clean_data(
        df: pd.DataFrame,
        passage_col: str,
        label_columns: List[str],
        remove_duplicates: bool = True,
        remove_missing: bool = True,
        min_length: int = 20
) -> pd.DataFrame:
    """Clean and prepare data"""
    df_clean = df.copy()

    # Remove missing passages
    if remove_missing:
        df_clean = df_clean[df_clean[passage_col].notna()]

    # Remove duplicates
    if remove_duplicates:
        df_clean = df_clean.drop_duplicates(subset=[passage_col])

    # Remove very short passages
    lengths = df_clean[passage_col].str.len()
    df_clean = df_clean[lengths >= min_length]

    # Ensure labels are binary
    for label in label_columns:
        df_clean[label] = pd.to_numeric(df_clean[label], errors='coerce').fillna(0).astype(int)

    print(f"Cleaned: {len(df)} → {len(df_clean)} passages")

    return df_clean