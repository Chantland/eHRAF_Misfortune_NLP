"""
Smart data loader for HRAF datasets
Handles both original multi-level headers and exported single-level headers
"""

import pandas as pd
from pathlib import Path
import json
from typing import Tuple, Optional, List
import streamlit as st


class SmartDataLoader:
    """Intelligent data loader that handles various HRAF data formats"""

    @staticmethod
    def detect_header_format(filepath: str) -> dict:
        """
        Detect header format by analyzing the file

        Returns dict with:
        - header_rows: int or list of ints
        - format: 'single' or 'multi'
        - confidence: float (0-1)
        """
        try:
            # Check for metadata file first
            meta_path = Path(filepath).with_suffix('')
            meta_file = Path(str(meta_path) + '_metadata.json')

            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                    if 'header_format' in metadata:
                        return {
                            'header_rows': metadata.get('header_row', 0),
                            'format': metadata.get('header_format', 'single'),
                            'confidence': 1.0,
                            'source': 'metadata'
                        }

            # Analyze file structure
            # Read first few rows without headers
            df_peek = pd.read_excel(filepath, header=None, nrows=3)

            # Check first row - if it has typical label keywords, likely single header
            first_row = df_peek.iloc[0].astype(str).str.lower()
            label_keywords = ['event', 'cause', 'action', 'illness', 'accident', 'passage', 'id']

            keyword_matches = sum(any(keyword in cell for keyword in label_keywords) for cell in first_row)

            if keyword_matches >= 3:
                return {
                    'header_rows': 0,
                    'format': 'single',
                    'confidence': 0.9,
                    'source': 'analysis'
                }

            # Check for multi-level structure
            # Second row might be sub-categories
            second_row = df_peek.iloc[1].astype(str).str.lower()
            second_row_keywords = sum(any(keyword in cell for keyword in label_keywords) for cell in second_row)

            if second_row_keywords >= 3:
                return {
                    'header_rows': [0, 1],
                    'format': 'multi',
                    'confidence': 0.8,
                    'source': 'analysis'
                }

            # Default to single header
            return {
                'header_rows': 0,
                'format': 'single',
                'confidence': 0.5,
                'source': 'default'
            }

        except Exception as e:
            print(f"Header detection error: {e}")
            return {
                'header_rows': 0,
                'format': 'single',
                'confidence': 0.3,
                'source': 'error_fallback'
            }

    @staticmethod
    def load_dataset(
            filepath: str,
            header_row: Optional[int] = None,
            passage_col_override: Optional[str] = None
    ) -> Tuple[pd.DataFrame, str, dict]:
        """
        Load dataset with smart header detection

        Returns:
        - DataFrame
        - detected passage column name
        - metadata dict with loading info
        """

        # Detect header format if not specified
        if header_row is None:
            detection = SmartDataLoader.detect_header_format(filepath)
            header_row = detection['header_rows']
            load_meta = {
                'header_detection': detection,
                'header_row_used': header_row
            }
        else:
            load_meta = {
                'header_detection': {'source': 'manual'},
                'header_row_used': header_row
            }

        # Load the data
        try:
            df = pd.read_excel(filepath, header=header_row)

            # Flatten column names if multi-level
            if isinstance(df.columns, pd.MultiIndex):
                # Create readable single-level names from multi-level
                df.columns = ['_'.join(map(str, col)).strip('_') for col in df.columns.values]
                load_meta['columns_flattened'] = True
            else:
                load_meta['columns_flattened'] = False

            # Clean column names
            df.columns = [str(col).strip() for col in df.columns]

            # Detect passage column
            passage_col = SmartDataLoader.detect_passage_column(df, passage_col_override)
            load_meta['passage_column'] = passage_col

            # Basic validation
            if passage_col is None:
                raise ValueError("Could not detect passage column")

            load_meta['success'] = True
            load_meta['num_rows'] = len(df)
            load_meta['num_columns'] = len(df.columns)

            return df, passage_col, load_meta

        except Exception as e:
            raise Exception(f"Failed to load dataset: {e}")

    @staticmethod
    def detect_passage_column(df: pd.DataFrame, override: Optional[str] = None) -> Optional[str]:
        """Detect which column contains passage text"""

        if override:
            if override in df.columns:
                return override
            else:
                print(f"Warning: Override column '{override}' not found")

        # Common passage column names (case-insensitive)
        possible_names = [
            'Passage', 'passage', 'PASSAGE',
            'Text', 'text', 'TEXT',
            'Content', 'content',
            'Passage_Text', 'passage_text'
        ]

        # Check exact matches first
        for name in possible_names:
            if name in df.columns:
                return name

        # Check case-insensitive matches
        df_cols_lower = {col.lower(): col for col in df.columns}
        for name in possible_names:
            if name.lower() in df_cols_lower:
                return df_cols_lower[name.lower()]

        # Look for columns with long text (likely passages)
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    non_null = df[col].dropna()
                    if len(non_null) > 0:
                        avg_length = non_null.astype(str).str.len().mean()
                        if avg_length > 100:  # Passages typically >100 chars
                            return col
            except:
                continue

        return None

    @staticmethod
    def validate_dataset(df: pd.DataFrame, passage_col: str) -> dict:
        """Validate dataset quality and return report"""

        report = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }

        # Check passage column exists
        if passage_col not in df.columns:
            report['valid'] = False
            report['issues'].append(f"Passage column '{passage_col}' not found")
            return report

        # Check for missing passages
        missing = df[passage_col].isna().sum()
        report['stats']['missing_passages'] = missing
        if missing > 0:
            pct = (missing / len(df)) * 100
            report['warnings'].append(f"{missing} passages missing ({pct:.1f}%)")

        # Check passage lengths
        lengths = df[passage_col].dropna().astype(str).str.len()
        report['stats']['passage_lengths'] = {
            'mean': float(lengths.mean()),
            'median': float(lengths.median()),
            'min': int(lengths.min()),
            'max': int(lengths.max())
        }

        # Check for very short passages
        very_short = (lengths < 20).sum()
        if very_short > 0:
            report['warnings'].append(f"{very_short} very short passages (<20 chars)")

        # Detect potential label columns (binary 0/1)
        potential_labels = []
        for col in df.columns:
            if col != passage_col and df[col].dtype in ['int64', 'float64']:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    if (df[col] == 1).sum() > 0:
                        potential_labels.append(col)

        report['stats']['potential_labels'] = potential_labels
        report['stats']['num_potential_labels'] = len(potential_labels)

        if len(potential_labels) == 0:
            report['issues'].append("No binary label columns detected")
            report['valid'] = False

        return report


def render_smart_data_loader_ui(key_prefix: str = "loader") -> Optional[Tuple[pd.DataFrame, str, List[str]]]:
    """
    Render UI for smart data loading

    Returns: (df, passage_col, label_columns) or None
    """

    st.markdown("### 📂 Load Dataset")

    # File selection (reuse existing browser or simple selector)
    uploaded_file = st.file_uploader(
        "Choose Excel file",
        type=['xlsx', 'xls'],
        key=f"{key_prefix}_upload"
    )

    if uploaded_file is None:
        return None

    # Save temporarily
    temp_path = Path(f"temp_{uploaded_file.name}")
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Detect header format
        with st.spinner("Analyzing file format..."):
            detection = SmartDataLoader.detect_header_format(str(temp_path))

        # Show detection results
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📋 Detected: **{detection['format'].title()}** header format")
            st.caption(f"Confidence: {detection['confidence']:.0%}")

        with col2:
            # Allow manual override
            manual_header = st.checkbox("Specify header row manually", key=f"{key_prefix}_manual")
            if manual_header:
                header_row = st.number_input("Header row (0-indexed):", 0, 5, 0, key=f"{key_prefix}_header")
            else:
                header_row = None

        # Load data
        if st.button("📂 Load Data", type="primary", key=f"{key_prefix}_load"):
            with st.spinner("Loading dataset..."):
                df, passage_col, load_meta = SmartDataLoader.load_dataset(
                    str(temp_path),
                    header_row=header_row
                )

                # Validate
                validation = SmartDataLoader.validate_dataset(df, passage_col)

                # Show results
                if validation['valid']:
                    st.success(f"✅ Loaded: {len(df)} passages")

                    # Show detected info
                    with st.expander("📊 Dataset Info"):
                        st.json({
                            'rows': len(df),
                            'columns': len(df.columns),
                            'passage_column': passage_col,
                            'potential_labels': validation['stats']['num_potential_labels']
                        })

                    # Warnings
                    if validation['warnings']:
                        for warning in validation['warnings']:
                            st.warning(warning)

                    # Clean up temp file
                    temp_path.unlink()

                    # Return data
                    label_columns = validation['stats']['potential_labels']
                    return df, passage_col, label_columns

                else:
                    st.error("❌ Dataset validation failed")
                    for issue in validation['issues']:
                        st.error(issue)

                    # Clean up temp file
                    temp_path.unlink()
                    return None

    except Exception as e:
        st.error(f"Error loading data: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return None

    return None