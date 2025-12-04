"""
Data Loader - Stable ID Generation
"""
import hashlib
import pandas as pd
from typing import Tuple


class SmartDataLoader:
    """Generates stable passage IDs based on content hash"""

    @staticmethod
    def add_stable_ids(df: pd.DataFrame, passage_col: str) -> pd.DataFrame:
        """
        Add stable passage IDs based on content hash

        These IDs survive cleaning, filtering, and reordering operations
        because they're derived from passage content, not row position.

        Args:
            df: DataFrame with passages
            passage_col: Name of passage text column

        Returns:
            DataFrame with 'passage_id' column added
        """
        df = df.copy()

        def generate_stable_id(text):
            """Generate 16-character stable ID from passage text"""
            if pd.isna(text):
                return None

            # Use MD5 hash of text content (first 16 chars for readability)
            text_str = str(text).strip()
            return hashlib.md5(text_str.encode('utf-8')).hexdigest()[:16]

        # Generate stable IDs
        df['passage_id'] = df[passage_col].apply(generate_stable_id)

        # Check for duplicates (same passage appearing multiple times)
        duplicate_ids = df['passage_id'].duplicated().sum()
        if duplicate_ids > 0:
            print(f"⚠️  Found {duplicate_ids} duplicate passages (same text)")

        return df