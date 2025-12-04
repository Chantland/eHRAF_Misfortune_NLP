"""
Persistent Cache Manager for Embeddings and Scores

Handles saving/loading of:
- Embeddings (passage_id_map) to/from disk
- Quality scores to/from disk
- Auto-detection of existing caches
- Checkpoint system for long operations
"""
import streamlit as st
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import shutil


class CacheManager:
    """Manages persistent caching of embeddings and scores"""

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup fallback cache directory
        self.fallback_cache_dir = Path(cache_dir.replace("data", "_data"))

    def load_embeddings(self, namespace: str) -> Optional[Dict[str, str]]:
        """
        Load stable_id -> pinecone_id mapping from disk
        Checks both primary and fallback locations
        """
        # Try primary first
        cache_file = self.cache_dir / f"{namespace}_embeddings.json"

        # Fall back to _data if not found
        if not cache_file.exists() and self.fallback_cache_dir.exists():
            cache_file = self.fallback_cache_dir / f"{namespace}_embeddings.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            format_version = data.get('format_version', '1.0')

            if format_version == '1.0':
                st.warning(f"⚠️ Found old embedding cache format for {namespace}")
                st.error("❌ Old cache uses DataFrame indices (broken)")
                st.info("💡 Delete this cache and re-generate embeddings with updated code")
                return None

            stable_id_map = data.get('stable_id_map', {})
            return stable_id_map

        except Exception as e:
            print(f"Error loading embeddings cache: {e}")
            return None

    def has_embeddings(self, namespace: str) -> bool:
        """Check if embeddings cache exists (in either location)"""
        primary = self.cache_dir / f"{namespace}_embeddings.json"
        fallback = self.fallback_cache_dir / f"{namespace}_embeddings.json"
        return primary.exists() or (self.fallback_cache_dir.exists() and fallback.exists())

    # ========================================================================
    # EMBEDDINGS CACHE
    # ========================================================================

    def save_embeddings(self, namespace: str, stable_id_map: Dict[str, str]) -> Path:
        """
        Save stable_id -> pinecone_id mapping to disk

        Args:
            namespace: Unique identifier for this dataset
            stable_id_map: Dict mapping stable_id -> pinecone_id

        Returns:
            Path to saved file
        """
        cache_file = self.cache_dir / f"{namespace}_embeddings.json"

        # All keys are already strings (stable_ids are hex strings)
        with open(cache_file, 'w') as f:
            json.dump({
                'namespace': namespace,
                'created_at': datetime.now().isoformat(),
                'num_passages': len(stable_id_map),
                'stable_id_map': stable_id_map,  # Changed from passage_id_map
                'format_version': '2.0',  # Mark as new format
            }, f, indent=2)

        return cache_file

    # ========================================================================
    # SCORES CACHE
    # ========================================================================

    def save_scores(self, namespace: str, scores_df: pd.DataFrame) -> Path:
        """
        Save quality scores to disk

        Args:
            namespace: Unique identifier for this dataset
            scores_df: DataFrame with quality scores

        Returns:
            Path to saved file
        """
        cache_file = self.cache_dir / f"{namespace}_scores.parquet"

        # Add metadata
        scores_df_copy = scores_df.copy()
        scores_df_copy.attrs['namespace'] = namespace
        scores_df_copy.attrs['created_at'] = datetime.now().isoformat()

        scores_df_copy.to_parquet(cache_file, index=False)

        return cache_file

    def load_scores(self, namespace: str) -> Optional[pd.DataFrame]:
        """
        Load quality scores from disk
        Checks both primary and fallback locations
        """
        # Try primary first
        cache_file = self.cache_dir / f"{namespace}_scores.parquet"

        # Fall back to _data if not found
        if not cache_file.exists() and self.fallback_cache_dir.exists():
            cache_file = self.fallback_cache_dir / f"{namespace}_scores.parquet"

        if not cache_file.exists():
            return None

        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Error loading scores cache: {e}")
            return None

    def has_scores(self, namespace: str) -> bool:
        """Check if scores cache exists (in either location)"""
        primary = self.cache_dir / f"{namespace}_scores.parquet"
        fallback = self.fallback_cache_dir / f"{namespace}_scores.parquet"
        return primary.exists() or (self.fallback_cache_dir.exists() and fallback.exists())


