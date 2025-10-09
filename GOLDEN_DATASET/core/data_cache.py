"""
Persistent Cache Manager for Embeddings and Scores

Handles saving/loading of:
- Embeddings (passage_id_map) to/from disk
- Quality scores to/from disk
- Auto-detection of existing caches
- Checkpoint system for long operations
"""

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

    # ========================================================================
    # EMBEDDINGS CACHE
    # ========================================================================

    def save_embeddings(self, namespace: str, passage_id_map: Dict[int, str]) -> Path:
        """
        Save passage_id_map to disk

        Args:
            namespace: Unique identifier for this dataset
            passage_id_map: Dict mapping df indices to Pinecone IDs

        Returns:
            Path to saved file
        """
        cache_file = self.cache_dir / f"{namespace}_embeddings.json"

        # Convert int keys to strings for JSON
        serializable_map = {str(k): v for k, v in passage_id_map.items()}

        with open(cache_file, 'w') as f:
            json.dump({
                'namespace': namespace,
                'created_at': datetime.now().isoformat(),
                'num_passages': len(passage_id_map),
                'passage_id_map': serializable_map
            }, f, indent=2)

        return cache_file

    def load_embeddings(self, namespace: str) -> Optional[Dict[int, str]]:
        """
        Load passage_id_map from disk

        Args:
            namespace: Unique identifier for this dataset

        Returns:
            Dict mapping df indices to Pinecone IDs, or None if not found
        """
        cache_file = self.cache_dir / f"{namespace}_embeddings.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Convert string keys back to ints
            passage_id_map = {int(k): v for k, v in data['passage_id_map'].items()}

            return passage_id_map
        except Exception as e:
            print(f"Error loading embeddings cache: {e}")
            return None

    def has_embeddings(self, namespace: str) -> bool:
        """Check if embeddings cache exists"""
        cache_file = self.cache_dir / f"{namespace}_embeddings.json"
        return cache_file.exists()

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

        Args:
            namespace: Unique identifier for this dataset

        Returns:
            DataFrame with quality scores, or None if not found
        """
        cache_file = self.cache_dir / f"{namespace}_scores.parquet"

        if not cache_file.exists():
            return None

        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Error loading scores cache: {e}")
            return None

    def has_scores(self, namespace: str) -> bool:
        """Check if scores cache exists"""
        cache_file = self.cache_dir / f"{namespace}_scores.parquet"
        return cache_file.exists()

    # ========================================================================
    # CHECKPOINT SYSTEM
    # ========================================================================

    def save_checkpoint(
        self,
        namespace: str,
        checkpoint_type: str,
        data: Dict
    ) -> Path:
        """
        Save a checkpoint during long operations

        Args:
            namespace: Dataset namespace
            checkpoint_type: Type of checkpoint (e.g., 'embedding', 'scoring')
            data: Checkpoint data

        Returns:
            Path to checkpoint file
        """
        checkpoint_file = self.cache_dir / f"{namespace}_{checkpoint_type}_checkpoint.json"

        with open(checkpoint_file, 'w') as f:
            json.dump({
                'namespace': namespace,
                'type': checkpoint_type,
                'timestamp': datetime.now().isoformat(),
                **data
            }, f, indent=2)

        return checkpoint_file

    def load_checkpoint(
        self,
        namespace: str,
        checkpoint_type: str
    ) -> Optional[Dict]:
        """Load a checkpoint"""
        checkpoint_file = self.cache_dir / f"{namespace}_{checkpoint_type}_checkpoint.json"

        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None

    def clear_checkpoint(self, namespace: str, checkpoint_type: str):
        """Clear a checkpoint after successful completion"""
        checkpoint_file = self.cache_dir / f"{namespace}_{checkpoint_type}_checkpoint.json"

        if checkpoint_file.exists():
            checkpoint_file.unlink()

    # ========================================================================
    # CACHE MANAGEMENT
    # ========================================================================

    def list_caches(self) -> List[Dict]:
        """List all cached datasets"""
        caches = []

        # Find all embeddings files
        for emb_file in self.cache_dir.glob("*_embeddings.json"):
            namespace = emb_file.stem.replace('_embeddings', '')

            cache_info = {
                'namespace': namespace,
                'has_embeddings': True,
                'has_scores': self.has_scores(namespace)
            }

            # Get metadata from embeddings file
            try:
                with open(emb_file, 'r') as f:
                    data = json.load(f)
                    cache_info['created_at'] = data.get('created_at')
                    cache_info['num_passages'] = data.get('num_passages')
            except:
                pass

            caches.append(cache_info)

        return caches

    def clear_cache(self, namespace: str):
        """Clear all caches for a namespace"""
        patterns = [
            f"{namespace}_embeddings.json",
            f"{namespace}_scores.parquet",
            f"{namespace}_*_checkpoint.json"
        ]

        for pattern in patterns:
            for file in self.cache_dir.glob(pattern):
                file.unlink()

    def get_cache_size(self, namespace: str) -> int:
        """Get total size of caches for a namespace in bytes"""
        total_size = 0

        patterns = [
            f"{namespace}_embeddings.json",
            f"{namespace}_scores.parquet"
        ]

        for pattern in patterns:
            for file in self.cache_dir.glob(pattern):
                total_size += file.stat().st_size

        return total_size

    def cleanup_old_caches(self, days: int = 30):
        """Remove caches older than specified days"""
        cutoff = datetime.now().timestamp() - (days * 86400)

        for file in self.cache_dir.glob("*"):
            if file.stat().st_mtime < cutoff:
                file.unlink()
                print(f"Removed old cache: {file.name}")