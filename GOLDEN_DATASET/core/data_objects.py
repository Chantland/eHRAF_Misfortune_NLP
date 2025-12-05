"""
Data Objects & Pipeline Management

Defines:
- DataObject: Immutable data at a specific pipeline stage
- DataPipeline: Manages transformations between stages
- Integration with CacheManager and DataExperiment
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


class PipelineStage(Enum):
    """Data pipeline stages"""
    RAW = "raw"
    CLEANED = "cleaned"
    EMBEDDED = "embedded"
    SCORED = "scored"
    TIERED = "tiered"


@dataclass
class DataObject:
    """
    Immutable data object at a specific pipeline stage

    Represents data + metadata + caches at one point in the pipeline
    """
    name: str
    stage: PipelineStage
    df: pd.DataFrame

    # Configuration
    passage_col: str
    label_columns: List[str]
    metadata_columns: List[str]

    # Lineage
    parent: Optional[str] = None  # Parent data object name
    created_at: str = None

    # Namespace for embeddings/scores
    namespace: str = None

    # Caches (loaded from disk if available)
    embeddings_cache: Optional[Dict] = None  # passage_id_map
    scores_cache: Optional[pd.DataFrame] = None  # quality scores

    # Metadata
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

        if self.namespace is None:
            self.namespace = self.name.lower().replace(' ', '_')

        if self.metadata is None:
            self.metadata = {}

    @property
    def has_embeddings(self) -> bool:
        """Check if embeddings are available"""
        return self.embeddings_cache is not None and len(self.embeddings_cache) > 0

    @property
    def has_scores(self) -> bool:
        """Check if scores are available"""
        return self.scores_cache is not None and len(self.scores_cache) > 0

    def summary(self) -> Dict[str, Any]:
        """Get summary information"""
        return {
            'name': self.name,
            'stage': self.stage.value,
            'num_passages': len(self.df),
            'num_labels': len(self.label_columns),
            'has_embeddings': self.has_embeddings,
            'has_scores': self.has_scores,
            'parent': self.parent,
            'created_at': self.created_at
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'stage': self.stage.value,
            'passage_col': self.passage_col,
            'label_columns': self.label_columns,
            'metadata_columns': self.metadata_columns,
            'parent': self.parent,
            'created_at': self.created_at,
            'namespace': self.namespace,
            'has_embeddings': self.has_embeddings,
            'has_scores': self.has_scores,
            'num_passages': len(self.df),
            'metadata': self.metadata
        }


class DataObjectManager:
    """
    Manages saving/loading of DataObjects

    Integrates:
    - CacheManager for embeddings/scores
    - DataExperiment for directory structure
    - File I/O for data and metadata
    """

    def __init__(self, base_dir: str = "./data/objects"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Import here to avoid circular dependency
        from core.data_cache import CacheManager
        self.cache_manager = CacheManager()

    def save(self, data_obj: DataObject) -> Path:
        """
        Save DataObject to disk with all caches

        Returns:
            Path to saved object directory
        """
        # Create directory
        obj_dir = self.base_dir / data_obj.stage.value / data_obj.name
        obj_dir.mkdir(parents=True, exist_ok=True)

        # Save dataframe
        df_path = obj_dir / "data.xlsx"
        data_obj.df.to_excel(df_path, index=False, engine='openpyxl')

        # Save embeddings cache if exists
        if data_obj.has_embeddings:
            self.cache_manager.save_embeddings(
                data_obj.namespace,
                data_obj.embeddings_cache
            )

        # Save scores cache if exists
        if data_obj.has_scores:
            self.cache_manager.save_scores(
                data_obj.namespace,
                data_obj.scores_cache
            )

        # Save metadata
        metadata = data_obj.to_dict()
        metadata['directory'] = str(obj_dir)

        with open(obj_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Generate README
        self._generate_readme(obj_dir, data_obj)

        return obj_dir

    def load(self, name: str, stage: PipelineStage) -> Optional[DataObject]:
        """
        Load DataObject from disk

        Args:
            name: Object name
            stage: Pipeline stage

        Returns:
            DataObject or None if not found
        """
        obj_dir = self.base_dir / stage.value / name

        if not obj_dir.exists():
            return None

        # Load metadata
        meta_path = obj_dir / "metadata.json"
        if not meta_path.exists():
            return None

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # Load dataframe
        df_path = obj_dir / "data.xlsx"
        if not df_path.exists():
            return None

        df = pd.read_excel(df_path)

        # Load caches
        namespace = metadata.get('namespace')
        embeddings = self.cache_manager.load_embeddings(namespace)
        scores = self.cache_manager.load_scores(namespace)

        # Create DataObject
        return DataObject(
            name=metadata['name'],
            stage=PipelineStage(metadata['stage']),
            df=df,
            passage_col=metadata['passage_col'],
            label_columns=metadata['label_columns'],
            metadata_columns=metadata.get('metadata_columns', []),
            parent=metadata.get('parent'),
            created_at=metadata.get('created_at'),
            namespace=namespace,
            embeddings_cache=embeddings,
            scores_cache=scores,
            metadata=metadata.get('metadata', {})
        )

    def list_objects(self, stage: Optional[PipelineStage] = None) -> List[Dict]:
        """
        List all saved DataObjects

        Args:
            stage: Optional filter by stage

        Returns:
            List of object summaries
        """
        objects = []

        if stage:
            stages = [stage]
        else:
            stages = list(PipelineStage)

        for stage in stages:
            stage_dir = self.base_dir / stage.value

            if not stage_dir.exists():
                continue

            for obj_dir in stage_dir.iterdir():
                if not obj_dir.is_dir():
                    continue

                meta_path = obj_dir / "metadata.json"
                if not meta_path.exists():
                    continue

                try:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)

                    objects.append({
                        'name': metadata['name'],
                        'stage': metadata['stage'],
                        'directory': str(obj_dir),
                        **metadata
                    })
                except Exception as e:
                    print(f"Error loading {obj_dir}: {e}")

        return sorted(objects, key=lambda x: x.get('created_at', ''), reverse=True)

    def delete(self, name: str, stage: PipelineStage):
        """Delete a DataObject and its caches"""
        obj_dir = self.base_dir / stage.value / name

        if obj_dir.exists():
            # Load metadata to get namespace
            meta_path = obj_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                    namespace = metadata.get('namespace')

                    # Clear caches
                    if namespace:
                        self.cache_manager.clear_cache(namespace)

            # Remove directory
            import shutil
            shutil.rmtree(obj_dir)

    def _generate_readme(self, obj_dir: Path, data_obj: DataObject):
        """Generate README for data object"""
        readme = f"""# {data_obj.name}

**Stage:** {data_obj.stage.value.upper()}  
**Created:** {data_obj.created_at}  
**Parent:** {data_obj.parent or 'None'}

## Overview

- **Passages:** {len(data_obj.df):,}
- **Labels:** {len(data_obj.label_columns)}
- **Embeddings:** {'✅ Yes' if data_obj.has_embeddings else '❌ No'}
- **Quality Scores:** {'✅ Yes' if data_obj.has_scores else '❌ No'}

## Configuration

- **Passage Column:** `{data_obj.passage_col}`
- **Namespace:** `{data_obj.namespace}`

## Labels

"""
        for label in data_obj.label_columns:
            count = int((data_obj.df[label] == 1).sum())
            pct = (count / len(data_obj.df) * 100)
            readme += f"- **{label}**: {count:,} ({pct:.1f}%)\n"

        readme += f"""

## Files

- `data.xlsx` - Dataset with {len(data_obj.df):,} passages
- `metadata.json` - Complete metadata
- `README.md` - This file

"""

        if data_obj.has_embeddings:
            readme += f"- Embeddings cached in: `data/cache/{data_obj.namespace}_embeddings.json`\n"

        if data_obj.has_scores:
            readme += f"- Scores cached in: `data/cache/{data_obj.namespace}_scores.parquet`\n"

        readme += """

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
data_obj = manager.load("{name}", PipelineStage.{stage})
```
""".format(name=data_obj.name, stage=data_obj.stage.value.upper())

        with open(obj_dir / "README.md", 'w') as f:
            f.write(readme)


class DataPipeline:
    """
    Manages data transformations through pipeline stages

    Handles:
    - Stage transitions
    - Cache propagation
    - Lineage tracking
    """

    def __init__(self):
        self.manager = DataObjectManager()

    def create_raw(
            self,
            name: str,
            df: pd.DataFrame,
            passage_col: str,
            label_columns: List[str],
            metadata_columns: List[str],
            **kwargs
    ) -> DataObject:
        """Create a raw data object from initial load"""
        data_obj = DataObject(
            name=name,
            stage=PipelineStage.RAW,
            df=df,
            passage_col=passage_col,
            label_columns=label_columns,
            metadata_columns=metadata_columns,
            metadata=kwargs
        )

        self.manager.save(data_obj)
        return data_obj

    def create_cleaned(
            self,
            name: str,
            parent_obj: DataObject,
            df_cleaned: pd.DataFrame,
            cleaning_steps: List[str]
    ) -> DataObject:
        """Create cleaned data object"""
        data_obj = DataObject(
            name=name,
            stage=PipelineStage.CLEANED,
            df=df_cleaned,
            passage_col=parent_obj.passage_col,
            label_columns=parent_obj.label_columns,
            metadata_columns=parent_obj.metadata_columns,
            parent=parent_obj.name,
            namespace=parent_obj.namespace,  # ✅ PRESERVE namespace
            metadata={
                'cleaning_steps': cleaning_steps,
                'removed_passages': len(parent_obj.df) - len(df_cleaned)
            }
        )

        self.manager.save(data_obj)
        return data_obj

    def create_embedded(
            self,
            name: str,
            parent_obj: DataObject,
            embeddings_cache: Dict[int, str]
    ) -> DataObject:
        """Create embedded data object"""
        data_obj = DataObject(
            name=name,
            stage=PipelineStage.EMBEDDED,
            df=parent_obj.df,
            passage_col=parent_obj.passage_col,
            label_columns=parent_obj.label_columns,
            metadata_columns=parent_obj.metadata_columns,
            parent=parent_obj.name,
            namespace=parent_obj.namespace,  # ✅ PRESERVE namespace
            embeddings_cache=embeddings_cache,
            metadata={
                'num_embedded': len(embeddings_cache),
                'embedding_namespace': parent_obj.namespace  # Track where embeddings live
            }
        )

        self.manager.save(data_obj)
        return data_obj

    def create_scored(
            self,
            name: str,
            parent_obj: DataObject,
            scores_df: pd.DataFrame
    ) -> DataObject:
        """Create scored data object"""
        data_obj = DataObject(
            name=name,
            stage=PipelineStage.SCORED,
            df=parent_obj.df,
            passage_col=parent_obj.passage_col,
            label_columns=parent_obj.label_columns,
            metadata_columns=parent_obj.metadata_columns,
            parent=parent_obj.name,
            namespace=parent_obj.namespace,  # ✅ PRESERVE namespace
            embeddings_cache=parent_obj.embeddings_cache,
            scores_cache=scores_df,
            metadata={
                'num_scored': len(scores_df),
                'mean_consistency': float(scores_df['consistency_avg'].mean()),
                'mean_rerank': float(scores_df['rerank_avg'].mean()),
                'embedding_namespace': parent_obj.namespace
            }
        )

        self.manager.save(data_obj)
        return data_obj

    def create_tiered(
            self,
            name: str,
            parent_obj: DataObject,
            tier1_df: pd.DataFrame,
            tier2_df: pd.DataFrame,
            inference_df: pd.DataFrame,
            tier_config: Dict
    ) -> DataObject:
        """Create tiered data object"""
        combined_df = pd.concat([tier1_df, tier2_df])

        data_obj = DataObject(
            name=name,
            stage=PipelineStage.TIERED,
            df=combined_df,
            passage_col=parent_obj.passage_col,
            label_columns=parent_obj.label_columns,
            metadata_columns=parent_obj.metadata_columns,
            parent=parent_obj.name,
            namespace=parent_obj.namespace,  # ✅ PRESERVE namespace
            embeddings_cache=parent_obj.embeddings_cache,
            scores_cache=parent_obj.scores_cache,
            metadata={
                'tier1_size': len(tier1_df),
                'tier2_size': len(tier2_df),
                'inference_size': len(inference_df),
                'tier_config': tier_config,
                'embedding_namespace': parent_obj.namespace
            }
        )

        obj_dir = self.manager.save(data_obj)
        tier1_df.to_excel(obj_dir / "tier1.xlsx", index=False)
        tier2_df.to_excel(obj_dir / "tier2.xlsx", index=False)
        inference_df.to_excel(obj_dir / "inference.xlsx", index=False)

        return data_obj