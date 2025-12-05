"""
Data Preparation Module for HRAF Golden Dataset Discovery
Intelligent data manipulation, cleaning, tiering, and export
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import io
import shutil
import yaml


# ============================================================================
# DATA ANALYSIS & QUALITY
# ============================================================================

class DataAnalyzer:
    """Analyzes data quality and suggests improvements"""

    def __init__(self, df: pd.DataFrame, label_columns: List[str], passage_col: str):
        self.df = df
        self.label_columns = label_columns
        self.passage_col = passage_col

    def analyze_quality(self) -> Dict:
        """Comprehensive data quality analysis"""
        issues = []
        suggestions = []
        stats = {}

        # Check for missing passages
        missing_passages = self.df[self.passage_col].isna().sum()
        if missing_passages > 0:
            pct = (missing_passages / len(self.df)) * 100
            issues.append(f"Missing passages: {missing_passages} ({pct:.1f}%)")
            suggestions.append("Remove rows with missing passages")

        # Check passage lengths
        lengths = self.df[self.passage_col].dropna().str.len()
        stats['passage_length'] = {
            'mean': float(lengths.mean()),
            'median': float(lengths.median()),
            'min': int(lengths.min()),
            'max': int(lengths.max()),
            'std': float(lengths.std())
        }

        # Very short passages
        very_short = (lengths < 50).sum()
        if very_short > 0:
            pct = (very_short / len(lengths)) * 100
            issues.append(f"Very short passages (<50 chars): {very_short} ({pct:.1f}%)")
            suggestions.append("Consider removing passages with <50 characters")

        # Very long passages (may be truncated)
        very_long = (lengths > 2000).sum()
        if very_long > 0:
            pct = (very_long / len(lengths)) * 100
            issues.append(f"Very long passages (>2000 chars): {very_long} ({pct:.1f}%)")
            suggestions.append("Long passages will be truncated at 512 tokens (~2048 chars)")

        # Check for duplicates
        duplicates = self.df[self.passage_col].duplicated().sum()
        if duplicates > 0:
            pct = (duplicates / len(self.df)) * 100
            issues.append(f"Duplicate passages: {duplicates} ({pct:.1f}%)")
            suggestions.append("Consider deduplicating passages")

        # Label distribution
        label_stats = {}
        imbalanced_labels = []

        for label in self.label_columns:
            count = self.df[label].sum()
            pct = (count / len(self.df)) * 100
            label_stats[label] = {
                'count': int(count),
                'percentage': float(pct)
            }

            if pct < 2:
                imbalanced_labels.append(f"{label} ({count}, {pct:.1f}%)")

        stats['label_distribution'] = label_stats

        if imbalanced_labels:
            issues.append(f"Severely imbalanced labels: {len(imbalanced_labels)}")
            suggestions.append("Use weighted loss or focal loss for training")

        # Passages with no labels
        no_labels = (self.df[self.label_columns].sum(axis=1) == 0).sum()
        if no_labels > 0:
            pct = (no_labels / len(self.df)) * 100
            issues.append(f"Passages with no labels: {no_labels} ({pct:.1f}%)")
            suggestions.append("Remove passages with no labels")

        # Passages with many labels
        many_labels = (self.df[self.label_columns].sum(axis=1) > 8).sum()
        if many_labels > 0:
            pct = (many_labels / len(self.df)) * 100
            issues.append(f"Passages with >8 labels: {many_labels} ({pct:.1f}%)")
            suggestions.append("Multi-label passages may be harder to learn")

        return {
            'issues': issues,
            'suggestions': suggestions,
            'stats': stats
        }

    def suggest_cleaning_steps(self, analysis: Dict) -> List[Dict]:
        """Generate actionable cleaning steps"""
        steps = []

        # Remove missing
        missing_passages = self.df[self.passage_col].isna().sum()
        if missing_passages > 0:
            steps.append({
                'name': 'Remove Missing Passages',
                'description': f'Remove {missing_passages} passages with missing text',
                'action': 'remove_missing',
                'impact': missing_passages,
                'recommended': True
            })

        # Remove duplicates
        duplicates = self.df[self.passage_col].duplicated().sum()
        if duplicates > 0:
            steps.append({
                'name': 'Remove Duplicates',
                'description': f'Remove {duplicates} duplicate passages',
                'action': 'remove_duplicates',
                'impact': duplicates,
                'recommended': True
            })

        # Remove unlabeled
        no_labels = (self.df[self.label_columns].sum(axis=1) == 0).sum()
        if no_labels > 0:
            steps.append({
                'name': 'Remove Unlabeled',
                'description': f'Remove {no_labels} passages with no labels',
                'action': 'remove_unlabeled',
                'impact': no_labels,
                'recommended': True
            })

        # Remove very short
        lengths = self.df[self.passage_col].dropna().str.len()
        very_short = (lengths < 50).sum()
        if very_short > 0:
            steps.append({
                'name': 'Remove Very Short',
                'description': f'Remove {very_short} passages with <50 characters',
                'action': 'remove_short',
                'impact': very_short,
                'recommended': True
            })

        # Optional: Remove very long
        very_long = (lengths > 2000).sum()
        if very_long > 0:
            steps.append({
                'name': 'Remove Very Long',
                'description': f'Remove {very_long} passages with >2000 characters',
                'action': 'remove_long',
                'impact': very_long,
                'recommended': False
            })

        return steps

    def apply_cleaning(self, selected_actions: List[str]) -> pd.DataFrame:
        """Apply selected cleaning steps"""
        df_clean = self.df.copy()

        if 'remove_missing' in selected_actions:
            df_clean = df_clean[df_clean[self.passage_col].notna()]

        if 'remove_duplicates' in selected_actions:
            df_clean = df_clean.drop_duplicates(subset=[self.passage_col], keep='first')

        if 'remove_unlabeled' in selected_actions:
            df_clean = df_clean[df_clean[self.label_columns].sum(axis=1) > 0]

        if 'remove_short' in selected_actions:
            lengths = df_clean[self.passage_col].str.len()
            df_clean = df_clean[lengths >= 50]

        if 'remove_long' in selected_actions:
            lengths = df_clean[self.passage_col].str.len()
            df_clean = df_clean[lengths <= 2000]

        return df_clean


# ============================================================================
# DATA SEGMENTATION & TIERING
# ============================================================================

class DataSegmenter:
    """Create quality-based data segments and tiers"""

    def __init__(self, df: pd.DataFrame, scores_df: Optional[pd.DataFrame], label_columns: List[str]):
        self.df = df
        self.scores_df = scores_df
        self.label_columns = label_columns

    def create_quality_tiers(
            self,
            tier1_config: Dict,
            tier2_config: Dict,
            label_targets: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Create quality-based tiers for curriculum learning

        Args:
            tier1_config: Config for elite tier (min/max consistency/rerank, target_size)
            tier2_config: Config for expansion tier
            label_targets: Optional label-specific targets {'tier1': {...}, 'tier2': {...}}

        Returns:
            (tier1_df, tier2_df, inference_df, metadata)
        """
        if self.scores_df is None or len(self.scores_df) == 0:
            raise ValueError("Quality scores required. Compute scores first.")

        if len(self.df) == 0:
            raise ValueError("Dataset is empty")

        # Get valid scored indices
        valid_indices = self.scores_df['passage_idx'].tolist()

        if len(valid_indices) == 0:
            raise ValueError("No valid scored passages found")

        scores_df = self.scores_df.copy()
        scores_df['composite'] = (scores_df['consistency_avg'] + scores_df['rerank_avg']) / 2

        # TIER 1: Elite training data
        tier1_mask = (
                (scores_df['consistency_avg'] >= tier1_config['min_consistency']) &
                (scores_df['consistency_avg'] <= tier1_config.get('max_consistency', 1.0)) &
                (scores_df['rerank_avg'] >= tier1_config['min_rerank']) &
                (scores_df['rerank_avg'] <= tier1_config.get('max_rerank', 1.0))
        )

        tier1_candidates = scores_df[tier1_mask].copy()

        # Apply label targeting if specified
        if label_targets and 'tier1' in label_targets:
            tier1_indices = self._apply_label_targeting(
                tier1_candidates, label_targets['tier1'], tier1_config.get('target_size', 1000)
            )
        else:
            tier1_candidates = tier1_candidates.sort_values('composite', ascending=False)
            target_count = tier1_config.get('target_size', int(len(valid_indices) * 0.12))
            tier1_indices = tier1_candidates.head(target_count)['passage_idx'].tolist()

        # TIER 2: Expansion training data
        remaining_indices = [idx for idx in valid_indices if idx not in tier1_indices]
        remaining_scores = scores_df[scores_df['passage_idx'].isin(remaining_indices)]

        tier2_mask = (
                (remaining_scores['consistency_avg'] >= tier2_config['min_consistency']) &
                (remaining_scores['consistency_avg'] <= tier2_config.get('max_consistency', 1.0)) &
                (remaining_scores['rerank_avg'] >= tier2_config['min_rerank']) &
                (remaining_scores['rerank_avg'] <= tier2_config.get('max_rerank', 1.0))
        )

        tier2_candidates = remaining_scores[tier2_mask].copy()

        # Apply label targeting if specified
        if label_targets and 'tier2' in label_targets:
            tier2_indices = self._apply_label_targeting(
                tier2_candidates, label_targets['tier2'], tier2_config.get('target_size', 2000)
            )
        else:
            tier2_candidates = tier2_candidates.sort_values('composite', ascending=False)
            target_count = tier2_config.get('target_size', int(len(valid_indices) * 0.25))
            tier2_indices = tier2_candidates.head(target_count)['passage_idx'].tolist()

        # INFERENCE: Everything else
        inference_indices = [idx for idx in valid_indices
                             if idx not in tier1_indices and idx not in tier2_indices]

        # Create dataframes
        tier1_df = self.df.loc[tier1_indices].copy()
        tier2_df = self.df.loc[tier2_indices].copy()
        inference_df = self.df.loc[inference_indices].copy()

        # Add confidence scores
        for tier_df, indices in [(tier1_df, tier1_indices),
                                 (tier2_df, tier2_indices),
                                 (inference_df, inference_indices)]:
            for idx in indices:
                if idx in scores_df['passage_idx'].values:
                    score_row = scores_df[scores_df['passage_idx'] == idx].iloc[0]
                    tier_df.loc[idx, 'confidence_composite'] = score_row['composite']
                    tier_df.loc[idx, 'confidence_consistency'] = score_row['consistency_avg']
                    tier_df.loc[idx, 'confidence_rerank'] = score_row['rerank_avg']

        # Generate metadata
        metadata = self._generate_tier_metadata(tier1_df, tier2_df, inference_df)

        return tier1_df, tier2_df, inference_df, metadata

    def _apply_label_targeting(
            self,
            candidates: pd.DataFrame,
            targets: Dict,
            target_size: int
    ) -> List[int]:
        """Select passages to meet label-specific targets"""
        selected_indices = []
        remaining_candidates = candidates.copy()

        # Priority labels first
        for label, target_count in sorted(targets.items(), key=lambda x: x[1], reverse=True):
            if label not in self.label_columns:
                continue

            # Find candidates with this label
            label_candidates = []
            for idx in remaining_candidates['passage_idx'].tolist():
                if idx in self.df.index and self.df.loc[idx, label] == 1:
                    label_candidates.append(idx)

            # Take up to target_count
            selected = label_candidates[:target_count]
            selected_indices.extend(selected)

            # Remove from candidates
            remaining_candidates = remaining_candidates[
                ~remaining_candidates['passage_idx'].isin(selected)
            ]

        # Fill remaining with top-scoring passages
        remaining_needed = target_size - len(selected_indices)
        if remaining_needed > 0:
            remaining_candidates = remaining_candidates.sort_values('composite', ascending=False)
            additional = remaining_candidates.head(remaining_needed)['passage_idx'].tolist()
            selected_indices.extend(additional)

        return selected_indices[:target_size]

    def _generate_tier_metadata(
            self,
            tier1_df: pd.DataFrame,
            tier2_df: pd.DataFrame,
            inference_df: pd.DataFrame
    ) -> Dict:
        """Generate comprehensive tier metadata"""
        from datetime import datetime

        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_passages': len(tier1_df) + len(tier2_df) + len(inference_df),
            'tiers': {}
        }

        for tier_name, tier_df in [('tier1', tier1_df), ('tier2', tier2_df), ('inference', inference_df)]:
            tier_meta = {
                'count': len(tier_df),
                'percentage': len(tier_df) / metadata['total_passages'] * 100,
            }

            # Quality statistics
            if 'confidence_consistency' in tier_df.columns:
                tier_meta['quality'] = {
                    'consistency_mean': float(tier_df['confidence_consistency'].mean()),
                    'consistency_median': float(tier_df['confidence_consistency'].median()),
                    'rerank_mean': float(tier_df['confidence_rerank'].mean()),
                    'rerank_median': float(tier_df['confidence_rerank'].median()),
                    'composite_mean': float(tier_df['confidence_composite'].mean()),
                }

            # Label distribution
            label_dist = {}
            for label in self.label_columns:
                if label in tier_df.columns:
                    count = int((tier_df[label] == 1).sum())
                    label_dist[label] = {
                        'count': count,
                        'percentage': count / len(tier_df) * 100 if len(tier_df) > 0 else 0
                    }
            tier_meta['label_distribution'] = label_dist

            metadata['tiers'][tier_name] = tier_meta

        return metadata

    def create_custom_segment(self, filters: Dict) -> pd.DataFrame:
        """Create custom data segment based on filters"""
        df_filtered = self.df.copy()

        # Label filters
        if 'required_labels' in filters and filters['required_labels']:
            for label in filters['required_labels']:
                df_filtered = df_filtered[df_filtered[label] == 1]

        if 'excluded_labels' in filters and filters['excluded_labels']:
            for label in filters['excluded_labels']:
                df_filtered = df_filtered[df_filtered[label] == 0]

        # Label count filter
        if 'min_labels' in filters:
            label_count = df_filtered[self.label_columns].sum(axis=1)
            df_filtered = df_filtered[label_count >= filters['min_labels']]

        if 'max_labels' in filters:
            label_count = df_filtered[self.label_columns].sum(axis=1)
            df_filtered = df_filtered[label_count <= filters['max_labels']]

        # Quality filters (if scores available)
        if self.scores_df is not None:
            scored_indices = self.scores_df['passage_idx'].tolist()
            df_filtered = df_filtered[df_filtered.index.isin(scored_indices)]

            if 'min_consistency' in filters:
                valid_indices = self.scores_df[
                    self.scores_df['consistency_avg'] >= filters['min_consistency']
                    ]['passage_idx'].tolist()
                df_filtered = df_filtered[df_filtered.index.isin(valid_indices)]

            if 'min_rerank' in filters:
                valid_indices = self.scores_df[
                    self.scores_df['rerank_avg'] >= filters['min_rerank']
                    ]['passage_idx'].tolist()
                df_filtered = df_filtered[df_filtered.index.isin(valid_indices)]

        return df_filtered

class DataExperiment:
    """Manages data experiments with full lineage tracking"""

    def __init__(self, base_dir: str = "data/experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(
            self,
            name: str,
            df: pd.DataFrame,
            experiment_type: str,
            metadata: Dict,
            session_state: Dict
    ) -> Path:
        """
        Create a new data experiment with full tracking

        Args:
            name: Experiment name (will be sanitized)
            df: DataFrame to save
            experiment_type: 'cleaned', 'segment', 'tier', etc.
            metadata: Additional metadata
            session_state: Streamlit session state for lineage

        Returns:
            Path to experiment directory
        """
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = self._sanitize_name(name)
        exp_dir = self.base_dir / f"{safe_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save data
        data_path = exp_dir / "data.xlsx"
        df.to_excel(data_path, index=False, engine='openpyxl')

        # Build comprehensive metadata
        full_metadata = self._build_metadata(
            df, experiment_type, metadata, session_state
        )

        # Save metadata
        meta_path = exp_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)

        # Generate README
        readme_path = exp_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(self._generate_readme(safe_name, full_metadata))

        return exp_dir

    def create_tier_experiment(
            self,
            name: str,
            tier1: pd.DataFrame,
            tier2: pd.DataFrame,
            inference: pd.DataFrame,
            tier_metadata: Dict,
            session_state: Dict
    ) -> Path:
        """Create experiment for tiered datasets"""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = self._sanitize_name(name)
        exp_dir = self.base_dir / f"{safe_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save all tiers
        tier1.to_excel(exp_dir / "tier1.xlsx", index=False)
        tier2.to_excel(exp_dir / "tier2.xlsx", index=False)
        inference.to_excel(exp_dir / "inference.xlsx", index=False)

        # Combined training set
        combined = pd.concat([tier1, tier2])
        combined.to_excel(exp_dir / "tier1_tier2_combined.xlsx", index=False)

        # Build metadata
        full_metadata = {
            'experiment_name': safe_name,
            'experiment_type': 'tiered_training',
            'created_at': datetime.now().isoformat(),
            'timestamp': timestamp,

            # Provenance
            'source': {
                'original_file': session_state.get('selected_file', 'unknown'),
                'original_namespace': session_state.get('namespace', 'unknown'),
                'working_dataset': 'cleaned' if 'cleaned_df' in session_state else 'original'
            },

            # Tier statistics
            'tiers': {
                'tier1': {
                    'count': len(tier1),
                    'percentage': len(tier1) / (len(tier1) + len(tier2) + len(inference)) * 100,
                    'file': 'tier1.xlsx'
                },
                'tier2': {
                    'count': len(tier2),
                    'percentage': len(tier2) / (len(tier1) + len(tier2) + len(inference)) * 100,
                    'file': 'tier2.xlsx'
                },
                'inference': {
                    'count': len(inference),
                    'percentage': len(inference) / (len(tier1) + len(tier2) + len(inference)) * 100,
                    'file': 'inference.xlsx'
                },
                'combined': {
                    'count': len(combined),
                    'file': 'tier1_tier2_combined.xlsx'
                }
            },

            # Configuration used
            'tier_configuration': tier_metadata,

            # Data characteristics
            'label_columns': session_state.get('label_columns', []),
            'passage_column': session_state.get('passage_col', 'Passage'),

            # Quality scores (if available)
            'quality_scores_used': session_state.get('cache') is not None,

            # Format info
            'format': {
                'header_type': 'single',
                'header_row': 0,
                'export_tool': 'HRAF_Data_Preparation_v1',
                'compatible_with': ['model_training', 'compute_scores']
            },

            # Usage recommendations
            'recommended_usage': {
                'tier1_only': 'Initial training on highest quality data',
                'tier1_tier2_combined': 'Full training with quality-stratified data',
                'inference': 'Model evaluation and testing',
                'curriculum_learning': 'Train on tier1 first, then fine-tune on combined'
            }
        }

        # Save metadata
        meta_path = exp_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)

        # Generate README
        readme_content = self._generate_tier_readme(safe_name, full_metadata)
        with open(exp_dir / "README.md", 'w') as f:
            f.write(readme_content)

        return exp_dir

    def _build_metadata(
            self,
            df: pd.DataFrame,
            experiment_type: str,
            custom_metadata: Dict,
            session_state: Dict
    ) -> Dict:
        """Build comprehensive metadata for experiment"""

        label_columns = session_state.get('label_columns', [])

        metadata = {
            'experiment_name': custom_metadata.get('name', 'unknown'),
            'experiment_type': experiment_type,
            'created_at': datetime.now().isoformat(),
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),

            # Provenance - track lineage
            'provenance': {
                'source_file': session_state.get('selected_file', 'unknown'),
                'source_namespace': session_state.get('namespace', 'unknown'),
                'parent_experiment': custom_metadata.get('parent_experiment'),
                'transformations_applied': custom_metadata.get('transformations', []),
                'working_dataset_type': 'cleaned' if 'cleaned_df' in session_state else 'original'
            },

            # Dataset statistics
            'statistics': {
                'num_passages': len(df),
                'num_columns': len(df.columns),
                'columns': list(df.columns),
                'label_columns': label_columns,
                'passage_column': session_state.get('passage_col', 'Passage')
            },

            # Label distribution
            'label_distribution': {},

            # Quality metrics (if available)
            'quality_metrics': {},

            # Configuration
            'configuration': custom_metadata.get('configuration', {}),

            # Format information
            'format': {
                'header_type': 'single',
                'header_row': 0,
                'export_tool': 'HRAF_Data_Preparation_v1',
                'compatible_with': ['model_training', 'compute_scores', 'data_prep']
            }
        }

        # Calculate label distribution
        for label in label_columns:
            if label in df.columns:
                count = int((df[label] == 1).sum())
                metadata['label_distribution'][label] = {
                    'count': count,
                    'percentage': float(count / len(df) * 100) if len(df) > 0 else 0
                }

        # Add quality metrics if available
        cache = session_state.get('cache')
        if cache and 'df_summary' in cache:
            scores_df = cache['df_summary']
            valid_indices = [idx for idx in df.index if idx in scores_df['passage_idx'].values]

            if valid_indices:
                subset_scores = scores_df[scores_df['passage_idx'].isin(valid_indices)]
                metadata['quality_metrics'] = {
                    'consistency_mean': float(subset_scores['consistency_avg'].mean()),
                    'consistency_median': float(subset_scores['consistency_avg'].median()),
                    'rerank_mean': float(subset_scores['rerank_avg'].mean()),
                    'rerank_median': float(subset_scores['rerank_avg'].median()),
                    'scored_passages': len(subset_scores)
                }

        # Merge custom metadata
        metadata.update(custom_metadata)

        return metadata

    def _generate_readme(self, name: str, metadata: Dict) -> str:
        """Generate README for experiment"""

        readme = f"""# Data Experiment: {name}

**Created:** {metadata['created_at']}  
**Type:** {metadata['experiment_type']}

## Overview

This dataset was created using the HRAF Data Preparation tool.

### Dataset Statistics
- **Passages:** {metadata['statistics']['num_passages']:,}
- **Labels:** {len(metadata['statistics']['label_columns'])}
- **Columns:** {metadata['statistics']['num_columns']}

## Provenance

**Source File:** `{metadata['provenance']['source_file']}`  
**Working Dataset:** {metadata['provenance']['working_dataset_type']}

"""

        # Add transformations if any
        if metadata['provenance'].get('transformations_applied'):
            readme += "\n### Transformations Applied\n\n"
            for transform in metadata['provenance']['transformations_applied']:
                readme += f"- {transform}\n"

        # Add label distribution
        readme += "\n## Label Distribution\n\n"
        readme += "| Label | Count | Percentage |\n"
        readme += "|-------|-------|------------|\n"

        for label, info in metadata['label_distribution'].items():
            readme += f"| {label} | {info['count']} | {info['percentage']:.1f}% |\n"

        # Add quality metrics if available
        if metadata.get('quality_metrics'):
            qm = metadata['quality_metrics']
            readme += f"\n## Quality Metrics\n\n"
            readme += f"- **Consistency Mean:** {qm['consistency_mean']:.3f}\n"
            readme += f"- **Consistency Median:** {qm['consistency_median']:.3f}\n"
            readme += f"- **Rerank Mean:** {qm['rerank_mean']:.3f}\n"
            readme += f"- **Rerank Median:** {qm['rerank_median']:.3f}\n"
            readme += f"- **Scored Passages:** {qm['scored_passages']}\n"

        # Add usage instructions
        readme += f"\n## Usage\n\n"
        readme += f"### Loading in Python\n\n"
        readme += f"```python\n"
        readme += f"import pandas as pd\n\n"
        readme += f"df = pd.read_excel('data.xlsx')\n"
        readme += f"```\n\n"
        readme += f"### Using in HRAF Tool\n\n"
        readme += f"1. Go to **Train Model** page\n"
        readme += f"2. Select this experiment directory\n"
        readme += f"3. File: `data.xlsx`\n\n"
        readme += f"### Metadata\n\n"
        readme += f"Full metadata available in `metadata.json`\n"

        return readme

    def _generate_tier_readme(self, name: str, metadata: Dict) -> str:
        """Generate README for tiered experiment"""

        tier_info = metadata['tiers']

        readme = f"""# Tiered Training Experiment: {name}

        **Created:** {metadata['created_at']}  
        **Type:** Quality-Based Tiered Training Data

        ## Overview

        This experiment contains quality-stratified training data for curriculum learning.

        ### Tier Statistics

        | Tier | Count | Percentage | Purpose |
        |------|-------|------------|---------|
        | Tier 1 (Elite) | {tier_info['tier1']['count']:,} | {tier_info['tier1']['percentage']:.1f}% | Initial training |
        | Tier 2 (Expansion) | {tier_info['tier2']['count']:,} | {tier_info['tier2']['percentage']:.1f}% | Generalization |
        | Inference (Test) | {tier_info['inference']['count']:,} | {tier_info['inference']['percentage']:.1f}% | Evaluation |
        | **Combined** | {tier_info['combined']['count']:,} | - | Full training |

        ## Files

        - **`tier1.xlsx`** - {tier_info['tier1']['count']} highest quality passages
        - **`tier2.xlsx`** - {tier_info['tier2']['count']} good quality passages  
        - **`tier1_tier2_combined.xlsx`** - {tier_info['combined']['count']} combined training data
        - **`inference.xlsx`** - {tier_info['inference']['count']} test/validation data
        - **`metadata.json`** - Complete experiment metadata
        - **`README.md`** - This file

        ## Provenance

        **Source:** `{metadata['source']['original_file']}`  
        **Dataset Type:** {metadata['source']['working_dataset']}  
        **Quality Scores:** {'Yes' if metadata['quality_scores_used'] else 'No'}

        ## Training Strategies

        ### Strategy 1: Curriculum Learning (Recommended)Stage 1 (Epochs 1-5): Train on tier1.xlsx
        └─ Learn from highest quality examplesStage 2 (Epochs 6-10): Fine-tune on tier1_tier2_combined.xlsx
        └─ Generalize to broader patternsStage 3: Evaluate on inference.xlsx
        └─ Final model testing

        ### Strategy 2: Single-Pass TrainingTrain on tier1_tier2_combined.xlsx for full epochs
        └─ Use all training data from start

        ### Strategy 3: Elite-Only TrainingTrain on tier1.xlsx only
        └─ Maximum quality, smaller dataset

        ## Label Distribution

        """

        # Add label distribution (would need to be calculated)
        readme += "\nSee `metadata.json` for detailed label distribution per tier.\n"

        readme += f"""
            ## Usage in HRAF Tool

            ### Loading for Training

            1. Navigate to **Train Model** page
            2. Under "Dataset Selection", choose **Tiered Datasets**
            3. Select training strategy:
               - **Tier 1 Only** → Use `tier1.xlsx`
               - **Tier 1 + Tier 2** → Use `tier1_tier2_combined.xlsx`
               - **Curriculum** → Train on tier1 first, then combined

            ### Configuration

            Tier configuration used to create this dataset is in `metadata.json` under `tier_configuration`.

            ## Quality Thresholds

            This experiment was created with the following quality criteria:

            """

        # Add tier config if available
        if 'tier_configuration' in metadata:
            tier_config = metadata['tier_configuration']
            if 'tiers' in tier_config:
                for tier_name, tier_data in tier_config['tiers'].items():
                    if 'quality' in tier_data:
                        q = tier_data['quality']
                        readme += f"\n### {tier_name.title()}\n"
                        readme += f"- Consistency: {q['consistency_mean']:.3f}\n"
                        readme += f"- Rerank: {q['rerank_mean']:.3f}\n"

        return readme

    def list_experiments(self) -> List[Dict]:
        """List all experiments with metadata"""
        experiments = []

        if not self.base_dir.exists():
            return experiments

        for exp_dir in sorted(self.base_dir.iterdir(), reverse=True):
            if exp_dir.is_dir():
                meta_path = exp_dir / "metadata.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, 'r') as f:
                            metadata = json.load(f)

                        experiments.append({
                            'directory': exp_dir,
                            'name': exp_dir.name,
                            'metadata': metadata
                        })
                    except:
                        pass

        return experiments

    def _sanitize_name(self, name: str) -> str:
        """Sanitize experiment name for filesystem"""
        import re
        safe = re.sub(r'[^\w\-_\.]', '_', name)
        safe = re.sub(r'_+', '_', safe)
        return safe.strip('_')[:50]  # Limit length


# Update save functions to use DataExperiment
def save_to_data_directory(df: pd.DataFrame, name: str, session_state: Dict, experiment_type: str = 'custom'):
    """Save dataframe as data experiment"""

    experiment = DataExperiment()

    # Gather metadata
    metadata = {
        'name': name,
        'experiment_type': experiment_type,
        'transformations': session_state.get('applied_transformations', []),
        'configuration': session_state.get('segment_filters', {})
    }

    try:
        exp_dir = experiment.create_experiment(
            name=name,
            df=df,
            experiment_type=experiment_type,
            metadata=metadata,
            session_state=session_state
        )

        # FIX: Resolve to absolute path before displaying
        exp_dir_abs = exp_dir.resolve()

        # Try to get relative path, but fall back to absolute if it fails
        try:
            display_path = exp_dir_abs.relative_to(Path.cwd().resolve())
        except ValueError:
            # If not in subpath, just show the full path
            display_path = exp_dir_abs

        st.success(f"✅ Experiment created: `{exp_dir.name}`")
        st.info(f"""
        📁 **Experiment Directory:** `{display_path}`

        **Files created:**
        - `data.xlsx` - Dataset
        - `metadata.json` - Full metadata with lineage
        - `README.md` - Human-readable documentation

        💡 This experiment is now available for:
        - Training models (Train Model page)
        - Computing scores (Compute Scores page)
        - Further data preparation
        """)

        # Add to session state for tracking
        if 'data_experiments' not in session_state:
            session_state['data_experiments'] = []

        session_state['data_experiments'].append({
            'name': name,
            'path': str(exp_dir_abs),
            'created_at': datetime.now().isoformat()
        })

    except Exception as e:
        st.error(f"Error creating experiment: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())

def save_tiers_to_data_directory(
        tier1: pd.DataFrame,
        tier2: pd.DataFrame,
        inference: pd.DataFrame,
        name: str,
        session_state: Dict
):
    """Save tiers as data experiment"""

    experiment = DataExperiment()

    # Get tier metadata from session
    tier_metadata = session_state.get('tier_metadata', {})

    try:
        exp_dir = experiment.create_tier_experiment(
            name=name,
            tier1=tier1,
            tier2=tier2,
            inference=inference,
            tier_metadata=tier_metadata,
            session_state=session_state
        )

        # FIX: Resolve to absolute path before displaying
        exp_dir_abs = exp_dir.resolve()

        # Try to get relative path, but fall back to absolute if it fails
        try:
            display_path = exp_dir_abs.relative_to(Path.cwd().resolve())
        except ValueError:
            # If not in subpath, just show the full path
            display_path = exp_dir_abs

        st.success(f"✅ Tier experiment created: `{exp_dir.name}`")
        st.info(f"""
        📁 **Experiment Directory:** `{display_path}`

        **Files created:**
        - `tier1.xlsx` ({len(tier1)} passages)
        - `tier2.xlsx` ({len(tier2)} passages)
        - `inference.xlsx` ({len(inference)} passages)
        - `tier1_tier2_combined.xlsx` ({len(tier1) + len(tier2)} passages)
        - `metadata.json` - Complete tier configuration
        - `README.md` - Training strategies and usage

        💡 Use these files in Train Model page with different strategies
        """)

        # Add to session state
        if 'data_experiments' not in session_state:
            session_state['data_experiments'] = []

        session_state['data_experiments'].append({
            'name': name,
            'path': str(exp_dir_abs),
            'type': 'tiered',
            'created_at': datetime.now().isoformat()
        })

    except Exception as e:
        st.error(f"Error creating tier experiment: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())


def create_training_package(session_state: Dict, name: str):
    """Create complete training package with README"""

    import zipfile

    tier1 = session_state['tier1_dataset']
    tier2 = session_state['tier2_dataset']
    inference = session_state['inference_dataset']
    label_columns = session_state.get('label_columns', [])

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Save datasets
        for tier_name, tier_df in [('tier1', tier1), ('tier2', tier2), ('inference', inference)]:
            excel_buffer = io.BytesIO()
            tier_df.to_excel(excel_buffer, index=False)
            zip_file.writestr(f'datasets/{name}_{tier_name}.xlsx', excel_buffer.getvalue())

        # Combined training set
        combined = pd.concat([tier1, tier2])
        combined_buffer = io.BytesIO()
        combined.to_excel(combined_buffer, index=False)
        zip_file.writestr(f'datasets/{name}_tier1_tier2_combined.xlsx', combined_buffer.getvalue())

        # Metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'tier1_count': len(tier1),
            'tier2_count': len(tier2),
            'inference_count': len(inference),
            'label_columns': label_columns
        }
        zip_file.writestr('metadata.json', json.dumps(metadata, indent=2))

        # README
        readme = f"""# HRAF Training Package: {name}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Contents

### Datasets
- `tier1.xlsx` - {len(tier1)} elite training passages (highest quality)
- `tier2.xlsx` - {len(tier2)} expansion passages (good quality)
- `tier1_tier2_combined.xlsx` - {len(combined)} combined training passages
- `inference.xlsx` - {len(inference)} test/validation passages

### Training Protocol

1. **Stage 1: Foundation** (Epochs 1-5)
   - Use: tier1.xlsx
   - Learn from highest quality examples

2. **Stage 2: Expansion** (Epochs 6-10)
   - Use: tier1_tier2_combined.xlsx
   - Generalize to broader patterns

3. **Stage 3: Evaluation**
   - Use: inference.xlsx
   - Final model testing

### Label Columns
{chr(10).join('- ' + label for label in label_columns)}

## Usage

Load any dataset file on the Train Model page to begin training.
"""
        zip_file.writestr('README.md', readme)

    st.download_button(
        label="📥 Download Training Package",
        data=zip_buffer.getvalue(),
        file_name=f"{name}.zip",
        mime="application/zip"
    )

