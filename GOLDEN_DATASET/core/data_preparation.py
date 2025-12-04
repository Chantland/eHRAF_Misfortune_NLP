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
        max_no_label_passages: int = 0

    def filter_no_label_passages(
            self,
            df: pd.DataFrame,
            max_no_label: int = 0,
            random_state: int = 42
    ) -> pd.DataFrame:
        """
        Filter out excess passages with no labels

        Args:
            df: DataFrame with label columns
            max_no_label: Maximum number of no-label passages to keep (0 = remove all, -1 = keep all)
            random_state: Random seed for sampling

        Returns:
            Filtered DataFrame
        """
        if max_no_label == -1:
            # Keep all passages
            return df.copy()

        if max_no_label == 0:
            # Remove all no-label passages
            return df[df[self.label_columns].sum(axis=1) > 0].copy()

        # Separate labeled and unlabeled
        has_labels = df[df[self.label_columns].sum(axis=1) > 0]
        no_labels = df[df[self.label_columns].sum(axis=1) == 0]

        # Sample max_no_label from unlabeled
        if len(no_labels) > max_no_label:
            no_labels_sample = no_labels.sample(n=max_no_label, random_state=random_state)
        else:
            no_labels_sample = no_labels

        # Combine
        result = pd.concat([has_labels, no_labels_sample])
        return result

    def create_quality_tiers(
            self,
            tier1_config: Dict,
            tier2_config: Dict,
            label_targets: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Create quality-based tiers for curriculum learning using STABLE IDs

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

        # ✅ FIX: Ensure we have stable_id column
        if 'passage_id' not in self.df.columns:
            raise ValueError("DataFrame must have 'passage_id' column for stable references")

        # ✅ FIX: Work with stable_ids, not DataFrame indices
        # Build mapping: stable_id -> current df.index
        stable_id_to_index = {}
        for idx in self.df.index:
            stable_id = self.df.loc[idx, 'passage_id']
            if pd.notna(stable_id):
                stable_id_to_index[stable_id] = idx

        # ✅ FIX: Ensure scores_df has stable_id column
        if 'stable_id' not in self.scores_df.columns:
            # Try to reconstruct from passage_idx
            if 'passage_idx' in self.scores_df.columns:
                self.scores_df['stable_id'] = self.scores_df['passage_idx'].map(
                    lambda idx: self.df.loc[idx, 'passage_id'] if idx in self.df.index else None
                )
            else:
                raise ValueError("scores_df missing both 'stable_id' and 'passage_idx' columns")

        # Get valid scored stable_ids (those that still exist in current DataFrame)
        valid_stable_ids = [
            sid for sid in self.scores_df['stable_id'].dropna().unique()
            if sid in stable_id_to_index
        ]

        if len(valid_stable_ids) == 0:
            raise ValueError("No valid scored passages found in current DataFrame")

        # Work with scores for valid passages only
        scores_df = self.scores_df[self.scores_df['stable_id'].isin(valid_stable_ids)].copy()
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
            tier1_stable_ids = self._apply_label_targeting_stable(
                tier1_candidates,
                label_targets['tier1'],
                tier1_config.get('target_size', 1000),
                stable_id_to_index
            )
        else:
            tier1_candidates = tier1_candidates.sort_values('composite', ascending=False)
            target_count = tier1_config.get('target_size', int(len(valid_stable_ids) * 0.12))
            tier1_stable_ids = tier1_candidates.head(target_count)['stable_id'].tolist()

        # TIER 2: Expansion training data
        remaining_stable_ids = [sid for sid in valid_stable_ids if sid not in tier1_stable_ids]
        remaining_scores = scores_df[scores_df['stable_id'].isin(remaining_stable_ids)]

        tier2_mask = (
                (remaining_scores['consistency_avg'] >= tier2_config['min_consistency']) &
                (remaining_scores['consistency_avg'] <= tier2_config.get('max_consistency', 1.0)) &
                (remaining_scores['rerank_avg'] >= tier2_config['min_rerank']) &
                (remaining_scores['rerank_avg'] <= tier2_config.get('max_rerank', 1.0))
        )

        tier2_candidates = remaining_scores[tier2_mask].copy()

        # Apply label targeting if specified
        if label_targets and 'tier2' in label_targets:
            tier2_stable_ids = self._apply_label_targeting_stable(
                tier2_candidates,
                label_targets['tier2'],
                tier2_config.get('target_size', 2000),
                stable_id_to_index
            )
        else:
            tier2_candidates = tier2_candidates.sort_values('composite', ascending=False)
            target_count = tier2_config.get('target_size', int(len(valid_stable_ids) * 0.25))
            tier2_stable_ids = tier2_candidates.head(target_count)['stable_id'].tolist()

        # INFERENCE: Everything else
        inference_stable_ids = [
            sid for sid in valid_stable_ids
            if sid not in tier1_stable_ids and sid not in tier2_stable_ids
        ]

        # ✅ FIX: Create dataframes using stable_id -> current index mapping
        tier1_indices = [stable_id_to_index[sid] for sid in tier1_stable_ids if sid in stable_id_to_index]
        tier2_indices = [stable_id_to_index[sid] for sid in tier2_stable_ids if sid in stable_id_to_index]
        inference_indices = [stable_id_to_index[sid] for sid in inference_stable_ids if sid in stable_id_to_index]

        tier1_df = self.df.loc[tier1_indices].copy()
        tier2_df = self.df.loc[tier2_indices].copy()
        inference_df = self.df.loc[inference_indices].copy()

        # Add confidence scores using stable_id
        for tier_df, stable_ids in [
            (tier1_df, tier1_stable_ids),
            (tier2_df, tier2_stable_ids),
            (inference_df, inference_stable_ids)
        ]:
            for stable_id in stable_ids:
                if stable_id not in stable_id_to_index:
                    continue

                idx = stable_id_to_index[stable_id]

                if idx not in tier_df.index:
                    continue

                # Find score row for this stable_id
                score_rows = scores_df[scores_df['stable_id'] == stable_id]
                if not score_rows.empty:
                    score_row = score_rows.iloc[0]
                    tier_df.loc[idx, 'confidence_composite'] = score_row['composite']
                    tier_df.loc[idx, 'confidence_consistency'] = score_row['consistency_avg']
                    tier_df.loc[idx, 'confidence_rerank'] = score_row['rerank_avg']

                # Filter no-label passages if requested
                if max_no_label_passages >= 0:
                    original_t1_len = len(tier1_df)
                    original_t2_len = len(tier2_df)

                    tier1_df = self.filter_no_label_passages(tier1_df, max_no_label_passages)
                    tier2_df = self.filter_no_label_passages(tier2_df, max_no_label_passages)

                    removed_t1 = original_t1_len - len(tier1_df)
                    removed_t2 = original_t2_len - len(tier2_df)

                    if removed_t1 > 0 or removed_t2 > 0:
                        st.info(f"🧹 Filtered no-label passages: Tier1 -{removed_t1}, Tier2 -{removed_t2}")

                # Generate metadata
                metadata = self._generate_tier_metadata(tier1_df, tier2_df, inference_df)

                # ADD THIS:
                metadata['no_label_filtering'] = {
                    'max_allowed': max_no_label_passages,
                    'tier1_no_labels': int((tier1_df[self.label_columns].sum(axis=1) == 0).sum()),
                    'tier2_no_labels': int((tier2_df[self.label_columns].sum(axis=1) == 0).sum()),
                    'inference_no_labels': int((inference_df[self.label_columns].sum(axis=1) == 0).sum())
                }

                return tier1_df, tier2_df, inference_df, metadata

        # Generate metadata
        metadata = self._generate_tier_metadata(tier1_df, tier2_df, inference_df)

        return tier1_df, tier2_df, inference_df, metadata

    def calculate_label_frequencies(self) -> Dict[str, Dict]:
        """Calculate label statistics for tier planning"""

        stats = {}

        for label in self.label_columns:
            count = int((self.df[label] == 1).sum())
            freq = count / len(self.df)

            stats[label] = {
                'count': count,
                'frequency': freq,
                'rarity': 'very_rare' if freq < 0.05 else 'rare' if freq < 0.15 else 'common',
                'recommended_min_tier1': max(30, int(count * 0.3)),  # At least 30 or 30% of total
                'recommended_min_tier2': max(50, int(count * 0.5))  # At least 50 or 50% of total
            }

        return stats

    def _apply_label_targeting_stable(
            self,
            candidates: pd.DataFrame,  # Has 'stable_id' column
            targets: Dict,
            target_size: int,
            stable_id_to_index: Dict[str, int]
    ) -> List[str]:
        """
        Select passages to meet label-specific targets using stable IDs

        Args:
            candidates: DataFrame with 'stable_id' and score columns
            targets: Dict mapping label -> target_count
            target_size: Total target size
            stable_id_to_index: Mapping of stable_id -> current df.index

        Returns:
            List of selected stable_ids
        """
        selected_stable_ids = []
        remaining_candidates = candidates.copy()

        # Priority labels first
        for label, target_count in sorted(targets.items(), key=lambda x: x[1], reverse=True):
            if label not in self.label_columns:
                continue

            # Find candidates with this label
            label_stable_ids = []
            for stable_id in remaining_candidates['stable_id'].tolist():
                if stable_id not in stable_id_to_index:
                    continue

                idx = stable_id_to_index[stable_id]

                if idx not in self.df.index:
                    continue

                if self.df.loc[idx, label] == 1:
                    label_stable_ids.append(stable_id)

            # Take up to target_count
            selected = label_stable_ids[:target_count]
            selected_stable_ids.extend(selected)

            # Remove from candidates
            remaining_candidates = remaining_candidates[
                ~remaining_candidates['stable_id'].isin(selected)
            ]

        # Fill remaining with top-scoring passages
        remaining_needed = target_size - len(selected_stable_ids)
        if remaining_needed > 0:
            remaining_candidates = remaining_candidates.sort_values('composite', ascending=False)
            additional = remaining_candidates.head(remaining_needed)['stable_id'].tolist()
            selected_stable_ids.extend(additional)

        return selected_stable_ids[:target_size]

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

    def create_stratified_quality_tiers(
            self,
            tier1_config: Dict,
            tier2_config: Dict,
            label_targets: Optional[Dict] = None,
            max_no_label_passages: int = 0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Enhanced tiering with GUARANTEED rare label representation

        Combines quality scores with label diversity to ensure rare labels are learned.

        Args:
            tier1_config: {'min_consistency', 'min_rerank', 'target_size'}
            tier2_config: Same as tier1
            label_targets: {'tier1': {'Material_Physical': 50, ...}, 'tier2': {...}}

        Returns:
            (tier1_df, tier2_df, inference_df, metadata)
        """

        if self.scores_df is None:
            raise ValueError("Quality scores required")

        if 'passage_id' not in self.df.columns:
            raise ValueError("DataFrame must have 'passage_id' for stable references")

        # Build stable_id mappings
        stable_id_to_index = {}
        for idx in self.df.index:
            stable_id = self.df.loc[idx, 'passage_id']
            if pd.notna(stable_id):
                stable_id_to_index[stable_id] = idx

        # Ensure scores_df has stable_id
        if 'stable_id' not in self.scores_df.columns:
            if 'passage_idx' in self.scores_df.columns:
                self.scores_df['stable_id'] = self.scores_df['passage_idx'].map(
                    lambda idx: self.df.loc[idx, 'passage_id'] if idx in self.df.index else None
                )
            else:
                raise ValueError("scores_df missing both 'stable_id' and 'passage_idx'")

        # Get valid scored stable_ids
        valid_stable_ids = [
            sid for sid in self.scores_df['stable_id'].dropna().unique()
            if sid in stable_id_to_index
        ]

        if len(valid_stable_ids) == 0:
            raise ValueError("No valid scored passages")

        scores_df = self.scores_df[self.scores_df['stable_id'].isin(valid_stable_ids)].copy()

        # Calculate label frequencies for diversity bonus
        label_freqs = {
            label: (self.df[label] == 1).sum() / len(self.df)
            for label in self.label_columns
        }

        # Enhanced scoring: quality + diversity
        scores_df['quality_score'] = (scores_df['consistency_avg'] + scores_df['rerank_avg']) / 2
        scores_df['diversity_score'] = 0.0

        # Add diversity bonus for rare labels
        for stable_id in scores_df['stable_id']:
            if stable_id not in stable_id_to_index:
                continue

            idx = stable_id_to_index[stable_id]
            rare_label_count = 0

            for label, freq in label_freqs.items():
                if freq < 0.15 and idx in self.df.index and self.df.loc[idx, label] == 1:
                    # Weight by rarity: rarer = higher bonus
                    rare_label_count += (0.15 - freq) * 10  # Scale to 0-1.5 range

            scores_df.loc[scores_df['stable_id'] == stable_id, 'diversity_score'] = rare_label_count

        # Combined score: 70% quality, 30% diversity
        scores_df['combined_score'] = 0.7 * scores_df['quality_score'] + 0.3 * (
                scores_df['quality_score'] + scores_df['diversity_score']
        )

        # TIER 1: With label guarantees
        tier1_stable_ids = self._select_tier_with_guarantees(
            scores_df=scores_df,
            target_size=tier1_config['target_size'],
            min_consistency=tier1_config['min_consistency'],
            min_rerank=tier1_config['min_rerank'],
            label_targets=label_targets.get('tier1', {}) if label_targets else {},
            label_freqs=label_freqs,
            stable_id_to_index=stable_id_to_index,
            tier_name="Tier 1"
        )

        # TIER 2: Remaining + guarantees
        remaining_scores = scores_df[~scores_df['stable_id'].isin(tier1_stable_ids)].copy()

        # Adjust tier2 targets (reduce by what tier1 already has)
        tier2_targets = {}
        if label_targets and 'tier2' in label_targets:
            for label, target in label_targets['tier2'].items():
                # Count how many of this label are in tier1
                tier1_count = sum(
                    1 for sid in tier1_stable_ids
                    if sid in stable_id_to_index
                    and stable_id_to_index[sid] in self.df.index
                    and self.df.loc[stable_id_to_index[sid], label] == 1
                )
                # Reduce tier2 target accordingly
                tier2_targets[label] = max(0, target - tier1_count)

        tier2_stable_ids = self._select_tier_with_guarantees(
            scores_df=remaining_scores,
            target_size=tier2_config['target_size'],
            min_consistency=tier2_config['min_consistency'],
            min_rerank=tier2_config['min_rerank'],
            label_targets=tier2_targets,
            label_freqs=label_freqs,
            stable_id_to_index=stable_id_to_index,
            tier_name="Tier 2"
        )

        # INFERENCE: Rest
        all_stable_ids = set(valid_stable_ids)
        inference_stable_ids = list(
            all_stable_ids - set(tier1_stable_ids) - set(tier2_stable_ids)
        )

        # Create DataFrames
        tier1_indices = [stable_id_to_index[sid] for sid in tier1_stable_ids if sid in stable_id_to_index]
        tier2_indices = [stable_id_to_index[sid] for sid in tier2_stable_ids if sid in stable_id_to_index]
        inference_indices = [stable_id_to_index[sid] for sid in inference_stable_ids if sid in stable_id_to_index]

        tier1_df = self.df.loc[tier1_indices].copy()
        tier2_df = self.df.loc[tier2_indices].copy()
        inference_df = self.df.loc[inference_indices].copy()

        # Add confidence scores
        for tier_df, stable_ids in [(tier1_df, tier1_stable_ids),
                                    (tier2_df, tier2_stable_ids),
                                    (inference_df, inference_stable_ids)]:
            for stable_id in stable_ids:
                if stable_id not in stable_id_to_index:
                    continue

                idx = stable_id_to_index[stable_id]
                if idx not in tier_df.index:
                    continue

                score_rows = scores_df[scores_df['stable_id'] == stable_id]
                if not score_rows.empty:
                    score_row = score_rows.iloc[0]
                    tier_df.loc[idx, 'confidence_quality'] = score_row['quality_score']
                    tier_df.loc[idx, 'confidence_diversity'] = score_row['diversity_score']
                    tier_df.loc[idx, 'confidence_combined'] = score_row['combined_score']

            # Filter no-label passages if requested
        if max_no_label_passages >= 0:
            original_t1_len = len(tier1_df)
            original_t2_len = len(tier2_df)

            tier1_df = self.filter_no_label_passages(tier1_df, max_no_label_passages)
            tier2_df = self.filter_no_label_passages(tier2_df, max_no_label_passages)

            removed_t1 = original_t1_len - len(tier1_df)
            removed_t2 = original_t2_len - len(tier2_df)

            if removed_t1 > 0 or removed_t2 > 0:
                st.info(f"🧹 Filtered no-label passages: Tier1 -{removed_t1}, Tier2 -{removed_t2}")

            # Generate metadata with verification
        metadata = self._generate_stratified_metadata_with_verification(
            tier1_df, tier2_df, inference_df, label_freqs, label_targets
        )

        # ADD THIS:
        metadata['no_label_filtering'] = {
            'max_allowed': max_no_label_passages,
            'tier1_no_labels': int((tier1_df[self.label_columns].sum(axis=1) == 0).sum()),
            'tier2_no_labels': int((tier2_df[self.label_columns].sum(axis=1) == 0).sum()),
            'inference_no_labels': int((inference_df[self.label_columns].sum(axis=1) == 0).sum())
        }

        return tier1_df, tier2_df, inference_df, metadata

    def _select_tier_with_guarantees(
            self,
            scores_df: pd.DataFrame,
            target_size: int,
            min_consistency: float,
            min_rerank: float,
            label_targets: Dict[str, int],
            label_freqs: Dict[str, float],
            stable_id_to_index: Dict,
            tier_name: str
    ) -> List[str]:
        """
        Select passages ensuring label guarantees WITHOUT sacrificing too much quality

        Strategy:
        1. Filter by minimum quality thresholds first
        2. Reserve slots for rare label minimums from quality-filtered pool
        3. Fill remaining with highest combined scores
        """

        # Phase 0: Filter by quality minimums
        quality_filtered = scores_df[
            (scores_df['consistency_avg'] >= min_consistency) &
            (scores_df['rerank_avg'] >= min_rerank)
            ].copy()

        if len(quality_filtered) == 0:
            st.warning(f"⚠️ {tier_name}: No passages meet quality minimums. Relaxing constraints...")
            quality_filtered = scores_df.copy()

        selected_stable_ids = []
        reserved_by_label = {}

        # Phase 1: Reserve for rare labels (from quality-filtered pool only)
        if label_targets:
            # Sort labels by rarity (rarest first) to prioritize
            sorted_labels = sorted(
                label_targets.keys(),
                key=lambda l: label_freqs.get(l, 1.0)
            )

            for label in sorted_labels:
                min_needed = label_targets[label]

                if min_needed == 0:
                    continue

                # Find quality-filtered candidates with this label
                label_candidates = []
                for stable_id in quality_filtered['stable_id']:
                    if stable_id in selected_stable_ids:
                        continue

                    if stable_id not in stable_id_to_index:
                        continue

                    idx = stable_id_to_index[stable_id]
                    if idx not in self.df.index:
                        continue

                    if self.df.loc[idx, label] == 1:
                        score_row = quality_filtered[quality_filtered['stable_id'] == stable_id]
                        if not score_row.empty:
                            label_candidates.append((stable_id, score_row.iloc[0]['combined_score']))

                # Sort by combined score, take best
                label_candidates.sort(key=lambda x: x[1], reverse=True)
                reserved = [sid for sid, _ in label_candidates[:min_needed]]

                reserved_by_label[label] = reserved
                selected_stable_ids.extend(reserved)

                # Show what we achieved
                if len(reserved) < min_needed:
                    st.warning(
                        f"⚠️ {tier_name} - {label}: Only found {len(reserved)}/{min_needed} in quality-filtered pool")
                else:
                    st.success(f"✅ {tier_name} - {label}: Reserved {len(reserved)}/{min_needed}")

        # Remove duplicates
        selected_stable_ids = list(set(selected_stable_ids))

        # Phase 2: Fill remaining with top quality scores
        remaining_slots = target_size - len(selected_stable_ids)

        if remaining_slots > 0:
            remaining = quality_filtered[
                ~quality_filtered['stable_id'].isin(selected_stable_ids)
            ].copy()

            remaining = remaining.sort_values('combined_score', ascending=False)
            additional = remaining.head(remaining_slots)['stable_id'].tolist()
            selected_stable_ids.extend(additional)

        return selected_stable_ids[:target_size]

    def _generate_stratified_metadata_with_verification(
            self,
            tier1_df: pd.DataFrame,
            tier2_df: pd.DataFrame,
            inference_df: pd.DataFrame,
            label_freqs: Dict[str, float],
            label_targets: Optional[Dict]
    ) -> Dict:
        """Generate metadata showing whether targets were met"""

        metadata = self._generate_tier_metadata(tier1_df, tier2_df, inference_df)

        # Add verification
        if label_targets:
            metadata['label_targeting'] = {
                'enabled': True,
                'verification': {}
            }

            for tier_name, tier_df in [('tier1', tier1_df), ('tier2', tier2_df)]:
                tier_key = 'tier1' if tier_name == 'tier1' else 'tier2'

                if tier_key not in label_targets:
                    continue

                verification = {}
                for label, target in label_targets[tier_key].items():
                    actual = int((tier_df[label] == 1).sum())
                    met = actual >= target

                    verification[label] = {
                        'target': target,
                        'actual': actual,
                        'met': met,
                        'percentage': (actual / len(tier_df) * 100) if len(tier_df) > 0 else 0,
                        'global_frequency': label_freqs.get(label, 0) * 100
                    }

                metadata['label_targeting']['verification'][tier_name] = verification

        return metadata
