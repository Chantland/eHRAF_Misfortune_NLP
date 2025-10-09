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
# DATA ANALYSIS & CLEANING
# ============================================================================

class DataAnalyzer:
    """Intelligent data analysis and cleaning suggestions"""

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

        # Identify very short passages
        very_short = (lengths < 50).sum()
        if very_short > 0:
            pct = (very_short / len(lengths)) * 100
            issues.append(f"Very short passages (<50 chars): {very_short} ({pct:.1f}%)")
            suggestions.append("Consider removing passages with <50 characters")

        # Identify very long passages (may be truncated in training)
        very_long = (lengths > 2000).sum()
        if very_long > 0:
            pct = (very_long / len(lengths)) * 100
            issues.append(f"Very long passages (>2000 chars): {very_long} ({pct:.1f}%)")
            suggestions.append("Long passages will be truncated at 512 tokens (~2048 chars)")

        # Check for duplicate passages
        duplicates = self.df[self.passage_col].duplicated().sum()
        if duplicates > 0:
            pct = (duplicates / len(self.df)) * 100
            issues.append(f"Duplicate passages: {duplicates} ({pct:.1f}%)")
            suggestions.append("Consider deduplicating passages")

        # Check label distribution
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
            suggestions.append("Consider oversampling rare labels in tiered datasets")

        # Check for passages with no labels
        no_labels = (self.df[self.label_columns].sum(axis=1) == 0).sum()
        if no_labels > 0:
            pct = (no_labels / len(self.df)) * 100
            issues.append(f"Passages with no labels: {no_labels} ({pct:.1f}%)")
            suggestions.append("Remove passages with no labels")

        # Check for passages with many labels
        many_labels = (self.df[self.label_columns].sum(axis=1) > 8).sum()
        if many_labels > 0:
            pct = (many_labels / len(self.df)) * 100
            issues.append(f"Passages with >8 labels: {many_labels} ({pct:.1f}%)")
            suggestions.append("Multi-label passages may be harder to learn - verify quality")

        return {
            'issues': issues,
            'suggestions': suggestions,
            'stats': stats
        }

    def suggest_cleaning_steps(self, analysis: Dict) -> List[Dict]:
        """Generate cleaning step recommendations"""
        steps = []

        # Remove missing passages
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

        # Remove no-label passages
        no_labels = (self.df[self.label_columns].sum(axis=1) == 0).sum()
        if no_labels > 0:
            steps.append({
                'name': 'Remove Unlabeled',
                'description': f'Remove {no_labels} passages with no labels',
                'action': 'remove_unlabeled',
                'impact': no_labels,
                'recommended': True
            })

        # Remove very short passages
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

        # Optional: Remove very long passages
        very_long = (lengths > 2000).sum()
        if very_long > 0:
            steps.append({
                'name': 'Remove Very Long',
                'description': f'Remove {very_long} passages with >2000 characters (optional)',
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


class DataSegmenter:
    """Intelligent data segmentation and tiering"""

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
        """Create quality-based tiers"""

        if self.scores_df is None or len(self.scores_df) == 0:
            raise ValueError("Quality scores required for tiering. Compute scores first.")

        if len(self.df) == 0:
            raise ValueError("Dataset is empty")

        # Ensure scores_df has required columns
        required_cols = ['passage_idx', 'consistency_avg', 'rerank_avg']
        missing_cols = [col for col in required_cols if col not in self.scores_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required score columns: {missing_cols}")

        valid_indices = self.scores_df['passage_idx'].tolist()

        if len(valid_indices) == 0:
            raise ValueError("No valid scored passages found")

        valid_indices = self.scores_df['passage_idx'].tolist()
        scores_df = self.scores_df.copy()
        scores_df['composite'] = (scores_df['consistency_avg'] + scores_df['rerank_avg']) / 2

        # Tier 1: Elite training data
        tier1_mask = (
                (scores_df['consistency_avg'] >= tier1_config['min_consistency']) &
                (scores_df['consistency_avg'] <= tier1_config['max_consistency']) &
                (scores_df['rerank_avg'] >= tier1_config['min_rerank']) &
                (scores_df['rerank_avg'] <= tier1_config['max_rerank'])
        )

        tier1_candidates = scores_df[tier1_mask].copy()

        # Apply label targeting for Tier 1 if specified
        if label_targets and 'tier1' in label_targets:
            tier1_indices = self._apply_label_targeting(
                tier1_candidates, label_targets['tier1'], tier1_config.get('target_size', 1000)
            )
        else:
            tier1_candidates = tier1_candidates.sort_values('composite', ascending=False)
            target_count = int(len(valid_indices) * tier1_config.get('target_pct', 12) / 100)
            tier1_indices = tier1_candidates.head(target_count)['passage_idx'].tolist()

        # Tier 2: Expansion training data
        remaining_indices = [idx for idx in valid_indices if idx not in tier1_indices]
        remaining_scores = scores_df[scores_df['passage_idx'].isin(remaining_indices)]

        tier2_mask = (
                (remaining_scores['consistency_avg'] >= tier2_config['min_consistency']) &
                (remaining_scores['consistency_avg'] <= tier2_config['max_consistency']) &
                (remaining_scores['rerank_avg'] >= tier2_config['min_rerank']) &
                (remaining_scores['rerank_avg'] <= tier2_config['max_rerank'])
        )

        tier2_candidates = remaining_scores[tier2_mask].copy()

        # Apply label targeting for Tier 2 if specified
        if label_targets and 'tier2' in label_targets:
            tier2_indices = self._apply_label_targeting(
                tier2_candidates, label_targets['tier2'], tier2_config.get('target_size', 2000)
            )
        else:
            tier2_candidates = tier2_candidates.sort_values('composite', ascending=False)
            target_count = int(len(valid_indices) * tier2_config.get('target_pct', 25) / 100)
            tier2_indices = tier2_candidates.head(target_count)['passage_idx'].tolist()

        # Inference: Everything else
        inference_indices = [idx for idx in valid_indices
                             if idx not in tier1_indices and idx not in tier2_indices]

        # Create dataframes
        tier1_df = self.df.loc[tier1_indices].copy()
        tier2_df = self.df.loc[tier2_indices].copy()
        inference_df = self.df.loc[inference_indices].copy()

        # Add confidence scores
        for idx in tier1_indices:
            if idx in scores_df['passage_idx'].values:
                score_row = scores_df[scores_df['passage_idx'] == idx].iloc[0]
                tier1_df.loc[idx, 'confidence_composite'] = score_row['composite']
                tier1_df.loc[idx, 'confidence_consistency'] = score_row['consistency_avg']
                tier1_df.loc[idx, 'confidence_rerank'] = score_row['rerank_avg']
                tier1_df.loc[idx, 'tier'] = 1

        for idx in tier2_indices:
            if idx in scores_df['passage_idx'].values:
                score_row = scores_df[scores_df['passage_idx'] == idx].iloc[0]
                tier2_df.loc[idx, 'confidence_composite'] = score_row['composite']
                tier2_df.loc[idx, 'confidence_consistency'] = score_row['consistency_avg']
                tier2_df.loc[idx, 'confidence_rerank'] = score_row['rerank_avg']
                tier2_df.loc[idx, 'tier'] = 2

        for idx in inference_indices:
            if idx in scores_df['passage_idx'].values:
                score_row = scores_df[scores_df['passage_idx'] == idx].iloc[0]
                inference_df.loc[idx, 'confidence_composite'] = score_row['composite']
                inference_df.loc[idx, 'confidence_consistency'] = score_row['consistency_avg']
                inference_df.loc[idx, 'confidence_rerank'] = score_row['rerank_avg']
                inference_df.loc[idx, 'tier'] = 3

        # Generate metadata
        metadata = self._generate_tier_metadata(tier1_df, tier2_df, inference_df)

        return tier1_df, tier2_df, inference_df, metadata

    def _apply_label_targeting(self, candidates: pd.DataFrame, targets: Dict, target_size: int) -> List[int]:
        """Apply label-specific targeting to select passages"""
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

    def _generate_tier_metadata(self, tier1_df, tier2_df, inference_df) -> Dict:
        """Generate comprehensive metadata about tiers"""
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
                    'consistency_std': float(tier_df['confidence_consistency'].std()),
                    'rerank_mean': float(tier_df['confidence_rerank'].mean()),
                    'rerank_median': float(tier_df['confidence_rerank'].median()),
                    'rerank_std': float(tier_df['confidence_rerank'].std()),
                    'composite_mean': float(tier_df['confidence_composite'].mean()),
                    'composite_median': float(tier_df['confidence_composite'].median()),
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

    def create_custom_segments(self, filters: Dict) -> pd.DataFrame:
        """Create custom data segments based on filters"""
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


# ============================================================================
# STREAMLIT UI
# ============================================================================

def render_data_preparation_page(session_state: Dict):
    """Render comprehensive data preparation page"""

    st.markdown("## 🛠️ Data Preparation & Export")

    if not session_state.get('initialized', False):
        st.warning("⚠️ Load a dataset first")
        st.info("Go to the Overview page and load a dataset")
        return

    df = session_state.get('df')
    label_columns = session_state.get('label_columns', [])
    passage_col = session_state.get('passage_col', 'Passage')
    cache = session_state.get('cache')
    scores_df = cache.get('df_summary') if cache else None

    # Create tabs - ADD EXPERIMENT BROWSER
    tabs = st.tabs([
        "🔍 Analyze & Clean",
        "✂️ Segment Data",
        "📊 Quality Tiers",
        "🧪 Experiments",  # NEW TAB
        "💾 Export"
    ])

    with tabs[0]:
        render_analysis_cleaning_tab(session_state, df, label_columns, passage_col)

    with tabs[2]:
        render_quality_tiers_tab(session_state, df, label_columns, scores_df)

    with tabs[3]:
        render_export_tab(session_state, df, label_columns, passage_col)

def render_analysis_cleaning_tab(session_state: Dict, df: pd.DataFrame, label_columns: List[str], passage_col: str):
    """Render data analysis and cleaning tab"""

    st.markdown("### 🔍 Data Quality Analysis")

    # Initialize analyzer
    analyzer = DataAnalyzer(df, label_columns, passage_col)

    # Run analysis - ADD KEY
    if st.button("🔎 Analyze Data Quality", type="primary", key="analyze_quality_btn"):
        with st.spinner("Analyzing..."):
            analysis = analyzer.analyze_quality()
            session_state['quality_analysis'] = analysis

    # Display analysis results
    analysis = session_state.get('quality_analysis')

    if analysis:
        st.markdown("#### 📊 Analysis Results")

        # Issues
        if analysis['issues']:
            st.markdown("**⚠️ Issues Found:**")
            for issue in analysis['issues']:
                st.warning(issue)
        else:
            st.success("✅ No major issues found!")

        # Statistics
        with st.expander("📈 Detailed Statistics"):
            stats = analysis['stats']

            # Passage length stats
            if 'passage_length' in stats:
                st.markdown("**Passage Lengths:**")
                length_stats = stats['passage_length']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean", f"{length_stats['mean']:.0f} chars")
                with col2:
                    st.metric("Median", f"{length_stats['median']:.0f} chars")
                with col3:
                    st.metric("Std Dev", f"{length_stats['std']:.0f} chars")

            # Label distribution
            if 'label_distribution' in stats:
                st.markdown("**Label Distribution:**")
                dist_data = []
                for label, info in stats['label_distribution'].items():
                    dist_data.append({
                        'Label': label,
                        'Count': info['count'],
                        'Percentage': f"{info['percentage']:.1f}%"
                    })
                st.dataframe(pd.DataFrame(dist_data), hide_index=True, width='stretch')

        # Suggestions
        if analysis['suggestions']:
            st.markdown("#### 💡 Recommendations")
            for suggestion in analysis['suggestions']:
                st.info(suggestion)

        st.markdown("---")

        # Cleaning steps
        st.markdown("### 🧹 Data Cleaning")

        cleaning_steps = analyzer.suggest_cleaning_steps(analysis)

        if cleaning_steps:
            st.markdown("**Select cleaning steps to apply:**")

            selected_actions = []
            for step in cleaning_steps:
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected = st.checkbox(
                        step['name'],
                        value=step['recommended'],
                        key=f"clean_step_{step['action']}",  # UNIQUE KEY
                        help=step['description']
                    )
                    if selected:
                        selected_actions.append(step['action'])
                with col2:
                    impact_color = "🟢" if step['impact'] < len(df) * 0.05 else "🟡" if step['impact'] < len(
                        df) * 0.1 else "🔴"
                    st.caption(f"{impact_color} -{step['impact']}")

            if selected_actions:
                st.markdown("---")

                col1, col2 = st.columns(2)

                with col1:
                    total_removed = sum(step['impact'] for step in cleaning_steps if step['action'] in selected_actions)
                    st.metric("Total Passages to Remove", total_removed)
                    st.metric("Remaining", len(df) - total_removed)

                with col2:
                    # ADD KEY
                    if st.button("🧹 Apply Cleaning", type="primary", key="apply_cleaning_btn"):
                        with st.spinner("Cleaning data..."):
                            df_clean = analyzer.apply_cleaning(selected_actions)
                            session_state['cleaned_df'] = df_clean
                            session_state['df'] = df_clean  # Update working df
                            st.success(f"✅ Cleaned! {len(df)} → {len(df_clean)} passages")
                            st.rerun()
        else:
            st.success("✅ Data is clean! No cleaning steps needed.")

def render_quality_tiers_tab(session_state: Dict, df: pd.DataFrame, label_columns: List[str],
                             scores_df: Optional[pd.DataFrame]):
    """Render quality-based tiering tab with full label control"""

    st.markdown("### 📊 Quality-Based Tiered Training Data")

    if scores_df is None:
        st.warning("⚠️ Quality scores required for tiering")
        st.info("Go to 'Compute Scores' page first to generate quality scores")
        return

    st.markdown("""
    Create tiered training datasets based on quality scores.
    - **Tier 1**: Elite data for initial training
    - **Tier 2**: Expansion data for generalization
    - **Inference**: Validation/test set
    """)

    # Use working_df if available
    working_df = session_state.get('working_df', df)
    segmenter = DataSegmenter(working_df, scores_df, label_columns)

    # Configuration presets
    TIER_PRESETS = {
        'balanced': {
            'name': 'Balanced',
            'tier1': {'min_consistency': 0.65, 'max_consistency': 1.0, 'min_rerank': 0.45, 'max_rerank': 1.0,
                      'target_pct': 12},
            'tier2': {'min_consistency': 0.45, 'max_consistency': 0.65, 'min_rerank': 0.30, 'max_rerank': 0.45,
                      'target_pct': 25}
        },
        'conservative': {
            'name': 'Conservative (High Quality)',
            'tier1': {'min_consistency': 0.70, 'max_consistency': 1.0, 'min_rerank': 0.50, 'max_rerank': 1.0,
                      'target_pct': 10},
            'tier2': {'min_consistency': 0.50, 'max_consistency': 0.70, 'min_rerank': 0.35, 'max_rerank': 0.50,
                      'target_pct': 22}
        },
        'aggressive': {
            'name': 'Aggressive (More Data)',
            'tier1': {'min_consistency': 0.60, 'max_consistency': 1.0, 'min_rerank': 0.40, 'max_rerank': 1.0,
                      'target_pct': 15},
            'tier2': {'min_consistency': 0.40, 'max_consistency': 0.60, 'min_rerank': 0.25, 'max_rerank': 0.40,
                      'target_pct': 28}
        }
    }

    # Preset selection
    preset = st.selectbox(
        "Configuration preset:",
        options=['custom'] + list(TIER_PRESETS.keys()),
        format_func=lambda x: 'Custom' if x == 'custom' else TIER_PRESETS[x]['name']
    )

    if preset != 'custom':
        tier1_config = TIER_PRESETS[preset]['tier1'].copy()
        tier2_config = TIER_PRESETS[preset]['tier2'].copy()
    else:
        tier1_config = {}
        tier2_config = {}

    # Calculate total available passages
    total_available = len([idx for idx in scores_df['passage_idx'].tolist() if idx in working_df.index])

    st.markdown("---")
    st.markdown("### ⚙️ Tier Configuration")

    # Tier size mode selection
    size_mode = st.radio(
        "Tier sizing mode:",
        ["Percentage-based", "Absolute counts"],
        horizontal=True,
        help="Choose whether to specify tier sizes as percentages or absolute numbers"
    )

    use_absolute_sizes = (size_mode == "Absolute counts")

    st.markdown("---")

    # Tier 1 Configuration
    st.markdown("#### 🥇 Tier 1 (Elite Training Data)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Quality Thresholds:**")
        t1_min_cons = st.slider(
            "Min consistency:",
            0.0, 1.0,
            tier1_config.get('min_consistency', 0.65),
            0.05,
            key="t1_cons"
        )
        t1_min_rerank = st.slider(
            "Min rerank:",
            0.0, 1.0,
            tier1_config.get('min_rerank', 0.45),
            0.05,
            key="t1_rerank"
        )

    with col2:
        st.markdown("**Tier Size:**")
        if use_absolute_sizes:
            default_t1_count = int(total_available * 0.12)
            t1_target_count = st.number_input(
                "Target count:",
                min_value=100,
                max_value=total_available,
                value=min(default_t1_count, total_available),
                step=50,
                key="t1_count"
            )
            t1_pct = (t1_target_count / total_available * 100) if total_available > 0 else 0
            st.caption(f"≈ {t1_pct:.1f}% of dataset")
        else:
            t1_pct = st.slider(
                "Target %:",
                5, 30,
                tier1_config.get('target_pct', 12),
                1,
                key="t1_pct"
            )
            t1_target_count = int(total_available * t1_pct / 100)
            st.caption(f"≈ {t1_target_count:,} passages")

    with col3:
        st.markdown("**Quality Check:**")
        # Show how many passages meet criteria
        tier1_candidates_mask = (
                (scores_df['consistency_avg'] >= t1_min_cons) &
                (scores_df['rerank_avg'] >= t1_min_rerank)
        )
        tier1_meeting_criteria = tier1_candidates_mask.sum()

        if tier1_meeting_criteria >= t1_target_count:
            st.success(f"✅ {tier1_meeting_criteria:,} passages meet criteria")
        elif tier1_meeting_criteria > 0:
            st.warning(f"⚠️ Only {tier1_meeting_criteria:,} meet criteria")
        else:
            st.error(f"❌ No passages meet criteria")

    tier1_config = {
        'min_consistency': t1_min_cons,
        'max_consistency': 1.0,
        'min_rerank': t1_min_rerank,
        'max_rerank': 1.0,
        'target_pct': t1_pct,
        'target_size': t1_target_count
    }

    st.markdown("---")

    # Tier 2 Configuration
    st.markdown("#### 📚 Tier 2 (Expansion Training Data)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Quality Thresholds:**")
        t2_min_cons = st.slider(
            "Min consistency:",
            0.0, t1_min_cons,
            min(tier2_config.get('min_consistency', 0.45), t1_min_cons),
            0.05,
            key="t2_cons"
        )
        t2_min_rerank = st.slider(
            "Min rerank:",
            0.0, t1_min_rerank,
            min(tier2_config.get('min_rerank', 0.30), t1_min_rerank),
            0.05,
            key="t2_rerank"
        )

    with col2:
        st.markdown("**Tier Size:**")
        if use_absolute_sizes:
            default_t2_count = int(total_available * 0.25)
            remaining_after_t1 = total_available - t1_target_count
            t2_target_count = st.number_input(
                "Target count:",
                min_value=100,
                max_value=remaining_after_t1,
                value=min(default_t2_count, remaining_after_t1),
                step=50,
                key="t2_count"
            )
            t2_pct = (t2_target_count / total_available * 100) if total_available > 0 else 0
            st.caption(f"≈ {t2_pct:.1f}% of dataset")
        else:
            t2_pct = st.slider(
                "Target %:",
                10, 50,
                tier2_config.get('target_pct', 25),
                1,
                key="t2_pct"
            )
            t2_target_count = int(total_available * t2_pct / 100)
            st.caption(f"≈ {t2_target_count:,} passages")

    with col3:
        st.markdown("**Quality Check:**")
        remaining_indices = [idx for idx in scores_df['passage_idx'].tolist()
                             if idx in working_df.index]
        remaining_scores = scores_df[scores_df['passage_idx'].isin(remaining_indices)]

        tier2_candidates_mask = (
                (remaining_scores['consistency_avg'] >= t2_min_cons) &
                (remaining_scores['rerank_avg'] >= t2_min_rerank)
        )
        tier2_meeting_criteria = tier2_candidates_mask.sum()

        if tier2_meeting_criteria >= t2_target_count:
            st.success(f"✅ {tier2_meeting_criteria:,} passages meet criteria")
        elif tier2_meeting_criteria > 0:
            st.warning(f"⚠️ Only {tier2_meeting_criteria:,} meet criteria")
        else:
            st.error(f"❌ No passages meet criteria")

    tier2_config = {
        'min_consistency': t2_min_cons,
        'max_consistency': t1_min_cons,
        'min_rerank': t2_min_rerank,
        'max_rerank': t1_min_rerank,
        'target_pct': t2_pct,
        'target_size': t2_target_count
    }

    st.markdown("---")

    # Inference tier info
    inference_count = total_available - t1_target_count - t2_target_count
    inference_pct = (inference_count / total_available * 100) if total_available > 0 else 0

    st.markdown(f"#### 🎯 Inference Set (Automatic)")
    st.info(f"**{inference_count:,} passages** ({inference_pct:.1f}%) - All remaining passages for validation/testing")

    st.markdown("---")

    # COMPLETE LABEL CONTROL
    st.markdown("### 🏷️ Label Targeting (Full Control)")

    use_label_targeting = st.checkbox(
        "Enable Label Targeting",
        value=False,
        help="Select which labels to include and set minimum counts for each tier"
    )

    label_targets = None

    if use_label_targeting:
        st.markdown("""
        **Select exactly which labels you want in each tier and how many examples.**
        - ✅ Check the box to include a label
        - Set the minimum number of passages needed for that label
        - Leave unchecked to exclude from targeting (label will still appear naturally in data)
        """)

        tab1, tab2 = st.tabs(["🥇 Tier 1 Labels", "📚 Tier 2 Labels"])

        with tab1:
            st.markdown("#### Tier 1: Select Labels for High-Quality Training")
            st.caption("Choose which labels need elite training data")

            tier1_targets = {}

            # Organize labels by category
            event_labels = [l for l in label_columns if
                            any(x in l for x in ['Illness', 'Accident']) and 'Other' not in l]
            cause_labels = [l for l in label_columns if any(x in l for x in
                                                            ['Material_Physical', 'Spirits_Gods', 'Witchcraft_Sorcery',
                                                             'Rule_Violation_Taboo', 'Technical_Specialist',
                                                             'Divination']) and 'Other' not in l]
            action_labels = [l for l in label_columns if any(x in l for x in
                                                             ['Physical_Material', 'Shaman_Medium_Healer',
                                                              'Priest_High_Religion']) and 'Other' not in l]
            other_labels = [l for l in label_columns if
                            'Other' in l or l not in event_labels + cause_labels + action_labels]

            # EVENT Labels
            if event_labels:
                st.markdown("**🔴 EVENT Labels:**")
                cols = st.columns(3)
                for i, label in enumerate(event_labels):
                    with cols[i % 3]:
                        # Calculate availability
                        label_available = 0
                        for idx in scores_df[tier1_candidates_mask]['passage_idx'].tolist():
                            if idx in working_df.index and working_df.loc[idx, label] == 1:
                                label_available += 1

                        include = st.checkbox(
                            f"Include {label}",
                            value=True,  # Default to including non-Other labels
                            key=f"t1_include_{label}",
                            help=f"{label_available} passages available with this label"
                        )

                        if include:
                            target = st.number_input(
                                f"Min count:",
                                min_value=0,
                                max_value=label_available if label_available > 0 else 1000,
                                value=min(200, label_available) if label_available > 0 else 100,
                                step=10,
                                key=f"t1_target_{label}",
                                help=f"Available: {label_available}"
                            )
                            if target > 0:
                                tier1_targets[label] = target

                        if label_available > 0:
                            st.caption(f"✓ {label_available} available")
                        else:
                            st.caption("⚠️ None available")

            # CAUSE Labels
            if cause_labels:
                st.markdown("**🔵 CAUSE Labels:**")
                cols = st.columns(3)
                for i, label in enumerate(cause_labels):
                    with cols[i % 3]:
                        label_available = 0
                        for idx in scores_df[tier1_candidates_mask]['passage_idx'].tolist():
                            if idx in working_df.index and working_df.loc[idx, label] == 1:
                                label_available += 1

                        # Default to True for non-Just_Happens
                        default_include = 'Just_Happens' not in label

                        include = st.checkbox(
                            f"Include {label}",
                            value=default_include,
                            key=f"t1_include_{label}",
                            help=f"{label_available} passages available with this label"
                        )

                        if include:
                            target = st.number_input(
                                f"Min count:",
                                min_value=0,
                                max_value=label_available if label_available > 0 else 1000,
                                value=min(200, label_available) if label_available > 0 else 100,
                                step=10,
                                key=f"t1_target_{label}",
                                help=f"Available: {label_available}"
                            )
                            if target > 0:
                                tier1_targets[label] = target

                        if label_available > 0:
                            st.caption(f"✓ {label_available} available")
                        else:
                            st.caption("⚠️ None available")

            # ACTION Labels
            if action_labels:
                st.markdown("**🟢 ACTION Labels:**")
                cols = st.columns(3)
                for i, label in enumerate(action_labels):
                    with cols[i % 3]:
                        label_available = 0
                        for idx in scores_df[tier1_candidates_mask]['passage_idx'].tolist():
                            if idx in working_df.index and working_df.loc[idx, label] == 1:
                                label_available += 1

                        include = st.checkbox(
                            f"Include {label}",
                            value=True,
                            key=f"t1_include_{label}",
                            help=f"{label_available} passages available with this label"
                        )

                        if include:
                            target = st.number_input(
                                f"Min count:",
                                min_value=0,
                                max_value=label_available if label_available > 0 else 1000,
                                value=min(200, label_available) if label_available > 0 else 100,
                                step=10,
                                key=f"t1_target_{label}",
                                help=f"Available: {label_available}"
                            )
                            if target > 0:
                                tier1_targets[label] = target

                        if label_available > 0:
                            st.caption(f"✓ {label_available} available")
                        else:
                            st.caption("⚠️ None available")

            # Other/Miscellaneous Labels (show separately)
            if other_labels:
                with st.expander("⚪ Other/Miscellaneous Labels (usually excluded)"):
                    st.caption("These are typically excluded from Tier 1 targeting")
                    cols = st.columns(3)
                    for i, label in enumerate(other_labels):
                        with cols[i % 3]:
                            label_available = 0
                            for idx in scores_df[tier1_candidates_mask]['passage_idx'].tolist():
                                if idx in working_df.index and working_df.loc[idx, label] == 1:
                                    label_available += 1

                            include = st.checkbox(
                                f"Include {label}",
                                value=False,  # Default to NOT including Other labels
                                key=f"t1_include_{label}",
                                help=f"{label_available} passages available"
                            )

                            if include:
                                target = st.number_input(
                                    f"Min count:",
                                    min_value=0,
                                    max_value=label_available if label_available > 0 else 1000,
                                    value=min(100, label_available) if label_available > 0 else 50,
                                    step=10,
                                    key=f"t1_target_{label}",
                                    help=f"Available: {label_available}"
                                )
                                if target > 0:
                                    tier1_targets[label] = target

            # Show Tier 1 summary
            if tier1_targets:
                st.markdown("---")
                st.markdown("**Tier 1 Selected Labels Summary:**")
                summary_data = []
                for label, target in tier1_targets.items():
                    summary_data.append({
                        'Label': label,
                        'Target': target
                    })
                st.dataframe(pd.DataFrame(summary_data), hide_index=True, width='stretch')
                st.caption(f"Total: {len(tier1_targets)} labels selected for targeting")

        with tab2:
            st.markdown("#### Tier 2: Select Labels for Expansion Training")
            st.caption("Choose which labels need additional coverage")

            tier2_targets = {}

            # Same organization but for Tier 2
            st.markdown("**All Labels:**")
            st.caption("Select labels that need broader coverage in Tier 2")

            cols = st.columns(3)
            for i, label in enumerate(label_columns):
                with cols[i % 3]:
                    # Calculate availability in Tier 2 range
                    label_available = 0
                    for idx in remaining_scores[tier2_candidates_mask]['passage_idx'].tolist():
                        if idx in working_df.index and working_df.loc[idx, label] == 1:
                            label_available += 1

                    # Default to including if not Other and not Just_Happens
                    default_include = 'Other' not in label and 'Just_Happens' not in label

                    include = st.checkbox(
                        f"Include {label}",
                        value=default_include,
                        key=f"t2_include_{label}",
                        help=f"{label_available} passages available with this label"
                    )

                    if include:
                        target = st.number_input(
                            f"Min count:",
                            min_value=0,
                            max_value=label_available if label_available > 0 else 1000,
                            value=min(150, label_available) if label_available > 0 else 75,
                            step=10,
                            key=f"t2_target_{label}",
                            help=f"Available: {label_available}"
                        )
                        if target > 0:
                            tier2_targets[label] = target

                    if label_available > 0:
                        st.caption(f"✓ {label_available} available")
                    else:
                        st.caption("⚠️ None available")

            # Show Tier 2 summary
            if tier2_targets:
                st.markdown("---")
                st.markdown("**Tier 2 Selected Labels Summary:**")
                summary_data = []
                for label, target in tier2_targets.items():
                    summary_data.append({
                        'Label': label,
                        'Target': target
                    })
                st.dataframe(pd.DataFrame(summary_data), hide_index=True, width='stretch')
                st.caption(f"Total: {len(tier2_targets)} labels selected for targeting")

        # Build label_targets dict
        if tier1_targets or tier2_targets:
            label_targets = {}
            if tier1_targets:
                label_targets['tier1'] = tier1_targets
            if tier2_targets:
                label_targets['tier2'] = tier2_targets

    st.markdown("---")

    # Summary before creation
    st.markdown("### 📊 Configuration Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Tier 1**")
        st.metric("Target Size", f"{t1_target_count:,}")
        st.caption(f"Quality: ≥{t1_min_cons:.2f} cons, ≥{t1_min_rerank:.2f} rerank")
        if label_targets and 'tier1' in label_targets:
            st.caption(f"**{len(label_targets['tier1'])} labels** selected for targeting")

    with col2:
        st.markdown("**Tier 2**")
        st.metric("Target Size", f"{t2_target_count:,}")
        st.caption(f"Quality: ≥{t2_min_cons:.2f} cons, ≥{t2_min_rerank:.2f} rerank")
        if label_targets and 'tier2' in label_targets:
            st.caption(f"**{len(label_targets['tier2'])} labels** selected for targeting")

    with col3:
        st.markdown("**Inference**")
        st.metric("Size", f"{inference_count:,}")
        st.caption("All remaining passages")

    # Create tiers button
    if st.button("🎯 Create Quality Tiers", type="primary", key="create_quality_tiers_btn"):
        with st.spinner("Creating tiers..."):
            try:
                # Validate
                if t1_target_count + t2_target_count >= total_available:
                    st.error("❌ Tier 1 + Tier 2 sizes exceed available passages!")
                    return

                st.info(f"Creating tiers from {total_available} scored passages...")

                tier1, tier2, inference, metadata = segmenter.create_quality_tiers(
                    tier1_config, tier2_config, label_targets
                )

                # Validate results
                if tier1 is None or len(tier1) == 0:
                    st.error("❌ Tier 1 is empty. Try adjusting thresholds or targets.")
                    return

                if tier2 is None or len(tier2) == 0:
                    st.warning("⚠️ Tier 2 is empty. Try adjusting thresholds.")

                # Save to session state
                session_state['tier1_dataset'] = tier1
                session_state['tier2_dataset'] = tier2
                session_state['inference_dataset'] = inference
                session_state['tier_metadata'] = metadata

                st.success(f"✅ Tiers created! Tier 1: {len(tier1)}, Tier 2: {len(tier2)}, Inference: {len(inference)}")

                # Show label coverage if targets were set
                if label_targets:
                    st.markdown("---")
                    st.markdown("### 🏷️ Label Target Achievement")

                    if 'tier1' in label_targets:
                        st.markdown("**Tier 1:**")
                        target_results = []
                        for label, target in label_targets['tier1'].items():  # Fixed: was tier1_targets['tier1']
                            actual = (tier1[label] == 1).sum() if label in tier1.columns else 0
                            status = "✅" if actual >= target else "⚠️" if actual > 0 else "❌"
                            pct_achieved = (actual / target * 100) if target > 0 else 0
                            target_results.append({
                                'Label': label,
                                'Target': target,
                                'Actual': actual,
                                'Achievement': f"{pct_achieved:.0f}%",
                                'Status': status
                            })
                        st.dataframe(pd.DataFrame(target_results), hide_index=True, width='stretch')

                    if 'tier2' in label_targets:
                        st.markdown("**Tier 2:**")
                        target_results = []
                        for label, target in label_targets[
                            'tier2'].items():  # Fixed: was tier2_targets (which doesn't exist)
                            actual = (tier2[label] == 1).sum() if label in tier2.columns else 0
                            status = "✅" if actual >= target else "⚠️" if actual > 0 else "❌"
                            pct_achieved = (actual / target * 100) if target > 0 else 0
                            target_results.append({
                                'Label': label,
                                'Target': target,
                                'Actual': actual,
                                'Achievement': f"{pct_achieved:.0f}%",
                                'Status': status
                            })
                        st.dataframe(pd.DataFrame(target_results), hide_index=True, width='stretch')

                st.rerun()

            except ValueError as e:
                st.error(f"❌ Configuration error: {e}")
                st.info("Try adjusting threshold values or tier sizes")
            except Exception as e:
                st.error(f"❌ Error creating tiers: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

    # Display existing tiers (rest of your existing code)
    if 'tier1_dataset' in session_state and session_state['tier1_dataset'] is not None:
        tier1 = session_state['tier1_dataset']
        tier2 = session_state.get('tier2_dataset')
        inference = session_state.get('inference_dataset')

        if tier1 is None or tier2 is None or inference is None:
            st.error("❌ Tier data is incomplete. Please recreate tiers.")
            return

        st.markdown("---")
        st.markdown("### 📊 Current Tier Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🥇 Tier 1 (Elite)**")
            st.metric("Count", len(tier1))
            if 'confidence_composite' in tier1.columns:
                st.metric("Avg Quality", f"{tier1['confidence_composite'].mean():.3f}")

        with col2:
            st.markdown("**📚 Tier 2 (Expansion)**")
            st.metric("Count", len(tier2))
            if 'confidence_composite' in tier2.columns:
                st.metric("Avg Quality", f"{tier2['confidence_composite'].mean():.3f}")

        with col3:
            st.markdown("**🎯 Inference (Test)**")
            st.metric("Count", len(inference))
            if 'confidence_composite' in inference.columns:
                st.metric("Avg Quality", f"{inference['confidence_composite'].mean():.3f}")

        # Label distribution
        with st.expander("📊 Label Distribution by Tier"):
            dist_data = []
            for label in label_columns:
                tier1_count = (tier1[label] == 1).sum() if label in tier1.columns else 0
                tier2_count = (tier2[label] == 1).sum() if label in tier2.columns else 0
                inference_count = (inference[label] == 1).sum() if label in inference.columns else 0

                dist_data.append({
                    'Label': label,
                    'Tier 1': int(tier1_count),
                    'Tier 2': int(tier2_count),
                    'Inference': int(inference_count),
                    'Total': int(tier1_count + tier2_count + inference_count)
                })

            st.dataframe(pd.DataFrame(dist_data), hide_index=True, width='stretch')

        # Save options
        st.markdown("---")
        st.markdown("### 💾 Save Tiers")

        tier_name = st.text_input(
            "Tier set name:",
            value=f"tiers_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            key="tier_save_name"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save All Tiers to Data Directory", type="primary", key="save_all_tiers"):
                save_tiers_to_data_directory(tier1, tier2, inference, tier_name, session_state)

        with col2:
            if st.button("💾 Save Tier 1 Only", key="save_tier1_only"):
                save_to_data_directory(tier1, f"{tier_name}_tier1", session_state)


def render_export_tab(session_state: Dict, df: pd.DataFrame, label_columns: List[str], passage_col: str):
    """Render export options tab"""

    st.markdown("### 💾 Export Data")

    # Select what to export
    st.markdown("#### 📦 Select Data to Export")

    export_options = ["Current Dataset"]

    if 'cleaned_df' in session_state:
        export_options.append("Cleaned Dataset")

    if 'custom_segment' in session_state:
        export_options.append("Custom Segment")

    if 'tier1_dataset' in session_state:
        export_options.extend(["Tier 1", "Tier 2", "Inference Set", "All Tiers (ZIP)"])

    selected_export = st.selectbox("Export:", export_options)

    # Get the dataframe to export
    if selected_export == "Current Dataset":
        export_df = df
    elif selected_export == "Cleaned Dataset":
        export_df = session_state['cleaned_df']
    elif selected_export == "Custom Segment":
        export_df = session_state['custom_segment']
    elif selected_export == "Tier 1":
        export_df = session_state['tier1_dataset']
    elif selected_export == "Tier 2":
        export_df = session_state['tier2_dataset']
    elif selected_export == "Inference Set":
        export_df = session_state['inference_dataset']
    else:
        export_df = None

    if export_df is not None:
        st.info(f"Selected: {len(export_df)} passages")

        # Column selection
        st.markdown("#### 📋 Select Columns")

        col1, col2 = st.columns(2)

        with col1:
            include_passage = st.checkbox("Include passage text", value=True)
            include_labels = st.checkbox("Include all labels", value=True)

        with col2:
            include_metadata = st.checkbox("Include metadata", value=False)
            include_scores = st.checkbox("Include quality scores", value='confidence' in export_df.columns)

        if not include_labels:
            selected_labels = st.multiselect("Select specific labels:", label_columns)
        else:
            selected_labels = label_columns

        # Build column list
        export_columns = []

        if 'ID' in export_df.columns:
            export_columns.append('ID')

        if include_passage:
            export_columns.append(passage_col)

        export_columns.extend(selected_labels)

        if include_scores:
            score_cols = [c for c in export_df.columns if 'confidence' in c or c == 'tier']
            export_columns.extend([c for c in score_cols if c in export_df.columns])

        if include_metadata:
            meta_cols = [c for c in export_df.columns if c not in export_columns and c not in label_columns]
            export_columns.extend(meta_cols)

        # Filter columns that exist
        export_columns = [c for c in export_columns if c in export_df.columns]

        final_df = export_df[export_columns].copy()

        st.markdown("---")
        st.markdown("#### 💾 Export Options")

        export_name = st.text_input("Filename:", value=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Download button
            output = io.BytesIO()
            final_df.to_excel(output, index=False, engine='openpyxl')

            st.download_button(
                label="📥 Download Excel",
                data=output.getvalue(),
                file_name=f"{export_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col2:
            # Save to data directory button
            if st.button("💾 Save to Data Directory", key="save_to_data_dir_btn"):
                save_to_data_directory(final_df, export_name, session_state)

        with col3:
            # CSV download
            csv_output = io.StringIO()
            final_df.to_csv(csv_output, index=False)

            st.download_button(
                label="📥 Download CSV",
                data=csv_output.getvalue(),
                file_name=f"{export_name}.csv",
                mime="text/csv"
            )

    elif selected_export == "All Tiers (ZIP)":
        # Export all tiers as ZIP
        st.markdown("#### 📦 Export Complete Tier Package")

        tier_name = st.text_input("Package name:", value=f"training_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        if st.button("💾 Create Training Package", key="create_training_pkg_btn"):
            create_training_package(session_state, tier_name)

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

