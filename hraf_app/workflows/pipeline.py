"""
Core workflow pipeline - orchestrates the entire process
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pickle

from core.quality import QualityScorer, PassageQuality
from core.training import ModelTrainer
from core.experiments import ExperimentTracker
from workflows.active_learning import ActiveLearner


@dataclass
class PipelineState:
    """Track pipeline state across workflow steps"""

    # Data
    df: Optional[pd.DataFrame] = None
    passage_col: Optional[str] = None
    label_columns: List[str] = field(default_factory=list)

    # Quality scoring
    quality_scores: Optional[Dict[int, PassageQuality]] = None
    quality_distribution: Optional[Dict] = None

    # Data selection
    selected_indices: Optional[List[int]] = None
    selection_criteria: Optional[Dict] = None

    # Training
    current_model: Optional[any] = None
    training_history: List[Dict] = field(default_factory=list)
    best_metrics: Optional[Dict] = None

    # Iteration
    failure_analysis: Optional[Dict] = None
    improvement_suggestions: List[Dict] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


class HRAFPipeline:
    """
    Main workflow orchestrator
    Manages state and coordinates between components
    """

    def __init__(self, cache_dir: str = "./cache"):
        self.state = PipelineState()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize components
        self.quality_scorer = None
        self.trainer = None
        self.active_learner = None
        self.experiment_tracker = ExperimentTracker()

        # Try to load cached state
        self._load_state()

    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================

    def get_current_step(self) -> str:
        """Determine which workflow step we're on"""
        if not self.has_data():
            return "load"
        elif not self.has_quality_scores():
            return "quality"
        elif not self.has_explored():
            return "explore"
        elif not self.has_model():
            return "train"
        else:
            return "iterate"

    def has_data(self) -> bool:
        return self.state.df is not None and len(self.state.df) > 0

    def has_quality_scores(self) -> bool:
        return self.state.quality_scores is not None

    def has_explored(self) -> bool:
        return self.state.selected_indices is not None

    def has_model(self) -> bool:
        return self.state.current_model is not None

    def has_results(self) -> bool:
        return self.state.best_metrics is not None

    def get_stats(self) -> Dict:
        """Get current pipeline statistics"""
        stats = {
            'num_passages': 0,
            'num_labels': 0,
            'avg_quality': 0.0,
            'best_f1': 0.0
        }

        if self.has_data():
            stats['num_passages'] = len(self.state.df)
            stats['num_labels'] = len(self.state.label_columns)

        if self.has_quality_scores():
            qualities = [q.overall_quality for q in self.state.quality_scores.values()]
            stats['avg_quality'] = np.mean(qualities) if qualities else 0.0

        if self.has_results():
            stats['best_f1'] = self.state.best_metrics.get('f1_micro', 0.0)

        return stats

    def _save_state(self):
        """Save pipeline state to disk"""
        cache_file = self.cache_dir / "pipeline_state.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(self.state, f)

        self.state.last_updated = datetime.now().isoformat()

    def _load_state(self):
        """Load pipeline state from disk"""
        cache_file = self.cache_dir / "pipeline_state.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.state = pickle.load(f)
            except:
                pass  # Start fresh if load fails

    # ========================================================================
    # STEP 1: DATA LOADING
    # ========================================================================

    def load_data(
            self,
            filepath: str,
            passage_col: str = None,
            label_columns: List[str] = None
    ) -> Dict:
        """
        Load and validate dataset

        Returns:
            Dict with validation results
        """
        df = pd.read_excel(filepath)

        # Auto-detect passage column
        if passage_col is None:
            passage_col = self._detect_passage_column(df)

        if passage_col not in df.columns:
            raise ValueError(f"Passage column '{passage_col}' not found")

        # Auto-detect label columns
        if label_columns is None:
            label_columns = self._detect_label_columns(df)

        if not label_columns:
            raise ValueError("No binary label columns found")

        # Validate
        validation = self._validate_data(df, passage_col, label_columns)

        # Store
        self.state.df = df
        self.state.passage_col = passage_col
        self.state.label_columns = label_columns

        self._save_state()

        return validation

    def _detect_passage_column(self, df: pd.DataFrame) -> str:
        """Auto-detect passage column"""
        candidates = ['Passage', 'passage', 'Text', 'text', 'Content']

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

    def _detect_label_columns(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect binary label columns"""
        label_cols = []

        exclude = {'ID', 'Passage', 'Text', 'Description', 'Culture', 'Region'}

        for col in df.columns:
            if col in exclude:
                continue

            if df[col].dtype in ['int64', 'float64']:
                unique_vals = df[col].dropna().unique()
                if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    if (df[col] == 1).sum() > 0:
                        label_cols.append(col)

        return label_cols

    def _validate_data(
            self,
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

        # Check for duplicates
        duplicates = df[passage_col].duplicated().sum()
        if duplicates > 0:
            validation['warnings'].append(
                f"{duplicates} duplicate passages found"
            )

        # Check label distribution
        label_stats = {}
        for label in label_columns:
            count = int((df[label] == 1).sum())
            pct = (count / len(df)) * 100
            label_stats[label] = {'count': count, 'percentage': pct}

            if pct < 2:
                validation['warnings'].append(
                    f"Label '{label}' very rare ({count} examples, {pct:.1f}%)"
                )

        validation['stats']['label_distribution'] = label_stats

        return validation

    # ========================================================================
    # STEP 2: QUALITY SCORING
    # ========================================================================

    def compute_quality(
            self,
            use_embeddings: bool = True,
            k_similar: int = 15
    ) -> Dict:
        """
        Compute quality scores for all passages

        Returns:
            Dict with scoring results and recommendations
        """
        if not self.has_data():
            raise RuntimeError("Load data first")

        # Initialize quality scorer
        self.quality_scorer = QualityScorer(
            df=self.state.df,
            passage_col=self.state.passage_col,
            label_columns=self.state.label_columns,
            use_embeddings=use_embeddings
        )

        # Compute scores
        quality_scores = self.quality_scorer.compute_all(k_similar=k_similar)

        # Analyze distribution
        distribution = self._analyze_quality_distribution(quality_scores)

        # Store
        self.state.quality_scores = quality_scores
        self.state.quality_distribution = distribution

        self._save_state()

        return {
            'num_scored': len(quality_scores),
            'distribution': distribution,
            'recommendations': self._get_quality_recommendations(distribution)
        }

    def _analyze_quality_distribution(
            self,
            quality_scores: Dict[int, PassageQuality]
    ) -> Dict:
        """Analyze quality score distribution"""
        qualities = [q.overall_quality for q in quality_scores.values()]

        distribution = {
            'mean': float(np.mean(qualities)),
            'median': float(np.median(qualities)),
            'std': float(np.std(qualities)),
            'tiers': {
                'elite': sum(1 for q in qualities if q >= 0.75),
                'good': sum(1 for q in qualities if 0.60 <= q < 0.75),
                'fair': sum(1 for q in qualities if 0.45 <= q < 0.60),
                'low': sum(1 for q in qualities if q < 0.45)
            }
        }

        total = len(qualities)
        distribution['tier_percentages'] = {
            tier: (count / total * 100) if total > 0 else 0
            for tier, count in distribution['tiers'].items()
        }

        return distribution

    def _get_quality_recommendations(self, distribution: Dict) -> List[str]:
        """Generate recommendations based on quality distribution"""
        recs = []

        elite_pct = distribution['tier_percentages']['elite']
        good_pct = distribution['tier_percentages']['good']

        if elite_pct < 10:
            recs.append(
                "⚠️ Less than 10% elite quality data. Consider data cleaning or higher thresholds."
            )
        elif elite_pct > 20:
            recs.append(
                "✅ Good amount of elite quality data. Can use conservative approach."
            )

        if good_pct + elite_pct < 30:
            recs.append(
                "⚠️ Limited high-quality data overall. May need aggressive selection or focus on specific labels."
            )

        if distribution['median'] < 0.5:
            recs.append(
                "⚠️ Low median quality. Consider reviewing labeling consistency or using different quality metrics."
            )

        return recs

    # ========================================================================
    # STEP 3: EXPLORE & FILTER
    # ========================================================================

    def select_training_data(
            self,
            min_quality: float = 0.60,
            tier_strategy: str = "balanced",
            label_targets: Dict = None
    ) -> Dict:
        """
        Select training data based on quality

        Args:
            min_quality: Minimum quality threshold
            tier_strategy: 'conservative', 'balanced', or 'aggressive'
            label_targets: Optional per-label minimum counts

        Returns:
            Selection results
        """
        if not self.has_quality_scores():
            raise RuntimeError("Compute quality scores first")

        # Apply quality filter
        selected = {
            idx: q for idx, q in self.state.quality_scores.items()
            if q.overall_quality >= min_quality
        }

        # Apply label targeting if specified
        if label_targets:
            selected = self._apply_label_targeting(selected, label_targets)

        # Store selection
        self.state.selected_indices = list(selected.keys())
        self.state.selection_criteria = {
            'min_quality': min_quality,
            'tier_strategy': tier_strategy,
            'label_targets': label_targets,
            'timestamp': datetime.now().isoformat()
        }

        self._save_state()

        # Analyze selection
        return self._analyze_selection(selected)

    def _apply_label_targeting(
            self,
            selected: Dict[int, PassageQuality],
            targets: Dict[str, int]
    ) -> Dict[int, PassageQuality]:
        """Ensure minimum counts for specific labels"""
        # Group by label
        by_label = {label: [] for label in self.state.label_columns}

        for idx, quality in selected.items():
            for label in self.state.label_columns:
                if self.state.df.loc[idx, label] == 1:
                    by_label[label].append((idx, quality))

        # Ensure targets met
        final_selected = {}

        for label, target_count in targets.items():
            if label not in by_label:
                continue

            # Sort by quality
            sorted_passages = sorted(
                by_label[label],
                key=lambda x: x[1].overall_quality,
                reverse=True
            )

            # Take top passages up to target
            for idx, quality in sorted_passages[:target_count]:
                final_selected[idx] = quality

        # Add remaining high-quality passages
        for idx, quality in selected.items():
            if idx not in final_selected:
                final_selected[idx] = quality

        return final_selected

    def _analyze_selection(self, selected: Dict[int, PassageQuality]) -> Dict:
        """Analyze data selection"""
        selected_df = self.state.df.loc[list(selected.keys())]

        analysis = {
            'num_selected': len(selected),
            'percentage': len(selected) / len(self.state.df) * 100,
            'avg_quality': np.mean([q.overall_quality for q in selected.values()]),
            'label_coverage': {}
        }

        # Check label coverage
        for label in self.state.label_columns:
            count = (selected_df[label] == 1).sum()
            analysis['label_coverage'][label] = {
                'count': int(count),
                'percentage': float(count / len(selected_df) * 100)
            }

        return analysis

    # ========================================================================
    # STEP 4: TRAIN MODEL
    # ========================================================================

    def train_model(
            self,
            config: Dict = None,
            experiment_name: str = None
    ) -> Dict:
        """
        Train model on selected data

        Returns:
            Training results
        """
        if not self.has_explored():
            raise RuntimeError("Select training data first")

        # Prepare training data
        train_df = self.state.df.loc[self.state.selected_indices]

        # Initialize trainer
        self.trainer = ModelTrainer(
            label_columns=self.state.label_columns,
            passage_col=self.state.passage_col,
            config=config or {}
        )

        # Train
        results = self.trainer.train(train_df)

        # Store results
        self.state.current_model = self.trainer.model
        self.state.training_history.append(results)
        self.state.best_metrics = results['test_metrics']

        # Log experiment
        if experiment_name:
            self.experiment_tracker.log_experiment(
                name=experiment_name,
                config=config,
                data_selection=self.state.selection_criteria,
                results=results
            )

        self._save_state()

        return results

    # ========================================================================
    # STEP 5: ITERATE
    # ========================================================================

    def analyze_failures(self) -> Dict:
        """Analyze model failures and suggest improvements"""
        if not self.has_model():
            raise RuntimeError("Train a model first")

        # Initialize active learner
        self.active_learner = ActiveLearner(
            model=self.state.current_model,
            trainer=self.trainer,
            quality_scorer=self.quality_scorer
        )

        # Analyze failures
        analysis = self.active_learner.analyze_failures(
            test_df=self.state.df,
            quality_scores=self.state.quality_scores
        )

        # Generate suggestions
        suggestions = self.active_learner.suggest_improvements(analysis)

        # Store
        self.state.failure_analysis = analysis
        self.state.improvement_suggestions = suggestions

        self._save_state()

        return {
            'analysis': analysis,
            'suggestions': suggestions
        }

    def apply_improvement(self, improvement_id: int) -> Dict:
        """Apply a suggested improvement and prepare for retraining"""
        if not self.state.improvement_suggestions:
            raise RuntimeError("Run failure analysis first")

        suggestion = self.state.improvement_suggestions[improvement_id]

        # Apply the improvement
        if suggestion['type'] == 'remove_low_quality':
            # Remove low quality passages with this label
            self.state.selected_indices = [
                idx for idx in self.state.selected_indices
                if self.state.quality_scores[idx].overall_quality >= suggestion['threshold']
            ]

        elif suggestion['type'] == 'add_more_examples':
            # Add more examples for underrepresented label
            # Find high-quality passages with this label
            label = suggestion['label']
            additional = []
            for idx, quality in self.state.quality_scores.items():
                if idx not in self.state.selected_indices:
                    if self.state.df.loc[idx, label] == 1:
                        if quality.overall_quality >= 0.65:
                            additional.append(idx)

            # Add top N
            additional_sorted = sorted(
                additional,
                key=lambda idx: self.state.quality_scores[idx].overall_quality,
                reverse=True
            )
            self.state.selected_indices.extend(additional_sorted[:suggestion['target_additional']])

        self._save_state()

        return {
            'improvement_applied': suggestion,
            'new_selection_size': len(self.state.selected_indices),
            'ready_for_retraining': True
        }