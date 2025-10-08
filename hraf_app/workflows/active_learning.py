"""
Active learning loop - closes the gap between training and data selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from collections import defaultdict


class ActiveLearner:
    """
    Analyzes model failures and suggests data improvements
    The missing link between training results and data selection
    """

    def __init__(self, model, trainer, quality_scorer):
        self.model = model
        self.trainer = trainer
        self.quality_scorer = quality_scorer

    def analyze_failures(
            self,
            test_df: pd.DataFrame,
            quality_scores: Dict
    ) -> Dict:
        """
        Analyze where and why the model fails

        Returns:
            Structured failure analysis with actionable insights
        """
        print("🔍 Analyzing model failures...")

        # Get predictions for test set
        predictions = self._get_predictions(test_df)

        # Categorize failures
        failures = {
            'by_label': defaultdict(list),
            'by_error_type': defaultdict(list),
            'by_quality': defaultdict(list),
            'summary': {}
        }

        for idx, pred in predictions.items():
            actual_labels = set(
                label for label in self.trainer.label_columns
                if test_df.loc[idx, label] == 1
            )
            predicted_labels = set(pred['predicted_labels'])

            # Identify errors
            false_positives = predicted_labels - actual_labels
            false_negatives = actual_labels - predicted_labels

            if false_positives or false_negatives:
                quality = quality_scores.get(idx)
                quality_score = quality.overall_quality if quality else 0.5

                error_info = {
                    'idx': idx,
                    'quality': quality_score,
                    'actual': list(actual_labels),
                    'predicted': list(predicted_labels),
                    'false_positives': list(false_positives),
                    'false_negatives': list(false_negatives)
                }

                # Group by label
                for label in false_positives:
                    failures['by_label'][label].append({
                        **error_info,
                        'error_type': 'false_positive'
                    })

                for label in false_negatives:
                    failures['by_label'][label].append({
                        **error_info,
                        'error_type': 'false_negative'
                    })

                # Group by error type
                if false_positives:
                    failures['by_error_type']['over_prediction'].append(error_info)
                if false_negatives:
                    failures['by_error_type']['under_prediction'].append(error_info)

                # Group by quality
                quality_tier = quality.tier if quality else 'unknown'
                failures['by_quality'][quality_tier].append(error_info)

        # Generate summary
        failures['summary'] = self._summarize_failures(failures)

        print(f"✅ Found {len(failures['by_error_type']['over_prediction'])} over-predictions")
        print(f"✅ Found {len(failures['by_error_type']['under_prediction'])} under-predictions")

        return failures

    def _get_predictions(self, test_df: pd.DataFrame) -> Dict:
        """Get model predictions for test set"""
        predictions = {}

        for idx in test_df.index:
            text = test_df.loc[idx, self.trainer.passage_col]
            if pd.notna(text):
                pred = self.trainer.predict_passage(text)
                predictions[idx] = pred

        return predictions

    def _summarize_failures(self, failures: Dict) -> Dict:
        """Create high-level summary of failures"""
        summary = {}

        # Most problematic labels
        label_error_counts = {
            label: len(errors)
            for label, errors in failures['by_label'].items()
        }
        summary['worst_labels'] = sorted(
            label_error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Quality correlation
        quality_tiers = failures['by_quality']
        summary['failures_by_tier'] = {
            tier: len(errors)
            for tier, errors in quality_tiers.items()
        }

        # Overall error rates
        total_over = len(failures['by_error_type']['over_prediction'])
        total_under = len(failures['by_error_type']['under_prediction'])
        summary['total_errors'] = total_over + total_under
        summary['over_prediction_rate'] = total_over
        summary['under_prediction_rate'] = total_under

        return summary

    def suggest_improvements(
            self,
            failure_analysis: Dict
    ) -> List[Dict]:
        """
        Generate actionable improvement suggestions

        Returns:
            List of improvement suggestions with specific actions
        """
        print("💡 Generating improvement suggestions...")

        suggestions = []

        # Check quality-based patterns
        quality_failures = failure_analysis['by_quality']

        # Suggestion 1: Remove low-quality examples with high error rates
        if 'low' in quality_failures and len(quality_failures['low']) > 5:
            suggestions.append({
                'id': len(suggestions),
                'type': 'remove_low_quality',
                'priority': 'high',
                'title': 'Remove low-quality passages causing errors',
                'description': (
                    f"Found {len(quality_failures['low'])} low-quality passages "
                    f"causing prediction errors. Removing these may improve overall performance."
                ),
                'action': 'Remove passages with quality < 0.45',
                'threshold': 0.45,
                'affected_passages': [e['idx'] for e in quality_failures['low']],
                'expected_impact': 'Reduce noise in training data'
            })

        # Suggestion 2: Label-specific improvements
        worst_labels = failure_analysis['summary']['worst_labels']

        for label, error_count in worst_labels[:3]:  # Top 3 problematic labels
            label_failures = failure_analysis['by_label'][label]

            # Check if it's mostly false negatives (under-represented)
            false_negatives = sum(
                1 for f in label_failures
                if f['error_type'] == 'false_negative'
            )

            if false_negatives > error_count * 0.6:
                # Need more examples
                suggestions.append({
                    'id': len(suggestions),
                    'type': 'add_more_examples',
                    'priority': 'medium',
                    'title': f'Add more "{label}" examples',
                    'description': (
                        f"Model struggles with {label} (mostly missed predictions). "
                        f"Adding more high-quality examples may help."
                    ),
                    'action': f'Add 20-50 more high-quality {label} examples',
                    'label': label,
                    'target_additional': 30,
                    'expected_impact': f'Improve {label} recall'
                })
            else:
                # Check quality of existing examples
                avg_quality = np.mean([f['quality'] for f in label_failures])

                if avg_quality < 0.55:
                    suggestions.append({
                        'id': len(suggestions),
                        'type': 'improve_label_quality',
                        'priority': 'high',
                        'title': f'Improve quality of "{label}" examples',
                        'description': (
                            f"Model confuses {label} (avg quality: {avg_quality:.2f}). "
                            f"Existing examples may be ambiguous. Consider reviewing labels."
                        ),
                        'action': f'Review and relabel or remove low-quality {label} examples',
                        'label': label,
                        'affected_passages': [f['idx'] for f in label_failures],
                        'expected_impact': f'Reduce {label} confusion'
                    })

        # Suggestion 3: Systematic over-prediction
        over_pred_rate = failure_analysis['summary']['over_prediction_rate']
        under_pred_rate = failure_analysis['summary']['under_prediction_rate']

        if over_pred_rate > under_pred_rate * 2:
            suggestions.append({
                'id': len(suggestions),
                'type': 'adjust_thresholds',
                'priority': 'low',
                'title': 'Adjust prediction thresholds',
                'description': (
                    f"Model tends to over-predict (ratio {over_pred_rate}:{under_pred_rate}). "
                    f"Consider raising prediction thresholds."
                ),
                'action': 'Increase decision threshold from 0.5 to 0.55-0.60',
                'expected_impact': 'Reduce false positives'
            })

        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        suggestions.sort(key=lambda x: priority_order[x['priority']])

        print(f"✅ Generated {len(suggestions)} improvement suggestions")

        return suggestions

    def predict_improvement_impact(
            self,
            suggestion: Dict,
            current_metrics: Dict
    ) -> Dict:
        """
        Estimate the impact of applying a suggestion

        This is a simple heuristic-based prediction
        More sophisticated: use held-out validation or bootstrap
        """
        impact = {
            'estimated_f1_change': 0.0,
            'confidence': 'low'
        }

        if suggestion['type'] == 'remove_low_quality':
            # Removing noise typically helps 2-5% F1
            num_removed = len(suggestion['affected_passages'])
            impact['estimated_f1_change'] = min(0.05, num_removed / 1000 * 0.03)
            impact['confidence'] = 'medium'

        elif suggestion['type'] == 'add_more_examples':
            # Adding examples helps if severely underrepresented
            impact['estimated_f1_change'] = 0.03
            impact['confidence'] = 'medium'

        elif suggestion['type'] == 'improve_label_quality':
            # Biggest potential impact
            impact['estimated_f1_change'] = 0.07
            impact['confidence'] = 'medium'

        elif suggestion['type'] == 'adjust_thresholds':
            # Quick fix, modest impact
            impact['estimated_f1_change'] = 0.02
            impact['confidence'] = 'high'

        # Add to suggestion
        new_f1 = current_metrics.get('f1_micro', 0.6) + impact['estimated_f1_change']
        impact['predicted_new_f1'] = min(new_f1, 0.95)  # Cap at 0.95

        return impact