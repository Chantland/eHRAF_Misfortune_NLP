"""
Experiment tracking and management
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


class ExperimentTracker:
    """Track and compare experiments"""

    def __init__(self, experiments_dir: str = "./experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)

    def log_experiment(
            self,
            name: str,
            config: Dict,
            data_selection: Dict,
            results: Dict
    ) -> Path:
        """
        Log an experiment

        Returns:
            Path to experiment directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = self.experiments_dir / f"{name}_{timestamp}"
        exp_dir.mkdir(exist_ok=True)

        # Save metadata
        metadata = {
            'name': name,
            'timestamp': timestamp,
            'config': config,
            'data_selection': data_selection,
            'results': {
                k: float(v) if isinstance(v, (float, int)) else v
                for k, v in results['test_metrics'].items()
            }
        }

        with open(exp_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save training history
        if 'history' in results:
            history_df = pd.DataFrame(results['history'])
            history_df.to_csv(exp_dir / 'training_history.csv', index=False)

        print(f"✅ Experiment logged: {exp_dir}")

        return exp_dir

    def list_experiments(self) -> List[Dict]:
        """List all experiments"""
        experiments = []

        for exp_dir in sorted(self.experiments_dir.iterdir(), reverse=True):
            if exp_dir.is_dir():
                metadata_file = exp_dir / 'metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    experiments.append({
                        'directory': exp_dir,
                        'name': metadata['name'],
                        'timestamp': metadata['timestamp'],
                        'f1_micro': metadata['results'].get('eval_f1_micro', 0)
                    })

        return experiments

    def compare_experiments(
            self,
            exp_names: List[str]
    ) -> pd.DataFrame:
        """Compare multiple experiments"""
        comparisons = []

        for exp in self.list_experiments():
            if exp['name'] in exp_names:
                metadata_file = exp['directory'] / 'metadata.json'
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                comparisons.append({
                    'Name': metadata['name'],
                    'Timestamp': metadata['timestamp'],
                    'F1 Micro': metadata['results'].get('eval_f1_micro', 0),
                    'F1 Macro': metadata['results'].get('eval_f1_macro', 0),
                    'Num Passages': metadata['data_selection'].get('num_selected', 0),
                    'Min Quality': metadata['data_selection'].get('min_quality', 0)
                })

        return pd.DataFrame(comparisons)