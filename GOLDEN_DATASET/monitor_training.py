#!/usr/bin/env python3
"""
Training Monitor Script
Checks training progress and reports metrics
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

def find_latest_training_dir(models_dir: str = "models") -> Path:
    """Find the most recent training directory"""
    models_path = Path(models_dir)
    if not models_path.exists():
        return None

    training_dirs = [d for d in models_path.iterdir()
                     if d.is_dir() and d.name.startswith("training_")]

    if not training_dirs:
        return None

    return max(training_dirs, key=lambda d: d.stat().st_mtime)


def get_latest_checkpoint(training_dir: Path) -> Path:
    """Find the latest checkpoint in a training directory"""
    checkpoints = list(training_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda d: int(d.name.split("-")[1]))


def read_trainer_state(checkpoint_dir: Path) -> dict:
    """Read trainer state from checkpoint"""
    state_file = checkpoint_dir / "trainer_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return None


def format_metrics(metrics: dict, prefix: str = "eval_") -> str:
    """Format metrics for display"""
    lines = []

    # Key metrics first
    f1_micro = metrics.get(f"{prefix}f1_micro", 0)
    f1_macro = metrics.get(f"{prefix}f1_macro", 0)

    lines.append(f"  F1 Micro: {f1_micro:.4f}  |  F1 Macro: {f1_macro:.4f}")

    # Per-label F1
    label_f1s = [(k.replace(f"{prefix}f1_", ""), v)
                 for k, v in metrics.items()
                 if k.startswith(f"{prefix}f1_") and k not in [f"{prefix}f1_micro", f"{prefix}f1_macro"]]

    if label_f1s:
        label_f1s.sort(key=lambda x: x[1], reverse=True)
        lines.append("  Per-label F1:")
        for label, f1 in label_f1s:
            bar = "█" * int(f1 * 20) + "░" * (20 - int(f1 * 20))
            lines.append(f"    {label:25s} {bar} {f1:.3f}")

    return "\n".join(lines)


def monitor(interval: int = 60, max_checks: int = 1000):
    """Monitor training progress"""

    print("=" * 60)
    print("TRAINING MONITOR")
    print("=" * 60)
    print(f"Checking every {interval} seconds...")
    print("Press Ctrl+C to stop monitoring\n")

    last_step = -1

    for check_num in range(max_checks):
        try:
            # Find latest training directory
            training_dir = find_latest_training_dir()

            if training_dir is None:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No training directory found. Waiting...")
                time.sleep(interval)
                continue

            # Check if training is still running
            import subprocess
            result = subprocess.run(
                ["pgrep", "-f", "train_model.py"],
                capture_output=True,
                text=True
            )
            is_running = result.returncode == 0

            # Get latest checkpoint
            checkpoint = get_latest_checkpoint(training_dir)

            if checkpoint is None:
                status = "🔄 Training starting..." if is_running else "⏸️ No checkpoints yet"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {training_dir.name}: {status}")
                time.sleep(interval)
                continue

            # Read trainer state
            state = read_trainer_state(checkpoint)

            if state is None:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Could not read trainer state")
                time.sleep(interval)
                continue

            current_step = state.get("global_step", 0)
            total_steps = state.get("max_steps", 0)
            epoch = state.get("epoch", 0)

            # Only report if step changed
            if current_step != last_step:
                last_step = current_step

                print("\n" + "=" * 60)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {training_dir.name}")
                print("=" * 60)

                status_icon = "🏃" if is_running else "✅"
                print(f"{status_icon} Step {current_step}/{total_steps} | Epoch {epoch:.2f}")

                # Show progress bar
                if total_steps > 0:
                    progress = current_step / total_steps
                    bar_width = 40
                    filled = int(progress * bar_width)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    print(f"   [{bar}] {progress*100:.1f}%")

                # Show best metrics
                best_metric = state.get("best_metric", None)
                if best_metric is not None:
                    print(f"\n📊 Best F1 Micro so far: {best_metric:.4f}")

                # Show log history (last eval)
                log_history = state.get("log_history", [])
                eval_logs = [log for log in log_history if "eval_f1_micro" in log]

                if eval_logs:
                    latest_eval = eval_logs[-1]
                    print(f"\n📈 Latest Evaluation (step {latest_eval.get('step', '?')}):")
                    print(format_metrics(latest_eval))

                    # Check target
                    f1_micro = latest_eval.get("eval_f1_micro", 0)
                    if f1_micro > 0.72:
                        print(f"\n🎯 TARGET EXCEEDED! F1 Micro = {f1_micro:.4f} > 0.72")

                if not is_running:
                    print("\n✅ Training completed!")

                    # Check for final results
                    final_model = training_dir / "final_model"
                    if final_model.exists():
                        info_file = final_model / "training_info.json"
                        if info_file.exists():
                            with open(info_file) as f:
                                info = json.load(f)
                            test_results = info.get("test_results", {})
                            if test_results:
                                print("\n📊 FINAL TEST RESULTS:")
                                print(format_metrics(test_results))
                    break

            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    monitor(interval=interval)
