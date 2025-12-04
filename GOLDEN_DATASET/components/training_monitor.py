"""
Intelligent Training Monitor - FIXED
Real-time progress tracking with proper step vs epoch handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl


class IntelligentMonitorCallback(TrainerCallback):
    """
    Callback that updates Streamlit UI during training
    FIXED: Properly handles step logs vs epoch logs
    """

    def __init__(self, placeholder_key: str = "monitor"):
        self.placeholder_key = placeholder_key
        self.epoch_history = []  # Only epoch-level logs
        self.step_history = []   # Step-level logs for smoothing
        self.start_time = None
        self.should_stop = False
        self.warnings = []
        self.last_completed_epoch = -1  # Track completed epochs

        # Thresholds for problem detection
        self.plateau_threshold = 3  # epochs without improvement
        self.overfitting_ratio = 1.2  # eval_loss / train_loss
        self.min_improvement = 0.001  # minimum F1 improvement

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        self.epoch_history = []
        self.step_history = []
        self.warnings = []
        self.last_completed_epoch = -1

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called after logging - update UI"""
        if logs is None:
            return

        if 'epoch' not in logs:
            return

        # Check if user requested stop
        if st.session_state.get('request_training_stop', False):
            self.should_stop = True
            control.should_training_stop = True
            st.session_state['training_stopped_by_user'] = True
            return control

        # ✅ FIX: Distinguish step logs from epoch logs
        is_eval_log = 'eval_loss' in logs or 'eval_f1_micro' in logs
        current_epoch = int(logs['epoch'])  # Integer epoch number

        if is_eval_log:
            # This is an epoch-level evaluation log
            if current_epoch > self.last_completed_epoch:
                # New epoch completed
                self.last_completed_epoch = current_epoch
                self.epoch_history.append(logs.copy())

                # Analyze and update UI
                analysis = self._analyze_training_state()
                self._update_streamlit_ui(logs, analysis, is_epoch_end=True)
        else:
            # This is a step-level training log
            self.step_history.append(logs.copy())

            # Update UI less frequently (every ~10 steps)
            if len(self.step_history) % 10 == 0:
                self._update_streamlit_ui(logs, {'status': 'training'}, is_epoch_end=False)

        return control

    def _analyze_training_state(self) -> Dict:
        """Analyze training progress using ONLY epoch-level logs"""

        if len(self.epoch_history) < 2:
            return {
                'status': 'starting',
                'message': '🟢 Training started...',
                'problems': [],
                'recommendations': []
            }

        latest = self.epoch_history[-1]
        problems = []
        recommendations = []

        # Extract metrics from LATEST EPOCH
        train_loss = latest.get('loss', latest.get('train_loss', 0))
        eval_loss = latest.get('eval_loss', 0)
        f1_micro = latest.get('eval_f1_micro', 0)

        # Check 1: Overfitting
        if eval_loss > 0 and train_loss > 0:
            loss_ratio = eval_loss / train_loss
            if loss_ratio > self.overfitting_ratio:
                problems.append({
                    'severity': 'warning',
                    'type': 'overfitting',
                    'message': f'Eval loss {loss_ratio:.2f}x higher than train loss'
                })
                recommendations.append('Consider: early stopping, more dropout, or more training data')

        # Check 2: Plateau (using EPOCH history only)
        if len(self.epoch_history) >= self.plateau_threshold:
            recent_f1s = [h.get('eval_f1_micro', 0) for h in self.epoch_history[-self.plateau_threshold:]]
            improvements = [recent_f1s[i] - recent_f1s[i-1] for i in range(1, len(recent_f1s))]

            if all(abs(imp) < self.min_improvement for imp in improvements):
                problems.append({
                    'severity': 'info',
                    'type': 'plateau',
                    'message': f'F1 hasn\'t improved significantly in {self.plateau_threshold} epochs'
                })
                recommendations.append('Model may have converged - consider stopping')

        # Check 3: Degradation
        if len(self.epoch_history) >= 3:
            last_3_f1s = [h.get('eval_f1_micro', 0) for h in self.epoch_history[-3:]]
            if last_3_f1s[-1] < last_3_f1s[-3] - 0.05:
                problems.append({
                    'severity': 'error',
                    'type': 'degradation',
                    'message': f'F1 dropped by {(last_3_f1s[-3] - last_3_f1s[-1]):.3f}'
                })
                recommendations.append('STOP TRAINING - model degrading. Load earlier checkpoint.')

        # Check 4: No learning (check step-level train loss)
        if len(self.step_history) >= 20:
            recent_losses = [h.get('loss', 0) for h in self.step_history[-20:]]
            loss_change = abs(recent_losses[-1] - recent_losses[0])

            if loss_change < 0.001:
                problems.append({
                    'severity': 'warning',
                    'type': 'no_learning',
                    'message': 'Training loss not decreasing'
                })
                recommendations.append('Check: learning rate too low? Data quality issues?')

        # Check 5: Exploding loss
        if train_loss > 10 or eval_loss > 10:
            problems.append({
                'severity': 'error',
                'type': 'exploding_loss',
                'message': 'Loss values extremely high - training unstable'
            })
            recommendations.append('STOP TRAINING - reduce learning rate')

        # Check 6: Poor performance (use epoch count)
        if len(self.epoch_history) >= 5 and f1_micro < 0.3:
            problems.append({
                'severity': 'warning',
                'type': 'poor_performance',
                'message': f'F1 still below 0.3 after {len(self.epoch_history)} epochs'
            })
            recommendations.append('Consider: weighted loss, focal loss, or better data quality')

        # Determine overall status
        if any(p['severity'] == 'error' for p in problems):
            status = 'critical'
            message = '🔴 Critical issues detected - consider stopping'
        elif any(p['severity'] == 'warning' for p in problems):
            status = 'warning'
            message = '🟡 Warning signs detected - monitor closely'
        elif any(p['type'] == 'plateau' for p in problems):
            status = 'plateau'
            message = '🔵 Training plateaued - may have converged'
        else:
            status = 'healthy'
            message = '🟢 Training looks healthy'

        return {
            'status': status,
            'message': message,
            'problems': problems,
            'recommendations': recommendations
        }

    def _update_streamlit_ui(self, logs: Dict, analysis: Dict, is_epoch_end: bool = False):
        """Update Streamlit interface with current status"""

        placeholder = st.session_state.get(f'{self.placeholder_key}_placeholder')
        if placeholder is None:
            return

        with placeholder.container():
            # ✅ IMPROVED HEADER
            if is_epoch_end:
                # Full update on epoch completion
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.markdown(f"### {analysis.get('message', '🟢 Training in progress')}")

                with col2:
                    elapsed = (datetime.now() - self.start_time).total_seconds() / 60
                    st.metric("Elapsed", f"{elapsed:.1f}m")

                with col3:
                    if st.button("🛑 Stop Training", key=f"stop_btn_{len(self.epoch_history)}"):
                        st.session_state['request_training_stop'] = True
                        st.warning("⚠️ Stop requested...")

                st.markdown("---")

                # ✅ CURRENT METRICS (epoch-level)
                st.markdown("#### 📊 Current Metrics")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    epoch = logs.get('epoch', 0)
                    st.metric("Epoch", f"{int(epoch)}")

                with col2:
                    train_loss = logs.get('loss', logs.get('train_loss', 0))
                    st.metric("Train Loss", f"{train_loss:.4f}")

                with col3:
                    eval_loss = logs.get('eval_loss', 0)
                    delta_loss = None
                    if len(self.epoch_history) >= 2:
                        prev_loss = self.epoch_history[-2].get('eval_loss', 0)
                        delta_loss = eval_loss - prev_loss
                    st.metric("Eval Loss", f"{eval_loss:.4f}",
                              delta=f"{delta_loss:.4f}" if delta_loss else None,
                              delta_color="inverse")

                with col4:
                    f1_micro = logs.get('eval_f1_micro', 0)
                    delta_f1 = None
                    if len(self.epoch_history) >= 2:
                        prev_f1 = self.epoch_history[-2].get('eval_f1_micro', 0)
                        delta_f1 = f1_micro - prev_f1
                    st.metric("F1 Micro", f"{f1_micro:.3f}",
                              delta=f"{delta_f1:.3f}" if delta_f1 else None)

                # ✅ PROBLEMS & RECOMMENDATIONS
                if analysis.get('problems'):
                    st.markdown("---")
                    st.markdown("#### ⚠️ Issues Detected")

                    for problem in analysis['problems']:
                        if problem['severity'] == 'error':
                            st.error(f"🔴 {problem['message']}")
                        elif problem['severity'] == 'warning':
                            st.warning(f"🟡 {problem['message']}")
                        else:
                            st.info(f"🔵 {problem['message']}")

                    if analysis.get('recommendations'):
                        st.markdown("**Recommendations:**")
                        for rec in analysis['recommendations']:
                            st.markdown(f"- {rec}")

                # ✅ TRAINING CURVES (only if we have multiple epochs)
                if len(self.epoch_history) >= 2:
                    st.markdown("---")
                    st.markdown("#### 📈 Training Progress")

                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                    epochs = [h.get('epoch', i) for i, h in enumerate(self.epoch_history)]
                    train_losses = [h.get('loss', h.get('train_loss', 0)) for h in self.epoch_history]
                    eval_losses = [h.get('eval_loss', 0) for h in self.epoch_history]
                    f1_scores = [h.get('eval_f1_micro', 0) for h in self.epoch_history]

                    # Loss plot
                    axes[0].plot(epochs, train_losses, 'b-', label='Train', linewidth=2, marker='o')
                    axes[0].plot(epochs, eval_losses, 'r-', label='Eval', linewidth=2, marker='s')
                    axes[0].set_xlabel('Epoch')
                    axes[0].set_ylabel('Loss')
                    axes[0].set_title('Loss Over Time')
                    axes[0].legend()
                    axes[0].grid(alpha=0.3)

                    # F1 plot
                    axes[1].plot(epochs, f1_scores, 'g-', linewidth=2, marker='o')
                    axes[1].set_xlabel('Epoch')
                    axes[1].set_ylabel('F1 Micro')
                    axes[1].set_title('F1 Score Over Time')
                    axes[1].set_ylim([0, 1])
                    axes[1].grid(alpha=0.3)

                    # Mark best F1
                    if f1_scores:
                        best_idx = np.argmax(f1_scores)
                        axes[1].scatter([epochs[best_idx]], [f1_scores[best_idx]],
                                        c='red', s=100, marker='*', zorder=5)
                        axes[1].text(epochs[best_idx], f1_scores[best_idx] + 0.05,
                                     f'Best: {f1_scores[best_idx]:.3f}',
                                     ha='center', fontsize=9)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # ✅ PER-LABEL PERFORMANCE
                with st.expander("📊 Per-Label Performance"):
                    label_metrics = []
                    for key, value in logs.items():
                        if key.startswith('eval_f1_') and key not in ['eval_f1_micro', 'eval_f1_macro',
                                                                        'eval_f1_micro_all', 'eval_f1_macro_all',
                                                                        'eval_f1_micro_sublabels', 'eval_f1_macro_sublabels']:
                            label_name = key.replace('eval_f1_', '')

                            # Get previous value for delta
                            delta = None
                            if len(self.epoch_history) >= 2:
                                prev_value = self.epoch_history[-2].get(key, 0)
                                delta = value - prev_value

                            quality = '🟢' if value > 0.7 else '🟡' if value > 0.5 else '🔴'

                            label_metrics.append({
                                'Label': label_name,
                                'F1': f"{value:.3f}",
                                'Change': f"{delta:+.3f}" if delta is not None else '—',
                                'Status': quality
                            })

                    if label_metrics:
                        # Sort by F1 (worst first)
                        label_metrics.sort(key=lambda x: float(x['F1']))
                        st.dataframe(
                            pd.DataFrame(label_metrics),
                            hide_index=True,
                            use_container_width=True
                        )

            else:
                # ✅ LIGHTWEIGHT STEP UPDATES
                current_step_epoch = logs.get('epoch', 0)
                current_step_loss = logs.get('loss', 0)

                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.markdown(f"### 🔄 Training... Epoch {current_step_epoch:.1f}")

                with col2:
                    st.metric("Current Loss", f"{current_step_loss:.4f}")

                with col3:
                    elapsed = (datetime.now() - self.start_time).total_seconds() / 60
                    st.caption(f"⏱️ {elapsed:.1f}m elapsed")

                # Show mini progress bar
                integer_epoch = int(current_step_epoch)
                progress_in_epoch = current_step_epoch - integer_epoch
                st.progress(progress_in_epoch, text=f"Epoch {integer_epoch + 1} progress")


def create_training_monitor() -> tuple:
    """
    Create monitoring infrastructure for training

    Returns:
        (callback, placeholder) - callback to pass to trainer, placeholder to update
    """

    # Create placeholder in Streamlit
    placeholder = st.empty()

    # Store in session state
    st.session_state['monitor_placeholder'] = placeholder
    st.session_state['request_training_stop'] = False

    # Create callback
    callback = IntelligentMonitorCallback(placeholder_key='monitor')

    return callback, placeholder