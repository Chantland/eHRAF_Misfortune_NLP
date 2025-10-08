"""
Proactive Claude assistant - provides context-aware guidance
"""

import anthropic
import os
from typing import Dict, Optional


class ProactiveAssistant:
    """
    AI assistant that analyzes workflow state and provides
    proactive suggestions without being asked
    """

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
            self.enabled = True
        else:
            self.client = None
            self.enabled = False

    def get_suggestion(self, pipeline) -> Optional[str]:
        """
        Analyze pipeline state and return proactive suggestion

        Returns:
            Suggestion string or None
        """
        if not self.enabled:
            return None

        # Different suggestions based on workflow state
        current_step = pipeline.get_current_step()

        if current_step == "load":
            return self._suggest_load(pipeline)

        elif current_step == "quality":
            return self._suggest_quality(pipeline)

        elif current_step == "explore":
            return self._suggest_explore(pipeline)

        elif current_step == "train":
            return self._suggest_train(pipeline)

        elif current_step == "iterate":
            return self._suggest_iterate(pipeline)

        return None

    def _suggest_load(self, pipeline) -> Optional[str]:
        """Suggestions for data loading step"""
        return (
            "👋 **Getting started:** Upload your Excel file with passages and labels. "
            "The tool will auto-detect your passage text and label columns."
        )

    def _suggest_quality(self, pipeline) -> Optional[str]:
        """Suggestions for quality scoring step"""
        if not pipeline.has_data():
            return None

        num_passages = len(pipeline.state.df)

        if num_passages > 20000:
            return (
                f"📊 You have {num_passages:,} passages. Quality scoring will take 10-15 minutes. "
                "Consider starting with a sample for faster iteration."
            )
        elif num_passages < 1000:
            return (
                f"⚠️ Only {num_passages} passages. This may be too small for robust training. "
                "Consider gathering more data if possible."
            )

        return "🔬 Quality scoring uses semantic similarity to find your best training examples."

    def _suggest_explore(self, pipeline) -> Optional[str]:
        """Suggestions for data exploration step"""
        if not pipeline.has_quality_scores():
            return None

        dist = pipeline.state.quality_distribution
        elite_pct = dist['tier_percentages']['elite']
        median = dist['median']

        if median < 0.50:
            return (
                f"⚠️ **Low median quality ({median:.2f})** suggests high labeling disagreement. "
                "Consider reviewing your label definitions or focusing on high-quality examples only."
            )

        elif elite_pct < 10:
            return (
                f"📊 Only {elite_pct:.1f}% elite quality data. "
                "You may need to lower your quality threshold or use an aggressive selection strategy."
            )

        elif elite_pct > 25:
            return (
                f"✅ Great! {elite_pct:.1f}% elite quality data. "
                "You can afford to be selective - try a conservative approach for best results."
            )

        return "🎯 Balance quality and quantity: Higher threshold = better learning but less data."

    def _suggest_train(self, pipeline) -> Optional[str]:
        """Suggestions for training step"""
        if not pipeline.has_explored():
            return None

        num_selected = len(pipeline.state.selected_indices)
        num_labels = len(pipeline.state.label_columns)

        # Check if data size is appropriate
        examples_per_label = num_selected / num_labels

        if examples_per_label < 50:
            return (
                f"⚠️ Only ~{examples_per_label:.0f} examples per label on average. "
                "This might not be enough. Consider lowering quality threshold or training on specific labels only."
            )

        elif examples_per_label > 500:
            return (
                f"✅ Strong coverage: ~{examples_per_label:.0f} examples per label. "
                "Consider using a smaller learning rate (1e-5) to avoid overfitting."
            )

        return "🎓 Default settings work well for most cases. Click 'Advanced Settings' only if needed."

    def _suggest_iterate(self, pipeline) -> Optional[str]:
        """Suggestions for iteration step"""
        if not pipeline.has_results():
            return None

        f1_micro = pipeline.state.best_metrics.get('f1_micro', 0)

        if f1_micro < 0.60:
            return (
                f"📉 **F1 {f1_micro:.3f} is below target.** "
                "Run failure analysis to find specific issues. Typical causes: "
                "low-quality training data, insufficient examples for rare labels, or ambiguous label definitions."
            )

        elif f1_micro < 0.70:
            return (
                f"📊 **F1 {f1_micro:.3f} is decent but improvable.** "
                "Failure analysis will show which labels need attention. "
                "Often removing the lowest-quality examples gives a quick boost."
            )

        elif f1_micro >= 0.75:
            return (
                f"🎉 **Excellent! F1 {f1_micro:.3f}** "
                "Your model performs well. Failure analysis can still identify edge cases for perfection."
            )

        return "🔍 Run failure analysis to understand where the model struggles and how to improve."

    def analyze_with_claude(
            self,
            pipeline,
            question: str
    ) -> str:
        """
        Ask Claude to analyze current state and answer question

        This is for more complex queries that need reasoning
        """
        if not self.enabled:
            return "Claude API not configured. Set ANTHROPIC_API_KEY."

        # Build context about current state
        context = self._build_context(pipeline)

        # Ask Claude
        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1000,
            temperature=1.0,
            system=f"""You are an expert in NLP data quality and model training.
You're helping a researcher with their HRAF project.

Current pipeline state:
{context}

Provide concise, actionable advice.""",
            messages=[
                {"role": "user", "content": question}
            ]
        )

        return response.content[0].text

    def _build_context(self, pipeline) -> str:
        """Build context string about pipeline state"""
        context_parts = []

        if pipeline.has_data():
            context_parts.append(
                f"- Loaded {len(pipeline.state.df)} passages with {len(pipeline.state.label_columns)} labels"
            )

        if pipeline.has_quality_scores():
            dist = pipeline.state.quality_distribution
            context_parts.append(
                f"- Quality: median={dist['median']:.2f}, "
                f"{dist['tier_percentages']['elite']:.1f}% elite"
            )

        if pipeline.has_explored():
            context_parts.append(
                f"- Selected {len(pipeline.state.selected_indices)} passages for training"
            )

        if pipeline.has_model():
            f1 = pipeline.state.best_metrics.get('f1_micro', 0)
            context_parts.append(
                f"- Trained model: F1 micro = {f1:.3f}"
            )

        if pipeline.state.improvement_suggestions:
            context_parts.append(
                f"- Generated {len(pipeline.state.improvement_suggestions)} improvement suggestions"
            )

        return "\n".join(context_parts) if context_parts else "- No data loaded yet"


def render_chat_interface(pipeline, assistant):
    """
    Optional chat interface for asking questions
    Can be added to sidebar or as separate page
    """
    import streamlit as st

    st.markdown("### 💬 Ask the Assistant")

    question = st.text_input(
        "Ask a question about your data or model:",
        placeholder="e.g., Why is my F1 score low?"
    )

    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            answer = assistant.analyze_with_claude(pipeline, question)
            st.markdown(answer)