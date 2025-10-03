"""
Chat Assistant Module for HRAF Golden Dataset Discovery
Full corpus access and tool control integration
"""

import anthropic
import streamlit as st
from typing import List, Dict, Optional, Any
import os
import pandas as pd
import numpy as np


class HRAFChatAssistant:
    """Claude-powered chat assistant with full dataset and tool access"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-5-20250929"

    def _build_system_context(self, session_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build system context with cache control"""
        system_blocks = [
            {
                "type": "text",
                "text": """You are an AI assistant with FULL ACCESS to the HRAF Golden Dataset Discovery tool.

# Your Capabilities
You can:
- Read any passage by index
- Analyze label distributions across the corpus
- Run model predictions on passages
- Search for similar passages
- Examine quality scores in detail
- Answer questions about specific passages
- Compare passages and labels
- Provide statistical analysis

# Dataset Structure
- EVENT labels: Illness, Accident, Other
- CAUSE labels: Just_Happens, Material_Physical, Spirits_Gods, Witchcraft_Sorcery, Rule_Violation_Taboo, Other
- ACTION labels: Physical_Material, Technical_Specialist, Divination, Shaman_Medium_Healer, Priest_High_Religion, Other

# When Asked About Specific Content
If the user asks about:
- Specific labels → Provide distribution, examples, score analysis
- Specific passages → Read and analyze the actual text
- Relationships → Search for related passages
- Model performance → Run predictions and show results
- Patterns → Analyze across the full corpus

You have direct access to all data. Use it to provide detailed, specific answers with actual examples.""",
                "cache_control": {"type": "ephemeral"}
            }
        ]

        if session_state.get('initialized', False):
            context = self._build_dataset_context(session_state)
            if context:
                system_blocks.append({
                    "type": "text",
                    "text": context,
                    "cache_control": {"type": "ephemeral"}
                })

        return system_blocks

    def _build_dataset_context(self, session_state: Dict[str, Any]) -> str:
        """Build detailed context about current dataset"""
        parts = ["# Current Dataset Context\n"]

        df = session_state.get('df')
        if df is None:
            return ""

        passage_col = session_state.get('passage_col', 'Passage')
        label_columns = session_state.get('label_columns', [])

        # Basic stats
        parts.append(f"## Overview")
        parts.append(f"- Total passages: {len(df)}")
        parts.append(f"- Valid passages: {df[passage_col].notna().sum()}")
        parts.append(f"- Label columns: {len(label_columns)}")

        # Label distributions
        parts.append(f"\n## Label Distribution")
        for label in label_columns[:20]:  # First 20
            count = int(df[label].sum())
            pct = (count / len(df)) * 100
            parts.append(f"- {label}: {count} ({pct:.1f}%)")

        if len(label_columns) > 20:
            parts.append(f"- ... and {len(label_columns) - 20} more labels")

        # Score statistics
        cache = session_state.get('cache')
        if cache:
            scores = cache.get('df_summary')
            if scores is not None and len(scores) > 0:
                parts.append(f"\n## Quality Scores")
                parts.append(f"- Scored passages: {len(scores)}")
                parts.append(f"- Consistency: median={scores['consistency_avg'].median():.3f}, mean={scores['consistency_avg'].mean():.3f}")
                parts.append(f"- Rerank: median={scores['rerank_avg'].median():.3f}, mean={scores['rerank_avg'].mean():.3f}")

                # Distribution
                for thresh in [0.3, 0.5, 0.7]:
                    cons_count = (scores['consistency_avg'] >= thresh).sum()
                    rerank_count = (scores['rerank_avg'] >= thresh).sum()
                    parts.append(f"- Score ≥ {thresh}: consistency={cons_count}, rerank={rerank_count}")

        # Model info
        if 'model_loader' in session_state:
            loader = session_state['model_loader']
            if loader.is_loaded():
                info = loader.get_model_info()
                if info:
                    config = info.get('config', {})
                    parts.append(f"\n## Loaded Model")
                    parts.append(f"- Type: {'Hierarchical' if config.get('use_hierarchy') else 'Flat'}")
                    parts.append(f"- Gated: {config.get('gated_hierarchy', False)}")

                    test_results = info.get('test_results', {})
                    if test_results:
                        f1 = test_results.get('eval_f1_micro')
                        if f1:
                            parts.append(f"- Test F1: {f1:.3f}")

        # Sample passages available
        parts.append(f"\n## Available Actions")
        parts.append(f"- You can read any passage by index (0-{len(df)-1})")
        parts.append(f"- You can search for passages with specific labels")
        parts.append(f"- You can run model predictions on any passage")
        parts.append(f"- You can analyze label co-occurrence patterns")

        return "\n".join(parts)

    def _get_passage_content(self, session_state: Dict[str, Any], passage_idx: int) -> Optional[Dict[str, Any]]:
        """Get full passage content and metadata"""
        df = session_state.get('df')
        if df is None or passage_idx not in df.index:
            return None

        passage_col = session_state.get('passage_col', 'Passage')
        label_columns = session_state.get('label_columns', [])

        passage_text = df.loc[passage_idx, passage_col]
        if pd.isna(passage_text):
            return None

        # Get labels
        labels = {}
        for label in label_columns:
            if label in df.columns:
                val = df.loc[passage_idx, label]
                labels[label] = int(val) if pd.notna(val) else 0

        # Get scores if available
        scores = None
        cache = session_state.get('cache')
        if cache:
            scores_df = cache.get('df_summary')
            if scores_df is not None and passage_idx in scores_df['passage_idx'].values:
                score_row = scores_df[scores_df['passage_idx'] == passage_idx].iloc[0]
                scores = {
                    'consistency': float(score_row['consistency_avg']),
                    'rerank': float(score_row['rerank_avg'])
                }

        return {
            'index': passage_idx,
            'text': str(passage_text),
            'labels': labels,
            'active_labels': [l for l, v in labels.items() if v == 1],
            'scores': scores,
            'length': len(str(passage_text))
        }

    def _search_passages_by_label(self, session_state: Dict[str, Any], label: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for passages with a specific label using semantic search"""
        finder = session_state.get('finder')
        namespace = session_state.get('namespace', 'main')

        if finder is None:
            # Fallback to simple pandas search
            return self._simple_search_by_label(session_state, label, limit)

        try:
            # Use semantic search with reranking
            results = finder.search_by_label_semantic(
                label=label,
                namespace=namespace,
                top_k_vector=50,
                top_k_rerank=limit
            )

            # Convert to passage data format
            passages = []
            for result in results:
                passage_data = self._get_passage_content(session_state, result['passage_idx'])
                if passage_data:
                    passage_data['vector_score'] = result.get('vector_score', 0)
                    passage_data['rerank_score'] = result.get('rerank_score', 0)
                    passage_data['combined_score'] = result.get('combined_score', 0)
                    passages.append(passage_data)

            return passages

        except Exception as e:
            # Fallback on error
            return self._simple_search_by_label(session_state, label, limit)

    def _simple_search_by_label(self, session_state: Dict[str, Any], label: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fallback simple pandas search"""
        df = session_state.get('df')
        label_columns = session_state.get('label_columns', [])

        if df is None or label not in label_columns:
            return []

        mask = df[label] == 1
        matching_indices = df[mask].index.tolist()[:limit]

        results = []
        for idx in matching_indices:
            passage_data = self._get_passage_content(session_state, idx)
            if passage_data:
                results.append(passage_data)

        return results

    def _search_similar_passages(self, session_state: Dict[str, Any], passage_idx: int, k: int = 5) -> List[Dict[str, Any]]:
        """Find similar passages using vector search"""
        finder = session_state.get('finder')
        namespace = session_state.get('namespace', 'main')

        if finder is None:
            return []

        try:
            results = finder.search_similar_to_passage(
                passage_idx=passage_idx,
                namespace=namespace,
                k=k
            )

            passages = []
            for result in results:
                passage_data = self._get_passage_content(session_state, result['passage_idx'])
                if passage_data:
                    passage_data['similarity'] = result['similarity']
                    passages.append(passage_data)

            return passages

        except Exception as e:
            return []

    def _semantic_search(self, session_state: Dict[str, Any], query: str, limit: int = 10, label_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform semantic search with optional label filtering"""
        finder = session_state.get('finder')
        namespace = session_state.get('namespace', 'main')

        if finder is None:
            return []

        try:
            results = finder.search_with_filters(
                query=query,
                namespace=namespace,
                label_filter=label_filter,
                top_k_vector=100,
                top_k_rerank=limit,
                rerank_instruction="Prioritize passages with clear, detailed descriptions relevant to the query"
            )

            passages = []
            for result in results:
                passage_data = self._get_passage_content(session_state, result['passage_idx'])
                if passage_data:
                    passage_data['vector_score'] = result.get('vector_score', 0)
                    passage_data['rerank_score'] = result.get('rerank_score', 0)
                    passage_data['combined_score'] = result.get('combined_score', 0)
                    passages.append(passage_data)

            return passages

        except Exception as e:
            return []

    def _get_label_statistics(self, session_state: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
        """Get detailed statistics for a label"""
        df = session_state.get('df')
        label_columns = session_state.get('label_columns', [])

        if df is None or label not in label_columns:
            return None

        count = int(df[label].sum())
        percentage = (count / len(df)) * 100

        stats = {
            'label': label,
            'count': count,
            'percentage': percentage,
            'total_passages': len(df)
        }

        # Get score statistics if available
        cache = session_state.get('cache')
        if cache:
            scores_df = cache.get('df_summary')
            if scores_df is not None:
                # Find passages with this label
                label_mask = df[label] == 1
                label_indices = df[label_mask].index.tolist()

                # Get scores for these passages
                label_scores = scores_df[scores_df['passage_idx'].isin(label_indices)]

                if len(label_scores) > 0:
                    stats['score_stats'] = {
                        'consistency_mean': float(label_scores['consistency_avg'].mean()),
                        'consistency_median': float(label_scores['consistency_avg'].median()),
                        'rerank_mean': float(label_scores['rerank_avg'].mean()),
                        'rerank_median': float(label_scores['rerank_avg'].median()),
                        'scored_count': len(label_scores)
                    }

        # Co-occurrence with other labels
        co_occurring = {}
        for other_label in label_columns:
            if other_label != label:
                both = ((df[label] == 1) & (df[other_label] == 1)).sum()
                if both > 0:
                    co_occurring[other_label] = int(both)

        stats['co_occurring'] = dict(sorted(co_occurring.items(), key=lambda x: x[1], reverse=True)[:10])

        return stats

    def _run_model_prediction(self, session_state: Dict[str, Any], passage_text: str) -> Optional[Dict[str, Any]]:
        """Run model prediction on a passage"""
        if 'model_loader' not in session_state:
            return None

        loader = session_state['model_loader']
        if not loader.is_loaded():
            return None

        try:
            result = loader.predict_passage(passage_text)
            return result
        except Exception as e:
            return {'error': str(e)}

    def _enhance_message_with_data(self, message: str, session_state: Dict[str, Any]) -> str:
        """Enhance user message with relevant data based on their question"""
        enhanced_parts = [message]

        df = session_state.get('df')
        if df is None:
            return message

        label_columns = session_state.get('label_columns', [])

        # Check if asking about specific passage indices
        if 'passage' in message.lower() and any(str(i) in message for i in range(len(df))):
            # Extract numbers from message
            import re
            numbers = [int(n) for n in re.findall(r'\b\d+\b', message) if int(n) < len(df)]

            if numbers:
                enhanced_parts.append("\n\n# Requested Passage Data:")
                for idx in numbers[:5]:  # Limit to 5 passages
                    passage_data = self._get_passage_content(session_state, idx)
                    if passage_data:
                        enhanced_parts.append(f"\n## Passage {idx}")
                        enhanced_parts.append(f"Text: {passage_data['text'][:500]}..." if len(passage_data['text']) > 500 else f"Text: {passage_data['text']}")
                        enhanced_parts.append(f"Labels: {', '.join(passage_data['active_labels'])}")
                        if passage_data['scores']:
                            enhanced_parts.append(f"Scores: consistency={passage_data['scores']['consistency']:.3f}, rerank={passage_data['scores']['rerank']:.3f}")

        # Check if asking about specific labels
        mentioned_labels = [label for label in label_columns if label.lower() in message.lower()]
        if mentioned_labels:
            enhanced_parts.append("\n\n# Label Statistics:")
            for label in mentioned_labels[:5]:  # Limit to 5 labels
                stats = self._get_label_statistics(session_state, label)
                if stats:
                    enhanced_parts.append(f"\n## {label}")
                    enhanced_parts.append(f"Count: {stats['count']} ({stats['percentage']:.1f}%)")
                    if 'score_stats' in stats:
                        enhanced_parts.append(f"Avg consistency: {stats['score_stats']['consistency_mean']:.3f}")
                        enhanced_parts.append(f"Avg rerank: {stats['score_stats']['rerank_mean']:.3f}")
                    if stats['co_occurring']:
                        top_co = list(stats['co_occurring'].items())[:3]
                        enhanced_parts.append(f"Often with: {', '.join([f'{l} ({c})' for l, c in top_co])}")

                    # Add example passages
                    examples = self._search_passages_by_label(session_state, label, limit=2)
                    if examples:
                        enhanced_parts.append(f"\nExample passages:")
                        for ex in examples:
                            text_preview = ex['text'][:200] + "..." if len(ex['text']) > 200 else ex['text']
                            enhanced_parts.append(f"- [{ex['index']}] {text_preview}")

        return "\n".join(enhanced_parts)

    def send_message(
        self,
        message: str,
        conversation_history: List[Dict[str, str]],
        session_state: Dict[str, Any],
        temperature: float = 1.0,
        max_tokens: int = 16000
    ) -> Dict[str, Any]:
        """Send message to Claude with full data access"""

        # Enhance message with relevant data
        enhanced_message = self._enhance_message_with_data(message, session_state)

        system_blocks = self._build_system_context(session_state)

        messages = []
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": enhanced_message})

        try:
            api_params = {
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system_blocks,
                "messages": messages
            }

            # Only enable thinking at temperature=1.0
            if temperature == 1.0:
                api_params["temperature"] = 1.0
                api_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 10000
                }
            else:
                api_params["temperature"] = temperature

            response = self.client.messages.create(**api_params)

            response_text = ""
            thinking_text = ""

            for block in response.content:
                if block.type == "thinking":
                    thinking_text = block.thinking
                elif block.type == "text":
                    response_text = block.text

            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_creation_input_tokens": getattr(response.usage, 'cache_creation_input_tokens', 0),
                "cache_read_input_tokens": getattr(response.usage, 'cache_read_input_tokens', 0)
            }

            return {
                "response": response_text,
                "thinking": thinking_text,
                "usage": usage,
                "success": True,
                "enhanced_message": enhanced_message != message
            }

        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "thinking": "",
                "usage": {},
                "success": False,
                "error": str(e)
            }


def format_usage_stats(usage: Dict[str, int]) -> str:
    """Format token usage"""
    if not usage:
        return ""

    total_in = usage.get('input_tokens', 0)
    cache_read = usage.get('cache_read_input_tokens', 0)
    out = usage.get('output_tokens', 0)

    parts = [f"{total_in:,} in / {out:,} out"]
    if cache_read > 0:
        parts.append(f"Cache: {cache_read:,}")

    return " | ".join(parts)


def render_chat_page(session_state: Dict[str, Any]):
    """Render chat interface"""
    st.markdown("## 💬 AI Assistant")

    # Check if data loaded
    if not session_state.get('initialized', False):
        st.info("💡 Load a dataset first to enable full assistant capabilities")

    if 'chat_assistant' not in session_state:
        try:
            session_state['chat_assistant'] = HRAFChatAssistant()
            st.success("✅ Claude Sonnet 4.5 with full corpus access")
        except ValueError as e:
            st.error(f"❌ {e}")
            st.info("Set ANTHROPIC_API_KEY in .env")
            return

    if 'chat_history' not in session_state:
        session_state['chat_history'] = []

    # Sidebar
    with st.sidebar:
        st.markdown("### 💬 Chat")

        temperature = st.slider("Temperature", 0.0, 1.0, 1.0, 0.1)

        if temperature == 1.0:
            show_thinking = st.checkbox("Show thinking", value=False)
            st.info("🧠 Thinking enabled")
        else:
            show_thinking = False
            st.warning("⚠️ Set to 1.0 for thinking")

        session_state['show_thinking'] = show_thinking

        if st.button("🗑️ Clear"):
            session_state['chat_history'] = []
            st.rerun()

        if session_state['chat_history']:
            st.markdown("---")
            st.metric("Messages", len(session_state['chat_history']))

    # Suggested questions if no history
    if len(session_state['chat_history']) == 0 and session_state.get('initialized'):
        st.markdown("### 💡 Try asking:")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Show me passages about spirits"):
                st.session_state['suggested_q'] = "Show me some example passages tagged with Spirits_Gods"
                st.rerun()
            if st.button("What labels co-occur most?"):
                st.session_state['suggested_q'] = "Which labels most frequently appear together?"
                st.rerun()

        with col2:
            if st.button("Analyze Rule_Violation_Taboo"):
                st.session_state['suggested_q'] = "Tell me about the Rule_Violation_Taboo label - how many passages, what are they about?"
                st.rerun()
            if st.button("Compare Illness vs Accident"):
                st.session_state['suggested_q'] = "Compare the Illness and Accident labels - differences, examples, scores"
                st.rerun()

        st.markdown("---")

    # Display history
    for msg in session_state['chat_history']:
        if msg['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(msg['content'])
        elif msg['role'] == 'assistant':
            with st.chat_message("assistant"):
                st.markdown(msg['content'])
                if show_thinking and msg.get('thinking'):
                    with st.expander("🧠 Thinking"):
                        st.markdown(msg['thinking'])
                if msg.get('usage'):
                    with st.expander("📊 Usage"):
                        st.caption(format_usage_stats(msg['usage']))

    # Handle suggested question
    if 'suggested_q' in session_state and session_state['suggested_q']:
        user_message = session_state['suggested_q']
        session_state['suggested_q'] = None
    else:
        user_message = st.chat_input("Ask about your data...")

    if user_message:
        session_state['chat_history'].append({
            "role": "user",
            "content": user_message
        })

        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                assistant = session_state['chat_assistant']

                conversation = [
                    {"role": m["role"], "content": m["content"]}
                    for m in session_state['chat_history'][:-1]
                ]

                result = assistant.send_message(
                    message=user_message,
                    conversation_history=conversation,
                    session_state=dict(session_state),
                    temperature=temperature
                )

                if result['success']:
                    st.markdown(result['response'])

                    if show_thinking and result.get('thinking'):
                        with st.expander("🧠 Thinking"):
                            st.markdown(result['thinking'])

                    if result.get('usage'):
                        with st.expander("📊 Usage"):
                            st.caption(format_usage_stats(result['usage']))

                        # Show if data was injected
                        if result.get('enhanced_message'):
                            st.caption("ℹ️ Answer includes actual passage/label data")

                    session_state['chat_history'].append({
                        "role": "assistant",
                        "content": result['response'],
                        "thinking": result.get('thinking', ''),
                        "usage": result.get('usage', {})
                    })
                else:
                    st.error(result['response'])

        st.rerun()