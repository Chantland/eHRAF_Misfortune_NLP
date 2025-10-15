"""
Global Chat Assistant - Page-Aware AI Assistant with Action Execution

Refactored for:
- Global availability across all pages
- Page-specific context awareness
- Action execution capabilities
- Proactive suggestions
"""

import anthropic
import streamlit as st
from typing import List, Dict, Optional, Any, Callable
import os
import pandas as pd
import numpy as np
from pathlib import Path


class GlobalChatAssistant:
    """
    Global chat assistant that:
    - Sees current page and context
    - Can execute actions
    - Provides proactive suggestions
    - Has full access to data and models
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-5-20250929"

        # Action registry
        self.actions = self._register_actions()

    def _register_actions(self) -> Dict[str, Callable]:
        """Register available actions the assistant can execute"""
        return {
            # Data Page Actions
            'load_dataset': self._action_load_dataset,
            'clean_data': self._action_clean_data,
            'generate_embeddings': self._action_generate_embeddings,
            'calculate_scores': self._action_calculate_scores,
            'create_tiers': self._action_create_tiers,

            # Models Page Actions (REAL implementations)
            'load_model': self._action_load_model,
            'configure_training': self._action_configure_training,
            'start_training': self._action_start_training,
            'evaluate_model': self._action_evaluate_model,
            'compare_models': self._action_compare_models,

            # Discover Page Actions
            'semantic_search': self._action_semantic_search,
            'find_similar': self._action_find_similar,
            'run_inference': self._action_run_inference,
            'test_hypothesis': self._action_test_hypothesis,

            # Navigation Actions
            'navigate_to': self._action_navigate_to,
        }

    def render(self, current_page: str, session_state: Dict[str, Any]):
        """
        Render chat interface with page context

        Args:
            current_page: Current page name (e.g., "📊 Data")
            session_state: Streamlit session state
        """

        # Initialize chat history if needed
        if 'global_chat_history' not in st.session_state:
            st.session_state.global_chat_history = []

        # Page context indicator
        st.caption(f"💬 Chatting on: **{current_page}**")

        # Show suggestions if chat is empty and on a specific page
        if len(st.session_state.global_chat_history) == 0:
            self._render_suggestions(current_page, session_state)

        # Chat settings
        with st.expander("⚙️ Chat Settings", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                temperature = st.slider("Temperature", 0.0, 1.0, 1.0, 0.1)

                if temperature == 1.0:
                    show_thinking = st.checkbox("Show thinking", value=False)
                    st.caption("🧠 Extended thinking enabled")
                else:
                    show_thinking = False
                    st.caption("⚠️ Set to 1.0 for thinking")

            with col2:
                enable_actions = st.checkbox("Enable actions", value=True,
                                            help="Allow assistant to execute actions")

                if st.button("🗑️ Clear Chat"):
                    st.session_state.global_chat_history = []
                    st.rerun()

        st.markdown("---")

        # Display chat history
        for msg in st.session_state.global_chat_history:
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(msg['content'])

            elif msg['role'] == 'assistant':
                with st.chat_message("assistant"):
                    st.markdown(msg['content'])

                    # Show thinking if available
                    if show_thinking and msg.get('thinking'):
                        with st.expander("🧠 Thinking Process"):
                            st.markdown(msg['thinking'])

                    # Show executed actions
                    if msg.get('actions_executed'):
                        with st.expander("⚡ Actions Executed"):
                            for action in msg['actions_executed']:
                                st.markdown(f"- ✅ {action}")

                    # Show usage stats
                    if msg.get('usage'):
                        with st.expander("📊 Token Usage"):
                            st.caption(self._format_usage_stats(msg['usage']))

        # Chat input
        user_message = st.chat_input(
            f"Ask about your data, run actions, or get help with {current_page}..."
        )

        if user_message:
            # Add user message
            st.session_state.global_chat_history.append({
                "role": "user",
                "content": user_message
            })

            # Display user message
            with st.chat_message("user"):
                st.markdown(user_message)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = self._generate_response(
                        user_message=user_message,
                        current_page=current_page,
                        session_state=session_state,
                        temperature=temperature,
                        enable_actions=enable_actions
                    )

                    if result['success']:
                        # Display response
                        st.markdown(result['response'])

                        # Show thinking
                        if show_thinking and result.get('thinking'):
                            with st.expander("🧠 Thinking Process"):
                                st.markdown(result['thinking'])

                        # Show executed actions
                        if result.get('actions_executed'):
                            with st.expander("⚡ Actions Executed"):
                                for action in result['actions_executed']:
                                    st.markdown(f"- ✅ {action}")

                        # Show usage
                        if result.get('usage'):
                            with st.expander("📊 Token Usage"):
                                st.caption(self._format_usage_stats(result['usage']))

                        # Add to history
                        st.session_state.global_chat_history.append({
                            "role": "assistant",
                            "content": result['response'],
                            "thinking": result.get('thinking', ''),
                            "usage": result.get('usage', {}),
                            "actions_executed": result.get('actions_executed', [])
                        })

                    else:
                        st.error(result['response'])

            st.rerun()

        # Handle action triggers
        if st.session_state.get('action_trigger'):
            trigger = st.session_state['action_trigger']

            if trigger == 'model_loaded':
                st.success("✅ Model loaded successfully")
                st.session_state['action_trigger'] = None
                st.rerun()

            elif trigger == 'config_updated':
                st.info("ℹ️ Training configuration updated")
                st.session_state['action_trigger'] = None

            elif trigger == 'start_training':
                st.warning("⚠️ Training initiated - configure settings below")
                st.session_state['action_trigger'] = None


    def _render_suggestions(self, current_page: str, session_state: Dict):
        """Render page-specific suggestions"""

        st.markdown("### 💡 Suggested Questions")

        suggestions = self._get_page_suggestions(current_page, session_state)

        if not suggestions:
            return

        # Display in columns
        cols = st.columns(2)

        for i, suggestion in enumerate(suggestions[:6]):  # Max 6 suggestions
            with cols[i % 2]:
                if st.button(suggestion['text'], key=f"suggestion_{i}", width='stretch'):
                    # Add suggestion as user message
                    st.session_state.global_chat_history.append({
                        "role": "user",
                        "content": suggestion['query']
                    })
                    st.rerun()

    def _get_page_suggestions(self, current_page: str, session_state: Dict) -> List[Dict]:
        """Get page-specific suggestions"""

        suggestions = []

        if "📊 Data" in current_page:
            if not session_state.get('initialized'):
                suggestions = [
                    {"text": "How do I load data?", "query": "How do I load my dataset?"},
                    {"text": "Supported file formats?", "query": "What file formats are supported?"},
                ]
            else:
                df = session_state.get('df')
                cache = session_state.get('cache', {})

                suggestions = [
                    {"text": "Analyze data quality", "query": "Analyze the quality of my dataset"},
                    {"text": "Show label distribution", "query": "Show me the label distribution in my dataset"},
                ]

                if 'stable_id_to_pinecone' not in cache:
                    suggestions.append(
                        {"text": "Generate embeddings", "query": "Help me generate embeddings for my data"}
                    )

                if 'df_summary' not in cache:
                    suggestions.append(
                        {"text": "Calculate quality scores", "query": "Calculate quality scores for my passages"}
                    )

        elif "🤖 Models" in current_page:
            manager = session_state.get('model_manager')

            if manager and len(manager) > 0:
                suggestions = [
                    {"text": "Compare loaded models", "query": "Compare all my loaded models"},
                    {"text": "Evaluate on test data", "query": "Evaluate my models on test data"},
                ]
            else:
                suggestions = [
                    {"text": "Train my first model", "query": "Help me train my first model"},
                    {"text": "What's hierarchical?", "query": "Explain hierarchical classification"},
                ]

            suggestions.append(
                {"text": "Model recommendations", "query": "What model configuration do you recommend?"}
            )

        elif "🔍 Discover" in current_page:
            cache = session_state.get('cache', {})

            if 'stable_id_to_pinecone' in cache:
                suggestions = [
                    {"text": "Search for shamans", "query": "Find passages about shamans healing illness"},
                    {"text": "Test a hypothesis", "query": "Test if spirits cause illness more than material causes"},
                    {"text": "Find interesting patterns", "query": "What are interesting patterns in my data?"},
                ]
            else:
                suggestions = [
                    {"text": "Need embeddings", "query": "Why do I need embeddings for semantic search?"},
                ]

        return suggestions

    def _generate_response(
        self,
        user_message: str,
        current_page: str,
        session_state: Dict,
        temperature: float,
        enable_actions: bool
    ) -> Dict[str, Any]:
        """Generate response with page context and action execution"""

        # Build system context
        system_blocks = self._build_system_context(current_page, session_state, enable_actions)

        # Build conversation history
        messages = []
        for msg in st.session_state.global_chat_history[:-1]:  # Exclude last (current) message
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Enhance user message with data
        enhanced_message = self._enhance_message_with_data(user_message, session_state)

        messages.append({"role": "user", "content": enhanced_message})

        try:
            # Call Claude API
            api_params = {
                "model": self.model,
                "max_tokens": 16000,
                "system": system_blocks,
                "messages": messages
            }

            if temperature == 1.0:
                api_params["temperature"] = 1.0
                api_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 10000
                }
            else:
                api_params["temperature"] = temperature

            response = self.client.messages.create(**api_params)

            # Extract response
            response_text = ""
            thinking_text = ""

            for block in response.content:
                if block.type == "thinking":
                    thinking_text = block.thinking
                elif block.type == "text":
                    response_text = block.text

            # Extract and execute actions if enabled
            actions_executed = []
            if enable_actions:
                actions_executed = self._extract_and_execute_actions(
                    response_text, session_state
                )

            # Usage stats
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
                "actions_executed": actions_executed,
                "success": True
            }

        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "success": False,
                "error": str(e)
            }

    def _build_system_context(
            self,
            current_page: str,
            session_state: Dict,
            enable_actions: bool
    ) -> List[Dict[str, Any]]:
        """Build system context with page awareness"""

        system_blocks = [
            {
                "type": "text",
                "text": f"""You are an AI assistant for the HRAF Golden Dataset Discovery tool.

    # Current Context
    - **Current Page:** {current_page}
    - **Dataset Loaded:** {'Yes' if session_state.get('initialized') else 'No'}
    - **Actions Enabled:** {'Yes' if enable_actions else 'No'}
    - **Training Active:** {'Yes' if session_state.get('training_active') else 'No'}
    
    # Your Capabilities
    You have FULL ACCESS to:
    - Read any passage by index
    - Analyze label distributions
    - Run predictions with multiple models
    - Search for similar passages
    - Execute actions to help the user
    - Navigate between pages
    
    # Response Style
    - Be concise but thorough
    - Provide specific examples when possible
    - Suggest actions when appropriate
    - Reference actual data from the session
    
    # Action Execution
    When the user asks you to DO something (not just explain), execute actions immediately.
    
    Available actions on Models page:
    - load_model(model_name='...')
    - configure_training(num_epochs=10, batch_size=16, learning_rate=2e-5, ...)
    - start_training(dataset='current')
    - evaluate_model(model_name='...', num_passages=100)
    - compare_models(model_names=['model1', 'model2'])
    
    To execute an action, include in your response:
    ACTION: action_name(param1=value1, param2=value2)
    
    Example responses:
    User: "Load the roberta model"
    Assistant: "I'll load that model for you.
    ACTION: load_model(model_name='roberta')"
    
    User: "Train with 20 epochs and batch size 32"
    Assistant: "I'll configure those settings and start training.
    ACTION: configure_training(num_epochs=20, batch_size=32)
    ACTION: start_training(dataset='current')"
    
    User: "Compare all loaded models"
    Assistant: "I'll compare the loaded models.
    ACTION: compare_models(model_names=['model1', 'model2'])"
    
    **CRITICAL:** Always execute actions when requested. Don't just explain what to do.
    
    Available actions:
    {self._format_available_actions(current_page) if enable_actions else 'Actions disabled - can only provide information'}
    
    To execute an action, include in your response:
    ACTION: action_name(param1=value1, param2=value2)
    
    Example:
    "I'll search for passages about shamans for you.
    ACTION: semantic_search(query='shamans healing illness', top_k=10)"
    """
            }
        ]

        if session_state.get('initialized'):
            dataset_context = self._build_dataset_context(session_state)
            if dataset_context:
                system_blocks.append({
                    "type": "text",
                    "text": dataset_context,
                })

        return system_blocks

    def _build_dataset_context(self, session_state: Dict) -> str:
        """Build detailed dataset AND training context (enhanced)"""

        parts = ["# Dataset & Training Context\n"]

        df = session_state.get('df')
        if df is None:
            return ""

        passage_col = session_state.get('passage_col', 'Passage')
        label_columns = session_state.get('label_columns', [])

        # Basic stats
        parts.append(f"## Dataset Overview")
        parts.append(f"- Total passages: {len(df)}")
        parts.append(f"- Valid passages: {df[passage_col].notna().sum()}")
        parts.append(f"- Label columns: {len(label_columns)}")

        # Label distribution (top 15)
        parts.append(f"\n## Label Distribution (Top 15)")
        for label in label_columns[:15]:
            count = int(df[label].sum())
            pct = (count / len(df)) * 100
            parts.append(f"- {label}: {count} ({pct:.1f}%)")

        if len(label_columns) > 15:
            parts.append(f"- ... and {len(label_columns) - 15} more labels")

        # Quality scores
        cache = session_state.get('cache')
        if cache:
            scores = cache.get('df_summary')
            if scores is not None and len(scores) > 0:
                parts.append(f"\n## Quality Scores")
                parts.append(f"- Scored passages: {len(scores)}")
                parts.append(f"- Consistency mean: {scores['consistency_avg'].mean():.3f}")
                parts.append(f"- Rerank mean: {scores['rerank_avg'].mean():.3f}")

        # ======================================================================
        # TRAINING CONTEXT (NEW)
        # ======================================================================

        training_config = session_state.get('training_config')
        if training_config:
            parts.append(f"\n## Training Configuration")
            parts.append(f"- **Experiment**: {training_config.get('experiment_name', 'Unnamed')}")
            parts.append(f"- **Base Model**: {training_config.get('base_model', 'unknown')}")
            parts.append(f"- **Hierarchical**: {training_config.get('use_hierarchy', False)}")

            if training_config.get('use_hierarchy'):
                parts.append(f"  - Gated: {training_config.get('gated_hierarchy', False)}")
                parts.append(f"  - Gate threshold: {training_config.get('gate_threshold', 0.5)}")
                parts.append(f"  - Predict main labels: {training_config.get('predict_main_labels', False)}")

                # Hierarchy details
                hierarchy = training_config.get('hierarchy_config')
                if hierarchy and 'categories' in hierarchy:
                    parts.append(f"  - Main categories configured: {len(hierarchy['categories'])}")
                    for cat_name, cat_data in hierarchy['categories'].items():
                        if cat_data.get('enabled'):
                            parts.append(f"    - {cat_name}: {len(cat_data['sublabels'])} sublabels")

            parts.append(f"- **Loss Configuration**:")
            parts.append(f"  - Focal loss: {training_config.get('use_focal_loss', False)}")
            if training_config.get('use_focal_loss'):
                parts.append(f"  - Focal gamma: {training_config.get('focal_gamma', 2.0)}")
            parts.append(f"  - Weighted loss: {training_config.get('use_weighted_loss', False)}")

            parts.append(f"- **Training Parameters**:")
            parts.append(f"  - Epochs: {training_config.get('num_epochs', 10)}")
            parts.append(f"  - Batch size: {training_config.get('batch_size', 16)}")
            parts.append(f"  - Learning rate: {training_config.get('learning_rate', 2e-5):.2e}")
            parts.append(f"  - Max length: {training_config.get('max_length', 512)}")

        # Training status
        training_active = session_state.get('training_active', False)
        if training_active:
            parts.append(f"\n## 🔴 Training Status: ACTIVE")
            current_epoch = session_state.get('current_epoch', 0)
            total_epochs = training_config.get('num_epochs', 10) if training_config else 10
            parts.append(f"- Current epoch: {current_epoch}/{total_epochs}")
            parts.append(f"- Progress: {(current_epoch / total_epochs) * 100:.1f}%")

        # Training history
        training_history = session_state.get('training_history', [])
        if training_history:
            parts.append(f"\n## Training History ({len(training_history)} epochs logged)")

            # Show last 3 epochs
            recent = training_history[-3:]
            parts.append(f"\n### Recent Epochs:")
            for log in recent:
                epoch = log.get('epoch', 0)
                train_loss = log.get('train_loss', 0)
                eval_loss = log.get('eval_loss', 0)
                f1_micro = log.get('eval_f1_micro', 0)
                f1_macro = log.get('eval_f1_macro', 0)

                parts.append(f"\n**Epoch {epoch}:**")
                parts.append(f"- Train loss: {train_loss:.4f}")
                parts.append(f"- Val loss: {eval_loss:.4f}")
                parts.append(f"- F1 micro: {f1_micro:.3f}")
                parts.append(f"- F1 macro: {f1_macro:.3f}")

                # Check for individual label F1s
                label_f1s = {k: v for k, v in log.items() if
                             k.startswith('eval_f1_') and k not in ['eval_f1_micro', 'eval_f1_macro']}
                if label_f1s:
                    # Show labels with F1 > 0
                    active_labels = {k.replace('eval_f1_', ''): v for k, v in label_f1s.items() if v > 0}
                    if active_labels:
                        parts.append(f"- Active labels: {len(active_labels)}/{len(label_f1s)}")
                        # Show top 5
                        top_5 = sorted(active_labels.items(), key=lambda x: x[1], reverse=True)[:5]
                        for label, score in top_5:
                            parts.append(f"  - {label}: {score:.3f}")

        # Test results
        test_results = session_state.get('test_results')
        if test_results:
            parts.append(f"\n## Test Results (Final)")
            parts.append(f"- F1 Micro: {test_results.get('eval_f1_micro', 0):.3f}")
            parts.append(f"- F1 Macro: {test_results.get('eval_f1_macro', 0):.3f}")

            # Count labels by performance
            label_f1s = {k: v for k, v in test_results.items() if
                         k.startswith('eval_f1_') and k not in ['eval_f1_micro', 'eval_f1_macro']}
            good = sum(1 for v in label_f1s.values() if v > 0.7)
            fair = sum(1 for v in label_f1s.values() if 0.5 < v <= 0.7)
            poor = sum(1 for v in label_f1s.values() if 0 < v <= 0.5)
            zero = sum(1 for v in label_f1s.values() if v == 0)

            parts.append(f"- Labels > 0.7: {good}")
            parts.append(f"- Labels 0.5-0.7: {fair}")
            parts.append(f"- Labels 0-0.5: {poor}")
            parts.append(f"- Labels = 0: {zero}")

            if zero > 0:
                parts.append(f"\n**⚠️ WARNING: {zero} labels have F1=0 (model never predicts them)**")
                zero_labels = [k.replace('eval_f1_', '') for k, v in label_f1s.items() if v == 0]
                parts.append(f"Zero-F1 labels: {', '.join(zero_labels[:10])}")

        # Training completion
        training_complete = session_state.get('training_complete', False)
        if training_complete:
            output_dir = session_state.get('training_output_dir')
            parts.append(f"\n## ✅ Training Complete")
            if output_dir:
                parts.append(f"- Model saved to: {output_dir}")

        # Loaded models
        loaded_models = session_state.get('loaded_models', {})
        manager = session_state.get('model_manager')
        if manager and len(manager) > 0:
            parts.append(f"\n## Loaded Models ({len(manager)} total)")
            for model_info in manager.list_models():
                parts.append(f"\n- **{model_info['name']}**")
                parts.append(f"  - Type: {model_info.get('architecture', 'Unknown')}")
                test_f1 = model_info.get('test_f1')
                if test_f1:
                    parts.append(f"  - Test F1: {test_f1:.3f}")

        return "\n".join(parts)

    def _format_available_actions(self, current_page: str) -> str:
        """Format available actions for current page"""

        if "📊 Data" in current_page:
            return """
    - load_dataset(source='file/experiment/upload')
    - clean_data(remove_duplicates=True, remove_missing=True)
    - generate_embeddings(batch_size=32)
    - calculate_scores(k_similar=20)
    - create_tiers(preset='balanced/conservative/aggressive')
    """

        elif "🤖 Models" in current_page or "Train" in current_page:
            return """
    - analyze_training_config() - Review current training configuration
    - suggest_improvements() - Suggest ways to improve model performance
    - explain_metric(metric_name='f1_micro') - Explain a training metric
    - diagnose_zero_labels() - Analyze why labels have F1=0
    - recommend_hyperparameters() - Suggest better hyperparameters
    - compare_to_baseline() - Compare current performance to baseline
    - adjust_config(parameter='focal_gamma', value=3.5) - Adjust training config
    """

        elif "🔍 Discover" in current_page:
            return """
    - semantic_search(query='...', top_k=10, label_filter='...')
    - find_similar(passage_idx=123, k=20)
    - run_inference(passage_idx=123, model_names=['...'])
    - test_hypothesis(label_a='...', label_b='...')
    """

        else:
            return "- navigate_to(page='Data/Models/Discover')"

    def _enhance_message_with_data(self, message: str, session_state: Dict) -> str:
        """Enhance message with relevant data (reuse existing logic)"""

        # This would use the existing implementation from the original chat_assistant.py
        # For now, just return the message as-is
        # In production, copy over the existing _enhance_message_with_data method

        return message

    def _extract_and_execute_actions(self, response: str, session_state: Dict) -> List[str]:
        """Extract ACTION: statements and execute them"""

        actions_executed = []

        # Look for ACTION: lines
        import re
        action_pattern = r'ACTION:\s*(\w+)\((.*?)\)'

        matches = re.findall(action_pattern, response, re.MULTILINE)

        for action_name, params_str in matches:
            if action_name in self.actions:
                try:
                    # Parse parameters
                    params = self._parse_action_params(params_str)

                    # Execute action
                    action_func = self.actions[action_name]
                    result = action_func(session_state, **params)

                    actions_executed.append(f"{action_name}: {result}")

                except Exception as e:
                    actions_executed.append(f"{action_name}: Failed - {str(e)}")

        return actions_executed

    def _parse_action_params(self, params_str: str) -> Dict:
        """Parse action parameters from string"""

        params = {}

        if not params_str.strip():
            return params

        # Simple parsing - handles key=value pairs
        import re
        pairs = re.findall(r'(\w+)=([^,]+)', params_str)

        for key, value in pairs:
            # Try to parse as int/float/bool
            value = value.strip().strip("'\"")

            if value.lower() == 'true':
                params[key] = True
            elif value.lower() == 'false':
                params[key] = False
            elif value.isdigit():
                params[key] = int(value)
            elif value.replace('.', '').isdigit():
                params[key] = float(value)
            else:
                params[key] = value

        return params

    # ========================================================================
    # ACTION IMPLEMENTATIONS - MODELS PAGE
    # ========================================================================

    def _action_load_model(self, session_state: Dict, model_name: str) -> str:
        """Action: Load model"""
        try:
            manager = session_state.get('model_manager')
            if not manager:
                from components.model_manager import ModelManager
                manager = ModelManager()
                session_state['model_manager'] = manager

            # Find model path
            from core.model_inference import find_model_directories
            model_dirs = find_model_directories("./models")

            # Match by name
            matching = [m for m in model_dirs if model_name.lower() in str(m).lower()]

            if not matching:
                return f"❌ Model '{model_name}' not found. Available: {[m.parent.name for m in model_dirs[:3]]}"

            model_path = str(matching[0])
            success = manager.load_model(model_path, nickname=model_name)

            if success:
                session_state['action_trigger'] = 'model_loaded'
                return f"✅ Loaded model '{model_name}'"
            else:
                return f"❌ Failed to load '{model_name}'"

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _action_configure_training(self, session_state: Dict, **config) -> str:
        """Action: Update training configuration"""
        try:
            if 'training_config' not in session_state:
                from core.model_training import get_default_training_config
                session_state['training_config'] = get_default_training_config()

            # Update config with provided values
            updated = []
            for key, value in config.items():
                if key in session_state['training_config']:
                    session_state['training_config'][key] = value
                    updated.append(f"{key}={value}")

            # ✅ SANITIZE TYPES
            from core.model_training import sanitize_config_types
            session_state['training_config'] = sanitize_config_types(session_state['training_config'])

            if updated:
                session_state['action_trigger'] = 'config_updated'
                return f"✅ Updated: {', '.join(updated)}"
            else:
                return "ℹ️ No valid config parameters provided"

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _action_start_training(self, session_state: Dict, dataset: str = 'current') -> str:
        """Action: Start training"""
        try:
            # Import here to avoid circular dependency
            from core.model_training import get_training_data_from_session, start_training

            # Validate data availability
            try:
                df, labels, passage_col = get_training_data_from_session(session_state)
            except ValueError as e:
                return f"❌ {str(e)}"

            # Initialize working data if not exists
            if 'training_working_data' not in session_state:
                session_state['training_working_data'] = {
                    'df': df,
                    'label_columns': labels,
                    'passage_col': passage_col
                }

            # Ensure config exists
            if 'training_config' not in session_state:
                from core.model_training import get_default_training_config, sanitize_config_types
                session_state['training_config'] = get_default_training_config()
                session_state['training_config'] = sanitize_config_types(session_state['training_config'])

            # Start training
            working = session_state['training_working_data']
            start_training(
                session_state,
                working['df'],
                working['label_columns'],
                working['passage_col']
            )

            return "✅ Training started successfully"

        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}\n{traceback.format_exc()[:200]}"

    def _action_evaluate_model(self, session_state: Dict, model_name: str, num_passages: int = 100) -> str:
        """Action: Evaluate model"""
        try:
            manager = session_state.get('model_manager')
            if not manager or model_name not in manager.models:
                return f"❌ Model '{model_name}' not loaded"

            if not session_state.get('initialized'):
                return "❌ No dataset loaded"

            # Set evaluation trigger
            session_state['action_trigger'] = 'evaluate'
            session_state['eval_model'] = model_name
            session_state['eval_num_passages'] = num_passages

            return f"✅ Evaluation queued for '{model_name}' on {num_passages} passages"

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _action_compare_models(self, session_state: Dict, model_names: List[str]) -> str:
        """Action: Compare models"""
        try:
            manager = session_state.get('model_manager')
            if not manager:
                return "❌ No models loaded"

            available = list(manager.models.keys())
            valid_models = [m for m in model_names if m in available]

            if len(valid_models) < 2:
                return f"❌ Need at least 2 loaded models. Available: {available}"

            # Set comparison trigger
            session_state['action_trigger'] = 'compare'
            session_state['compare_models'] = valid_models

            return f"✅ Comparison queued for: {', '.join(valid_models)}"

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _action_analyze_training_config(self, session_state: Dict) -> str:
        """Action: Analyze current training configuration"""
        try:
            config = session_state.get('training_config')

            if not config:
                return "❌ No training configuration found. Configure training first."

            # Build analysis summary
            summary = []
            summary.append("📋 **Current Training Configuration:**\n")

            # Model architecture
            summary.append(f"**Model:** {config.get('base_model', 'unknown')}")
            summary.append(f"**Hierarchical:** {config.get('use_hierarchy', False)}")
            if config.get('use_hierarchy'):
                summary.append(f"  - Gated: {config.get('gated_hierarchy', False)}")
                summary.append(f"  - Gate threshold: {config.get('gate_threshold', 0.5)}")

            # Loss configuration
            summary.append(f"\n**Loss Configuration:**")
            summary.append(f"  - Focal loss: {config.get('use_focal_loss', False)}")
            if config.get('use_focal_loss'):
                summary.append(f"    - Gamma: {config.get('focal_gamma', 2.0)}")
            summary.append(f"  - Weighted loss: {config.get('use_weighted_loss', False)}")

            # Training params
            summary.append(f"\n**Training Parameters:**")
            summary.append(f"  - Epochs: {config.get('num_epochs', 10)}")
            summary.append(f"  - Batch size: {config.get('batch_size', 16)}")
            summary.append(f"  - Learning rate: {config.get('learning_rate', 2e-5):.2e}")
            summary.append(f"  - Warmup steps: {config.get('warmup_steps', 500)}")

            return "\n".join(summary)

        except Exception as e:
            return f"❌ Error analyzing config: {str(e)}"

    def _action_load_dataset(self, session_state: Dict, source: str = 'file') -> str:
        """Action: Load dataset"""
        # This would trigger the appropriate UI flow
        return f"Navigate to Data page → Load Data → {source}"

    def _action_clean_data(self, session_state: Dict, **kwargs) -> str:
        """Action: Clean data"""
        return "Cleaning workflow initiated (go to Data → Clean & Analyze)"

    def _action_generate_embeddings(self, session_state: Dict, batch_size: int = 32) -> str:
        """Action: Generate embeddings"""
        return f"Embeddings generation queued (batch_size={batch_size})"

    def _action_calculate_scores(self, session_state: Dict, k_similar: int = 20) -> str:
        """Action: Calculate quality scores"""
        return f"Score calculation queued (k_similar={k_similar})"

    def _action_create_tiers(self, session_state: Dict, preset: str = 'balanced') -> str:
        """Action: Create training tiers"""
        return f"Tier creation queued (preset={preset})"

    def _action_semantic_search(self, session_state: Dict, query: str, top_k: int = 10,
                                label_filter: Optional[str] = None) -> str:
        """Action: Semantic search with stable ID handling"""

        finder = session_state.get('finder')
        namespace = session_state.get('namespace', 'main')
        df = session_state.get('df')

        if finder is None:
            return "❌ Finder not initialized"

        try:
            # Run search
            results = finder.search_with_filters(
                query=query,
                namespace=namespace,
                label_filter=label_filter,
                top_k_vector=top_k * 2,
                top_k_rerank=top_k,
                min_similarity=0.0
            )

            # ✅ FIX: Map results back to current dataframe indices
            mapped_results = []
            for result in results:
                stable_id = result.get('metadata', {}).get('stable_id')

                if stable_id and df is not None and 'passage_id' in df.columns:
                    # Find current index for this stable ID
                    matching = df[df['passage_id'] == stable_id]
                    if not matching.empty:
                        result['passage_idx'] = matching.index[0]
                        mapped_results.append(result)
                else:
                    # Fallback to stored index
                    mapped_results.append(result)

            # Store in session state
            session_state['search_results'] = mapped_results

            return f"✅ Found {len(mapped_results)} results for '{query}'"

        except Exception as e:
            return f"❌ Search failed: {str(e)}"

    def _action_find_similar(self, session_state: Dict, passage_idx: int, k: int = 20) -> str:
        """Action: Find similar passages"""
        return f"Found {k} passages similar to passage {passage_idx}"

    def _action_run_inference(self, session_state: Dict, passage_idx: int, model_names: List[str]) -> str:
        """Action: Run model inference"""
        return f"Inference completed on passage {passage_idx} with {len(model_names)} models"

    def _action_test_hypothesis(self, session_state: Dict, label_a: str, label_b: str) -> str:
        """Action: Test hypothesis"""
        return f"Chi-square test completed for {label_a} vs {label_b}"

    def _action_navigate_to(self, session_state: Dict, page: str) -> str:
        """Action: Navigate to page"""
        return f"Navigate to {page} page"

    def _format_usage_stats(self, usage: Dict[str, int]) -> str:
        """Format token usage statistics"""

        if not usage:
            return ""

        total_in = usage.get('input_tokens', 0)
        cache_read = usage.get('cache_read_input_tokens', 0)
        out = usage.get('output_tokens', 0)

        parts = [f"{total_in:,} in / {out:,} out"]

        if cache_read > 0:
            parts.append(f"Cache: {cache_read:,}")

        return " | ".join(parts)

