# GOLDEN_DATASET/page_views/analysis_page.py
"""
Analysis Page - Interactive data exploration and diagnostics
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def render():
    """Main render function for Analysis page"""

    st.markdown("# 🔬 Data Analysis")
    st.caption("Explore data quality, label distributions, and model diagnostics")

    # Check if data loaded
    if not st.session_state.get('initialized'):
        st.warning("⚠️ Load a dataset first (Data page)")
        return

    # Create tabs
    tabs = st.tabs([
        "📊 Quick Stats",
        "🔍 Label Diagnostics",
        "💻 Code Playground"
    ])

    with tabs[0]:
        render_quick_stats()

    with tabs[1]:
        render_label_diagnostics()

    with tabs[2]:
        render_code_playground()


def render_quick_stats():
    """Dataset overview and basic statistics"""

    df = st.session_state.df
    label_columns = st.session_state.label_columns
    passage_col = st.session_state.passage_col

    st.markdown("### 📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Passages", f"{len(df):,}")

    with col2:
        st.metric("Total Labels", len(label_columns))

    with col3:
        avg_labels = df[label_columns].sum(axis=1).mean()
        st.metric("Avg Labels/Passage", f"{avg_labels:.2f}")

    with col4:
        total_annotations = df[label_columns].sum().sum()
        st.metric("Total Annotations", f"{int(total_annotations):,}")

    st.markdown("---")
    st.markdown("### 🏷️ Label Distribution")

    # Label frequency table
    label_stats = []
    for label in label_columns:
        count = int(df[label].sum())
        pct = (count / len(df)) * 100
        label_stats.append({
            'Label': label,
            'Count': count,
            'Percentage': f"{pct:.1f}%",
            'Rarity': '🔴 Rare' if pct < 5 else '🟡 Uncommon' if pct < 15 else '🟢 Common'
        })

    stats_df = pd.DataFrame(label_stats).sort_values('Count', ascending=False)
    st.dataframe(stats_df, hide_index=True, width='stretch')

    # Passage length distribution
    st.markdown("---")
    st.markdown("### 📏 Passage Length Distribution")

    import matplotlib.pyplot as plt

    if passage_col in df.columns:
        lengths = df[passage_col].astype(str).str.len()

        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(lengths, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
            ax.axvline(lengths.median(), color='red', linestyle='--', label=f'Median: {lengths.median():.0f}')
            ax.set_xlabel('Passage Length (characters)')
            ax.set_ylabel('Frequency')
            ax.set_title('Passage Length Distribution')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with col2:
            st.metric("Min Length", f"{lengths.min():.0f}")
            st.metric("Median Length", f"{lengths.median():.0f}")
            st.metric("Max Length", f"{lengths.max():.0f}")

            # Tokenization estimate (rough)
            est_tokens = lengths.median() / 4
            st.caption(f"Est. tokens: ~{est_tokens:.0f}")
            if est_tokens > 512:
                st.warning("⚠️ Some passages may exceed 512 token limit")


def render_label_diagnostics():
    """Deep dive into specific label quality"""

    df = st.session_state.df
    label_columns = st.session_state.label_columns
    passage_col = st.session_state.passage_col

    st.markdown("### 🔍 Label Quality Diagnostics")

    # Select label to analyze
    selected_label = st.selectbox(
        "Select label to analyze:",
        label_columns,
        key="diag_label_select"
    )

    if st.button("🔬 Run Diagnostics", type="primary"):
        with st.spinner(f"Analyzing {selected_label}..."):

            # Basic stats
            positive_count = int(df[selected_label].sum())
            negative_count = len(df) - positive_count
            prevalence = (positive_count / len(df)) * 100

            st.markdown("#### 📊 Basic Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Positive Examples", positive_count)

            with col2:
                st.metric("Negative Examples", negative_count)

            with col3:
                st.metric("Prevalence", f"{prevalence:.1f}%")
                if prevalence < 5:
                    st.error("⚠️ RARE LABEL")
                elif prevalence < 10:
                    st.warning("⚠️ Uncommon")

            st.markdown("---")

            # Co-occurrence analysis
            st.markdown("#### 🔗 Co-occurrence with Other Labels")

            positive_df = df[df[selected_label] == 1]

            cooccur = []
            for other_label in label_columns:
                if other_label == selected_label:
                    continue

                overlap = int((positive_df[other_label] == 1).sum())
                overlap_pct = (overlap / positive_count * 100) if positive_count > 0 else 0

                if overlap_pct > 10:  # Only show significant co-occurrences
                    cooccur.append({
                        'Label': other_label,
                        'Co-occurs': overlap,
                        'Percentage': f"{overlap_pct:.1f}%"
                    })

            if cooccur:
                cooccur_df = pd.DataFrame(cooccur).sort_values('Co-occurs', ascending=False)
                st.dataframe(cooccur_df, hide_index=True, width='stretch')

                # Check for high co-occurrence (potential confusion)
                high_cooccur = [c for c in cooccur if float(c['Percentage'].rstrip('%')) > 80]
                if high_cooccur:
                    st.warning(
                        f"⚠️ High co-occurrence detected! {selected_label} appears with {high_cooccur[0]['Label']} in {high_cooccur[0]['Percentage']} of cases. This may cause label confusion.")
            else:
                st.info("No significant co-occurrences found")

            st.markdown("---")

            # Sample passages
            st.markdown("#### 📝 Sample Passages")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Positive Examples ({selected_label}=1)**")
                positive_samples = positive_df.sample(min(3, positive_count))
                for idx, row in positive_samples.iterrows():
                    with st.expander(f"Example {idx}"):
                        st.text(str(row[passage_col])[:500])
                        # Show other active labels
                        other_active = [l for l in label_columns if l != selected_label and row[l] == 1]
                        if other_active:
                            st.caption(f"Also: {', '.join(other_active)}")

            with col2:
                st.markdown(f"**Negative Examples ({selected_label}=0)**")
                negative_df = df[df[selected_label] == 0]
                negative_samples = negative_df.sample(min(3, negative_count))
                for idx, row in negative_samples.iterrows():
                    with st.expander(f"Example {idx}"):
                        st.text(str(row[passage_col])[:500])
                        active = [l for l in label_columns if row[l] == 1]
                        if active:
                            st.caption(f"Labels: {', '.join(active)}")

            st.markdown("---")

            # Class imbalance severity
            st.markdown("#### ⚖️ Imbalance Assessment")

            imbalance_ratio = negative_count / positive_count if positive_count > 0 else float('inf')

            st.metric("Imbalance Ratio", f"{imbalance_ratio:.1f}:1")

            if imbalance_ratio > 50:
                st.error(f"🔴 **SEVERE IMBALANCE** - Model will struggle without aggressive weighting")
                st.info("Recommendations: focal_gamma ≥ 5.0, weighted loss, consider oversampling")
            elif imbalance_ratio > 20:
                st.warning(f"🟡 **HIGH IMBALANCE** - Use focal loss and weighted loss")
                st.info("Recommendations: focal_gamma = 4.0-5.0, weighted loss enabled")
            elif imbalance_ratio > 5:
                st.success(f"🟢 **MODERATE IMBALANCE** - Standard techniques sufficient")
                st.info("Recommendations: focal_gamma = 2.5-3.5, weighted loss optional")
            else:
                st.success(f"✅ **BALANCED** - No special handling needed")


def render_code_playground():
    """Interactive code execution for custom analysis"""

    st.markdown("### 💻 Code Playground")
    st.caption("Execute custom Python code with access to your data")

    df = st.session_state.df
    label_columns = st.session_state.label_columns
    passage_col = st.session_state.passage_col

    # Pre-built analysis templates
    st.markdown("#### 📋 Analysis Templates")

    templates = {
        "Check Material_Physical Quality": """# Material_Physical Diagnostic
mp_count = df['Material_Physical'].sum()
mp_pct = (mp_count / len(df)) * 100

print(f"Material_Physical Analysis:")
print(f"{'='*50}")
print(f"Positive examples: {mp_count} ({mp_pct:.1f}%)")
print()

# Co-occurrence with other CAUSE labels
cause_labels = ['Material_Physical', 'Spirits_Gods', 'Witchcraft_Sorcery', 'Rule_Violation_Taboo']
mp_positive = df[df['Material_Physical'] == 1]

print("Co-occurrence with other CAUSE labels:")
for label in cause_labels:
    if label == 'Material_Physical' or label not in df.columns:
        continue
    overlap = (mp_positive[label] == 1).sum()
    overlap_pct = (overlap / mp_count * 100) if mp_count > 0 else 0
    print(f"  {label}: {overlap} ({overlap_pct:.1f}%)")
""",

        "Find Similar Passages": """# Find passages similar to a target
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Select target passage
target_idx = 0
target_text = df[passage_col].iloc[target_idx]

# Compute similarity
vectorizer = TfidfVectorizer(max_features=100)
tfidf = vectorizer.fit_transform(df[passage_col].astype(str))
similarities = cosine_similarity(tfidf[target_idx:target_idx+1], tfidf)[0]

# Top 5 most similar
top_5_idx = np.argsort(similarities)[-6:-1][::-1]  # Exclude self

print("Most similar passages:")
for idx in top_5_idx:
    print(f"\\nSimilarity: {similarities[idx]:.3f}")
    print(df[passage_col].iloc[idx][:200] + "...")
""",

        "Class Weight Calculation": """# Calculate class weights for all labels
weights = []
for label in label_columns:
    pos = df[label].sum()
    neg = len(df) - pos
    weight = neg / pos if pos > 0 else 1.0
    weight = min(weight, 100)  # Cap at 100x
    weights.append({'Label': label, 'Weight': f"{weight:.2f}x"})

import pandas as pd
weights_df = pd.DataFrame(weights)
print(weights_df.to_string(index=False))
"""
    }

    selected_template = st.selectbox(
        "Load template:",
        ["Custom Code"] + list(templates.keys()),
        key="template_select"
    )

    # Code editor
    if selected_template == "Custom Code":
        default_code = f"""# Available variables:
# - df: DataFrame with {len(df)} passages
# - label_columns: {label_columns[:3]}...
# - passage_col: '{passage_col}'

import pandas as pd
import numpy as np

# Your code here
print(df.head())
"""
    else:
        default_code = templates[selected_template]

    code = st.text_area(
        "Python code:",
        value=default_code,
        height=300,
        key="code_editor"
    )

    col1, col2 = st.columns([1, 4])

    with col1:
        if st.button("▶️ Run Code", type="primary"):
            st.session_state['run_code'] = True

    with col2:
        st.caption("Available: pandas (pd), numpy (np), matplotlib (plt)")

    # Execute code
    if st.session_state.get('run_code'):
        st.markdown("---")
        st.markdown("#### 📤 Output")

        try:
            # Create execution context
            import io
            import sys
            from contextlib import redirect_stdout

            output_buffer = io.StringIO()

            # Prepare globals with data access
            exec_globals = {
                'df': df,
                'label_columns': label_columns,
                'passage_col': passage_col,
                'pd': pd,
                'np': np,
                'st': st,
                'plt': None  # Import only if used
            }

            # Execute code
            with redirect_stdout(output_buffer):
                exec(code, exec_globals)

            # Display output
            output = output_buffer.getvalue()
            if output:
                st.code(output, language="text")
            else:
                st.info("Code executed successfully (no output)")

            # Check for created figures
            import matplotlib.pyplot as plt
            if plt.get_fignums():
                st.pyplot(plt.gcf())
                plt.close('all')

        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

        finally:
            st.session_state['run_code'] = False