"""
Sidebar Selectors - Quick access to datasets and models

Provides simple dropdown selectors in the sidebar for:
- Available datasets (Excel files in data/)
- Available/loaded models (in models/)

No processing required - just select and use.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import json


# Default label columns for this project
DEFAULT_LABELS = [
    "Illness", "Accident", "Other", "Material_Physical", "Spirits_Gods",
    "Witchcraft_Sorcery", "Rule_Violation_Taboo", "Physical_Material",
    "Technical_Specialist", "Divination", "Shaman_Medium_Healer", "Priest_High_Religion"
]


def render_sidebar_selectors():
    """Render dataset and model selectors in sidebar"""

    st.markdown("### 📊 Dataset")
    render_dataset_selector()

    st.markdown("---")

    st.markdown("### 🤖 Model")
    render_model_selector()


def render_dataset_selector():
    """Simple dataset selector - just pick a file"""

    # Scan for available datasets in multiple locations
    datasets = []

    # 1. Direct files in data/
    data_dir = Path("data")
    if data_dir.exists():
        for f in data_dir.glob("*.xlsx"):
            datasets.append({
                'path': f,
                'name': f.stem,
                'size': f.stat().st_size / 1024,
                'modified': datetime.fromtimestamp(f.stat().st_mtime),
                'source': 'data'
            })

    # 2. Files in temp/
    temp_dir = Path("temp")
    if temp_dir.exists():
        for f in temp_dir.glob("*.xlsx"):
            datasets.append({
                'path': f,
                'name': f.stem,
                'size': f.stat().st_size / 1024,
                'modified': datetime.fromtimestamp(f.stat().st_mtime),
                'source': 'temp'
            })

    # 3. Tiered datasets (tier1, tier2, inference from objects)
    objects_dir = Path("data/objects")
    if objects_dir.exists():
        # Find tiered directories
        for tiered_dir in objects_dir.glob("tiered/*/"):
            for tier_file in ['tier1.xlsx', 'tier2.xlsx', 'inference.xlsx', 'data.xlsx']:
                tier_path = tiered_dir / tier_file
                if tier_path.exists():
                    # Create friendly name
                    tier_name = f"{tier_file.replace('.xlsx', '')}_{tiered_dir.name[:30]}"
                    datasets.append({
                        'path': tier_path,
                        'name': tier_name,
                        'size': tier_path.stat().st_size / 1024,
                        'modified': datetime.fromtimestamp(tier_path.stat().st_mtime),
                        'source': 'tiered'
                    })

        # Also check raw, cleaned, etc.
        for stage in ['raw', 'cleaned', 'embedded', 'scored']:
            stage_dir = objects_dir / stage
            if stage_dir.exists():
                for obj_dir in stage_dir.iterdir():
                    if obj_dir.is_dir():
                        data_file = obj_dir / "data.xlsx"
                        if data_file.exists():
                            datasets.append({
                                'path': data_file,
                                'name': f"{stage}_{obj_dir.name[:25]}",
                                'size': data_file.stat().st_size / 1024,
                                'modified': datetime.fromtimestamp(data_file.stat().st_mtime),
                                'source': stage
                            })

    # Sort by modification time
    datasets = sorted(datasets, key=lambda x: x['modified'], reverse=True)

    if not datasets:
        st.caption("No datasets found")
        # Show upload option
        if st.button("📤 Upload", key="upload_dataset_btn", use_container_width=True):
            st.session_state['show_upload'] = True
        return

    # Current selection - try to match by path
    current = st.session_state.get('selected_dataset_path')
    current_idx = 0

    # Create display names with source indicator
    def get_display_name(d):
        source_icon = {
            'data': '📄',
            'temp': '📁',
            'tiered': '🎯',
            'raw': '📦',
            'cleaned': '🧹',
            'embedded': '🔢',
            'scored': '📊'
        }.get(d['source'], '📄')
        return f"{source_icon} {d['name']}"

    display_names = [get_display_name(d) for d in datasets]

    if current:
        # Find matching dataset by path
        for i, d in enumerate(datasets):
            if str(d['path']) == current:
                current_idx = i
                break

    # Selector
    selected_display = st.selectbox(
        "Select dataset:",
        display_names,
        index=current_idx,
        key="sidebar_dataset_select",
        label_visibility="collapsed"
    )

    # Find selected dataset
    selected_idx = display_names.index(selected_display)
    selected = datasets[selected_idx]

    if selected:
        # Show info
        st.caption(f"{selected['size']:.0f} KB • {selected['source']}")

        # Load button if not already loaded or different dataset
        current_path = st.session_state.get('selected_dataset_path')

        if current_path != str(selected['path']):
            if st.button("📂 Load", key="load_dataset_btn", use_container_width=True):
                load_dataset(selected['path'])
        else:
            # Show loaded status
            df = st.session_state.get('df')
            if df is not None:
                st.success(f"✓ {len(df):,} rows", icon="✅")


def render_model_selector():
    """Simple model selector"""

    # Scan for available models
    models_dir = Path("models")
    models = []

    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                # Check for final_model subdirectory or direct model files
                final_model = model_dir / "final_model"
                if final_model.exists():
                    model_path = final_model
                elif (model_dir / "config.json").exists():
                    model_path = model_dir
                else:
                    continue

                # Try to get model info
                info = get_model_info(model_path)
                models.append({
                    'path': model_path,
                    'name': model_dir.name,
                    'f1': info.get('f1', None),
                    'labels': info.get('num_labels', 12)
                })

    if not models:
        st.caption("No models in models/")
        return

    # Current selection
    current = st.session_state.get('selected_model_path')
    current_idx = 0

    model_names = [m['name'] for m in models]
    if current:
        current_name = Path(current).parent.name if Path(current).name == "final_model" else Path(current).name
        if current_name in model_names:
            current_idx = model_names.index(current_name)

    # Selector
    selected_name = st.selectbox(
        "Select model:",
        model_names,
        index=current_idx,
        key="sidebar_model_select",
        label_visibility="collapsed"
    )

    # Find selected model
    selected = next((m for m in models if m['name'] == selected_name), None)

    if selected:
        # Show info
        if selected['f1']:
            st.caption(f"F1: {selected['f1']:.3f} • {selected['labels']} labels")
        else:
            st.caption(f"{selected['labels']} labels")

        # Load button if not already loaded
        current_path = st.session_state.get('selected_model_path')

        if current_path != str(selected['path']):
            if st.button("🔄 Load", key="load_model_btn", use_container_width=True):
                load_model(selected['path'])
        else:
            # Show loaded status
            if st.session_state.get('model_loaded'):
                st.success("✓ Loaded", icon="✅")


def get_model_info(model_path: Path) -> Dict:
    """Get basic model info from training_info.json"""
    info = {}

    try:
        info_file = model_path / "training_info.json"
        if info_file.exists():
            with open(info_file) as f:
                data = json.load(f)

            # Extract key metrics
            test_results = data.get('test_results', {})
            info['f1'] = test_results.get('eval_f1_micro')
            info['num_labels'] = len(data.get('label_columns', []))
    except:
        pass

    return info


def load_dataset(file_path: Path):
    """Load a dataset file into session state"""

    with st.spinner(f"Loading {file_path.name}..."):
        try:
            df = pd.read_excel(file_path)

            # Auto-detect passage column
            passage_col = None
            for col in df.columns:
                if 'passage' in col.lower():
                    passage_col = col
                    break

            if not passage_col:
                # Find longest text column
                for col in df.columns:
                    if df[col].dtype == 'object':
                        passage_col = col
                        break

            if not passage_col:
                passage_col = df.columns[0]

            # Auto-detect label columns
            label_columns = [c for c in df.columns if c in DEFAULT_LABELS]

            # If no labels found, check for binary columns
            if not label_columns:
                for col in df.columns:
                    if col == passage_col:
                        continue
                    if df[col].dtype in ['int64', 'float64']:
                        unique = df[col].dropna().unique()
                        if len(unique) <= 2 and set(unique).issubset({0, 1, 0.0, 1.0}):
                            label_columns.append(col)

            # Set session state
            st.session_state['selected_dataset_path'] = str(file_path)
            st.session_state['df'] = df
            st.session_state['passage_col'] = passage_col
            st.session_state['label_columns'] = label_columns
            st.session_state['initialized'] = True
            st.session_state['namespace'] = file_path.stem

            # Clear prediction mode flag
            st.session_state['prediction_mode'] = len(label_columns) == 0

            st.rerun()

        except Exception as e:
            st.error(f"Error loading: {e}")


def load_model(model_path: Path):
    """Load a model into session state"""

    with st.spinner(f"Loading model..."):
        try:
            # Initialize model manager if needed
            from components.model_manager import ModelManager

            if 'model_manager' not in st.session_state or st.session_state.model_manager is None:
                st.session_state.model_manager = ModelManager()

            manager = st.session_state.model_manager

            # Load the model
            success = manager.load_model(str(model_path))

            if success:
                st.session_state['selected_model_path'] = str(model_path)
                st.session_state['model_loaded'] = True
                st.rerun()
            else:
                st.error("Failed to load model")

        except Exception as e:
            st.error(f"Error loading model: {e}")


def get_current_dataset() -> Optional[pd.DataFrame]:
    """Get the currently loaded dataset"""
    return st.session_state.get('df')


def get_current_model():
    """Get the currently loaded model"""
    manager = st.session_state.get('model_manager')
    if manager and len(manager) > 0:
        # Return first loaded model
        model_names = list(manager.models.keys())
        if model_names:
            return manager.get_model(model_names[0])
    return None


def get_dataset_info() -> Dict:
    """Get info about current dataset"""
    df = st.session_state.get('df')
    if df is None:
        return {}

    return {
        'name': st.session_state.get('namespace', 'Unknown'),
        'rows': len(df),
        'passage_col': st.session_state.get('passage_col'),
        'label_columns': st.session_state.get('label_columns', []),
        'prediction_mode': st.session_state.get('prediction_mode', False)
    }
