# core/quality_scoring.py
"""
Quality Scoring System for HRAF Dataset
Uses VoyageAI embeddings + Pinecone vector search for passage quality assessment
"""

import voyageai
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os

load_dotenv()


class QualityScorer:
    """
    Calculates passage quality scores using:
    - VoyageAI embeddings (voyage-3-large)
    - Pinecone vector search
    - VoyageAI reranker (rerank-2.5)
    """

    # Label semantic descriptions for reranking
    LABEL_QUERIES = {
        # EVENT subcategories
        'Illness': 'Disease, sickness, illness, or mental/physical health problems',
        'Accident': 'Physical accidents, injuries, or harm not caused by illness',
        'Other': 'Other forms of misfortune, bad luck, or negative events',

        # CAUSE subcategories
        'Just_Happens': 'Events happening by chance, coincidence, or without specific cause',
        'Material_Physical': 'Physical, tangible, or natural causes for misfortune',
        'Spirits_Gods': 'Spirits, gods, deities, orixás, or supernatural entities causing problems',
        'Witchcraft_Sorcery': 'Witchcraft, sorcery, curses, or mystical malicious actions',
        'Rule_Violation_Taboo': 'Breaking rules, taboos, sins, or cultural prohibitions',
        'Other.1': 'Other causes for misfortune not covered above',

        # ACTION subcategories
        'Physical_Material': 'Physical remedies, medicine, washing wounds, protective objects, totems',
        'Technical_Specialist': 'Medical doctors, technical experts, or specialists with practical knowledge',
        'Divination': 'Divination, fortune telling, or procedures to reveal hidden information',
        'Shaman_Medium_Healer': 'Shamans, mediums, spirit healers, or people who interact with spirits',
        'Priest_High_Religion': 'Priests, ordained religious authorities, or organized religious figures',
        'Other.2': 'Other actions to prevent or mollify misfortune'
    }

    def __init__(self,
                 voyage_api_key: str = None,
                 pinecone_api_key: str = None,
                 index_name: str = "hraf-misfortune-test",
                 region: str = "us-east-1"):
        """Initialize scoring system with API clients"""

        # Get API keys from env if not provided
        voyage_api_key = voyage_api_key or os.getenv("VOYAGE_API_KEY")
        pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")

        if not voyage_api_key:
            raise ValueError("VOYAGE_API_KEY required")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY required")

        # Initialize clients
        self.voyage = voyageai.Client(api_key=voyage_api_key)
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.region = region

        # Setup Pinecone index
        self._setup_index()

    def _setup_index(self):
        """Create or connect to Pinecone index"""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,  # voyage-3-large dimension
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region=self.region)
            )

            # Wait for ready
            while True:
                status = self.pc.describe_index(self.index_name).status
                if hasattr(status, 'ready') and status.ready:
                    break
                elif isinstance(status, dict) and status.get('ready'):
                    break
                time.sleep(1)
            print("✅ Index created")
        else:
            print(f"Connected to existing index: {self.index_name}")

        # Connect to index
        index_info = self.pc.describe_index(self.index_name)
        self.index = self.pc.Index(name=self.index_name, host=index_info.host)

    def _get_vectors_from_fetch(self, fetch_result):
        """Extract vectors from Pinecone FetchResponse"""
        if hasattr(fetch_result, 'vectors'):
            return fetch_result.vectors or {}
        elif isinstance(fetch_result, dict):
            return fetch_result.get('vectors', {})
        else:
            try:
                return dict(fetch_result).get('vectors', {})
            except:
                return {}

    def embed_and_store_passages(self,
                                 df: pd.DataFrame,
                                 passage_column: str,
                                 label_columns: List[str],
                                 namespace: str,
                                 batch_size: int = 32) -> Dict[str, str]:
        """
        Generate embeddings and store in Pinecone

        Returns: {stable_id: pinecone_id} mapping
        """
        if 'passage_id' not in df.columns:
            raise ValueError("DataFrame must have 'passage_id' column")

        # Filter valid passages
        valid_mask = df[passage_column].notna() & df['passage_id'].notna()
        valid_df = df[valid_mask].copy()

        print(f"Embedding {len(valid_df)} passages...")

        stable_id_to_pinecone = {}

        # Process in batches
        for i in tqdm(range(0, len(valid_df), batch_size), desc="Embedding"):
            batch_df = valid_df.iloc[i:i + batch_size]
            batch_texts = [str(text) for text in batch_df[passage_column].tolist()]

            # Generate embeddings
            result = self.voyage.embed(
                texts=batch_texts,
                model="voyage-3-large",
                input_type="document"
            )

            # Prepare for Pinecone
            vectors = []
            for j, embedding in enumerate(result.embeddings):
                row_idx = valid_df.index[i + j]
                stable_id = valid_df.loc[row_idx, 'passage_id']
                pinecone_id = f"passage_{stable_id}"

                metadata = {
                    'stable_id': stable_id,
                    'text_preview': batch_texts[j][:1000],
                    'text_length': len(batch_texts[j])
                }

                # Add label values
                for label in label_columns:
                    val = batch_df.iloc[j][label]
                    metadata[f'label_{label}'] = int(val) if pd.notna(val) else 0

                vectors.append({
                    'id': pinecone_id,
                    'values': embedding,
                    'metadata': metadata
                })

                stable_id_to_pinecone[stable_id] = pinecone_id

            # Upsert to Pinecone
            self.index.upsert(vectors=vectors, namespace=namespace)

        print(f"✅ Stored {len(stable_id_to_pinecone)} vectors")
        return stable_id_to_pinecone

    def search_similar_to_passage(self,
                                  passage_idx: int,
                                  df: pd.DataFrame,
                                  namespace: str,
                                  k: int = 20,
                                  exclude_self: bool = True) -> List[Dict]:
        """Find passages similar to given passage"""

        if 'passage_id' not in df.columns:
            raise ValueError("DataFrame must have 'passage_id' column")

        if passage_idx not in df.index:
            raise ValueError(f"Passage {passage_idx} not in DataFrame")

        # Get stable ID
        stable_id = df.loc[passage_idx, 'passage_id']
        query_id = f"passage_{stable_id}"

        # Fetch vector
        fetch_result = self.index.fetch(ids=[query_id], namespace=namespace)
        vectors_dict = self._get_vectors_from_fetch(fetch_result)

        if query_id not in vectors_dict:
            raise ValueError(f"Passage {stable_id} not found in Pinecone")

        vector_data = vectors_dict[query_id]
        query_vector = vector_data.values if hasattr(vector_data, 'values') else vector_data['values']

        # Search
        search_results = self.index.query(
            vector=query_vector,
            top_k=k + (1 if exclude_self else 0),
            namespace=namespace,
            include_metadata=True
        )

        matches = search_results.matches if hasattr(search_results, 'matches') else search_results.get('matches', [])

        similar = []
        for match in matches:
            match_id = match.id if hasattr(match, 'id') else match['id']

            if exclude_self and match_id == query_id:
                continue

            score = match.score if hasattr(match, 'score') else match['score']
            metadata = match.metadata if hasattr(match, 'metadata') else match['metadata']

            # Map back to DataFrame using stable_id
            match_stable_id = metadata.get('stable_id')
            if not match_stable_id:
                continue

            matching_rows = df[df['passage_id'] == match_stable_id]
            if matching_rows.empty:
                continue

            match_df_idx = matching_rows.index[0]

            similar.append({
                'passage_idx': match_df_idx,
                'similarity': score,
                'metadata': metadata,
                'stable_id': match_stable_id
            })

        return similar[:k]

    def calculate_label_consistency(self,
                                    query_idx: int,
                                    similar_passages: List[Dict],
                                    label_columns: List[str],
                                    df: pd.DataFrame,
                                    namespace: str,
                                    label_frequencies: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate per-label consistency scores

        Strategy based on label frequency:
        - Very rare (<5%): High similarity to labeled neighbors required
        - Rare (5-15%): Hybrid agreement + similarity
        - Common (>15%): Traditional neighbor agreement
        """

        if 'passage_id' not in df.columns:
            raise ValueError("DataFrame must have 'passage_id' column")

        # Calculate frequencies if not provided
        if label_frequencies is None:
            label_frequencies = {label: df[label].mean() for label in label_columns}

        # Get query metadata from Pinecone
        stable_id = df.loc[query_idx, 'passage_id']
        query_id = f"passage_{stable_id}"

        fetch_result = self.index.fetch(ids=[query_id], namespace=namespace)
        vectors_dict = self._get_vectors_from_fetch(fetch_result)

        if query_id not in vectors_dict:
            return {label: 0.0 for label in label_columns}

        vector_data = vectors_dict[query_id]
        query_metadata = vector_data.metadata if hasattr(vector_data, 'metadata') else vector_data.get('metadata', {})

        # Get query labels
        query_labels = {
            label: query_metadata.get(f'label_{label}', 0)
            for label in label_columns
        }

        consistency = {}

        for label in label_columns:
            # Skip inactive labels
            if query_labels.get(label, 0) == 0:
                consistency[label] = 0.0
                continue

            freq = label_frequencies[label]

            # VERY RARE: <5%
            if freq < 0.05:
                labeled_neighbors = [
                    p for p in similar_passages
                    if p['metadata'].get(f'label_{label}', 0) == 1
                ]

                if len(labeled_neighbors) < 2:
                    consistency[label] = 0.0
                else:
                    similarities = [p['similarity'] for p in labeled_neighbors[:10]]
                    mean_sim = np.mean(similarities)
                    support_weight = min(len(labeled_neighbors) / 5.0, 1.0)
                    consistency[label] = mean_sim * support_weight

            # RARE: 5-15%
            elif freq < 0.15:
                agreements = sum(
                    1 for p in similar_passages[:15]
                    if p['metadata'].get(f'label_{label}', 0) == query_labels[label]
                )
                agreement_score = agreements / min(15, len(similar_passages))

                labeled_neighbors = [
                    p for p in similar_passages[:15]
                    if p['metadata'].get(f'label_{label}', 0) == 1
                ]

                if labeled_neighbors:
                    mean_sim = np.mean([p['similarity'] for p in labeled_neighbors])
                    consistency[label] = 0.7 * agreement_score + 0.3 * mean_sim
                else:
                    consistency[label] = agreement_score * 0.5

            # COMMON: >15%
            else:
                agreements = sum(
                    1 for p in similar_passages[:20]
                    if p['metadata'].get(f'label_{label}', 0) == query_labels[label]
                )
                consistency[label] = agreements / min(20, len(similar_passages))

        return consistency

    def calculate_rerank_scores(self,
                                passages: List[Tuple[int, str]],
                                label_columns: List[str],
                                df: pd.DataFrame,
                                batch_size: int = 32) -> Dict[int, Dict[str, float]]:
        """
        Calculate semantic relevance scores using reranker

        Groups by label for efficiency (reduces API calls)
        """

        # Initialize scores
        rerank_scores = {idx: {label: 0.0 for label in label_columns} for idx, _ in passages}

        # Group passages by label
        label_to_passages = {label: [] for label in label_columns}

        for df_idx, passage_text in passages:
            for label in label_columns:
                if df.loc[df_idx, label] == 1:
                    label_to_passages[label].append((df_idx, passage_text))

        # Process each label
        for label in tqdm(label_columns, desc="Reranking"):
            passages_with_label = label_to_passages[label]

            if not passages_with_label:
                continue

            label_query = self.LABEL_QUERIES.get(label, label)

            # Process in batches
            for i in range(0, len(passages_with_label), batch_size):
                batch = passages_with_label[i:i + batch_size]
                batch_texts = [text for _, text in batch]
                batch_indices = [idx for idx, _ in batch]

                try:
                    result = self.voyage.rerank(
                        query=label_query,
                        documents=batch_texts,
                        model="rerank-2.5",
                        top_k=len(batch_texts)
                    )

                    for res in result.results:
                        df_idx = batch_indices[res.index]
                        rerank_scores[df_idx][label] = res.relevance_score

                except Exception as e:
                    print(f"Reranking error for {label}: {e}")

                time.sleep(0.1)  # Rate limit protection

        return rerank_scores