"""
Unified quality scoring system
Single source of truth for passage quality
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import voyageai
from pinecone import Pinecone
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


@dataclass
class PassageQuality:
    """Single unified quality representation"""

    idx: int  # Passage index

    # Core metrics (0-1 normalized)
    semantic_consistency: float  # How similar to labeled neighbors
    label_confidence: float  # Reranker score for labels
    model_agreement: float  # If models exist, prediction consistency

    # Computed properties
    overall_quality: float
    tier: str  # "elite" / "good" / "fair" / "low"

    # Supporting data
    num_labels: int
    similar_passages: List[int] = None
    problematic_labels: List[str] = None

    def __post_init__(self):
        """Compute derived properties"""
        if self.overall_quality == 0:
            self._compute_overall_quality()
        if not self.tier:
            self._assign_tier()

    def _compute_overall_quality(self):
        """Weighted composite score"""
        # Weight based on what data is available
        if self.model_agreement > 0:
            # If we have model predictions, weight them heavily
            weights = [0.3, 0.3, 0.4]
            scores = [self.semantic_consistency, self.label_confidence, self.model_agreement]
        else:
            # Otherwise, equal weight
            weights = [0.5, 0.5, 0.0]
            scores = [self.semantic_consistency, self.label_confidence, 0]

        self.overall_quality = sum(w * s for w, s in zip(weights, scores))

    def _assign_tier(self):
        """Assign quality tier"""
        if self.overall_quality >= 0.75:
            self.tier = "elite"
        elif self.overall_quality >= 0.60:
            self.tier = "good"
        elif self.overall_quality >= 0.45:
            self.tier = "fair"
        else:
            self.tier = "low"

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'idx': self.idx,
            'semantic_consistency': self.semantic_consistency,
            'label_confidence': self.label_confidence,
            'model_agreement': self.model_agreement,
            'overall_quality': self.overall_quality,
            'tier': self.tier,
            'num_labels': self.num_labels
        }


class QualityScorer:
    """
    Unified quality scoring system
    Computes all quality metrics in one place
    """

    # Label semantic definitions for reranking
    LABEL_DEFINITIONS = {
        'Illness': 'Disease, sickness, illness, or mental/physical health problems',
        'Accident': 'Physical accidents, injuries, or harm not caused by illness',
        'Just_Happens': 'Events happening by chance, coincidence, or without specific cause',
        'Material_Physical': 'Physical, tangible, or natural causes for misfortune',
        'Spirits_Gods': 'Spirits, gods, deities, or supernatural entities causing problems',
        'Witchcraft_Sorcery': 'Witchcraft, sorcery, curses, or mystical malicious actions',
        'Rule_Violation_Taboo': 'Breaking rules, taboos, sins, or cultural prohibitions',
        'Physical_Material': 'Physical remedies, medicine, washing wounds, protective objects',
        'Technical_Specialist': 'Medical doctors, technical experts, or specialists',
        'Divination': 'Divination, fortune telling, or procedures to reveal hidden information',
        'Shaman_Medium_Healer': 'Shamans, mediums, spirit healers, or people who interact with spirits',
        'Priest_High_Religion': 'Priests, ordained religious authorities, or organized religious figures',
    }

    def __init__(
            self,
            df: pd.DataFrame,
            passage_col: str,
            label_columns: List[str],
            use_embeddings: bool = True,
            index_name: str = "hraf-passages"
    ):
        self.df = df
        self.passage_col = passage_col
        self.label_columns = label_columns
        self.use_embeddings = use_embeddings

        # Initialize embedding/reranking if enabled
        if use_embeddings:
            if not VOYAGE_API_KEY:
                raise ValueError(
                    "VOYAGE_API_KEY not found. Please set it in your .env file.\n"
                    "Get your API key from: https://dash.voyageai.com"
                )
            if not PINECONE_API_KEY:
                raise ValueError(
                    "PINECONE_API_KEY not found. Please set it in your .env file.\n"
                    "Get your API key from: https://app.pinecone.io"
                )

            self.voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            self.embeddings_cache = {}

            # CREATE OR CONNECT TO PINECONE INDEX
            self.index_name = index_name

            # Check if index exists
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]

            if index_name not in existing_indexes:
                print(f"📊 Creating new Pinecone index: {index_name}")
                from pinecone import ServerlessSpec

                self.pc.create_index(
                    name=index_name,
                    dimension=1024,  # ← Change from 2048 to 1024
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                print(f"✅ Created index: {index_name}")
            else:
                print(f"✅ Connected to existing index: {index_name}")

            # Get the index
            self.index = self.pc.Index(index_name)
            print(f"✅ Pinecone index ready: {self.index.describe_index_stats()}")
        else:
            self.voyage = None
            self.pc = None
            self.index = None

    def compute_all(
            self,
            k_similar: int = 15,
            use_cache: bool = True
    ) -> Dict[int, PassageQuality]:
        """
        Compute quality scores for all passages

        Returns:
            Dict mapping passage index to PassageQuality
        """
        quality_scores = {}

        # Get valid passages
        valid_mask = self.df[self.passage_col].notna()
        valid_indices = self.df[valid_mask].index.tolist()

        print(f"Computing quality for {len(valid_indices)} passages...")

        # Batch compute embeddings if needed
        if self.use_embeddings:
            self._compute_embeddings(valid_indices)

        # Compute scores for each passage
        for idx in valid_indices:
            try:
                quality = self._compute_passage_quality(idx, k_similar)
                quality_scores[idx] = quality
            except Exception as e:
                print(f"Error computing quality for passage {idx}: {e}")
                # Create low-quality placeholder
                quality_scores[idx] = PassageQuality(
                    idx=idx,
                    semantic_consistency=0.0,
                    label_confidence=0.0,
                    model_agreement=0.0,
                    overall_quality=0.0,
                    tier="low",
                    num_labels=0
                )

        print(f"✅ Computed quality for {len(quality_scores)} passages")

        return quality_scores

    def _compute_embeddings(self, indices: List[int]):
        """Batch compute embeddings for passages"""
        print("Computing embeddings...")

        # Get texts
        texts = [str(self.df.loc[idx, self.passage_col]) for idx in indices]

        # Batch embed
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_indices = indices[i:i + batch_size]

            result = self.voyage.embed(
                texts=batch_texts,
                model="voyage-3-large",
                input_type="document"
            )

            # Cache embeddings
            for idx, embedding in zip(batch_indices, result.embeddings):
                self.embeddings_cache[idx] = np.array(embedding)

        print("✅ Embeddings computed")

    def _compute_embeddings_with_progress(
            self,
            indices: List[int],
            progress_callback=None,
            namespace: str = "main"
    ):
        """
        Batch compute embeddings and STORE IN PINECONE
        Only computes embeddings that don't already exist
        """
        print(f"📊 Checking which passages need embedding...")

        # Check which passages already have embeddings in Pinecone
        existing_ids = set()
        if hasattr(self, 'index') and self.index:
            try:
                # Fetch in batches to check what exists
                batch_size = 100
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    ids_to_check = [f"passage_{idx}" for idx in batch_indices]

                    fetch_result = self.index.fetch(ids=ids_to_check, namespace=namespace)
                    vectors_dict = fetch_result.vectors if hasattr(fetch_result, 'vectors') else fetch_result.get(
                        'vectors', {})

                    existing_ids.update(vectors_dict.keys())
            except Exception as e:
                print(f"Could not check existing vectors: {e}")

        # Determine which passages need embedding
        indices_to_embed = [
            idx for idx in indices
            if f"passage_{idx}" not in existing_ids
        ]

        if not indices_to_embed:
            print(f"✅ All {len(indices)} passages already embedded in Pinecone!")
            return

        print(f"📊 Need to embed {len(indices_to_embed)} passages ({len(existing_ids)} already exist)")

        # Get texts for passages that need embedding
        texts = [str(self.df.loc[idx, self.passage_col]) for idx in indices_to_embed]

        # Batch embed and store
        batch_size = 32
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_indices = indices_to_embed[i:i + batch_size]

            # Generate embeddings
            result = self.voyage.embed(
                texts=batch_texts,
                model="voyage-3-large",
                input_type="document"
            )

            # Prepare vectors for Pinecone
            vectors_to_upsert = []

            for idx, embedding in zip(batch_indices, result.embeddings):
                # Cache in memory
                self.embeddings_cache[idx] = np.array(embedding)

                # Prepare for Pinecone
                passage_id = f"passage_{idx}"

                # Build metadata
                metadata = {
                    'passage_idx': int(idx),
                    'text_preview': str(self.df.loc[idx, self.passage_col])[:1000],
                    'text_length': len(str(self.df.loc[idx, self.passage_col])),
                }

                # Add labels to metadata
                for label in self.label_columns:
                    metadata[f'label_{label}'] = int(self.df.loc[idx, label])

                vectors_to_upsert.append({
                    'id': passage_id,
                    'values': embedding,
                    'metadata': metadata
                })

            # Upsert to Pinecone
            if hasattr(self, 'index') and self.index:
                try:
                    self.index.upsert(vectors=vectors_to_upsert, namespace=namespace)
                except Exception as e:
                    print(f"Warning: Could not upsert to Pinecone: {e}")

            # Progress callback
            if progress_callback:
                current_batch = (i // batch_size) + 1
                progress_callback(
                    current_batch,
                    total_batches,
                    f"Embedded batch {current_batch}/{total_batches}"
                )

        print(f"✅ Embeddings computed and stored in Pinecone!")

    def compute_all_with_progress(
            self,
            k_similar: int = 15,
            namespace: str = "main",
            progress_callback=None
    ) -> Dict[int, PassageQuality]:
        """
        Compute quality scores - REUSES STORED EMBEDDINGS

        Workflow:
        1. Compute/load embeddings (one-time cost)
        2. Use Pinecone for all similarity searches (fast!)
        3. Compute quality scores
        """
        quality_scores = {}

        # Get valid passages
        valid_mask = self.df[self.passage_col].notna()
        valid_indices = self.df[valid_mask].index.tolist()

        total = len(valid_indices)
        print(f"Computing quality for {total} passages...")

        # Check if we can use Pinecone
        can_use_pinecone = (
                self.use_embeddings and
                hasattr(self, 'index') and
                self.index is not None
        )

        if not can_use_pinecone:
            print("⚠️ Pinecone not available - quality scores will be less accurate")

        # Compute scores for each passage
        for i, idx in enumerate(valid_indices):
            try:
                # Get active labels
                active_labels = [
                    label for label in self.label_columns
                    if self.df.loc[idx, label] == 1
                ]

                # Compute semantic consistency using Pinecone (fast!)
                if can_use_pinecone:
                    semantic_consistency = self._compute_semantic_consistency_pinecone(
                        idx, k_similar, active_labels, namespace
                    )
                else:
                    semantic_consistency = 0.5  # Neutral default

                # For label confidence, use semantic consistency as proxy
                # (reranker is too slow for all passages)
                label_confidence = semantic_consistency

                quality = PassageQuality(
                    idx=idx,
                    semantic_consistency=semantic_consistency,
                    label_confidence=label_confidence,
                    model_agreement=0.0,
                    overall_quality=0.0,
                    tier="",
                    num_labels=len(active_labels)
                )

                quality_scores[idx] = quality

                # Progress callback every 50 passages
                if progress_callback and (i % 50 == 0 or i == total - 1):
                    progress_callback(
                        i + 1,
                        total,
                        f"Scored {i + 1}/{total} passages"
                    )

            except Exception as e:
                print(f"Error computing quality for passage {idx}: {e}")
                quality_scores[idx] = PassageQuality(
                    idx=idx,
                    semantic_consistency=0.0,
                    label_confidence=0.0,
                    model_agreement=0.0,
                    overall_quality=0.0,
                    tier="low",
                    num_labels=0
                )

        print(f"✅ Computed quality for {len(quality_scores)} passages")

        return quality_scores

    def _compute_semantic_consistency_pinecone(
            self,
            idx: int,
            k: int,
            active_labels: List[str],
            namespace: str = "main"
    ) -> float:
        """
        Compute semantic consistency using stored Pinecone vectors
        This is FAST because vectors are already computed and indexed!
        """
        try:
            query_id = f"passage_{idx}"

            # Fetch query vector from Pinecone
            fetch_result = self.index.fetch(ids=[query_id], namespace=namespace)
            vectors_dict = fetch_result.vectors if hasattr(fetch_result, 'vectors') else fetch_result.get('vectors', {})

            if query_id not in vectors_dict:
                return 0.5

            # Get vector
            vector_data = vectors_dict[query_id]
            query_vector = vector_data.values if hasattr(vector_data, 'values') else vector_data['values']

            # Search Pinecone for similar passages (super fast!)
            search_results = self.index.query(
                vector=query_vector,
                top_k=k + 1,  # +1 to exclude self
                namespace=namespace,
                include_metadata=True
            )

            if not active_labels:
                return 0.5

            # Calculate label agreement with similar passages
            matches = search_results.matches if hasattr(search_results, 'matches') else search_results.get('matches',
                                                                                                           [])

            agreements = []
            for match in matches:
                metadata = match.metadata if hasattr(match, 'metadata') else match['metadata']
                match_idx = metadata['passage_idx']

                # Skip self
                if match_idx == idx:
                    continue

                # Check label agreement
                agreement = 0
                for label in active_labels:
                    if metadata.get(f'label_{label}', 0) == 1:
                        agreement += 1

                # Weight by similarity score
                similarity = match.score if hasattr(match, 'score') else match['score']
                weighted_agreement = (agreement / len(active_labels)) * similarity
                agreements.append(weighted_agreement)

            return np.mean(agreements) if agreements else 0.5

        except Exception as e:
            print(f"Pinecone similarity failed for {idx}: {e}")
            return 0.5

    def _compute_semantic_consistency_fast(self, idx: int, k: int) -> float:
        """
        Fast semantic consistency using Pinecone vector search
        """
        try:
            # Query Pinecone for similar passages
            query_id = f"passage_{idx}"

            # Fetch query vector
            fetch_result = self.index.fetch(ids=[query_id])
            vectors_dict = fetch_result.vectors if hasattr(fetch_result, 'vectors') else fetch_result.get('vectors', {})

            if query_id not in vectors_dict:
                return 0.5

            # Get vector
            vector_data = vectors_dict[query_id]
            query_vector = vector_data.values if hasattr(vector_data, 'values') else vector_data['values']

            # Search for similar passages
            search_results = self.index.query(
                vector=query_vector,
                top_k=k + 1,  # +1 to exclude self
                include_metadata=True
            )

            # Calculate label agreement
            active_labels = [
                label for label in self.label_columns
                if self.df.loc[idx, label] == 1
            ]

            if not active_labels:
                return 0.5

            matches = search_results.matches if hasattr(search_results, 'matches') else search_results.get('matches',
                                                                                                           [])

            agreements = []
            for match in matches:
                match_idx = match.metadata['passage_idx'] if hasattr(match, 'metadata') else match['metadata'][
                    'passage_idx']

                if match_idx == idx:
                    continue

                if match_idx not in self.df.index:
                    continue

                # Check label agreement
                agreement = 0
                for label in active_labels:
                    if self.df.loc[match_idx, label] == 1:
                        agreement += 1

                if len(active_labels) > 0:
                    similarity = match.score if hasattr(match, 'score') else match['score']
                    weighted_agreement = (agreement / len(active_labels)) * similarity
                    agreements.append(weighted_agreement)

            return np.mean(agreements) if agreements else 0.5

        except Exception as e:
            print(f"Fast similarity failed for {idx}: {e}")
            return 0.5

    def _compute_passage_quality(
            self,
            idx: int,
            k_similar: int
    ) -> PassageQuality:
        """Compute quality for a single passage"""

        # Get active labels
        active_labels = [
            label for label in self.label_columns
            if self.df.loc[idx, label] == 1
        ]

        # Compute semantic consistency
        if self.use_embeddings and idx in self.embeddings_cache:
            semantic_consistency = self._compute_semantic_consistency(
                idx, k_similar, active_labels
            )
        else:
            semantic_consistency = 0.5  # Neutral default

        # Compute label confidence (reranker scores)
        if self.use_embeddings and active_labels:
            label_confidence = self._compute_label_confidence(idx, active_labels)
        else:
            label_confidence = 0.5  # Neutral default

        # Model agreement (computed later if model exists)
        model_agreement = 0.0  # Will be updated after training

        return PassageQuality(
            idx=idx,
            semantic_consistency=semantic_consistency,
            label_confidence=label_confidence,
            model_agreement=model_agreement,
            overall_quality=0.0,  # Will be computed in __post_init__
            tier="",  # Will be assigned in __post_init__
            num_labels=len(active_labels)
        )

    def _compute_semantic_consistency(
            self,
            idx: int,
            k: int,
            active_labels: List[str]
    ) -> float:
        """
        Compute how consistent this passage's labels are with similar passages

        High consistency = similar passages have similar labels
        """
        if not active_labels:
            return 0.5

        # Get embedding
        query_embedding = self.embeddings_cache[idx]

        # Find k most similar passages (excluding self)
        similarities = {}
        for other_idx, other_embedding in self.embeddings_cache.items():
            if other_idx == idx:
                continue

            # Cosine similarity
            sim = np.dot(query_embedding, other_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(other_embedding)
            )
            similarities[other_idx] = sim

        # Get top k
        top_k = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]

        if not top_k:
            return 0.5

        # Check label agreement
        agreements = []
        for other_idx, sim in top_k:
            # Check if labels match
            agreement = 0
            for label in active_labels:
                if self.df.loc[other_idx, label] == 1:
                    agreement += 1

            # Weight by similarity
            weighted_agreement = (agreement / len(active_labels)) * sim
            agreements.append(weighted_agreement)

        return np.mean(agreements)

    def _compute_label_confidence(
            self,
            idx: int,
            active_labels: List[str]
    ) -> float:
        """
        Use reranker to score how well passage matches label definitions

        High confidence = passage clearly demonstrates the labeled concepts
        """
        text = str(self.df.loc[idx, self.passage_col])

        # Score each active label
        scores = []
        for label in active_labels:
            if label not in self.LABEL_DEFINITIONS:
                scores.append(0.5)  # Neutral for unknown labels
                continue

            # Use reranker to score relevance
            try:
                result = self.voyage.rerank(
                    query=self.LABEL_DEFINITIONS[label],
                    documents=[text],
                    model="rerank-2.5"
                )
                scores.append(result.results[0].relevance_score)
            except:
                scores.append(0.5)  # Neutral on error

        return np.mean(scores) if scores else 0.5

    def update_with_model_predictions(
            self,
            quality_scores: Dict[int, PassageQuality],
            model_predictions: Dict[int, Dict]
    ):
        """
        Update quality scores with model prediction agreement

        After training a model, we can check if predictions agree with labels
        High agreement = labels are clear and learnable
        """
        for idx, quality in quality_scores.items():
            if idx not in model_predictions:
                continue

            pred = model_predictions[idx]
            actual_labels = set(
                label for label in self.label_columns
                if self.df.loc[idx, label] == 1
            )
            predicted_labels = set(pred['predicted_labels'])

            # Compute agreement (Jaccard similarity)
            if actual_labels or predicted_labels:
                agreement = len(actual_labels & predicted_labels) / len(actual_labels | predicted_labels)
            else:
                agreement = 1.0

            # Update quality
            quality.model_agreement = agreement
            quality._compute_overall_quality()
            quality._assign_tier()

    def get_quality_report(
            self,
            quality_scores: Dict[int, PassageQuality]
    ) -> Dict:
        """Generate quality distribution report"""
        qualities = [q.overall_quality for q in quality_scores.values()]
        tiers = [q.tier for q in quality_scores.values()]

        from collections import Counter
        tier_counts = Counter(tiers)

        return {
            'mean': float(np.mean(qualities)),
            'median': float(np.median(qualities)),
            'std': float(np.std(qualities)),
            'min': float(np.min(qualities)),
            'max': float(np.max(qualities)),
            'tier_distribution': dict(tier_counts),
            'tier_percentages': {
                tier: count / len(tiers) * 100
                for tier, count in tier_counts.items()
            }
        }