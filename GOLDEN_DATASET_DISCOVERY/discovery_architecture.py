"""
Golden Dataset Identification for HRAF Misfortune Classification

Uses VoyageAI embeddings (voyage-3-large) and reranker (rerank-2.5)
with Pinecone vector storage to identify high-confidence training passages.

Requirements:
- pinecone>=7.0.0
- voyageai>=0.2.0
"""

import voyageai
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import hashlib
import time
from dotenv import load_dotenv
import os
from collections import defaultdict

load_dotenv()

class GoldenDatasetFinder:
    """
    Identifies high-confidence labeled passages using:
    1. VoyageAI voyage-3-large embeddings (stored in Pinecone)
    2. VoyageAI rerank-2.5 for relevance scoring
    3. Similarity clustering for label consistency
    """

    # Label definitions for HRAF misfortune classification
    # Maps actual column names to their semantic descriptions
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

    # Columns to exclude from label detection (metadata, descriptions, etc.)
    EXCLUDE_COLUMNS = {
        'No_Info', 'No_Info.1', 'No_Info.2', 'Description', 'Local_Terms', 'Local_terms',
        'Other_Comments', 'Other_Comments.1', 'Run_Number', 'Finished', 'Coder',
        'Dataset', 'Info', 'Passage Number', 'Region', 'SubRegion',
        'Culture', 'DocTitle', 'Section', 'Author', 'Page', 'Year',
        'OCM', 'OWC', 'Passage', 'ID', 'CULTURE'
    }

    def __init__(self,
                 voyage_api_key: str,
                 pinecone_api_key: str,
                 index_name: str = "hraf-misfortune",
                 cloud: str = "aws",
                 region: str = "us-east-1"):
        """
        Initialize VoyageAI and Pinecone clients

        Args:
            voyage_api_key: VoyageAI API key
            pinecone_api_key: Pinecone API key
            index_name: Name for Pinecone index
            cloud: Cloud provider (aws, gcp, or azure)
            region: Region for serverless index
        """
        # Initialize VoyageAI
        self.voyage = voyageai.Client(api_key=voyage_api_key)

        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.cloud = cloud
        self.region = region

        # Setup or connect to Pinecone index
        self._setup_index()

    def _setup_index(self):
        """Create or connect to Pinecone index using v7 API"""
        # Check if index exists - list_indexes() returns objects with .name attribute
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,  # voyage-3-large dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=self.cloud,
                    region=self.region
                )
            )
            # Wait for index to be ready - describe_index returns object with .status attribute
            while True:
                status = self.pc.describe_index(self.index_name).status
                if hasattr(status, 'ready') and status.ready:
                    break
                elif isinstance(status, dict) and status.get('ready'):
                    break
                time.sleep(1)
            print("✅ Index created successfully")
        else:
            print(f"Connected to existing index: {self.index_name}")

        # Connect to index - requires host parameter in v6+
        index_info = self.pc.describe_index(self.index_name)
        self.index = self.pc.Index(name=self.index_name, host=index_info.host)

    def _generate_passage_id(self, text: str) -> str:
        """Generate consistent ID from passage text"""
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _auto_detect_label_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Auto-detect label columns (binary 0/1 columns that aren't metadata)

        Returns list of column names that appear to be labels
        """
        label_columns = []

        for col in df.columns:
            # Skip excluded columns
            if col in self.EXCLUDE_COLUMNS:
                continue

            # Check if column contains only 0/1 values (allowing NaN)
            if df[col].dtype in ['int64', 'float64', 'Int64']:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) > 0 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    # Additional filter: must have at least some positive values
                    if (df[col] == 1).sum() > 0:
                        label_columns.append(col)

        return label_columns

    def embed_and_store_passages(self,
                                 df: pd.DataFrame,
                                 passage_column: str = 'Passage',
                                 label_columns: List[str] = None,
                                 namespace: str = "main",
                                 batch_size: int = 32) -> Dict[int, str]:
        """
        Generate embeddings for all passages and store in Pinecone

        Args:
            df: DataFrame with passages and labels
            passage_column: Name of column containing passage text
            label_columns: List of label column names (auto-detect if None)
            namespace: Pinecone namespace for organization
            batch_size: Batch size for embedding generation

        Returns:
            Dictionary mapping dataframe index to passage ID
        """
        # Auto-detect label columns if not provided
        if label_columns is None:
            label_columns = self._auto_detect_label_columns(df)

        if not label_columns:
            raise ValueError(
                "No label columns detected! Please specify label_columns explicitly "
                "or ensure your DataFrame has binary (0/1) label columns."
            )

        # Filter out rows with missing passages
        valid_mask = df[passage_column].notna()
        valid_df = df[valid_mask].copy()
        skipped = len(df) - len(valid_df)

        if skipped > 0:
            print(f"⚠️  Skipped {skipped} passages with missing text")

        print(f"Embedding {len(valid_df)} passages using voyage-3-large...")
        print(f"Found {len(label_columns)} label columns: {label_columns[:5]}{'...' if len(label_columns) > 5 else ''}")

        passage_id_map = {}

        # Process in batches
        for i in tqdm(range(0, len(valid_df), batch_size), desc="Embedding batches"):
            batch_df = valid_df.iloc[i:i + batch_size]
            batch_texts = batch_df[passage_column].tolist()

            # Additional safety: convert to strings and handle any remaining NaN
            batch_texts = [str(text) if pd.notna(text) else "" for text in batch_texts]

            # Skip empty texts
            if not any(batch_texts):
                continue

            # Generate embeddings
            result = self.voyage.embed(
                texts=batch_texts,
                model="voyage-3-large",
                input_type="document"
            )
            embeddings = result.embeddings

            # Prepare vectors for Pinecone
            vectors = []
            for j, embedding in enumerate(embeddings):
                # Get original dataframe index
                original_idx = valid_df.index[i + j]
                text = batch_texts[j]
                passage_id = f"passage_{original_idx}"

                # Build metadata
                metadata = {
                    'text_preview': text[:1000],  # First 1000 chars
                    'passage_idx': int(original_idx),
                    'text_length': len(text),
                    'text_hash': self._generate_passage_id(text)
                }

                # Add all label values
                for label in label_columns:
                    if label in batch_df.columns:
                        val = batch_df.iloc[j][label]
                        # Handle NaN in labels
                        metadata[f"label_{label}"] = int(val) if pd.notna(val) else 0

                # Add additional metadata if available
                if 'ID' in batch_df.columns:
                    metadata['original_id'] = str(batch_df.iloc[j]['ID'])
                if 'Culture' in batch_df.columns:
                    metadata['culture'] = str(batch_df.iloc[j]['Culture'])

                vectors.append({
                    'id': passage_id,
                    'values': embedding,
                    'metadata': metadata
                })

                passage_id_map[original_idx] = passage_id

            # Upsert to Pinecone
            self.index.upsert(vectors=vectors, namespace=namespace)

        # Verify
        stats = self.index.describe_index_stats()
        # stats is a dict in v7
        total_vectors = stats.get('total_vector_count', 0) if isinstance(stats, dict) else stats.total_vector_count
        print(f"✅ Stored {total_vectors} vectors")

        return passage_id_map

    def find_similar_passages(self,
                              query_idx: int,
                              k: int = 20,
                              namespace: str = "main",
                              exclude_self: bool = True) -> List[Dict]:
        """
        Find similar passages using Pinecone vector search

        Args:
            query_idx: Index of query passage in original dataframe
            k: Number of similar passages to retrieve
            namespace: Pinecone namespace
            exclude_self: Whether to exclude the query itself

        Returns:
            List of dictionaries with similar passages and scores
        """
        query_id = f"passage_{query_idx}"

        # Fetch query vector - returns dict in v7
        fetch_result = self.index.fetch(ids=[query_id], namespace=namespace)

        # Check if vectors exist in response
        if 'vectors' not in fetch_result or query_id not in fetch_result['vectors']:
            raise ValueError(f"Passage {query_idx} not found in index")

        # Extract vector values from dict
        query_vector = fetch_result['vectors'][query_id]['values']

        # Search for similar vectors - returns dict in v7
        search_results = self.index.query(
            vector=query_vector,
            top_k=k + (1 if exclude_self else 0),
            namespace=namespace,
            include_metadata=True
        )

        # Process results - matches is a list of dicts
        similar_passages = []
        for match in search_results['matches']:
            if exclude_self and match['id'] == query_id:
                continue

            similar_passages.append({
                'passage_idx': match['metadata']['passage_idx'],
                'similarity': match['score'],
                'metadata': match['metadata']
            })

        return similar_passages[:k]

    def calculate_label_consistency(self,
                                    query_idx: int,
                                    similar_passages: List[Dict],
                                    label_columns: List[str],
                                    namespace: str = "main") -> Dict[str, float]:
        """
        Calculate label consistency between query and similar passages

        Returns consistency score (0-1) for each label
        """
        consistency = {}

        # Get query labels from Pinecone - returns dict in v7
        query_id = f"passage_{query_idx}"
        fetch_result = self.index.fetch(ids=[query_id], namespace=namespace)

        # Check if vectors exist
        if 'vectors' not in fetch_result or query_id not in fetch_result['vectors']:
            # If passage not found, return 0 consistency for all labels
            return {label: 0.0 for label in label_columns}

        # Extract metadata from dict
        query_metadata = fetch_result['vectors'][query_id].get('metadata', {})
        query_labels = {}
        for label in label_columns:
            query_labels[label] = query_metadata.get(f'label_{label}', 0)

        # Calculate agreement with similar passages
        for label in label_columns:
            if label not in query_labels:
                consistency[label] = 0.0
                continue

            query_value = query_labels[label]
            agreements = 0

            for passage in similar_passages:
                passage_value = passage['metadata'].get(f'label_{label}', 0)
                if passage_value == query_value:
                    agreements += 1

            consistency[label] = agreements / len(similar_passages) if similar_passages else 0.0

        return consistency

    def rerank_passages_for_label(self,
                                  passages: List[str],
                                  label: str,
                                  batch_size: int = 100) -> List[float]:
        """
        Use VoyageAI rerank-2.5 to score passage relevance for a specific label

        Args:
            passages: List of passage texts
            label: Label name (e.g., 'EVENT_Illness')
            batch_size: Max passages per rerank call

        Returns:
            List of relevance scores (0-1)
        """
        if label not in self.LABEL_QUERIES:
            raise ValueError(f"Unknown label: {label}")

        query = self.LABEL_QUERIES[label]
        all_scores = []

        # Process in batches (rerank-2.5 has limits)
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i + batch_size]

            result = self.voyage.rerank(
                query=query,
                documents=batch,
                model="rerank-2.5",
                top_k=len(batch)  # Return all with scores
            )

            # Extract relevance scores
            scores = [doc.relevance_score for doc in result.results]
            all_scores.extend(scores)

            # Rate limiting
            time.sleep(0.1)

        return all_scores

    def identify_golden_dataset(self,
                                df: pd.DataFrame,
                                passage_column: str = 'Passage',
                                label_columns: List[str] = None,
                                min_consistency: float = 0.7,
                                min_rerank_score: float = 0.6,
                                target_size: int = 1000,
                                k_similar: int = 20,
                                namespace: str = "main",
                                recompute_embeddings: bool = False) -> pd.DataFrame:
        """
        Identify golden dataset passages with high label confidence

        Args:
            df: DataFrame with passages and labels
            passage_column: Column name for passage text
            label_columns: List of label columns (auto-detect if None)
            min_consistency: Minimum consistency score for similar passages
            min_rerank_score: Minimum reranker relevance score
            target_size: Target number of golden passages
            k_similar: Number of similar passages to check
            namespace: Pinecone namespace
            recompute_embeddings: Whether to recompute and store embeddings

        Returns:
            DataFrame with golden passages and confidence scores
        """
        # Auto-detect label columns if not provided
        if label_columns is None:
            label_columns = self._auto_detect_label_columns(df)

        if not label_columns:
            raise ValueError(
                "No label columns detected! Please specify label_columns explicitly."
            )

        print(f"Found {len(label_columns)} labels to evaluate: {label_columns}")

        # Step 1: Embed and store if needed
        if recompute_embeddings:
            print("\n📊 Step 1: Generating and storing embeddings...")
            passage_id_map = self.embed_and_store_passages(
                df=df,
                passage_column=passage_column,
                label_columns=label_columns,
                namespace=namespace
            )
            embedded_indices = list(passage_id_map.keys())
        else:
            print("\n📊 Step 1: Using existing embeddings from Pinecone")
            valid_mask = df[passage_column].notna()
            embedded_indices = df[valid_mask].index.tolist()

        print(f"   Processing {len(embedded_indices)} embedded passages")

        # Step 2: Calculate consistency scores
        print("\n🔍 Step 2: Calculating label consistency from similar passages...")
        consistency_scores = {}

        for idx in tqdm(embedded_indices, desc="Checking consistency"):
            try:
                similar = self.find_similar_passages(
                    query_idx=idx,
                    k=k_similar,
                    namespace=namespace
                )

                consistency = self.calculate_label_consistency(
                    query_idx=idx,
                    similar_passages=similar,
                    label_columns=label_columns,
                    namespace=namespace
                )

                # Get active labels for this passage
                passage_labels = [label for label in label_columns
                                  if df.loc[idx, label] == 1]

                if passage_labels:
                    avg_consistency = np.mean([consistency[label]
                                               for label in passage_labels])
                else:
                    avg_consistency = 0.0

                consistency_scores[idx] = avg_consistency
            except Exception as e:
                print(f"Error processing passage {idx}: {e}")
                consistency_scores[idx] = 0.0

        # Step 3: Rerank for label relevance
        print("\n🎯 Step 3: Reranking passages for label relevance...")
        rerank_scores = {label: {} for label in label_columns}

        passages = df[passage_column].tolist()

        for label in tqdm(label_columns, desc="Reranking labels"):
            label_mask = df[label] == 1
            label_indices = [idx for idx in embedded_indices if idx in df.index and df.loc[idx, label] == 1]
            label_passages = [passages[idx] for idx in label_indices]

            if label_passages:
                scores = self.rerank_passages_for_label(label_passages, label)

                for idx, score in zip(label_indices, scores):
                    rerank_scores[label][idx] = score

        # Step 4: Combine scores and select golden passages
        print("\n✨ Step 4: Selecting golden passages...")

        golden_candidates = []
        for idx in embedded_indices:
            active_labels = [label for label in label_columns
                             if idx in df.index and df.loc[idx, label] == 1]

            if not active_labels:
                continue

            consistency_score = consistency_scores.get(idx, 0.0)
            rerank_values = [rerank_scores[label].get(idx, 0.0) for label in active_labels]
            avg_rerank = np.mean(rerank_values) if rerank_values else 0.0

            # Weighted composite: 50% consistency, 50% rerank
            composite_score = 0.5 * consistency_score + 0.5 * avg_rerank

            if consistency_score >= min_consistency and avg_rerank >= min_rerank_score:
                golden_candidates.append({
                    'passage_idx': idx,
                    'consistency_score': consistency_score,
                    'rerank_score': avg_rerank,
                    'composite_score': composite_score,
                    'num_labels': len(active_labels)
                })

        # Sort by composite score and take top target_size
        golden_candidates.sort(key=lambda x: x['composite_score'], reverse=True)
        golden_candidates = golden_candidates[:target_size]

        if not golden_candidates:
            print(
                f"\n⚠️  No passages met the criteria (consistency >= {min_consistency}, rerank >= {min_rerank_score})")
            print("   Consider lowering thresholds or checking data quality")
            return pd.DataFrame(
                columns=list(df.columns) + ['confidence_consistency', 'confidence_rerank', 'confidence_composite'])

        # Build golden dataset
        golden_indices = [c['passage_idx'] for c in golden_candidates]
        golden_df = df.loc[golden_indices].copy()

        # Add confidence metrics
        for candidate in golden_candidates:
            idx = candidate['passage_idx']
            golden_df.loc[idx, 'confidence_consistency'] = candidate['consistency_score']
            golden_df.loc[idx, 'confidence_rerank'] = candidate['rerank_score']
            golden_df.loc[idx, 'confidence_composite'] = candidate['composite_score']

        print(f"\n✅ Identified {len(golden_df)} golden passages")
        print(f"   Mean consistency: {golden_df['confidence_consistency'].mean():.3f}")
        print(f"   Mean rerank score: {golden_df['confidence_rerank'].mean():.3f}")

        return golden_df


# Example usage
if __name__ == "__main__":
    # Initialize
    finder = GoldenDatasetFinder(
        voyage_api_key=os.getenv("VOYAGE_API_KEY"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        index_name="hraf-misfortune",
        cloud="aws",
        region="us-east-1"
    )

    # Load data
    df = pd.read_excel("sample_dataset.xlsx")

    # Identify golden dataset
    golden_df = finder.identify_golden_dataset(
        df=df,
        passage_column='Passage',
        min_consistency=0.7,
        min_rerank_score=0.6,
        target_size=200,
        recompute_embeddings=True
    )

    # Save results
    golden_df.to_excel("golden_dataset_sample.xlsx", index=False)
    print("\n✅ Golden dataset saved!")