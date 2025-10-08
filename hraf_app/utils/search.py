"""
Smart search utilities for finding relevant passages
Combines semantic search, keyword search, and quality filtering
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import voyageai
from pinecone import Pinecone
import os
from collections import defaultdict
import re


class SmartSearch:
    """
    Unified search interface combining multiple search strategies
    """

    def __init__(
            self,
            df: pd.DataFrame,
            passage_col: str,
            label_columns: List[str],
            quality_scores: Optional[Dict] = None,
            use_embeddings: bool = True
    ):
        self.df = df
        self.passage_col = passage_col
        self.label_columns = label_columns
        self.quality_scores = quality_scores

        # Initialize embedding-based search if enabled
        self.use_embeddings = use_embeddings
        if use_embeddings:
            self._init_embedding_search()

        # Build keyword index
        self._build_keyword_index()

    def _init_embedding_search(self):
        """Initialize embedding-based search"""
        try:
            self.voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
            self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

            # Try to connect to existing index
            index_name = "hraf-misfortune"
            if index_name in [idx.name for idx in self.pc.list_indexes()]:
                index_info = self.pc.describe_index(index_name)
                self.index = self.pc.Index(name=index_name, host=index_info.host)
                print(f"✅ Connected to Pinecone index: {index_name}")
            else:
                self.index = None
                print("⚠️ Pinecone index not found. Semantic search disabled.")
        except Exception as e:
            print(f"⚠️ Could not initialize embedding search: {e}")
            self.use_embeddings = False

    def _build_keyword_index(self):
        """Build inverted index for keyword search"""
        self.keyword_index = defaultdict(list)

        for idx in self.df.index:
            text = str(self.df.loc[idx, self.passage_col]).lower()
            # Extract words (simple tokenization)
            words = re.findall(r'\b\w+\b', text)

            for word in set(words):  # Unique words per passage
                self.keyword_index[word].append(idx)

    def search(
            self,
            query: str,
            search_type: str = "hybrid",
            filters: Optional[Dict] = None,
            top_k: int = 20,
            min_quality: float = 0.0
    ) -> List[Dict]:
        """
        Unified search interface

        Args:
            query: Search query
            search_type: "semantic", "keyword", or "hybrid"
            filters: Optional filters (labels, quality, etc.)
            top_k: Number of results
            min_quality: Minimum quality threshold

        Returns:
            List of search results with scores
        """
        if search_type == "semantic":
            results = self._semantic_search(query, top_k * 2)
        elif search_type == "keyword":
            results = self._keyword_search(query, top_k * 2)
        else:  # hybrid
            results = self._hybrid_search(query, top_k * 2)

        # Apply filters
        if filters:
            results = self._apply_filters(results, filters)

        # Apply quality threshold
        if min_quality > 0 and self.quality_scores:
            results = [
                r for r in results
                if self.quality_scores.get(r['idx'], type('obj', (object,),
                                                          {'overall_quality': 0})()).overall_quality >= min_quality
            ]

        # Add quality scores if available
        if self.quality_scores:
            for result in results:
                idx = result['idx']
                if idx in self.quality_scores:
                    result['quality'] = self.quality_scores[idx].overall_quality
                    result['quality_tier'] = self.quality_scores[idx].tier

        return results[:top_k]

    def _semantic_search(self, query: str, k: int) -> List[Dict]:
        """Semantic search using embeddings"""
        if not self.use_embeddings or not self.index:
            print("⚠️ Semantic search not available, falling back to keyword search")
            return self._keyword_search(query, k)

        try:
            # Embed query
            query_result = self.voyage.embed(
                texts=[query],
                model="voyage-3-large",
                input_type="query"
            )
            query_vector = query_result.embeddings[0]

            # Search Pinecone
            search_results = self.index.query(
                vector=query_vector,
                top_k=k,
                include_metadata=True
            )

            # Format results
            results = []
            matches = search_results.matches if hasattr(search_results, 'matches') else search_results.get('matches',
                                                                                                           [])

            for match in matches:
                idx = match.metadata['passage_idx'] if hasattr(match, 'metadata') else match['metadata']['passage_idx']
                score = match.score if hasattr(match, 'score') else match['score']

                if idx in self.df.index:
                    results.append({
                        'idx': idx,
                        'score': float(score),
                        'search_type': 'semantic',
                        'text': str(self.df.loc[idx, self.passage_col])[:200]
                    })

            return results

        except Exception as e:
            print(f"⚠️ Semantic search failed: {e}")
            return self._keyword_search(query, k)

    def _keyword_search(self, query: str, k: int) -> List[Dict]:
        """Keyword-based search using BM25-like scoring"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))

        # Calculate scores for each passage
        passage_scores = defaultdict(float)

        for word in query_words:
            if word in self.keyword_index:
                # IDF-like weighting
                idf = np.log((len(self.df) + 1) / (len(self.keyword_index[word]) + 1))

                for idx in self.keyword_index[word]:
                    passage_scores[idx] += idf

        # Sort by score
        ranked = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)

        # Format results
        results = []
        for idx, score in ranked[:k]:
            if idx in self.df.index:
                results.append({
                    'idx': idx,
                    'score': float(score),
                    'search_type': 'keyword',
                    'text': str(self.df.loc[idx, self.passage_col])[:200]
                })

        return results

    def _hybrid_search(self, query: str, k: int) -> List[Dict]:
        """Hybrid search combining semantic and keyword"""
        # Get results from both methods
        semantic_results = self._semantic_search(query, k)
        keyword_results = self._keyword_search(query, k)

        # Combine with weighted scores
        combined_scores = {}

        # Normalize and combine scores
        sem_max = max([r['score'] for r in semantic_results], default=1.0)
        kw_max = max([r['score'] for r in keyword_results], default=1.0)

        for result in semantic_results:
            idx = result['idx']
            normalized_score = result['score'] / sem_max if sem_max > 0 else 0
            combined_scores[idx] = 0.7 * normalized_score  # 70% weight

        for result in keyword_results:
            idx = result['idx']
            normalized_score = result['score'] / kw_max if kw_max > 0 else 0
            combined_scores[idx] = combined_scores.get(idx, 0) + 0.3 * normalized_score  # 30% weight

        # Sort by combined score
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Format results
        results = []
        for idx, score in ranked[:k]:
            if idx in self.df.index:
                results.append({
                    'idx': idx,
                    'score': float(score),
                    'search_type': 'hybrid',
                    'text': str(self.df.loc[idx, self.passage_col])[:200]
                })

        return results

    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Apply filters to search results"""
        filtered = []

        for result in results:
            idx = result['idx']

            # Label filters
            if 'required_labels' in filters:
                if not all(self.df.loc[idx, label] == 1 for label in filters['required_labels']):
                    continue

            if 'excluded_labels' in filters:
                if any(self.df.loc[idx, label] == 1 for label in filters['excluded_labels']):
                    continue

            # Label count filters
            if 'min_labels' in filters or 'max_labels' in filters:
                label_count = sum(self.df.loc[idx, label] == 1 for label in self.label_columns)

                if 'min_labels' in filters and label_count < filters['min_labels']:
                    continue
                if 'max_labels' in filters and label_count > filters['max_labels']:
                    continue

            filtered.append(result)

        return filtered

    def find_similar(
            self,
            passage_idx: int,
            k: int = 20,
            same_labels_only: bool = False
    ) -> List[Dict]:
        """
        Find passages similar to a given passage

        Args:
            passage_idx: Index of reference passage
            k: Number of similar passages to return
            same_labels_only: Only return passages with same labels

        Returns:
            List of similar passages with similarity scores
        """
        if not self.use_embeddings or not self.index:
            return self._find_similar_keyword(passage_idx, k)

        try:
            # Get reference passage vector
            passage_id = f"passage_{passage_idx}"
            fetch_result = self.index.fetch(ids=[passage_id])

            # Handle both dict and object responses
            if hasattr(fetch_result, 'vectors'):
                vectors_dict = fetch_result.vectors or {}
            else:
                vectors_dict = fetch_result.get('vectors', {})

            if passage_id not in vectors_dict:
                print(f"⚠️ Passage {passage_idx} not in index")
                return self._find_similar_keyword(passage_idx, k)

            # Get vector
            vector_data = vectors_dict[passage_id]
            query_vector = vector_data.values if hasattr(vector_data, 'values') else vector_data['values']

            # Build filter if needed
            filter_dict = None
            if same_labels_only:
                # Get labels of reference passage
                active_labels = [
                    label for label in self.label_columns
                    if self.df.loc[passage_idx, label] == 1
                ]

                if active_labels:
                    # This is complex in Pinecone - would need all labels to match
                    # For simplicity, filter in post-processing
                    pass

            # Search
            search_results = self.index.query(
                vector=query_vector,
                top_k=k + 1,  # +1 to exclude self
                include_metadata=True,
                filter=filter_dict
            )

            # Format results
            results = []
            matches = search_results.matches if hasattr(search_results, 'matches') else search_results.get('matches',
                                                                                                           [])

            for match in matches:
                idx = match.metadata['passage_idx'] if hasattr(match, 'metadata') else match['metadata']['passage_idx']

                # Skip self
                if idx == passage_idx:
                    continue

                if idx not in self.df.index:
                    continue

                # Check label similarity if required
                if same_labels_only:
                    ref_labels = set(
                        label for label in self.label_columns
                        if self.df.loc[passage_idx, label] == 1
                    )
                    match_labels = set(
                        label for label in self.label_columns
                        if self.df.loc[idx, label] == 1
                    )

                    if ref_labels != match_labels:
                        continue

                score = match.score if hasattr(match, 'score') else match['score']

                results.append({
                    'idx': idx,
                    'similarity': float(score),
                    'text': str(self.df.loc[idx, self.passage_col])[:200]
                })

            return results[:k]

        except Exception as e:
            print(f"⚠️ Similar search failed: {e}")
            return self._find_similar_keyword(passage_idx, k)

    def _find_similar_keyword(self, passage_idx: int, k: int) -> List[Dict]:
        """Find similar passages using keyword overlap (fallback)"""
        if passage_idx not in self.df.index:
            return []

        ref_text = str(self.df.loc[passage_idx, self.passage_col]).lower()
        ref_words = set(re.findall(r'\b\w+\b', ref_text))

        # Calculate Jaccard similarity with other passages
        similarities = {}

        for idx in self.df.index:
            if idx == passage_idx:
                continue

            text = str(self.df.loc[idx, self.passage_col]).lower()
            words = set(re.findall(r'\b\w+\b', text))

            # Jaccard similarity
            intersection = len(ref_words & words)
            union = len(ref_words | words)

            if union > 0:
                similarities[idx] = intersection / union

        # Sort by similarity
        ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Format results
        results = []
        for idx, sim in ranked[:k]:
            results.append({
                'idx': idx,
                'similarity': float(sim),
                'text': str(self.df.loc[idx, self.passage_col])[:200]
            })

        return results

    def search_by_label(
            self,
            label: str,
            min_quality: float = 0.0,
            top_k: int = 50
    ) -> List[Dict]:
        """
        Find best examples of a specific label

        Combines label presence with quality scores
        """
        if label not in self.label_columns:
            raise ValueError(f"Label '{label}' not found")

        # Get passages with this label
        mask = self.df[label] == 1
        labeled_indices = self.df[mask].index.tolist()

        # Score by quality if available
        if self.quality_scores:
            scored = [
                (idx, self.quality_scores[idx].overall_quality)
                for idx in labeled_indices
                if idx in self.quality_scores
                   and self.quality_scores[idx].overall_quality >= min_quality
            ]

            # Sort by quality
            scored.sort(key=lambda x: x[1], reverse=True)

            results = []
            for idx, quality in scored[:top_k]:
                results.append({
                    'idx': idx,
                    'quality': quality,
                    'text': str(self.df.loc[idx, self.passage_col])[:200],
                    'label': label
                })
        else:
            # Just return labeled passages
            results = [
                {
                    'idx': idx,
                    'text': str(self.df.loc[idx, self.passage_col])[:200],
                    'label': label
                }
                for idx in labeled_indices[:top_k]
            ]

        return results

    def get_passage_details(self, idx: int) -> Dict:
        """Get full details for a passage"""
        if idx not in self.df.index:
            return None

        details = {
            'idx': idx,
            'text': str(self.df.loc[idx, self.passage_col]),
            'labels': {
                label: int(self.df.loc[idx, label])
                for label in self.label_columns
            }
        }

        # Add quality if available
        if self.quality_scores and idx in self.quality_scores:
            quality = self.quality_scores[idx]
            details['quality'] = {
                'overall': quality.overall_quality,
                'tier': quality.tier,
                'semantic_consistency': quality.semantic_consistency,
                'label_confidence': quality.label_confidence
            }

        return details


class LabelSearcher:
    """Specialized search for label-specific analysis"""

    def __init__(self, smart_search: SmartSearch):
        self.search = smart_search

    def find_ambiguous_passages(
            self,
            label: str,
            threshold: float = 0.55
    ) -> List[Dict]:
        """
        Find passages with this label that have low quality
        (ambiguous or potentially mislabeled)
        """
        results = self.search.search_by_label(label, min_quality=0.0)

        if not self.search.quality_scores:
            return []

        ambiguous = [
            r for r in results
            if r.get('quality', 1.0) < threshold
        ]

        return ambiguous

    def find_high_quality_examples(
            self,
            label: str,
            min_quality: float = 0.75,
            n: int = 20
    ) -> List[Dict]:
        """Find best training examples for a label"""
        return self.search.search_by_label(label, min_quality=min_quality, top_k=n)

    def compare_labels(
            self,
            label1: str,
            label2: str,
            n: int = 10
    ) -> Dict:
        """
        Compare two labels by finding passages that:
        - Have only label1
        - Have only label2
        - Have both
        - Have neither (but similar context)
        """
        df = self.search.df

        comparison = {
            'only_label1': [],
            'only_label2': [],
            'both': [],
            'neither': []
        }

        for idx in df.index:
            has_label1 = df.loc[idx, label1] == 1
            has_label2 = df.loc[idx, label2] == 1

            if has_label1 and not has_label2:
                comparison['only_label1'].append(idx)
            elif has_label2 and not has_label1:
                comparison['only_label2'].append(idx)
            elif has_label1 and has_label2:
                comparison['both'].append(idx)

        # Limit to n examples each
        for key in comparison:
            comparison[key] = comparison[key][:n]

        return comparison