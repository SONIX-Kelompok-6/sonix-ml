"""
Collaborative Filtering Module
------------------------------
Provides a User-Based Collaborative Filtering (UBCF) engine using NearestNeighbors.
Refactored for low cyclomatic complexity, early returns, and optimized payload (IDs only).
"""

import logging
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class UserCollaborativeRecommender:
    """
    User-Based Collaborative Filtering (UBCF) Engine.
    Evaluates real-time buffer and NearestNeighbors baseline to return top shoe IDs.
    """

    def __init__(self, df_interactions: pd.DataFrame, shoe_metadata: pd.DataFrame):
        self.cache: Dict[int, Tuple[List[str], float]] = {} 
        self.cache_ttl = 60 
        self.live_changes: Dict[int, Dict[str, float]] = {} 
        
        self.user_map: Dict[int, int] = {}
        self.item_map: Dict[str, int] = {}
        self.user_ids: List[int] = []
        self.item_ids: List[str] = []
        self.sparse_matrix: Optional[csr_matrix] = None
        self.model: Optional[NearestNeighbors] = None

        if df_interactions.empty:
            logger.warning("CF Engine: Passive Mode.")
            return

        self._initialize_matrix(df_interactions)

    def _initialize_matrix(self, df: pd.DataFrame) -> None:
        """Builds the sparse matrix and fits the NearestNeighbors model."""
        pivot_df = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        
        self.user_ids = list(pivot_df.index)
        self.item_ids = list(pivot_df.columns)
        self.user_map = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_map = {iid: i for i, iid in enumerate(self.item_ids)}
        
        self.sparse_matrix = csr_matrix(pivot_df.values)
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(self.sparse_matrix)
        
        logger.info(f"CF Engine Initialized: {len(self.user_ids)} Users.")

    def _convert_rating(self, rating: Optional[int], is_like_action: bool) -> float:
        """Converts user action into a numerical weight."""
        if is_like_action: 
            return 1.0
        mapping = {1: -2.0, 2: -1.0, 3: 0.1, 4: 1.0, 5: 2.0}
        return mapping.get(rating, 0.1) if rating else 0.1

    def _apply_live_buffer(self, user_id: int, user_vector: np.ndarray) -> np.ndarray:
        """Injects recent un-trained interactions into the user's vector."""
        if user_id not in self.live_changes: 
            return user_vector
            
        for item_id, rating in self.live_changes[user_id].items():
            if item_id in self.item_map:
                user_vector[0, self.item_map[item_id]] = rating
        return user_vector

    def _calculate_neighbor_scores(self, user_id: int, distances: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
        """Calculates similarity scores for recommended items based on neighbor vectors."""
        rec_scores: Dict[str, float] = {}
        for i, neighbor_idx in enumerate(indices[0]):
            if user_id in self.user_map and neighbor_idx == self.user_map[user_id]: 
                continue
                
            similarity = 1.0 - distances[0][i]
            if similarity <= 0: 
                continue

            neighbor_vector = self.sparse_matrix[neighbor_idx].toarray()[0]
            liked_indices = np.where(neighbor_vector > 0)[0]
            
            for it_idx in liked_indices:
                score = neighbor_vector[it_idx] * similarity
                it_id = self.item_ids[it_idx]
                rec_scores[it_id] = rec_scores.get(it_id, 0.0) + score
                
        return rec_scores

    def get_realtime_recommendations(self, user_id: int, new_item_id: Optional[str] = None, 
                                     new_rating_val: Optional[int] = None, is_like: bool = False,
                                     n_neighbors: int = 10) -> List[str]:
        """
        Main facade to retrieve recommendations. Handles caching, buffering, and inference.
        """
        current_time = time.time()
        
        # 1. Update Live Buffer
        if new_item_id and new_item_id in self.item_map:
            score_val = self._convert_rating(new_rating_val, is_like)
            self.live_changes.setdefault(user_id, {})[new_item_id] = score_val
            self.cache.pop(user_id, None)

        # 2. Check Cache
        if not new_item_id and user_id in self.cache:
            result, timestamp = self.cache[user_id]
            if current_time - timestamp < self.cache_ttl:
                return result

        # 3. Guards
        if self.sparse_matrix is None or self.model is None: 
            return []

        # 4. Vector Construction
        num_items = len(self.item_ids)
        if user_id in self.user_map:
            user_vector = self.sparse_matrix[self.user_map[user_id]].toarray().copy()
        else:
            user_vector = np.zeros((1, num_items))
            
        user_vector = self._apply_live_buffer(user_id, user_vector)

        if np.all(user_vector == 0): 
            return []

        # 5. KNN Inference
        effective_k = min(n_neighbors + 1, len(self.user_ids))
        try:
            distances, indices = self.model.kneighbors(user_vector, n_neighbors=effective_k)
        except Exception:
            return []

        # 6. Scoring & Filtering
        rec_scores = self._calculate_neighbor_scores(user_id, distances, indices)
        if not rec_scores: 
            return []
        
        seen_indices = np.where(user_vector[0] != 0)[0]
        seen_items = {self.item_ids[i] for i in seen_indices}

        candidates = [
            {'shoe_id': k, 'cf_score': v} 
            for k, v in rec_scores.items() if k not in seen_items
        ]
        
        if not candidates: 
            return []

        # 7. Extract IDs Only
        candidates_df = pd.DataFrame(candidates).sort_values('cf_score', ascending=False).head(20)
        final_results = candidates_df['shoe_id'].tolist()

        if not new_item_id:
            self.cache[user_id] = (final_results, current_time)

        return final_results