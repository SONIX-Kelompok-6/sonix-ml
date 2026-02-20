import logging
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

class UserCollaborativeRecommender:
    """
    User-Based Collaborative Filtering (UBCF) Engine.
    Refactored using Extract Method to minimize CC without altering logic.
    """

    def __init__(self, df_interactions: pd.DataFrame, shoe_metadata: pd.DataFrame):
        self.cache: Dict[int, Tuple[List[Any], float]] = {} 
        self.cache_ttl = 60 
        self.shoe_metadata = shoe_metadata
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

        self._build_matrix(df_interactions)

    def _build_matrix(self, df_interactions: pd.DataFrame) -> None:
        self.pivot_df = df_interactions.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        self.user_ids = list(self.pivot_df.index)
        self.item_ids = list(self.pivot_df.columns)
        self.user_map = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_map = {iid: i for i, iid in enumerate(self.item_ids)}
        
        self.sparse_matrix = csr_matrix(self.pivot_df.values)
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(self.sparse_matrix)
        logger.info(f"CF Engine Initialized: {len(self.user_ids)} Users.")

    def _convert_rating(self, rating: Optional[int], is_like_action: bool = False) -> float:
        if is_like_action: return 1.0
        if rating is None or rating == 0: return 0.1
        mapping = {1: -2.0, 2: -1.0, 3: 0.1, 4: 1.0, 5: 2.0}
        return mapping.get(rating, 0.1)

    def _update_buffer_and_cache(self, user_id: int, new_item_id: Optional[str], 
                                 new_rating_val: Optional[int], is_like: bool, current_time: float) -> Optional[List[Any]]:
        if new_item_id and new_item_id in self.item_map:
            score_val = self._convert_rating(new_rating_val, is_like_action=is_like)
            self.live_changes.setdefault(user_id, {})[new_item_id] = score_val
            self.cache.pop(user_id, None)

        if new_item_id is None and user_id in self.cache:
            result, timestamp = self.cache[user_id]
            if current_time - timestamp < self.cache_ttl:
                return result
        return None

    def _build_user_vector(self, user_id: int) -> np.ndarray:
        num_items = len(self.item_ids)
        if user_id in self.user_map:
            user_idx = self.user_map[user_id]
            user_vector = self.sparse_matrix[user_idx].toarray()
        else:
            user_vector = np.zeros((1, num_items))

        if user_id in self.live_changes:
            for item_id, rating in self.live_changes[user_id].items():
                if item_id in self.item_map:
                    idx = self.item_map[item_id]
                    user_vector[0, idx] = rating
        return user_vector

    def _compute_scores(self, user_id: int, distances: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
        rec_scores: Dict[str, float] = {}
        for i, neighbor_idx in enumerate(indices[0]):
            if user_id in self.user_map and neighbor_idx == self.user_map[user_id]: continue
            similarity = 1.0 - distances[0][i]
            if similarity <= 0: continue

            neighbor_vector = self.sparse_matrix[neighbor_idx].toarray()[0]
            liked_indices = np.where(neighbor_vector > 0)[0]
            
            for it_idx in liked_indices:
                score = neighbor_vector[it_idx] * similarity
                it_id = self.item_ids[it_idx]
                rec_scores[it_id] = rec_scores.get(it_id, 0.0) + score
        return rec_scores

    def get_realtime_recommendations(self, user_id: int, new_item_id: Optional[str] = None, 
                                     new_rating_val: Optional[int] = None, is_like: bool = False,
                                     n_neighbors: int = 10) -> List[Any]:
        current_time = time.time()
        
        cached_result = self._update_buffer_and_cache(user_id, new_item_id, new_rating_val, is_like, current_time)
        if cached_result is not None: return cached_result
        if self.sparse_matrix is None or self.model is None: return []

        user_vector = self._build_user_vector(user_id)
        if np.all(user_vector == 0): return []

        effective_k = min(n_neighbors + 1, len(self.user_ids))
        try:
            distances, indices = self.model.kneighbors(user_vector, n_neighbors=effective_k)
        except Exception:
            return []

        rec_scores = self._compute_scores(user_id, distances, indices)
        if not rec_scores: return []
        
        seen_indices = np.where(user_vector[0] != 0)[0]
        seen_items = {self.item_ids[i] for i in seen_indices}

        candidates = [
            {'shoe_id': k, 'cf_score': v} 
            for k, v in rec_scores.items() 
            if k not in seen_items
        ]
        if not candidates: return []

        candidates_df = pd.DataFrame(candidates).sort_values('cf_score', ascending=False).head(20)
        final_results = candidates_df['shoe_id'].tolist()

        if new_item_id is None:
            self.cache[user_id] = (final_results, current_time)

        return final_results