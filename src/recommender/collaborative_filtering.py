import logging
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Optional, Any, Tuple

# Configure logger
logger = logging.getLogger(__name__)

class UserCollaborativeRecommender:
    """
    User-Based Collaborative Filtering (UBCF) Engine.
    Optimized for 'IDs Only' return strategy.
    """

    def __init__(self, df_interactions: pd.DataFrame, shoe_metadata: pd.DataFrame):
        self.cache: Dict[int, Tuple[List[Any], float]] = {} 
        self.cache_ttl = 60 
        self.shoe_metadata = shoe_metadata

        # Buffer for Real-time changes
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

        # 1. Pivot & Matrix Build
        self.pivot_df = df_interactions.pivot(
            index='user_id', columns='item_id', values='rating'
        ).fillna(0)
        
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

    def get_realtime_recommendations(self, 
                                     user_id: int, 
                                     new_item_id: Optional[str] = None, 
                                     new_rating_val: Optional[int] = None,
                                     is_like: bool = False,
                                     n_neighbors: int = 10) -> List[Any]:
        """
        Returns: List of Shoe IDs (strings/ints) only.
        """
        current_time = time.time()
        
        # 1. Update Buffer
        if new_item_id and new_item_id in self.item_map:
            score_val = self._convert_rating(new_rating_val, is_like_action=is_like)
            if user_id not in self.live_changes: self.live_changes[user_id] = {}
            self.live_changes[user_id][new_item_id] = score_val
            if user_id in self.cache: del self.cache[user_id]

        # 2. Cache Check
        if new_item_id is None and user_id in self.cache:
            result, timestamp = self.cache[user_id]
            if current_time - timestamp < self.cache_ttl:
                return result

        # 3. Guards
        if self.sparse_matrix is None or self.model is None: return []

        # 4. Build User Vector
        num_items = len(self.item_ids)
        if user_id in self.user_map:
            user_idx = self.user_map[user_id]
            user_vector = self.sparse_matrix[user_idx].toarray()
        else:
            user_vector = np.zeros((1, num_items))

        # 5. Apply Live Buffer
        if user_id in self.live_changes:
            for item_id, rating in self.live_changes[user_id].items():
                if item_id in self.item_map:
                    idx = self.item_map[item_id]
                    user_vector[0, idx] = rating

        if np.all(user_vector == 0): return []

        # 6. Find Neighbors
        total_users = len(self.user_ids)
        effective_k = min(n_neighbors + 1, total_users)
        try:
            distances, indices = self.model.kneighbors(user_vector, n_neighbors=effective_k)
        except Exception:
            return []

        # 7. Calculate Scores
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

        # 8. Filter & Sort
        if not rec_scores: return []
        
        seen_indices = np.where(user_vector[0] != 0)[0]
        seen_items = {self.item_ids[i] for i in seen_indices}

        candidates = [
            {'shoe_id': k, 'cf_score': v} 
            for k, v in rec_scores.items() 
            if k not in seen_items
        ]
        
        if not candidates: return []

        candidates_df = pd.DataFrame(candidates)
        candidates_df = candidates_df.sort_values('cf_score', ascending=False).head(20)

        # 9. Extract IDs Only (Optimization)
        # We don't need full metadata enrichment anymore, just the IDs sorted by score
        final_results = candidates_df['shoe_id'].tolist()

        # 10. Save to Cache
        if new_item_id is None:
            self.cache[user_id] = (final_results, current_time)

        return final_results