import logging
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Optional, Any, Set, cast, Tuple

# Configure logger
logger = logging.getLogger(__name__)

class UserCollaborativeRecommender:
    """
    User-Based Collaborative Filtering (UBCF) Engine for Sonix.
    Includes Micro-Caching (TTL) to reduce 95%ile latency spikes.
    """

    def __init__(self, df_interactions: pd.DataFrame, shoe_metadata: pd.DataFrame):
        """
        Initializes the engine, builds the matrix, and fits the k-NN model.
        """
        self.cache: Dict[int, Tuple[List[Dict[str, Any]], float]] = {} 
        self.cache_ttl = 60  # Cache valid for 60 seconds
        self.shoe_metadata = shoe_metadata
        
        # --- Type Safe Initialization ---
        self.user_map: Dict[int, int] = {}
        self.item_map: Dict[str, int] = {}
        self.user_ids: List[int] = []
        self.item_ids: List[str] = []
        self.sparse_matrix: Optional[csr_matrix] = None
        self.model: Optional[NearestNeighbors] = None

        if df_interactions.empty:
            logger.warning("CF Engine: No interaction data found. Passive Mode.")
            return

        # 1. Pivot Data
        # Fillna(0) ensures we have a dense-compatible structure for sparse conversion
        self.pivot_df = df_interactions.pivot(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        # 2. Index Mapping
        self.user_ids = list(self.pivot_df.index)
        self.item_ids = list(self.pivot_df.columns)
        self.user_map = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_map = {iid: i for i, iid in enumerate(self.item_ids)}
        
        # 3. Model Fitting
        # CSR Matrix is used for memory efficiency
        self.sparse_matrix = csr_matrix(self.pivot_df.values)
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(self.sparse_matrix)
        
        logger.info(f"CF Engine Initialized: {len(self.user_ids)} Users, {len(self.item_ids)} Items.")

    def _convert_rating(self, rating: Optional[int], is_like_action: bool = False) -> float:
        """Maps raw signals to symmetric preference scale (-2.0 to +2.0)."""
        if is_like_action: return 1.0
        if rating is None or rating == 0: return 0.1
        # 1-2 (Negative), 3 (Neutral), 4-5 (Positive)
        mapping = {1: -2.0, 2: -1.0, 3: 0.1, 4: 1.0, 5: 2.0}
        return mapping.get(rating, 0.1)

    def get_realtime_recommendations(self, 
                                     user_id: int, 
                                     new_item_id: Optional[str] = None, 
                                     new_rating_val: Optional[int] = None,
                                     is_like: bool = False,
                                     n_neighbors: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves recommendations with Real-Time Injection and TTL Caching.
        """
        current_time = time.time()
        
        # --- 1. CACHE CHECK (Micro-Caching Strategy) ---
        if user_id in self.cache:
            result, timestamp = self.cache[user_id]
            # If cache is fresh (< 60s) and no new interaction forces an update
            if (current_time - timestamp < self.cache_ttl) and (new_item_id is None):
                return result

        # --- 2. INITIALIZATION GUARDS ---
        num_items = len(self.item_ids)
        if num_items == 0: return []
        
        # Pylance Guard: Ensure model exists
        if self.sparse_matrix is None or self.model is None:
            return []

        # --- 3. BUILD USER VECTOR ---
        # If user exists, copy their history. If new, start with zeros.
        if user_id in self.user_map:
            user_idx = self.user_map[user_id]
            # .toarray() returns a copy, so we can modify it safely
            user_vector = self.sparse_matrix[user_idx].toarray()
        else:
            user_vector = np.zeros((1, num_items))

        # --- 4. REAL-TIME INJECTION ---
        # Inject the latest interaction into the vector before finding neighbors
        if new_item_id and new_item_id in self.item_map:
            item_idx = self.item_map[new_item_id]
            score_val = self._convert_rating(new_rating_val, is_like_action=is_like)
            user_vector[0, item_idx] = score_val

        # If vector is empty (Cold Start User with no interaction), return empty
        if np.all(user_vector == 0): return []

        # --- 5. FIND NEIGHBORS ---
        total_users = len(self.user_ids)
        if total_users <= 1: return []

        # Request k+1 neighbors because the user themselves might be returned as result 1
        effective_k = min(n_neighbors + 1, total_users)

        try:
            distances, indices = self.model.kneighbors(user_vector, n_neighbors=effective_k)
        except Exception as e:
            logger.error(f"CF Neighbor search failed: {e}")
            return []

        # --- 6. CALCULATE SCORES ---
        rec_scores: Dict[str, float] = {}
        
        # indices[0] contains the index of neighbors in self.user_ids
        for i, neighbor_idx in enumerate(indices[0]):
            # Skip self if present
            if user_id in self.user_map and neighbor_idx == self.user_map[user_id]:
                continue
                
            similarity = 1.0 - distances[0][i]  # Convert distance to similarity
            if similarity <= 0: continue

            # Get neighbor's ratings
            neighbor_vector = self.sparse_matrix[neighbor_idx].toarray()[0]
            liked_indices = np.where(neighbor_vector > 0)[0]
            
            # Weighted Sum: Score = NeighborRating * Similarity
            for it_idx in liked_indices:
                score = neighbor_vector[it_idx] * similarity
                it_id = self.item_ids[it_idx]
                rec_scores[it_id] = rec_scores.get(it_id, 0.0) + score

        # --- 7. FILTER & SORT ---
        if not rec_scores: return []

        # Exclude items the user has already interacted with (including the injected one)
        seen_indices = np.where(user_vector[0] != 0)[0]
        seen_items = {self.item_ids[i] for i in seen_indices}

        # Convert to DataFrame for easy manipulation
        candidates = [
            {'shoe_id': k, 'cf_score': v} 
            for k, v in rec_scores.items() 
            if k not in seen_items
        ]
        
        if not candidates: return []

        candidates_df = pd.DataFrame(candidates)
        candidates_df = candidates_df.sort_values('cf_score', ascending=False).head(20)

        # --- 8. ENRICHMENT ---
        # Merge with metadata to get shoe details (name, image, etc.)
        top_ids = candidates_df['shoe_id'].tolist()
        enriched = self.shoe_metadata[self.shoe_metadata['shoe_id'].isin(top_ids)].copy()
        
        # Map scores back to the enriched dataframe
        score_map = dict(zip(candidates_df['shoe_id'], candidates_df['cf_score']))
        enriched['match_score'] = enriched['shoe_id'].map(score_map)
        enriched = enriched.sort_values('match_score', ascending=False)

        # Convert to list of dicts
        final_results = cast(List[Dict[str, Any]], enriched.to_dict(orient='records'))

        # --- 9. SAVE TO CACHE ---
        # Only cache if this wasn't a temporary injection request
        if new_item_id is None:
            self.cache[user_id] = (final_results, current_time)

        return final_results