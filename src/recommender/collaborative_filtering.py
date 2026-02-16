import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Optional, Any, Set, cast

# Configure logger for tracking engine initialization and runtime errors
logger = logging.getLogger(__name__)

class UserCollaborativeRecommender:
    """
    User-Based Collaborative Filtering (UBCF) Engine for the Sonix project.
    
    Identifies similar users based on interaction history and suggests items 
    based on neighbor preferences. Supports real-time profile injection.
    """

    def __init__(self, df_interactions: pd.DataFrame, shoe_metadata: pd.DataFrame):
        """
        Initializes the engine by constructing the User-Item Matrix and fitting the k-NN model.
        """
        self.shoe_metadata = shoe_metadata
        
        # --- Default Initialization ---
        # Explicitly defined to prevent AttributeError and satisfy strict type checking
        self.user_map: Dict[int, int] = {}
        self.item_map: Dict[str, int] = {}
        self.user_ids: List[int] = []
        self.item_ids: List[str] = []
        self.sparse_matrix: Optional[csr_matrix] = None
        self.model: Optional[NearestNeighbors] = None

        if df_interactions.empty:
            logger.warning("CF Engine: No interaction data found. Operating in Passive Mode.")
            return

        # 1. Pivot Data: Using 'item_id' exactly as per your requirement
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
        self.sparse_matrix = csr_matrix(self.pivot_df.values)
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(self.sparse_matrix)
        
        logger.info(f"CF Engine Initialized: {len(self.user_ids)} Users, {len(self.item_ids)} Items.")

    def _convert_rating(self, rating: Optional[int], is_like_action: bool = False) -> float:
        """Maps raw input signals to a symmetric preference scale (-2.0 to +2.0)."""
        if is_like_action: return 1.0
        if rating is None or rating == 0: return 0.1
        mapping = {1: -2.0, 2: -1.0, 3: 0.1, 4: 1.0, 5: 2.0}
        return mapping.get(rating, 0.1)

    def get_realtime_recommendations(self, 
                                     user_id: int, 
                                     new_item_id: Optional[str] = None, 
                                     new_rating_val: Optional[int] = None,
                                     is_like: bool = False,
                                     n_neighbors: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves recommendations by injecting latest user interactions in real-time.
        Uses Type Guards to resolve Pylance 'None' warnings.
        """
        
        # --- A. Initialization Guards ---
        num_items = len(self.item_ids)
        if num_items == 0: return []

        # Resolves Pylance red lines by ensuring objects are not None
        if self.sparse_matrix is None or self.model is None:
            return []

        # --- B. Initialize User Vector ---
        if user_id in self.user_map:
            user_idx = self.user_map[user_id]
            user_vector = self.sparse_matrix[user_idx].toarray()
        else:
            user_vector = np.zeros((1, num_items))

        # --- C. Real-time Injection ---
        if new_item_id and new_item_id in self.item_map:
            item_idx = self.item_map[new_item_id]
            score_val = self._convert_rating(new_rating_val, is_like_action=is_like)
            user_vector[0, item_idx] = score_val

        # Reject requests with zeroed vectors
        if np.all(user_vector == 0): return []

        # --- D. Dynamic Neighbor Capping ---
        total_users = len(self.user_ids)
        if total_users <= 1: return []

        # Adjust neighbors based on current dataset size to prevent ValueError
        effective_neighbors = min(n_neighbors, total_users - 1)

        try:
            # Query n+1 neighbors to ensure target exclusion
            distances, indices = self.model.kneighbors(user_vector, n_neighbors=effective_neighbors + 1)
        except Exception as e:
            logger.error(f"CF Inference failed: {e}")
            return []

        # --- E. Score Calculation ---
        rec_scores: Dict[str, float] = {}
        for i, neighbor_idx in enumerate(indices[0]):
            if user_id in self.user_map and neighbor_idx == self.user_map[user_id]:
                continue
                
            similarity = 1.0 - distances[0][i]
            if similarity <= 0: continue

            neighbor_vector = self.sparse_matrix[neighbor_idx].toarray()[0]
            liked_indices = np.where(neighbor_vector > 0)[0]
            
            for it_idx in liked_indices:
                it_id = self.item_ids[it_idx]
                score = neighbor_vector[it_idx] * similarity
                rec_scores[it_id] = rec_scores.get(it_id, 0.0) + score

        # --- F. Post-Processing & Enrichment ---
        seen_indices = np.where(user_vector[0] != 0)[0]
        seen_items: Set[str] = {self.item_ids[i] for i in seen_indices}
        
        if not rec_scores: return []

        # Prepare candidates for final enrichment
        candidates_df = pd.DataFrame(list(rec_scores.items()), columns=['shoe_id', 'match_score'])
        candidates_df = candidates_df[~candidates_df['shoe_id'].isin(seen_items)]
        candidates_df = candidates_df.sort_values('match_score', ascending=False).head(10)
        
        if candidates_df.empty: return []

        # Enrichment step using shoe_metadata
        top_ids = candidates_df['shoe_id'].tolist()
        enriched_results = self.shoe_metadata[self.shoe_metadata['shoe_id'].isin(top_ids)].copy()
        
        score_map = dict(zip(candidates_df['shoe_id'], candidates_df['match_score']))
        enriched_results['match_score'] = enriched_results['shoe_id'].map(score_map)
        
        enriched_results = enriched_results.sort_values('match_score', ascending=False)
        
        # Explicit type cast for clean returns
        return cast(List[Dict[str, Any]], enriched_results.to_dict(orient='records'))