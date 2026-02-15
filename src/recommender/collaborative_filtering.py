import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Optional, Set

# Configure logger for tracking engine initialization and runtime errors
logger = logging.getLogger(__name__)

class UserCollaborativeRecommender:
    """
    User-Based Collaborative Filtering (UBCF) Engine.
    
    This memory-based recommender identifies similar users based on interaction history
    and suggests items liked by those 'neighbors'.
    
    Features:
    - Symmetric rating scale (-2.0 to +2.0) to handle both positive and negative sentiment.
    - Sparse Matrix implementation for memory efficiency.
    - Real-time Injection: Instantly updates user profile in-memory for immediate feedback.
    """

    def __init__(self, df_interactions: pd.DataFrame):
        """
        Initializes the engine by pivoting data and fitting the k-Nearest Neighbors model.
        
        Args:
            df_interactions: DataFrame containing ['user_id', 'item_id', 'rating']
        """
        if df_interactions.empty:
            logger.warning("CF Engine: No interaction data found. Operating in Cold Start mode.")
            self.user_ids = []
            self.item_ids = []
            return

        # 1. Pivot Data: Reshape into a User x Item matrix
        # Columns (items) not interacted with are filled with 0 (neutral)
        self.pivot_df = df_interactions.pivot(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        # 2. Index Mapping: Bi-directional mapping between database IDs and matrix indices
        self.user_ids = list(self.pivot_df.index)
        self.item_ids = list(self.pivot_df.columns)
        self.user_map = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_map = {iid: i for i, iid in enumerate(self.item_ids)}
        
        # 3. Model Fitting: Using Cosine Similarity to find similar user vectors
        self.sparse_matrix = csr_matrix(self.pivot_df.values)
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(self.sparse_matrix)
        
        logger.info(f"CF Engine Initialized: {len(self.user_ids)} Users, {len(self.item_ids)} Items.")

    def _convert_rating(self, rating: Optional[int], is_like_action: bool = False) -> float:
        """
        Maps raw input (1-5 stars or 'likes') to a symmetric preference scale.
        
        Logic:
        - Symmetric scale (-2.0 to +2.0) allows the model to differentiate between 
          active dislike and simple lack of interaction.
        """
        if is_like_action:
            return 1.0  # Positive signal for "Like" button
        
        # Neutral-Positive default (0.1) for views or zero-ratings to avoid zero-vector issues
        if rating is None or rating == 0:
            return 0.1

        # Database mapping for 1-5 star reviews
        mapping = {
            1: -2.0, # Strong Dislike
            2: -1.0, # Dislike
            3:  0.1, # Neutral / Positive Interest
            4:  1.0, # Like
            5:  2.0  # Strong Like
        }
        return mapping.get(rating, 0.1)

    def get_realtime_recommendations(self, 
                                     user_id: int, 
                                     new_item_id: Optional[str] = None, 
                                     new_rating_val: Optional[int] = None,
                                     is_like: bool = False,
                                     n_neighbors: int = 5) -> List[Dict[str, float]]:
        """
        Retrieves recommendations by injecting the latest user interaction in real-time.
        
        Args:
            user_id: Target user ID
            new_item_id: ID of the shoe just interacted with
            new_rating_val: Raw rating value (1-5)
            is_like: Boolean flag for "Like" action
            n_neighbors: Number of similar users to analyze
            
        Returns:
            List of dictionaries containing shoe_id and calculated match score.
        """
        # A. Initialize User Vector
        if user_id in self.user_map:
            user_idx = self.user_map[user_id]
            user_vector = self.sparse_matrix[user_idx].toarray()
        else:
            # Handle New/Cold User: Start with a zero-vector
            user_vector = np.zeros((1, len(self.item_ids)))

        # B. Real-time Injection: Update the vector in-memory before inference
        if new_item_id and new_item_id in self.item_map:
            item_idx = self.item_map[new_item_id]
            score_val = self._convert_rating(new_rating_val, is_like_action=is_like)
            user_vector[0, item_idx] = score_val

        # C. Inference: Find the K-Nearest Neighbors
        try:
            # Query n+1 because the closest neighbor is usually the user themselves
            distances, indices = self.model.kneighbors(user_vector, n_neighbors=n_neighbors + 1)
        except ValueError:
            return [] # Edge case: model not fitted properly

        # D. Score Calculation: Aggregate item preferences from similar users
        rec_scores: Dict[str, float] = {}
        
        for i, neighbor_idx in enumerate(indices[0]):
            # Skip the target user to avoid recommending already seen items
            if user_id in self.user_map and neighbor_idx == self.user_map[user_id]:
                continue
                
            similarity = 1.0 - distances[0][i]
            if similarity <= 0: 
                continue

            neighbor_vector = self.sparse_matrix[neighbor_idx].toarray()[0]
            
            # Identify items liked by the neighbor (positive preference only)
            liked_indices = np.where(neighbor_vector > 0)[0]
            
            for it_idx in liked_indices:
                item_id_rec = self.item_ids[it_idx]
                rating_neighbor = neighbor_vector[it_idx]
                
                # Recommendation Score = Neighbor Interest * User-Neighbor Similarity
                score = rating_neighbor * similarity
                rec_scores[item_id_rec] = rec_scores.get(item_id_rec, 0.0) + score

        # E. Post-Processing: Filter items already interacted with by the target user
        seen_indices = np.where(user_vector[0] != 0)[0]
        seen_items: Set[str] = {self.item_ids[i] for i in seen_indices}
        
        final_recs = []
        for item, score in rec_scores.items():
            if item not in seen_items:
                final_recs.append({
                    'shoe_id': item, 
                    'cf_score': round(float(score), 3)
                })
        
        # Sort by match score (Descending)
        final_recs.sort(key=lambda x: x['cf_score'], reverse=True)
        
        return final_recs[:10]