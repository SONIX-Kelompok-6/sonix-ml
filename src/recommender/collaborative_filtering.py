import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Optional, Any, cast

# Configure logger for tracking engine initialization and runtime errors
logger = logging.getLogger(__name__)

class UserCollaborativeRecommender:
    """
    User-Based Collaborative Filtering (UBCF) Engine.
    
    This memory-based recommender identifies similar users based on interaction history
    and suggests items liked by those 'neighbors'. It supports real-time profile 
    injection to solve the immediate 'Cold Start' problem for active sessions.
    """

    def __init__(self, df_interactions: pd.DataFrame, shoe_metadata: pd.DataFrame):
        """
        Initializes the engine by constructing the User-Item Matrix and fitting the k-NN model.
        
        Args:
            df_interactions (pd.DataFrame): Historical data containing ['user_id', 'item_id', 'rating'].
            shoe_metadata (pd.DataFrame): Full catalog of shoes (cols: shoe_id, name, image_url, etc.)
                                          used for result enrichment.
        """
        self.shoe_metadata = shoe_metadata

        self.user_map = {}
        self.item_map = {}
        self.user_ids = []
        self.item_ids = []
        self.sparse_matrix = None
        self.model = None

        if df_interactions.empty:
            logger.warning("CF Engine: No interaction data found. Operating in Passive Mode.")
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
        # brute force is acceptable for small-to-medium datasets (<100k users)
        self.sparse_matrix = csr_matrix(self.pivot_df.values)
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(self.sparse_matrix)
        
        logger.info(f"CF Engine Initialized: {len(self.user_ids)} Users, {len(self.item_ids)} Items.")

    def _convert_rating(self, rating: Optional[int], is_like_action: bool = False) -> float:
        """
        Maps raw input (1-5 stars or 'likes') to a symmetric preference scale (-2.0 to +2.0).
        """
        if is_like_action:
            return 1.0  # Positive signal for "Like" button
        
        # Neutral-Positive default (0.1) for views/implicit signals
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
                                     n_neighbors: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves recommendations by injecting the latest user interaction in real-time.
        
        Args:
            user_id: Target user ID.
            new_item_id: ID of the shoe just interacted with (optional).
            new_rating_val: Raw rating value (optional).
            is_like: Boolean flag for "Like" action.
            n_neighbors: Number of similar users to analyze.
            
        Returns:
            List[Dict[str, Any]]: A list of full shoe objects sorted by match_score.
        """
        
        # --- A. Initialize User Vector ---
        # If the user exists in history, load their profile. Otherwise, initialize a zero-vector.
        if user_id in self.user_map:
            user_idx = self.user_map[user_id]
            user_vector = self.sparse_matrix[user_idx].toarray()
        else:
            user_vector = np.zeros((1, len(self.item_ids)))

        # --- B. Real-time Injection ---
        # Update the vector in-memory to reflect immediate interest (Active Mode)
        if new_item_id and new_item_id in self.item_map:
            item_idx = self.item_map[new_item_id]
            score_val = self._convert_rating(new_rating_val, is_like_action=is_like)
            user_vector[0, item_idx] = score_val

        # --- [CRITICAL UPDATE] Zero-Vector Validation ---
        # If the vector remains all zeros (New User + No Real-time Interaction),
        # we cannot compute similarity. Return empty to trigger fallback logic in the API.
        if np.all(user_vector == 0):
            return []

        # --- C. Inference (k-NN) ---
        try:
            # Query n+1 because the closest neighbor is typically the user themselves (distance=0)
            distances, indices = self.model.kneighbors(user_vector, n_neighbors=n_neighbors + 1)
        except ValueError:
            logger.error("CF Model inference failed. Ensure sparse matrix is populated.")
            return []

        # --- D. Score Calculation ---
        rec_scores: Dict[str, float] = {}
        
        for i, neighbor_idx in enumerate(indices[0]):
            # Skip the target user (self) to avoid data leakage
            if user_id in self.user_map and neighbor_idx == self.user_map[user_id]:
                continue
                
            # Convert distance to similarity (1 - distance)
            similarity = 1.0 - distances[0][i]
            
            # Filter out non-positive similarity to ensure relevance
            if similarity <= 0: continue

            neighbor_vector = self.sparse_matrix[neighbor_idx].toarray()[0]
            
            # Identify items liked by this neighbor
            liked_indices = np.where(neighbor_vector > 0)[0]
            
            for it_idx in liked_indices:
                item_id_rec = self.item_ids[it_idx]
                rating_neighbor = neighbor_vector[it_idx]
                
                # Weighted Score: Neighbor's Rating * Similarity Strength
                score = rating_neighbor * similarity
                rec_scores[item_id_rec] = rec_scores.get(item_id_rec, 0.0) + score

        # --- E. Post-Processing & Enrichment ---
        
        # 1. Identify seen items to filter them out
        seen_indices = np.where(user_vector[0] != 0)[0]
        seen_items: set = {self.item_ids[i] for i in seen_indices}
        
        if not rec_scores:
            return []

        # 2. Convert scores to DataFrame
        candidates_df = pd.DataFrame(list(rec_scores.items()), columns=['shoe_id', 'match_score'])
        
        # 3. Filter out items the user has already interacted with
        candidates_df = candidates_df[~candidates_df['shoe_id'].isin(seen_items)]
        
        # 4. Sort and Slice (Top 10)
        candidates_df = candidates_df.sort_values('match_score', ascending=False).head(10)
        
        if candidates_df.empty:
            return []

        # 5. Metadata Enrichment
        top_ids = candidates_df['shoe_id'].tolist()
        enriched_results = self.shoe_metadata[self.shoe_metadata['shoe_id'].isin(top_ids)].copy()
        
        # Map scores back to the enriched dataframe
        score_map = dict(zip(candidates_df['shoe_id'], candidates_df['match_score']))
        enriched_results['match_score'] = enriched_results['shoe_id'].map(score_map)
        
        # 6. Final Sort
        enriched_results = enriched_results.sort_values('match_score', ascending=False)
        
        # Return as strictly typed list of dictionaries
        return cast(List[Dict[str, Any]], enriched_results.to_dict(orient='records'))