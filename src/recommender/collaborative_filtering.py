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
    User-Based Collaborative Filtering recommendation engine with real-time 
    injection capability and TTL-based caching.
    
    Uses cosine similarity in sparse matrix space to find similar users and 
    aggregate their preferences. Supports real-time injection of new interactions 
    without requiring a full model rebuild.
    
    Attributes:
        cache (Dict[int, Tuple[List[Dict], float]]): User-level recommendation 
            cache with timestamps
        cache_ttl (int): Time-to-live for cache entries in seconds (default: 60)
        shoe_metadata (pd.DataFrame): Full shoe catalog with all attributes
        user_map (Dict[int, int]): Maps user IDs to matrix row indices
        item_map (Dict[str, int]): Maps shoe IDs to matrix column indices
        user_ids (List[int]): Ordered list of all user IDs in the system
        item_ids (List[str]): Ordered list of all shoe IDs in the system
        sparse_matrix (csr_matrix): Sparse user-item interaction matrix
        model (NearestNeighbors): Fitted k-NN model for similarity search
    """

    def __init__(self, df_interactions: pd.DataFrame, shoe_metadata: pd.DataFrame):
        """
        Initializes the collaborative filtering engine by building the user-item 
        interaction matrix and fitting the k-NN model.
        
        Args:
            df_interactions (pd.DataFrame): Interaction history with columns:
                - user_id (int): User identifier
                - item_id (str): Shoe identifier
                - rating (float): Interaction score (-2.0 to +2.0 scale)
            shoe_metadata (pd.DataFrame): Complete shoe catalog for enrichment
        
        Returns:
            None
        
        Side Effects:
            - Creates sparse CSR matrix from interaction data
            - Fits NearestNeighbors model with cosine metric
            - Logs initialization status
        
        Example:
            >>> interactions = pd.DataFrame({
            ...     'user_id': [1, 1, 2],
            ...     'item_id': ['R001', 'R002', 'R001'],
            ...     'rating': [1.0, 2.0, -1.0]
            ... })
            >>> recommender = UserCollaborativeRecommender(interactions, shoe_df)
        """
        self.cache: Dict[int, Tuple[List[Dict[str, Any]], float]] = {} 
        self.cache_ttl = 60
        self.shoe_metadata = shoe_metadata
        
        self.user_map: Dict[int, int] = {}
        self.item_map: Dict[str, int] = {}
        self.user_ids: List[int] = []
        self.item_ids: List[str] = []
        self.sparse_matrix: Optional[csr_matrix] = None
        self.model: Optional[NearestNeighbors] = None

        if df_interactions.empty:
            logger.warning("CF Engine: No interaction data found. Passive Mode.")
            return

        self.pivot_df = df_interactions.pivot(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        self.user_ids = list(self.pivot_df.index)
        self.item_ids = list(self.pivot_df.columns)
        self.user_map = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_map = {iid: i for i, iid in enumerate(self.item_ids)}
        
        self.sparse_matrix = csr_matrix(self.pivot_df.values)
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(self.sparse_matrix)
        
        logger.info(f"CF Engine Initialized: {len(self.user_ids)} Users, {len(self.item_ids)} Items.")

    def _convert_rating(self, rating: Optional[int], is_like_action: bool = False) -> float:
        """
        Converts raw user signals (likes or 1-5 star ratings) into a symmetric 
        preference scale suitable for collaborative filtering.
        
        Args:
            rating (Optional[int]): Star rating from 1 to 5, or None
            is_like_action (bool): Whether this is a binary "like" action 
                (default: False)
        
        Returns:
            float: Converted score on scale:
                - Like: 1.0
                - 5 stars: 2.0
                - 4 stars: 1.0
                - 3 stars: 0.1 (neutral)
                - 2 stars: -1.0
                - 1 star: -2.0
                - None/0: 0.1 (default neutral)
        
        Example:
            >>> self._convert_rating(5, is_like_action=False)
            2.0
            >>> self._convert_rating(None, is_like_action=True)
            1.0
        """
        if is_like_action: 
            return 1.0
        if rating is None or rating == 0: 
            return 0.1
        mapping = {1: -2.0, 2: -1.0, 3: 0.1, 4: 1.0, 5: 2.0}
        return mapping.get(rating, 0.1)

    def _check_cache(self, user_id: int, new_item_id: Optional[str], 
                     current_time: float) -> Optional[List[Dict[str, Any]]]:
        """
        Checks if a valid cached recommendation exists for the user. Cache is 
        invalidated if TTL expired or if a new interaction is being injected.
        
        Args:
            user_id (int): Target user identifier
            new_item_id (Optional[str]): Shoe ID of new interaction 
                (None if just fetching feed)
            current_time (float): Current Unix timestamp
        
        Returns:
            Optional[List[Dict[str, Any]]]: 
                - List of cached recommendations if valid
                - None if cache miss, expired, or invalidated by new interaction
        
        Logic:
            1. Check if user exists in cache
            2. Verify timestamp is within TTL window (60 seconds)
            3. Ensure no new interaction forces invalidation
            4. Return cached result only if all conditions pass
        
        Example:
            >>> cached = self._check_cache(user_id=8, new_item_id=None, 
            ...                            current_time=time.time())
            >>> if cached:
            ...     return cached  # Fast path
        """
        if user_id not in self.cache:
            return None
        
        result, timestamp = self.cache[user_id]
        is_fresh = (current_time - timestamp < self.cache_ttl)
        no_new_interaction = (new_item_id is None)
        
        if is_fresh and no_new_interaction:
            return result
        return None

    def _validate_prerequisites(self) -> Optional[str]:
        """
        Validates that the engine has been properly initialized and has 
        sufficient data to generate recommendations.
        
        Args:
            None
        
        Returns:
            Optional[str]: 
                - Error code string if prerequisites not met:
                    - "no_items": No shoes in the catalog
                    - "model_not_ready": Sparse matrix or KNN model not initialized
                    - "insufficient_users": Less than 2 users (KNN requires neighbors)
                - None if all prerequisites are satisfied
        
        Example:
            >>> error = self._validate_prerequisites()
            >>> if error:
            ...     logger.warning(f"Cannot generate recommendations: {error}")
            ...     return []
        """
        if len(self.item_ids) == 0:
            return "no_items"
        if self.sparse_matrix is None or self.model is None:
            return "model_not_ready"
        if len(self.user_ids) <= 1:
            return "insufficient_users"
        return None

    def _get_or_create_user_vector(self, user_id: int) -> np.ndarray:
        """
        Retrieves the existing interaction vector for a user, or creates a 
        zero-initialized vector for cold-start users.
        
        Args:
            user_id (int): Target user identifier
        
        Returns:
            np.ndarray: User vector of shape (1, num_items) where each element 
                represents the user's preference score for that item. Zero vector 
                returned for new users.
        
        Example:
            >>> user_vector = self._get_or_create_user_vector(user_id=8)
            >>> user_vector.shape
            (1, 250)  # 250 shoes in catalog
            >>> user_vector[0, 10]  # User's score for shoe at index 10
            1.0
        """
        num_items = len(self.item_ids)
        
        if user_id in self.user_map:
            user_idx = self.user_map[user_id]
            return self.sparse_matrix[user_idx].toarray()
        
        return np.zeros((1, num_items))

    def _inject_interaction(self, user_vector: np.ndarray, new_item_id: Optional[str],
                           new_rating_val: Optional[int], is_like: bool) -> None:
        """
        Injects a new user interaction into the user vector in-place, enabling 
        real-time recommendations without rebuilding the entire model.
        
        Args:
            user_vector (np.ndarray): User's preference vector (modified in-place)
            new_item_id (Optional[str]): Shoe ID being interacted with
            new_rating_val (Optional[int]): Star rating (1-5) if rating action
            is_like (bool): Whether this is a "like" action vs. a rating
        
        Returns:
            None (modifies user_vector in-place)
        
        Side Effects:
            Updates the element in user_vector corresponding to new_item_id 
            with the converted preference score.
        
        Example:
            >>> user_vector = self._get_or_create_user_vector(user_id=8)
            >>> self._inject_interaction(user_vector, new_item_id="R278", 
            ...                          new_rating_val=5, is_like=False)
            >>> # user_vector now contains 2.0 at the index for shoe R278
        """
        if not new_item_id or new_item_id not in self.item_map:
            return
        
        item_idx = self.item_map[new_item_id]
        score_val = self._convert_rating(new_rating_val, is_like_action=is_like)
        user_vector[0, item_idx] = score_val

    def _find_neighbors(self, user_vector: np.ndarray, n_neighbors: int
                       ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Finds k-nearest neighbors to the given user vector using cosine similarity.
        
        Args:
            user_vector (np.ndarray): Query vector of shape (1, num_items)
            n_neighbors (int): Number of neighbors to retrieve
        
        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]:
                - (distances, indices): Arrays of shape (1, k) containing 
                  cosine distances and user indices of k nearest neighbors
                - None: If KNN search fails (logged as error)
        
        Notes:
            - Automatically adjusts k if fewer users exist in the system
            - Requests k+1 neighbors since the user themselves might be included
        
        Example:
            >>> result = self._find_neighbors(user_vector, n_neighbors=10)
            >>> if result:
            ...     distances, indices = result
            ...     # distances[0, 0] is the closest neighbor's distance
            ...     # indices[0, 0] is the closest neighbor's user index
        """
        total_users = len(self.user_ids)
        effective_k = min(n_neighbors + 1, total_users)
        
        try:
            distances, indices = self.model.kneighbors(user_vector, n_neighbors=effective_k)
            return distances, indices
        except Exception as e:
            logger.error(f"CF Neighbor search failed: {e}")
            return None

    def _aggregate_scores(self, user_id: int, user_vector: np.ndarray,
                         distances: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
        """
        Aggregates preference scores from similar users using weighted voting. 
        Each neighbor's preferences are weighted by their similarity to the target user.
        
        Args:
            user_id (int): Target user identifier (to skip self in neighbors)
            user_vector (np.ndarray): Target user's preference vector
            distances (np.ndarray): Cosine distances from KNN search
            indices (np.ndarray): User indices of neighbors from KNN search
        
        Returns:
            Dict[str, float]: Mapping from shoe_id to aggregated weighted score. 
                Higher scores indicate stronger predicted preference.
        
        Algorithm:
            For each neighbor:
                similarity = 1 - cosine_distance
                for each item the neighbor liked:
                    score[item] += neighbor_rating[item] * similarity
        
        Example:
            >>> scores = self._aggregate_scores(user_id=8, user_vector=vec, 
            ...                                 distances=dists, indices=idxs)
            >>> scores
            {'R045': 3.2, 'R278': 2.8, 'T012': 1.5}
            >>> # User 8's similar users strongly prefer R045
        """
        rec_scores: Dict[str, float] = {}
        
        for i, neighbor_idx in enumerate(indices[0]):
            # Skip self if present
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

    def _filter_and_enrich(self, rec_scores: Dict[str, float], 
                          user_vector: np.ndarray) -> List[Dict[str, Any]]:
        """
        Filters out items the user has already interacted with, ranks remaining 
        candidates by score, and enriches with full shoe metadata.
        
        Args:
            rec_scores (Dict[str, float]): Raw aggregated scores from neighbors
            user_vector (np.ndarray): Target user's preference vector 
                (to identify seen items)
        
        Returns:
            List[Dict[str, Any]]: Top 10 recommended shoes with full metadata:
                - shoe_id (str): Shoe identifier
                - name (str): Shoe name
                - brand (str): Brand name
                - match_score (float): Predicted preference score
                - (other shoe attributes from metadata)
        
        Steps:
            1. Identify items user has already interacted with (non-zero in vector)
            2. Filter these from candidate scores
            3. Sort by score descending, take top 10
            4. Join with shoe_metadata to add names, images, etc.
            5. Return enriched list
        
        Example:
            >>> results = self._filter_and_enrich(rec_scores, user_vector)
            >>> results[0]
            {
                'shoe_id': 'R045',
                'name': 'Nike Vaporfly 3',
                'brand': 'Nike',
                'match_score': 3.2,
                'image_url': 'https://...'
            }
        """
        if not rec_scores:
            return []

        # Filter seen items
        seen_indices = np.where(user_vector[0] != 0)[0]
        seen_items = {self.item_ids[i] for i in seen_indices}
        
        candidates = [
            {'shoe_id': k, 'cf_score': v} 
            for k, v in rec_scores.items() 
            if k not in seen_items
        ]
        
        if not candidates:
            return []

        # Sort and take top 10
        candidates_df = pd.DataFrame(candidates)
        candidates_df = candidates_df.sort_values('cf_score', ascending=False).head(10)

        # Enrich with metadata
        top_ids = candidates_df['shoe_id'].tolist()
        enriched = self.shoe_metadata[self.shoe_metadata['shoe_id'].isin(top_ids)].copy()
        
        score_map = dict(zip(candidates_df['shoe_id'], candidates_df['cf_score']))
        enriched['match_score'] = enriched['shoe_id'].map(score_map)
        enriched = enriched.sort_values('match_score', ascending=False)

        return enriched.to_dict(orient='records')

    def get_realtime_recommendations(self, 
                                     user_id: int, 
                                     new_item_id: Optional[str] = None, 
                                     new_rating_val: Optional[int] = None,
                                     is_like: bool = False,
                                     n_neighbors: int = 10) -> List[Dict[str, Any]]:
        """
        Generates personalized recommendations for a user with optional real-time 
        injection of a new interaction. Utilizes TTL caching for performance.
        
        This is the main entry point for collaborative filtering recommendations. 
        It supports both "cold" queries (fetching a user's feed) and "hot" queries 
        (reflecting a just-submitted interaction immediately).
        
        Args:
            user_id (int): Target user identifier
            new_item_id (Optional[str]): Shoe ID of a new interaction to inject. 
                If None, returns cached/computed feed without injection.
            new_rating_val (Optional[int]): Star rating (1-5) if user rated the item. 
                Ignored if is_like=True.
            is_like (bool): Whether the interaction is a binary "like" (True) vs. 
                a star rating (False). Default: False.
            n_neighbors (int): Number of similar users to consider. Default: 10.
        
        Returns:
            List[Dict[str, Any]]: Top 10 recommended shoes with metadata and scores. 
                Returns empty list if:
                - User has no interaction history (cold start)
                - System has insufficient data (<2 users)
                - Model not initialized
        
        Workflow:
            1. Check cache (returns immediately if valid)
            2. Validate prerequisites (users, items, model)
            3. Load or create user preference vector
            4. Inject new interaction if provided (real-time update)
            5. Find k-nearest similar users via cosine similarity
            6. Aggregate weighted preferences from neighbors
            7. Filter seen items and enrich with metadata
            8. Cache result (if no injection performed)
            9. Return top 10 recommendations
        
        Performance:
            - Cached requests: ~0-5ms (in-memory lookup)
            - Uncached requests: ~20-50ms (KNN + aggregation)
            - Cache TTL: 60 seconds
        
        Example (Fetch feed):
            >>> recommender.get_realtime_recommendations(user_id=8)
            [
                {'shoe_id': 'R045', 'name': 'Nike Vaporfly 3', 'match_score': 3.2},
                {'shoe_id': 'R278', 'name': 'Nike Pegasus 41', 'match_score': 2.8}
            ]
        
        Example (Like + immediate recommendations):
            >>> recommender.get_realtime_recommendations(
            ...     user_id=8, 
            ...     new_item_id='R278', 
            ...     is_like=True
            ... )
            # Returns recommendations reflecting the new like immediately
        
        Example (Rate + immediate recommendations):
            >>> recommender.get_realtime_recommendations(
            ...     user_id=8, 
            ...     new_item_id='R045', 
            ...     new_rating_val=5, 
            ...     is_like=False
            ... )
            # Returns recommendations reflecting the 5-star rating immediately
        """
        current_time = time.time()
        
        # 1. Check cache
        cached = self._check_cache(user_id, new_item_id, current_time)
        if cached is not None:
            return cached

        # 2. Validate prerequisites
        error = self._validate_prerequisites()
        if error:
            return []

        # 3. Get user vector
        user_vector = self._get_or_create_user_vector(user_id)

        # 4. Inject new interaction if provided
        self._inject_interaction(user_vector, new_item_id, new_rating_val, is_like)

        # 5. Cold start check
        if np.all(user_vector == 0):
            return []

        # 6. Find neighbors
        neighbors_result = self._find_neighbors(user_vector, n_neighbors)
        if neighbors_result is None:
            return []
        
        distances, indices = neighbors_result

        # 7. Aggregate scores
        rec_scores = self._aggregate_scores(user_id, user_vector, distances, indices)

        # 8. Filter and enrich
        final_results = self._filter_and_enrich(rec_scores, user_vector)

        # 9. Cache if not a temporary injection
        if new_item_id is None:
            self.cache[user_id] = (final_results, current_time)

        return final_results