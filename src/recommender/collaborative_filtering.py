"""
Collaborative Filtering Module
------------------------------
Provides a Deep Learning-based Neural Collaborative Filtering (NCF) engine.
Refactored to minimize cyclomatic complexity and optimize inference speed.
"""

import logging
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class NeuralCollaborativeRecommender:
    """
    Neural Collaborative Filtering (NCF) Recommender Engine.
    
    Utilizes a pre-trained Deep Learning model to predict user-item interactions.
    Designed for O(1) cyclomatic complexity and optimized payload (IDs only).
    """

    def __init__(self, model_path: str, shoe_metadata: pd.DataFrame):
        """
        Initializes the NCF engine and loads the Keras model.
        
        Args:
            model_path (str): Filepath to the trained .keras NCF model.
            shoe_metadata (pd.DataFrame): DataFrame containing shoe details, 
                                          must include a 'shoe_id' column.
        
        Returns:
            None
        """
        self.cache: Dict[int, Tuple[List[str], float]] = {}
        self.cache_ttl = 60
        self.item_ids = shoe_metadata['shoe_id'].tolist() if 'shoe_id' in shoe_metadata.columns else []
        self.item_map = {iid: i for i, iid in enumerate(self.item_ids)}
        self.live_changes: Dict[int, Dict[str, float]] = {}

        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("Neural CF Engine Initialized.")
        except Exception as e:
            logger.error(f"Failed to load NCF model: {e}")
            self.model = None

    def _get_cached(self, user_id: int, current_time: float) -> Optional[List[str]]:
        """
        Retrieves recommendations from cache if they are still within TTL.
        
        Args:
            user_id (int): The unique identifier of the user.
            current_time (float): The current Unix timestamp.
        
        Returns:
            Optional[List[str]]: List of cached shoe IDs, or None if invalid/missing.
        """
        if user_id in self.cache:
            result, timestamp = self.cache[user_id]
            if current_time - timestamp < self.cache_ttl:
                return result
        return None

    def _update_live_buffer(self, user_id: int, item_id: Optional[str], is_like: bool) -> None:
        """
        Updates the real-time interaction buffer for a specific user.
        
        Args:
            user_id (int): The unique identifier of the user.
            item_id (Optional[str]): The ID of the shoe interacted with.
            is_like (bool): True if the interaction was a 'like', False otherwise.
            
        Returns:
            None
        """
        if not item_id or item_id not in self.item_map: 
            return
        
        weight = 1.0 if is_like else 0.5
        self.live_changes.setdefault(user_id, {})[item_id] = weight
        self.cache.pop(user_id, None)

    def _predict_ncf(self, user_id: int) -> List[str]:
        """
        Executes the Deep Learning inference to predict top shoe matches.
        
        Args:
            user_id (int): The unique identifier of the user.
            
        Returns:
            List[str]: A list of the top 20 recommended shoe IDs.
        """
        if not self.model or not self.item_ids: 
            return []
            
        user_tensor = np.full(len(self.item_ids), user_id)
        item_tensor = np.arange(len(self.item_ids))
        
        predictions = self.model.predict([user_tensor, item_tensor], verbose=0).flatten()
        
        top_indices = predictions.argsort()[-20:][::-1]
        return [self.item_ids[i] for i in top_indices]

    def get_realtime_recommendations(self, user_id: int, new_item_id: Optional[str] = None, 
                                     is_like: bool = False) -> List[str]:
        """
        Retrieves real-time recommendations for a specific user.
        
        Orchestrates cache checking, live buffer updates, and NCF model inference.
        
        Args:
            user_id (int): The unique identifier of the user.
            new_item_id (Optional[str], optional): ID of a newly interacted shoe. Defaults to None.
            is_like (bool, optional): Interaction type (True for like). Defaults to False.
        
        Returns:
            List[str]: A list of recommended shoe IDs.
        """
        current_time = time.time()
        
        self._update_live_buffer(user_id, new_item_id, is_like)

        if not new_item_id:
            cached_result = self._get_cached(user_id, current_time)
            if cached_result: 
                return cached_result

        final_ids = self._predict_ncf(user_id)

        if not new_item_id:
            self.cache[user_id] = (final_ids, current_time)

        return final_ids