"""
Content-Based Engine Module
---------------------------
Core recommendation pipeline leveraging a Deep Autoencoder for dimensionality 
reduction and K-Means for latent space routing. 
Refactored for low cyclomatic complexity using dictionary mapping.
"""

import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

def get_priority_val(user_input: Dict[str, Any], 
                     priority_list: List[str], 
                     mapping_dicts: Dict[str, Dict[str, float]]) -> float:
    """
    Resolves feature values heuristically based on a priority list.
    Replaces nested if-statements with flat dictionary lookups to reduce CC.

    Args:
        user_input (Dict[str, Any]): Raw user preferences.
        priority_list (List[str]): Keys to check in order of importance.
        mapping_dicts (Dict[str, Dict[str, float]]): Mapping definitions.

    Returns:
        float: Quantitative feature value (defaults to 0.5).
    """
    for source_key in priority_list:
        user_choice = user_input.get(source_key)
        if user_choice:
            val = mapping_dicts.get(source_key, {}).get(user_choice)
            if val is not None:
                return val
    return 0.5

def run_recommendation_pipeline(full_vector_raw: List[float], 
                                valid_indices: List[int], 
                                artifacts: Dict[str, Any]) -> List[str]:
    """
    Executes the core inference pipeline: Autoencoder -> K-Means -> Cosine Similarity.
    
    Args:
        full_vector_raw (List[float]): Unmasked numerical user vector.
        valid_indices (List[int]): Active indices for masked similarity.
        artifacts (Dict[str, Any]): Loaded ML models and metadata.

    Returns:
        List[str]: Top recommended shoe IDs.
    """
    df_data = artifacts['df_data']
    encoder_model = artifacts['encoder_model']
    kmeans_model = artifacts['kmeans_model']
    X_combined_data = artifacts['X_combined_data']

    # 1. Latent Space Projection
    full_vector_np = np.array([full_vector_raw])
    user_latent = encoder_model.predict(full_vector_np, verbose=0)
    
    # 2. Cluster Routing
    distances = kmeans_model.transform(user_latent)[0]
    n_select = math.ceil(kmeans_model.n_clusters / 3) 
    closest_clusters = np.argsort(distances)[:n_select]
    
    # 3. Filtering & Scoring
    candidates = df_data[df_data['cluster'].isin(closest_clusters)].copy()
    if candidates.empty: 
        return []
    
    user_vec_masked = full_vector_np[:, valid_indices]
    cand_vecs_masked = X_combined_data[candidates.index][:, valid_indices]
    
    if np.all(user_vec_masked == 0):
        candidates['match_score'] = 0.0
    else:
        candidates['match_score'] = cosine_similarity(user_vec_masked, cand_vecs_masked)[0]
    
    return candidates.sort_values('match_score', ascending=False).head(10)['shoe_id'].tolist()