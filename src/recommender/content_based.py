"""
Content-Based Engine Module
---------------------------
Core recommendation pipeline leveraging a Deep Autoencoder for dimensionality 
reduction and K-Means for latent space routing. 
Refactored for low cyclomatic complexity and linear execution.
"""

import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

def get_priority_val(user_input: Dict[str, Any], 
                     priority_list: List[str], 
                     mapping_dicts: Dict[str, Dict[str, float]]) -> float:
    """
    Resolves feature values heuristically based on a priority list to minimize cyclomatic complexity.

    Args:
        user_input (Dict[str, Any]): Raw input preferences from the user.
        priority_list (List[str]): Ordered list of keys to check in user_input.
        mapping_dicts (Dict[str, Dict[str, float]]): Mapping definitions for quantitative translation.

    Returns:
        float: The resolved quantitative feature value, defaulting to 0.5.
    """
    for source_key in priority_list:
        user_choice = user_input.get(source_key)
        if user_choice and source_key in mapping_dicts:
            mapping = mapping_dicts[source_key]
            if user_choice in mapping:
                return mapping[user_choice]
    return 0.5

def run_recommendation_pipeline(full_vector_raw: List[float], 
                                valid_indices: List[int], 
                                artifacts: Dict[str, Any]) -> List[str]:
    """
    Executes the core content-based recommendation pipeline.
    
    Projects the user vector into a latent space using an Autoencoder, 
    routes it to the nearest clusters via K-Means, and ranks candidates 
    using masked cosine similarity.

    Args:
        full_vector_raw (List[float]): The complete, unmasked numerical user vector.
        valid_indices (List[int]): Indices of features actively provided by the user.
        artifacts (Dict[str, Any]): Dictionary of loaded machine learning models and metadata.

    Returns:
        List[str]: A list containing the top recommended shoe IDs.
    """
    df_data = artifacts.get('df_data')
    if df_data is None or df_data.empty:
        return []

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
    
    # 3. Candidate Filtering via Guard Clause
    candidates = df_data[df_data['cluster'].isin(closest_clusters)].copy()
    if candidates.empty: 
        return []
    
    # 4. Masked Cosine Similarity
    candidate_indices = candidates.index
    candidate_vectors = X_combined_data[candidate_indices]
    
    user_vec_masked = full_vector_np[:, valid_indices]
    cand_vecs_masked = candidate_vectors[:, valid_indices]
    
    if np.all(user_vec_masked == 0):
        candidates['match_score'] = 0.0
    else:
        candidates['match_score'] = cosine_similarity(user_vec_masked, cand_vecs_masked)[0]
    
    # 5. Ranking and ID Extraction
    top_shoes = candidates.sort_values('match_score', ascending=False).head(10)
    return top_shoes['shoe_id'].tolist()