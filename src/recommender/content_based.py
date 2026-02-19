import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple

def get_priority_val(user_input: Dict[str, Any], 
                     priority_list: List[str], 
                     mapping_dicts: Dict[str, Dict[str, float]]) -> float:
    """Heuristic helper to resolve feature values."""
    for source_key in priority_list:
        if source_key in user_input and user_input[source_key]:
            user_choice = user_input[source_key]
            if source_key in mapping_dicts:
                mapping = mapping_dicts[source_key]
                if user_choice in mapping:
                    return mapping[user_choice]
    return 0.5

def run_recommendation_pipeline(full_vector_raw: List[float], 
                                valid_indices: List[int], 
                                artifacts: Dict[str, Any]) -> List[Any]:
    """
    Core Recommendation Pipeline.
    RETURNS: List of Shoe IDs (not full objects).
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
    
    # 3. Candidate Filtering
    candidates = df_data[df_data['cluster'].isin(closest_clusters)].copy()
    if candidates.empty: 
        return []
    
    # 4. Masked Cosine Similarity
    candidate_indices = candidates.index
    candidate_vectors = X_combined_data[candidate_indices]
    
    user_vec_masked = full_vector_np[:, valid_indices]
    cand_vecs_masked = candidate_vectors[:, valid_indices]
    
    if np.all(user_vec_masked == 0):
        scores = np.zeros(len(candidates))
    else:
        scores = cosine_similarity(user_vec_masked, cand_vecs_masked)[0]
    
    candidates['match_score'] = scores
    
    # 5. Ranking and ID Extraction
    # Optimized: Return only IDs to reduce payload size
    top_shoes = candidates.sort_values('match_score', ascending=False).head(10)
    
    # Ensure we return a clean list of IDs
    return top_shoes['shoe_id'].tolist()