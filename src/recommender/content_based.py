import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple

def get_priority_val(user_input: Dict[str, Any], 
                     priority_list: List[str], 
                     mapping_dicts: Dict[str, Dict[str, float]]) -> float:
    """
    Heuristic helper to resolve feature values based on a priority list of user inputs.
    
    This function traverses the priority list and returns the first mapped value 
    found in the user input. If no matches are found, it returns a neutral default.

    Args:
        user_input: Dictionary of raw answers from the user questionnaire.
        priority_list: Ordered list of keys to check in the user input.
        mapping_dicts: Nested dictionary mapping string answers to numerical values.

    Returns:
        float: The resolved numerical feature value (default 0.5).
    """
    for source_key in priority_list:
        if source_key in user_input and user_input[source_key]:
            user_choice = user_input[source_key]
            if source_key in mapping_dicts:
                mapping = mapping_dicts[source_key]
                if user_choice in mapping:
                    return mapping[user_choice]
    return 0.5  # Neutral default for missing information

def run_recommendation_pipeline(full_vector_raw: List[float], 
                                valid_indices: List[int], 
                                artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Core Recommendation Pipeline: Cluster Routing -> Candidate Filtering -> Masked Similarity.
    
    This pipeline handles high-dimensional shoe data by projecting it into a latent space
    for clustering, then calculating fine-grained similarity on the original feature space.

    Args:
        full_vector_raw: The processed numerical vector representing user preferences.
        valid_indices: Indices of features explicitly provided by the user (for masking).
        artifacts: Dictionary containing pre-loaded models (Encoder, KMeans) and datasets.

    Returns:
        List[Dict]: Top 10 recommended shoes with metadata and match scores.
    """
    df_data = artifacts['df_data']
    encoder_model = artifacts['encoder_model']
    kmeans_model = artifacts['kmeans_model']
    X_combined_data = artifacts['X_combined_data']

    # 1. Latent Space Projection (Encoding)
    # Project the high-dimensional user vector into an 8D latent space using the Deep Autoencoder.
    full_vector_np = np.array([full_vector_raw])
    user_latent = encoder_model.predict(full_vector_np, verbose=0)
    
    # 2. Cluster Routing (Search Space Reduction)
    # Calculate distance to cluster centroids and select the top 1/3 closest clusters.
    # This optimizes latency by filtering out irrelevant candidates.
    distances = kmeans_model.transform(user_latent)[0]
    n_select = math.ceil(kmeans_model.n_clusters / 3) 
    closest_clusters = np.argsort(distances)[:n_select]
    
    # 3. Candidate Filtering
    # Select shoes from the metadata pool that belong to the identified clusters.
    candidates = df_data[df_data['cluster'].isin(closest_clusters)].copy()
    if candidates.empty: 
        return []
    
    # 4. Masked Cosine Similarity
    # We calculate similarity ONLY on features the user has expressed a preference for.
    # This prevents 'noise' from default values in the similarity score.
    candidate_indices = candidates.index
    candidate_vectors = X_combined_data[candidate_indices]
    
    user_vec_masked = full_vector_np[:, valid_indices]
    cand_vecs_masked = candidate_vectors[:, valid_indices]
    
    # Safety check for zero-vectors to prevent division-by-zero errors in Cosine Similarity
    if np.all(user_vec_masked == 0):
        scores = np.zeros(len(candidates))
    else:
        scores = cosine_similarity(user_vec_masked, cand_vecs_masked)[0]
    
    candidates['match_score'] = scores
    
    # 5. Ranking and Formatting
    # Sort candidates by match_score and return the Top 10 as a dictionary for API consumption.
    top_shoes = candidates.sort_values('match_score', ascending=False).head(10)
    return top_shoes.to_dict(orient='records')