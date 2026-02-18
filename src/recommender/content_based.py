import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple

def get_priority_val(user_input: Dict[str, Any], 
                     priority_list: List[str], 
                     mapping_dicts: Dict[str, Dict[str, float]]) -> float:
    """
    Heuristic helper to resolve feature values based on a priority list of 
    user inputs.
    
    This function traverses the priority list and returns the first mapped value 
    found in the user input. If no matches are found, it returns a neutral default.
    
    Args:
        user_input (Dict[str, Any]): Dictionary of raw answers from the user 
            questionnaire. Keys are question identifiers (e.g., 'pace', 
            'arch_type'), values are user's selected answers (e.g., 'Fast', 
            'Normal').
        priority_list (List[str]): Ordered list of keys to check in the user 
            input. First match wins. Example: ['pace', 'running_purpose']
        mapping_dicts (Dict[str, Dict[str, float]]): Nested dictionary mapping 
            string answers to numerical values. Structure:
            {
                'pace': {'Easy': 0.2, 'Steady': 0.6, 'Fast': 1.0},
                'arch_type': {'Flat': 0.0, 'Normal': 0.5, 'High': 1.0}
            }
    
    Returns:
        float: The resolved numerical feature value. Returns 0.5 (neutral default) 
            if no match is found in any priority source.
    
    Example:
        >>> user_input = {'pace': 'Fast', 'terrain': 'Rocky'}
        >>> priority_list = ['pace', 'running_purpose']
        >>> mapping_dicts = {
        ...     'pace': {'Easy': 0.2, 'Steady': 0.6, 'Fast': 1.0},
        ...     'running_purpose': {'Daily': 0.2, 'Race': 1.0}
        ... }
        >>> get_priority_val(user_input, priority_list, mapping_dicts)
        1.0
        
        >>> # Example with missing input
        >>> user_input = {'terrain': 'Rocky'}
        >>> get_priority_val(user_input, priority_list, mapping_dicts)
        0.5
    """
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
                                artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Core content-based recommendation pipeline using Deep Autoencoder, K-Means 
    clustering, and masked cosine similarity.
    
    This pipeline handles high-dimensional shoe data by projecting it into a 
    latent space for clustering, then calculating fine-grained similarity on 
    the original feature space using only user-specified features (masking).
    
    Workflow:
        1. Latent Space Projection: Encode user vector to 8D via Deep Autoencoder
        2. Cluster Routing: Select top ⌈K/3⌉ nearest clusters via K-Means
        3. Candidate Filtering: Extract shoes from selected clusters
        4. Masked Cosine Similarity: Calculate similarity only on valid_indices
        5. Ranking: Sort by match_score descending, return top 10
    
    Args:
        full_vector_raw (List[float]): The processed numerical vector representing 
            user preferences. Length matches the number of features in the trained 
            model (e.g., 31 for road, 36 for trail). Values are normalized [0, 1] 
            for binary features or [0, 1] for continuous features.
        valid_indices (List[int]): Indices of features explicitly provided by the 
            user (for masking). Prevents noise from default/neutral values in 
            similarity calculation. Example: [0, 5, 12, 18] means only these 
            4 features should be used for similarity.
        artifacts (Dict[str, Any]): Dictionary containing pre-loaded models and 
            datasets:
            - 'df_data' (pd.DataFrame): Shoe metadata with cluster labels
            - 'encoder_model' (tf.keras.Model): Trained Autoencoder encoder
            - 'kmeans_model' (sklearn.cluster.KMeans): Trained K-Means (K=5)
            - 'X_combined_data' (np.ndarray): Scaled feature matrix for all shoes
    
    Returns:
        List[Dict[str, Any]]: Top 10 recommended shoes with metadata and scores:
            [
                {
                    'shoe_id': 'R045',
                    'name': 'Nike Vaporfly 3',
                    'brand': 'Nike',
                    'cluster': 2,
                    'match_score': 0.97,
                    ... (other shoe attributes from df_data)
                },
                ...
            ]
        Returns empty list [] if no candidates found in selected clusters.
    
    Performance:
        - Encoding: ~8-15ms (TensorFlow inference)
        - K-Means transform: ~2-5ms
        - Cosine similarity: ~5-10ms (depends on candidate pool size)
        - Total: ~20-30ms per request
    
    Example:
        >>> # Assume artifacts already loaded
        >>> user_vector = [0.5, 1.0, 0.0, ...]  # 31 features for road
        >>> valid_idx = [0, 5, 12, 18, 25]      # User provided 5 preferences
        >>> results = run_recommendation_pipeline(user_vector, valid_idx, artifacts)
        >>> results[0]
        {
            'shoe_id': 'R045',
            'name': 'Nike Vaporfly 3',
            'brand': 'Nike',
            'cluster': 2,
            'match_score': 0.97
        }
    
    Notes:
        - Zero-vector safety: If user_vec_masked is all zeros, assigns zero scores 
          to prevent division-by-zero in cosine similarity
        - Cluster selection: Using ⌈K/3⌉ nearest clusters balances precision 
          (staying close to user's latent space) with recall (not missing good 
          candidates)
        - Feature masking: Critical for accuracy when user provides sparse input. 
          Without masking, default values dominate similarity calculation.
    """
    df_data = artifacts['df_data']
    encoder_model = artifacts['encoder_model']
    kmeans_model = artifacts['kmeans_model']
    X_combined_data = artifacts['X_combined_data']

    # 1. Latent Space Projection (Encoding)
    # Project the high-dimensional user vector into an 8D latent space using 
    # the Deep Autoencoder.
    full_vector_np = np.array([full_vector_raw])
    user_latent = encoder_model.predict(full_vector_np, verbose=0)
    
    # 2. Cluster Routing (Search Space Reduction)
    # Calculate distance to cluster centroids and select the top 1/3 closest 
    # clusters. This optimizes latency by filtering out irrelevant candidates.
    distances = kmeans_model.transform(user_latent)[0]
    n_select = math.ceil(kmeans_model.n_clusters / 3) 
    closest_clusters = np.argsort(distances)[:n_select]
    
    # 3. Candidate Filtering
    # Select shoes from the metadata pool that belong to the identified clusters.
    candidates = df_data[df_data['cluster'].isin(closest_clusters)].copy()
    if candidates.empty: 
        return []
    
    # 4. Masked Cosine Similarity
    # Calculate similarity ONLY on features the user has expressed a preference for.
    # This prevents 'noise' from default values in the similarity score.
    candidate_indices = candidates.index
    candidate_vectors = X_combined_data[candidate_indices]
    
    user_vec_masked = full_vector_np[:, valid_indices]
    cand_vecs_masked = candidate_vectors[:, valid_indices]
    
    # Safety check for zero-vectors to prevent division-by-zero errors
    if np.all(user_vec_masked == 0):
        scores = np.zeros(len(candidates))
    else:
        scores = cosine_similarity(user_vec_masked, cand_vecs_masked)[0]
    
    candidates['match_score'] = scores
    
    # 5. Ranking and Formatting
    # Sort candidates by match_score and return the Top 10 as a dictionary 
    # for API consumption.
    top_shoes = candidates.sort_values('match_score', ascending=False).head(10)
    return top_shoes.to_dict(orient='records')