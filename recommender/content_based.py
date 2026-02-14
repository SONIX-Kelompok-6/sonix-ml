import numpy as np
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_priority_val(user_input, priority_list, mapping_dicts):
    """Fungsi helper untuk mengambil nilai prioritas."""
    for source_key in priority_list:
        if source_key in user_input and user_input[source_key]:
            user_choice = user_input[source_key]
            if source_key in mapping_dicts:
                mapping = mapping_dicts[source_key]
                if user_choice in mapping:
                    return mapping[user_choice]
    return 0.5

def run_recommendation_pipeline(full_vector, valid_idx, artifacts):
    """
    Logika inti rekomendasi.
    Menerima vector yang SUDAH diproses, lalu menjalankan pencarian kandidat.
    """
    # Unpack artifacts
    df_data = artifacts['df_data']
    encoder_model = artifacts['encoder_model']
    kmeans_model = artifacts['kmeans_model']
    X_combined_data = artifacts['X_combined_data']

    # 1. Cluster Routing
    full_vector_np = np.array([full_vector])
    user_latent = encoder_model.predict(full_vector_np, verbose=0)
    distances = kmeans_model.transform(user_latent)[0]
    
    n_select = math.ceil(kmeans_model.n_clusters / 3) 
    closest_clusters = np.argsort(distances)[:n_select]
    
    # 2. Filter Candidates
    candidates = df_data[df_data['cluster'].isin(closest_clusters)].copy()
    if candidates.empty: 
        return []
    
    # 3. Masked Similarity Calculation
    candidate_indices = candidates.index
    candidate_vectors = X_combined_data[candidate_indices]
    
    user_vec_masked = full_vector_np[:, valid_idx]
    cand_vecs_masked = candidate_vectors[:, valid_idx]
    
    if np.all(user_vec_masked == 0):
        scores = np.zeros(len(candidates))
    else:
        scores = cosine_similarity(user_vec_masked, cand_vecs_masked)[0]
    
    candidates['match_score'] = scores
    
    # 4. Return Top 10
    top_shoes = candidates.sort_values('match_score', ascending=False).head(10)
    return top_shoes.to_dict(orient='records')