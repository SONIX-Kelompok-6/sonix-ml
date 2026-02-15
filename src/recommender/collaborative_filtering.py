import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)

class UserCollaborativeRecommender:
    """
    Engine rekomendasi berbasis User-Based Collaborative Filtering (Memory Based).
    Fitur: Skala Simetris (-2 s/d +2), Sparse Matrix, Real-time Injection.
    """

    def __init__(self, df_interactions: pd.DataFrame):
        if df_interactions.empty:
            logger.warning("CF Engine Warning: No interaction data found. Starting in COLD START mode.")
            self.user_ids = []
            self.item_ids = []
            return

        # 1. Pivot Data (Users x Items) -> Sparse Matrix
        self.pivot_df = df_interactions.pivot(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        # 2. Index Mapping
        self.user_ids = list(self.pivot_df.index)
        self.item_ids = list(self.pivot_df.columns)
        self.user_map = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_map = {iid: i for i, iid in enumerate(self.item_ids)}
        
        # 3. Build Model
        self.sparse_matrix = csr_matrix(self.pivot_df.values)
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(self.sparse_matrix)
        
        logger.info(f"CF Engine initialized: {len(self.user_ids)} Users, {len(self.item_ids)} Items.")

    def _convert_rating(self, rating: Optional[int], is_like_action: bool = False) -> float:
        """
        Konversi input user ke skala simetris untuk In-Memory Injection.
        Harus konsisten dengan logika di database.py.
        """
        if is_like_action:
            return 1.0
        
        # Default 0.1 (Netral Positif) jika rating kosong
        if rating is None or rating == 0:
            return 0.1

        mapping = {
            1: -2.0, # Sangat Benci
            2: -1.0, # Tidak Suka
            3:  0.1, # Netral (sedikit positif)
            4:  1.0, # Suka
            5:  2.0  # Sangat Suka
        }
        return mapping.get(rating, 0.1)

    def get_realtime_recommendations(self, 
                                     user_id: int, 
                                     new_item_id: Optional[str] = None, 
                                     new_rating_val: Optional[int] = None,
                                     is_like: bool = False,
                                     n_neighbors: int = 5) -> List[Dict[str, float]]:
        """
        Mendapatkan rekomendasi tetangga terdekat.
        Melakukan 'Injection' interaksi terbaru ke memori sementara agar hasil update instan.
        """
        # A. Siapkan Vector User (Current State)
        if user_id in self.user_map:
            user_idx = self.user_map[user_id]
            user_vector = self.sparse_matrix[user_idx].toarray()
        else:
            # Cold User: Vector Nol
            user_vector = np.zeros((1, len(self.item_ids)))

        # B. Inject New Interaction (Temporary Update di RAM)
        if new_item_id and new_item_id in self.item_map:
            item_idx = self.item_map[new_item_id]
            score_val = self._convert_rating(new_rating_val, is_like_action=is_like)
            user_vector[0, item_idx] = score_val # Update nilai

        # C. Inference (Cari Tetangga)
        try:
            # n_neighbors+1 karena hasil terdekat biasanya diri sendiri
            distances, indices = self.model.kneighbors(user_vector, n_neighbors=n_neighbors + 1)
        except ValueError:
            return [] # Fail safe jika model belum fit sempurna

        # D. Kalkulasi Skor
        rec_scores: Dict[str, float] = {}
        
        for i, neighbor_idx in enumerate(indices[0]):
            # Skip diri sendiri
            if user_id in self.user_map and neighbor_idx == self.user_map[user_id]:
                continue
                
            similarity = 1.0 - distances[0][i]
            if similarity <= 0: continue # Abaikan yang tidak mirip

            # Ambil vector tetangga
            neighbor_vector = self.sparse_matrix[neighbor_idx].toarray()[0]
            
            # Hanya ambil item yang disukai tetangga (Rating > 0)
            liked_indices = np.where(neighbor_vector > 0)[0]
            
            for it_idx in liked_indices:
                item_id_rec = self.item_ids[it_idx]
                rating_neighbor = neighbor_vector[it_idx]
                
                # Rumus: Rating Tetangga * Similarity
                score = rating_neighbor * similarity
                rec_scores[item_id_rec] = rec_scores.get(item_id_rec, 0.0) + score

        # E. Filter Barang yang sudah dilihat/diinteraksikan user sendiri
        seen_indices = np.where(user_vector[0] != 0)[0]
        seen_items: Set[str] = {self.item_ids[i] for i in seen_indices}
        
        final_recs = []
        for item, score in rec_scores.items():
            if item not in seen_items:
                final_recs.append({'shoe_id': item, 'cf_score': round(float(score), 3)})
        
        # Sort Highest Score
        final_recs.sort(key=lambda x: x['cf_score'], reverse=True)
        
        return final_recs[:10]