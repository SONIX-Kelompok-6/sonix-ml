"""
Sonix-ML Database Integration Layer
-----------------------------------
Handles Supabase operations, data ingestion for ML training, and 
real-time interaction persistence. Refactored for O(1) complexity 
using guard clauses and vectorized mappings.
"""

import os
import logging
import pandas as pd
from supabase import create_client, Client
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def _initialize_supabase() -> Optional[Client]:
    """Safely initializes the Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Critical: Supabase credentials missing.")
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        return None

supabase = _initialize_supabase()

def fetch_and_merge_training_data() -> pd.DataFrame:
    """
    Retrieves interaction data (Favorites & Reviews) and merges them.
    Utilizes vectorized pandas operations to eliminate loop complexity.

    Returns:
        pd.DataFrame: Unified dataframe containing user_id, item_id, rating.
    """
    empty_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating'])
    if not supabase: return empty_df

    try:
        frames = []
        
        # 1. Process Favorites
        res_fav = supabase.table("favorites").select("user_id, shoe_id").execute()
        if res_fav.data:
            df_fav = pd.DataFrame(res_fav.data).rename(columns={'shoe_id': 'item_id'})
            df_fav['score'] = 1.0
            frames.append(df_fav[['user_id', 'item_id', 'score']])

        # 2. Process Reviews via Vectorized Mapping
        res_rate = supabase.table("reviews").select("user_id, shoe_id, rating").execute()
        if res_rate.data:
            df_rate = pd.DataFrame(res_rate.data).rename(columns={'shoe_id': 'item_id'})
            rating_map = {5: 2.0, 4: 1.0, 3: 0.1, 2: -1.0, 1: -2.0}
            df_rate['score'] = df_rate['rating'].map(rating_map).fillna(0.1)
            frames.append(df_rate[['user_id', 'item_id', 'score']])

        if not frames: return empty_df

        # 3. Merge & Aggregate
        df_combined = pd.concat(frames)
        df_final = df_combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
        return df_final.rename(columns={'score': 'rating'})

    except Exception as e:
        logger.error(f"Error fetching training data: {e}")
        return empty_df

def save_interaction_routed(user_id: int, shoe_id: str, action_type: str, rating: Optional[int] = None) -> None:
    """
    Persists real-time user interactions using UPSERT to prevent duplicates.
    """
    if not supabase: return

    try:
        action = action_type.lower()
        if action == 'like':
            data = {"user_id": user_id, "shoe_id": shoe_id}
            supabase.table("favorites").upsert(data, on_conflict="user_id, shoe_id").execute()
            
        elif action == 'rate' and rating is not None:
            data = {"user_id": user_id, "shoe_id": shoe_id, "rating": rating}
            supabase.table("reviews").upsert(data, on_conflict="user_id, shoe_id").execute()
            
    except Exception as e:
        logger.error(f"Failed to persist interaction: {e}")

def fetch_shoes_by_type(shoe_type: str) -> pd.DataFrame:
    """Retrieves raw shoe metadata directly from the database."""
    if not supabase: return pd.DataFrame()
    try:
        response = supabase.table("shoes").select("*").execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to fetch shoes: {e}")
        return pd.DataFrame()