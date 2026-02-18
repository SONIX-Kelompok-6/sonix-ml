"""
Sonix-ML Database Integration Layer
-----------------------------------
This module handles all Supabase operations, including data ingestion for 
training and real-time persistence of user interactions.
"""

import os
import logging
import pandas as pd
from supabase import create_client, Client
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION & INITIALIZATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Optional[Client] = None

try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info(">>> Supabase Client Initialized Successfully.")
    else:
        logger.error("Critical: SUPABASE_URL or SUPABASE_KEY environment variables are missing.")
except Exception as e:
    logger.error(f"Critical: Failed to initialize Supabase client. Error: {e}")
    supabase = None

def fetch_and_merge_training_data() -> pd.DataFrame:
    """
    Retrieves interaction data (Favorites & Reviews) and merges them 
    into a unified Interaction Score DataFrame for the ML Engine.
    """
    if not supabase:
        return pd.DataFrame(columns=['user_id', 'item_id', 'rating'])

    try:
        # 1. Process Favorites (Like = 1.0)
        res_fav = supabase.table("favorites").select("user_id, shoe_id").execute()
        df_fav = pd.DataFrame()
        
        if res_fav.data:
            df_fav = pd.DataFrame(res_fav.data)
            df_fav = df_fav.rename(columns={'shoe_id': 'item_id'})
            df_fav['score'] = 1.0  # Explicit like
            df_fav = df_fav[['user_id', 'item_id', 'score']]

        # 2. Process Reviews (Rating 1-5 -> Weighted Score)
        res_rate = supabase.table("reviews").select("user_id, shoe_id, rating").execute()
        df_rate = pd.DataFrame()
        
        if res_rate.data:
            df_rate = pd.DataFrame(res_rate.data)
            df_rate = df_rate.rename(columns={'shoe_id': 'item_id'})
            
            # Map 1-5 stars to -2.0 to +2.0 scale
            def map_to_symmetric_scale(r):
                mapping = {5: 2.0, 4: 1.0, 3: 0.1, 2: -1.0, 1: -2.0}
                return mapping.get(r, 0.1)
            
            df_rate['score'] = df_rate['rating'].apply(map_to_symmetric_scale)
            df_rate = df_rate[['user_id', 'item_id', 'score']]

        # 3. Merge & Aggregate
        frames = []
        if not df_fav.empty: frames.append(df_fav)
        if not df_rate.empty: frames.append(df_rate)
        
        if not frames:
            return pd.DataFrame(columns=['user_id', 'item_id', 'rating'])

        df_combined = pd.concat(frames)
        
        # Sum scores if user liked AND rated the same shoe
        df_final = df_combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
        df_final = df_final.rename(columns={'score': 'rating'})
        
        logger.info(f"Interaction data merged. Total records: {len(df_final)}")
        return df_final

    except Exception as e:
        logger.error(f"Error fetching training data: {e}")
        return pd.DataFrame(columns=['user_id', 'item_id', 'rating'])

def save_interaction_routed(user_id: int, shoe_id: str, action_type: str, rating: Optional[int] = None):
    """
    Persists user interactions (Like/Rate) to Supabase using UPSERT.
    This ensures data isn't lost when the ML server restarts.
    """
    if not supabase: 
        logger.warning("Supabase offline, skipping save.")
        return

    try:
        # Normalize action type
        action = action_type.lower()
        
        if action == 'like':
            # Upsert into favorites
            data = {"user_id": user_id, "shoe_id": shoe_id}
            supabase.table("favorites").upsert(data, on_conflict="user_id, shoe_id").execute()
            logger.info(f"Persisted LIKE for User {user_id} -> Shoe {shoe_id}")
            
        elif action == 'rate':
            if rating is None: return 
            # Upsert into reviews
            data = {"user_id": user_id, "shoe_id": shoe_id, "rating": rating}
            supabase.table("reviews").upsert(data, on_conflict="user_id, shoe_id").execute()
            logger.info(f"Persisted RATING {rating} for User {user_id} -> Shoe {shoe_id}")
            
    except Exception as e:
        logger.error(f"Failed to persist interaction for user {user_id}: {e}")

def fetch_shoes_by_type(shoe_type: str) -> pd.DataFrame:
    """
    Optional utility to fetch raw shoe data if needed.
    """
    if not supabase: return pd.DataFrame()
    try:
        # Fallback: Fetch all shoes, logic handled in python if needed
        # This avoids errors if your DB IDs are integers (1, 2, 3) instead of strings (R001)
        response = supabase.table("shoes").select("*").execute()
        if not response.data: return pd.DataFrame()
        return pd.DataFrame(response.data)
    except Exception as e:
        logger.error(f"Failed to fetch shoes: {e}")
        return pd.DataFrame()