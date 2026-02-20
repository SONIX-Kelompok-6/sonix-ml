"""
Sonix-ML Database Integration Layer
-----------------------------------
Handles Supabase operations, including data ingestion for machine learning 
training and real-time persistence of user interactions. Refactored with 
guard clauses and vectorized mapping for O(1) complexity per block.
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

    Utilizes vectorized pandas operations to eliminate loop complexity 
    when translating user ratings into normalized weights.

    Returns:
        pd.DataFrame: A unified dataframe containing 'user_id', 'item_id', 
                      and aggregated 'rating' scores. Returns an empty dataframe 
                      if no data or connection is available.
    """
    if not supabase:
        return pd.DataFrame(columns=['user_id', 'item_id', 'rating'])

    try:
        frames = []

        # 1. Process Favorites (Implicit Like = 1.0)
        res_fav = supabase.table("favorites").select("user_id, shoe_id").execute()
        if res_fav.data:
            df_fav = pd.DataFrame(res_fav.data).rename(columns={'shoe_id': 'item_id'})
            df_fav['score'] = 1.0
            frames.append(df_fav[['user_id', 'item_id', 'score']])

        # 2. Process Reviews (Explicit Rating 1-5 -> Weighted Score)
        res_rate = supabase.table("reviews").select("user_id, shoe_id, rating").execute()
        if res_rate.data:
            df_rate = pd.DataFrame(res_rate.data).rename(columns={'shoe_id': 'item_id'})
            
            # Vectorized mapping replacing internal functions and loops
            rating_map = {5: 2.0, 4: 1.0, 3: 0.1, 2: -1.0, 1: -2.0}
            df_rate['score'] = df_rate['rating'].map(rating_map).fillna(0.1)
            frames.append(df_rate[['user_id', 'item_id', 'score']])

        # 3. Merge & Aggregate (Guard Clause for empty states)
        if not frames:
            return pd.DataFrame(columns=['user_id', 'item_id', 'rating'])

        df_combined = pd.concat(frames)
        
        # Sum scores if a user liked AND rated the same shoe
        df_final = df_combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
        df_final = df_final.rename(columns={'score': 'rating'})
        
        logger.info(f"Interaction data merged. Total records: {len(df_final)}")
        return df_final

    except Exception as e:
        logger.error(f"Error fetching training data: {e}")
        return pd.DataFrame(columns=['user_id', 'item_id', 'rating'])


def save_interaction_routed(user_id: int, shoe_id: str, action_type: str, rating: Optional[int] = None) -> None:
    """
    Persists real-time user interactions (Like/Rate) to Supabase.
    
    Uses UPSERT methodology to prevent duplicate entries and ensure 
    data is preserved across ML server restarts.

    Args:
        user_id (int): Unique identifier of the user making the interaction.
        shoe_id (str): Unique identifier of the target shoe.
        action_type (str): Type of interaction ('like' or 'rate').
        rating (Optional[int]): Numerical rating (1-5), required if action_type is 'rate'.

    Returns:
        None
    """
    if not supabase: 
        logger.warning("Supabase offline, skipping save.")
        return

    try:
        action = action_type.lower()
        
        if action == 'like':
            data = {"user_id": user_id, "shoe_id": shoe_id}
            supabase.table("favorites").upsert(data, on_conflict="user_id, shoe_id").execute()
            logger.info(f"Persisted LIKE for User {user_id} -> Shoe {shoe_id}")
            
        elif action == 'rate' and rating is not None:
            data = {"user_id": user_id, "shoe_id": shoe_id, "rating": rating}
            supabase.table("reviews").upsert(data, on_conflict="user_id, shoe_id").execute()
            logger.info(f"Persisted RATING {rating} for User {user_id} -> Shoe {shoe_id}")
            
    except Exception as e:
        logger.error(f"Failed to persist interaction for user {user_id}: {e}")


def fetch_shoes_by_type(shoe_type: str) -> pd.DataFrame:
    """
    Retrieves raw shoe metadata directly from the database.
    
    Acts as a fallback fetching mechanism for the training pipeline 
    if specific category segregation is needed downstream.

    Args:
        shoe_type (str): Category of the shoe ('road', 'trail', etc.).

    Returns:
        pd.DataFrame: A dataframe containing all fetched shoe records. 
                      Returns an empty dataframe on failure.
    """
    if not supabase: 
        return pd.DataFrame()
        
    try:
        response = supabase.table("shoes").select("*").execute()
        if not response.data: 
            return pd.DataFrame()
        return pd.DataFrame(response.data)
    except Exception as e:
        logger.error(f"Failed to fetch shoes: {e}")
        return pd.DataFrame()