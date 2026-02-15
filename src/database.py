"""
Sonix-ML Database Integration Layer
-----------------------------------
This module handles all Supabase operations, including data ingestion for 
training and real-time persistence of user interactions.

Functionality:
- Automated Client Initialization.
- Hybrid Data Aggregation (Likes + Ratings).
- Interaction Routing (Upsert logic).
- Categorical Shoe Retrieval.
"""

import os
import logging
import pandas as pd
from supabase import create_client, Client
from typing import Optional, List

# Configure logging for database operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION & INITIALIZATION ---
# SECURITY NOTE: Never hardcode keys in production. Use environment variables.
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
    Retrieves interaction data from 'favorites' and 'reviews' tables and 
    merges them into a unified Interaction Score DataFrame.
    
    Logic:
    - 'Like' Action: Assigned a base score of +1.0.
    - 'Rating' (1-5): Converted to a symmetric scale (-2.0 to +2.0).
    - Aggregation: If a user has both liked and rated an item, the scores 
      are summed to represent a higher confidence in the preference.
    
    Returns:
        pd.DataFrame: A processed DataFrame with columns ['user_id', 'item_id', 'rating'].
    """
    if not supabase:
        return pd.DataFrame(columns=['user_id', 'item_id', 'rating'])

    try:
        # 1. Process Favorites (Binary Interaction)
        res_fav = supabase.table("favorites").select("user_id, shoe_id").execute()
        
        df_fav = pd.DataFrame()
        if res_fav.data:
            df_fav = pd.DataFrame(res_fav.data)
            df_fav = df_fav.rename(columns={'shoe_id': 'item_id'})
            df_fav['score'] = 1.0  # Constant weight for "Like"
            df_fav = df_fav[['user_id', 'item_id', 'score']]

        # 2. Process Reviews (Numerical Interaction)
        res_rate = supabase.table("reviews").select("user_id, shoe_id, rating").execute()
        
        df_rate = pd.DataFrame()
        if res_rate.data:
            df_rate = pd.DataFrame(res_rate.data)
            df_rate = df_rate.rename(columns={'shoe_id': 'item_id'})
            
            # Symmetric Scale Mapping: Star Rating -> Model Weights
            # Ensures consistency with UserCollaborativeRecommender._convert_rating
            def map_to_symmetric_scale(r):
                mapping = {5: 2.0, 4: 1.0, 3: 0.1, 2: -1.0, 1: -2.0}
                return mapping.get(r, 0.1)
            
            df_rate['score'] = df_rate['rating'].apply(map_to_symmetric_scale)
            df_rate = df_rate[['user_id', 'item_id', 'score']]

        # 3. Aggregation Logic
        frames = []
        if not df_fav.empty: frames.append(df_fav)
        if not df_rate.empty: frames.append(df_rate)
        
        if not frames:
            return pd.DataFrame(columns=['user_id', 'item_id', 'rating'])

        # Concatenate disparate interaction sources
        df_combined = pd.concat(frames)

        # Sum scores for identical User-Item pairs
        df_final = df_combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
        
        # Standardize nomenclature for Collaborative Filtering Engine
        df_final = df_final.rename(columns={'score': 'rating'})
        
        logger.info(f"Interaction data merged. Total records: {len(df_final)}")
        return df_final

    except Exception as e:
        logger.error(f"Error fetching training data: {e}")
        return pd.DataFrame(columns=['user_id', 'item_id', 'rating'])

def save_interaction_routed(user_id: int, shoe_id: str, action_type: str, rating: Optional[int]):
    """
    Persists user interactions into the appropriate table based on the action type.
    
    Utilizes Supabase 'Upsert' logic to prevent duplicate entries and update 
    existing preferences.
    """
    if not supabase: return

    try:
        if action_type == 'like':
            data = {"user_id": user_id, "shoe_id": shoe_id}
            # 'on_conflict' ensures the composite key (user, shoe) remains unique
            supabase.table("favorites").upsert(data, on_conflict="user_id, shoe_id").execute()
            
        elif action_type == 'rate':
            if rating is None: return 
            data = {"user_id": user_id, "shoe_id": shoe_id, "rating": rating}
            supabase.table("reviews").upsert(data, on_conflict="user_id, shoe_id").execute()
            
    except Exception as e:
        logger.error(f"Failed to persist interaction for user {user_id}: {e}")

def fetch_shoes_by_type(shoe_type: str) -> pd.DataFrame:
    """
    Retrieves shoe catalog data filtered by category prefix.
    
    Args:
        shoe_type: 'road' (id starts with 'R') or 'trail' (id starts with 'T').
        
    Returns:
        pd.DataFrame: Catalog data for the specified category.
    """
    if not supabase:
        logger.warning("Supabase client is offline.")
        return pd.DataFrame()

    # Prefix convention: R = Road, T = Trail
    prefix = "R" if shoe_type.lower() == "road" else "T"
    
    try:
        # Case-insensitive pattern matching on shoe_id
        response = supabase.table("shoes").select("*").ilike("shoe_id", f"{prefix}%").execute()
        
        if not response.data:
            logger.warning(f"No records found for shoe type: {shoe_type}")
            return pd.DataFrame()
            
        df = pd.DataFrame(response.data)
        logger.info(f"Loaded {len(df)} records for {shoe_type} category.")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch {shoe_type} shoes: {e}")
        return pd.DataFrame()