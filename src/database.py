"""
Sonix-ML Database Abstraction Layer (DAL)
-----------------------------------------
Serves as the centralized interface between the Machine Learning Engine and 
the Supabase persistent storage. This module handles the "Extract" and "Load" 
phases of the ELT pipeline.

Core Responsibilities:
1. Connection Management: Secure initialization of Supabase Client using Env Vars.
2. Data Aggregation (Read Path): Merges explicit feedback (Reviews) and 
   implicit feedback (Favorites) into a unified Interaction Matrix.
3. Event Persistence (Write Path): Real-time UPSERT operations for user actions.

Data Transformation Logic:
    The ML model requires a single continuous target variable (score). 
    This module implements a heuristic scoring system:
    - Favorites (Binary): +1.0
    - Reviews (1-5 Stars): Mapped to symmetric scale [-2.0 to +2.0]
    - Aggregate Score: Sum of both (e.g., Liked + 5 Stars = Score 3.0)
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
# Critical: Database connection requires strict environment variable availability.
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
    # Fail-safe: Initialize as None to allow app startup, but log critical error
    logger.error(f"Critical: Failed to initialize Supabase client. Error: {e}")
    supabase = None

def fetch_and_merge_training_data() -> pd.DataFrame:
    """
    Constructs the User-Item Interaction Matrix by aggregating multiple 
    feedback signals from Supabase tables.
    
    This function performs an in-memory ETL process:
    1. Extract: Fetches raw rows from 'favorites' and 'reviews'.
    2. Transform:
       - Normalizes column names to standard [user_id, item_id].
       - Converts 5-star ratings to a centered symmetric scale (-2.0 to +2.0).
       - Rationale: Center-scaling helps the model distinguish between 
         "bad" (negative) and "neutral" (zero) items effectively.
    3. Load (Return): Returns a clean DataFrame ready for Matrix Factorization.

    Scoring Logic:
        - Favorite: +1.0
        - 5 Stars:  +2.0
        - 4 Stars:  +1.0
        - 3 Stars:  +0.1 (Neutral-positive)
        - 2 Stars:  -1.0
        - 1 Star:   -2.0
        
    Returns:
        pd.DataFrame: A DataFrame with columns ['user_id', 'item_id', 'rating'].
        - If Supabase is offline or empty, returns an empty DataFrame with headers.
    
    Performance Note:
        - Performs client-side joining (pandas) to reduce database load complexity.
        - Scales linearly with the number of total interactions (O(N)).
    """
    if not supabase:
        return pd.DataFrame(columns=['user_id', 'item_id', 'rating'])

    try:
        # --- 1. Process Favorites (Implicit Positive) ---
        res_fav = supabase.table("favorites").select("user_id, shoe_id").execute()
        df_fav = pd.DataFrame()
        
        if res_fav.data:
            df_fav = pd.DataFrame(res_fav.data)
            df_fav = df_fav.rename(columns={'shoe_id': 'item_id'})
            df_fav['score'] = 1.0  # Base score for a "Like"
            df_fav = df_fav[['user_id', 'item_id', 'score']]

        # --- 2. Process Reviews (Explicit Rating) ---
        res_rate = supabase.table("reviews").select("user_id, shoe_id, rating").execute()
        df_rate = pd.DataFrame()
        
        if res_rate.data:
            df_rate = pd.DataFrame(res_rate.data)
            df_rate = df_rate.rename(columns={'shoe_id': 'item_id'})
            
            # Map 1-5 stars to Symmetric Scale
            # Goal: Make 3 stars near-zero, 1 star negative, 5 stars highly positive
            def map_to_symmetric_scale(r):
                mapping = {5: 2.0, 4: 1.0, 3: 0.1, 2: -1.0, 1: -2.0}
                return mapping.get(r, 0.1)
            
            df_rate['score'] = df_rate['rating'].apply(map_to_symmetric_scale)
            df_rate = df_rate[['user_id', 'item_id', 'score']]

        # --- 3. Merge & Aggregate ---
        frames = []
        if not df_fav.empty: frames.append(df_fav)
        if not df_rate.empty: frames.append(df_rate)
        
        if not frames:
            return pd.DataFrame(columns=['user_id', 'item_id', 'rating'])

        df_combined = pd.concat(frames)
        
        # Aggregation Strategy: SUM
        # Example: User Likes (+1) AND Rates 5 stars (+2) = Total Score 3.0
        df_final = df_combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
        df_final = df_final.rename(columns={'score': 'rating'})
        
        logger.info(f"Interaction data merged. Total records: {len(df_final)}")
        return df_final

    except Exception as e:
        logger.error(f"Error fetching training data: {e}")
        return pd.DataFrame(columns=['user_id', 'item_id', 'rating'])

def save_interaction_routed(user_id: int, shoe_id: str, action_type: str, rating: Optional[int] = None):
    """
    Persists a user interaction event to the database using an UPSERT strategy.
    
    This function is Idempotent:
    - If the record exists, it updates it.
    - If the record is new, it creates it.
    - This prevents duplicate key errors during rapid UI interactions.

    Args:
        user_id (int): The ID of the user performing the action.
        shoe_id (str): The SKU/ID of the target shoe.
        action_type (str): The type of interaction. Supported:
            - 'like': Adds/Updates entry in 'favorites' table.
            - 'rate': Adds/Updates entry in 'reviews' table (requires 'rating').
        rating (int, optional): 1-5 integer, required if action_type is 'rate'.

    Error Handling:
        - Silently logs errors to avoid crashing the API thread.
        - Checks for Supabase connectivity before attempting write.
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
    Utility function to retrieve the raw catalog for a specific category.
    
    Args:
        shoe_type (str): 'road' or 'trail'.
        
    Returns:
        pd.DataFrame: Raw catalog data containing all feature columns.
    """
    if not supabase: return pd.DataFrame()
    try:
        # Note: Actual filtering by type handled in Python logic or SQL view if optimized
        response = supabase.table("shoes").select("*").execute()
        if not response.data: return pd.DataFrame()
        return pd.DataFrame(response.data)
    except Exception as e:
        logger.error(f"Failed to fetch shoes: {e}")
        return pd.DataFrame()