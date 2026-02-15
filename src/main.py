import os
import glob
import pickle
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Local Module Imports ---
from recommender import road_recommender, trail_recommender, collaborative_filtering
from database import fetch_and_merge_training_data

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Global In-Memory Artifacts ---
# Artifacts are stored globally to be accessible by endpoint functions 
# after being initialized in the lifespan context.
road_artifacts: Dict[str, Any] = {}
trail_artifacts: Dict[str, Any] = {}
cf_engine: Optional[collaborative_filtering.UserCollaborativeRecommender] = None

# --- Utility Functions ---

def get_latest_model_path(base_path: str, prefix: str = 'v_') -> str:
    """Retrieves the most recent versioned model directory based on modification time."""
    search_pattern = os.path.join(base_path, f'{prefix}*')
    folders = glob.glob(search_pattern)
    if not folders:
        raise FileNotFoundError(f"No versioned model folders found in: {base_path}")
    return max(folders, key=os.path.getmtime)

def load_cb_artifacts(base_path: str) -> Dict[str, Any]:
    """Loads all required ML artifacts for Content-Based recommendation."""
    try:
        v_path = get_latest_model_path(base_path)
        logger.info(f"Loading Content-Based artifacts from: {v_path}")
        
        # Load shoe metadata and categorical attributes
        df_meta = pd.read_pickle(os.path.join(v_path, "shoe_metadata.pkl"))
        
        # Load pre-computed feature matrices for similarity calculation
        with open(os.path.join(v_path, "shoe_features.pkl"), "rb") as f:
            X_features = pickle.load(f)
            
        with open(os.path.join(v_path, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

        return {
            "df_data": df_meta, 
            "X_combined_data": X_features,
            "scaler": scaler,
            "encoder_model": tf.keras.models.load_model(os.path.join(v_path, "shoe_encoder.keras")),
            "kmeans_model": pickle.load(open(os.path.join(v_path, "kmeans_model.pkl"), "rb")),
            "binary_cols": df_meta.attrs.get('binary_cols', []),
            "continuous_cols": df_meta.attrs.get('continuous_cols', [])
        }
    except Exception as e:
        logger.error(f"Critical failure during CB artifact loading: {e}")
        raise RuntimeError(f"Initialization Error: {str(e)}")

# --- API Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Pre-loading models into RAM is critical for sub-100ms P95 latency.
    """
    global road_artifacts, trail_artifacts, cf_engine
    
    logger.info("Initializing Sonix-ML Hybrid Engine...")
    try:
        # 1. Load Content-Based Artifacts for Road and Trail categories
        road_artifacts = load_cb_artifacts("model_artifacts/road")
        trail_artifacts = load_cb_artifacts("model_artifacts/trail")
        
        # 2. Bootstrap Collaborative Filtering with Supabase Data
        logger.info("Syncing training data from Supabase...")
        interaction_df = fetch_and_merge_training_data()
        
        logger.info("Initializing Real-time Collaborative Filtering Engine...")
        cf_engine = collaborative_filtering.UserCollaborativeRecommender(interaction_df)
        
        logger.info("Sonix-ML API v2.0 is now READY.")
        
    except Exception as e:
        logger.critical(f"API Startup Failed: {e}")
        raise e

    yield  # API handles requests
    
    # Clean up resources on shutdown
    logger.info("Shutting down API. Clearing memory artifacts...")
    road_artifacts.clear()
    trail_artifacts.clear()
    cf_engine = None

# --- Application Initialization ---

app = FastAPI(
    title="Sonix-ML Hybrid Recommender", 
    version="2.0", 
    lifespan=lifespan
)

# --- Request Schemas ---

class RoadInput(BaseModel):
    pace: Optional[str] = None
    arch_type: Optional[str] = None
    strike_pattern: Optional[str] = None
    foot_width: Optional[str] = None
    season: Optional[str] = None
    orthotic_usage: Optional[str] = None
    running_purpose: Optional[str] = None
    cushion_preferences: Optional[str] = None
    stability_need: Optional[str] = None

class TrailInput(BaseModel):
    pace: Optional[str] = None
    arch_type: Optional[str] = None
    strike_pattern: Optional[str] = None
    foot_width: Optional[str] = None
    season: Optional[str] = None
    orthotic_usage: Optional[str] = None
    terrain: Optional[str] = None
    rock_sensitive: Optional[str] = None
    water_resistance: Optional[str] = None

class UserAction(BaseModel):
    user_id: int
    shoe_id: str
    action_type: str  # "rate" or "like"
    value: Optional[int] = None # 1-5 for rating, None for like actions

# --- API Endpoints ---

@app.post("/predict/road", tags=["Content-Based"])
async def predict_road(prefs: RoadInput):
    """Generates personalized road shoe recommendations based on user bio-mechanics."""
    if not road_artifacts: 
        raise HTTPException(503, "Road recommendation engine is not initialized.")
    try:
        results = road_recommender.get_recommendations(prefs.model_dump(), road_artifacts)
        return {"status": "success", "category": "road", "results": results}
    except Exception as e:
        logger.error(f"Road Prediction Error: {e}")
        raise HTTPException(500, "Internal prediction error.")

@app.post("/predict/trail", tags=["Content-Based"])
async def predict_trail(prefs: TrailInput):
    """Generates personalized trail shoe recommendations with terrain-specific logic."""
    if not trail_artifacts: 
        raise HTTPException(503, "Trail recommendation engine is not initialized.")
    try:
        results = trail_recommender.get_recommendations(prefs.model_dump(), trail_artifacts)
        return {"status": "success", "category": "trail", "results": results}
    except Exception as e:
        logger.error(f"Trail Prediction Error: {e}")
        raise HTTPException(500, "Internal prediction error.")

@app.post("/interact", tags=["Real-time Collaborative"])
async def user_interaction(payload: UserAction):
    """
    Processes real-time user interactions (Like/Rate) to update the session-based 
    collaborative filtering model instantly.
    """
    if not cf_engine: 
        raise HTTPException(503, "Collaborative Filtering engine is not initialized.")
    
    try:
        is_like = (payload.action_type.lower() == "like")
        
        # Immediate Memory Update: Ensures recommendations reflect the user's 
        # latest action without waiting for batch retraining.
        recommendations = cf_engine.get_realtime_recommendations(
            user_id=payload.user_id,
            new_item_id=payload.shoe_id,
            new_rating_val=payload.value,
            is_like=is_like,
            n_neighbors=10
        )
        
        return {
            "status": "success",
            "message": "Real-time user preferences updated.",
            "data": {
                "triggered_by": payload.shoe_id,
                "recommendations": recommendations
            }
        }
    except Exception as e:
        logger.error(f"Real-time Interaction Error: {e}")
        raise HTTPException(500, "Failed to process interaction.")

if __name__ == "__main__":
    import uvicorn
    # Local development server with auto-reload enabled
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)