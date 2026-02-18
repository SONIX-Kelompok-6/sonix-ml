import os
import glob
import pickle
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import RedirectResponse, UJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

# --- Project Imports ---
# NOTE: Ensure database.py is in the same folder (src)
from .recommender import road_recommender, trail_recommender, collaborative_filtering
from .database import fetch_and_merge_training_data, save_interaction_routed 

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
logger = logging.getLogger("sonix_ml_api")

road_artifacts: Dict[str, Any] = {}
trail_artifacts: Dict[str, Any] = {}
cf_engine: Optional[collaborative_filtering.UserCollaborativeRecommender] = None

# CT Logic: Trigger retraining every 50 new interactions
interaction_counter = 0 
REFRESH_THRESHOLD = 50 

# --- Core Utility Functions ---

def get_latest_model_path(base_path: str, prefix: str = 'v_') -> str:
    """Retrieves the most recent versioned model directory."""
    search_pattern = os.path.join(base_path, f'{prefix}*')
    folders = glob.glob(search_pattern)
    if not folders: 
        raise FileNotFoundError(f"Critical: No model folders found in {base_path}")
    latest_version = max(folders, key=os.path.getmtime)
    logger.info(f"Version Control: Selected latest artifact '{os.path.basename(latest_version)}'")
    return latest_version

def load_cb_artifacts(base_path: str) -> Dict[str, Any]:
    """Loads all serialized artifacts into memory for inference."""
    try:
        v_path = get_latest_model_path(base_path)
        logger.info(f"Loading from: {v_path}")
        
        # Load artifacts
        df_meta = pd.read_pickle(os.path.join(v_path, "shoe_metadata.pkl"))
        with open(os.path.join(v_path, "shoe_features.pkl"), "rb") as f:
            X_features = pickle.load(f)
        with open(os.path.join(v_path, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        
        encoder = tf.keras.models.load_model(os.path.join(v_path, "shoe_encoder.h5"), compile=False)
        
        with open(os.path.join(v_path, "kmeans_model.pkl"), "rb") as f:
            kmeans = pickle.load(f)

        return {
            "df_data": df_meta, "X_combined_data": X_features, "scaler": scaler,
            "encoder_model": encoder, "kmeans_model": kmeans,
            "binary_cols": df_meta.attrs.get('binary_cols', []),
            "continuous_cols": df_meta.attrs.get('continuous_cols', [])
        }
    except Exception as e:
        logger.critical(f"Artifact Loading Failure in {base_path}: {str(e)}")
        raise RuntimeError(f"Failed to initialize ML engine: {str(e)}")

async def refresh_global_cf_engine():
    """Background task to rebuild the CF engine without downtime."""
    global cf_engine
    logger.info("CT Process: Syncing global CF engine with latest Supabase data...")
    try:
        interaction_df = await run_in_threadpool(fetch_and_merge_training_data)
        # Re-initialize engine with fresh data
        new_cf_engine = collaborative_filtering.UserCollaborativeRecommender(
            df_interactions=interaction_df, 
            shoe_metadata=road_artifacts['df_data']
        )
        cf_engine = new_cf_engine
        logger.info("CT Success: Global community matrix has been updated.")
    except Exception as e:
        logger.error(f"CT Failure: Background synchronization failed: {e}")

def format_ids(raw_ids: List[Any], prefix: str) -> List[str]:
    """
    Formats raw IDs into strings with prefixes (e.g., 5 -> 'R005').
    Reduces payload size by sending strings instead of full objects.
    """
    formatted = []
    for rid in raw_ids:
        # Convert to string, zero pad to 3 digits, prepend prefix
        # Example: ID 12 -> "012" -> "R012"
        s_id = str(rid).zfill(3)
        formatted.append(f"{prefix}{s_id}")
    return formatted

# --- API Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global road_artifacts, trail_artifacts, cf_engine
    logger.info("--- Starting Sonix-ML Hybrid Engine ---")
    try:
        # Load Content-Based Models
        road_artifacts = load_cb_artifacts("model_artifacts/road")
        trail_artifacts = load_cb_artifacts("model_artifacts/trail")
        
        # Load Collaborative Engine
        interaction_df = fetch_and_merge_training_data()
        cf_engine = collaborative_filtering.UserCollaborativeRecommender(
            df_interactions=interaction_df, 
            shoe_metadata=road_artifacts['df_data']
        )
        logger.info("--- Sonix-ML API is READY ---")
    except Exception as e:
        logger.critical(f"Fatal Startup Error: {e}")
        raise e
    yield 
    # Cleanup
    road_artifacts.clear()
    trail_artifacts.clear()

# --- Application Definition ---

app = FastAPI(
    title="Sonix-ML Hybrid Recommender API",
    version="2.1.0",
    lifespan=lifespan,
    default_response_class=UJSONResponse 
)

# --- CORS OPTIMIZATION ---
# max_age=86400 tells the browser to cache the Preflight response for 24 hours.
# This eliminates the 300ms latency on every request.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400, 
)

# --- Detailed Input Schemas ---

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
    action_type: str 
    value: Optional[int] = None 

# --- API Endpoints ---

@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "ct_sync_progress": f"{interaction_counter}/{REFRESH_THRESHOLD}"
    }

@app.post("/recommend/road", tags=["Content-Based"], response_model=List[str])
async def recommend_road(prefs: RoadInput):
    if not road_artifacts: raise HTTPException(status_code=503, detail="Engine not ready")
    try:
        input_data = {k: v for k, v in prefs.model_dump().items() if v is not None}
        
        # 1. Get Raw IDs from ML Engine
        raw_ids = await run_in_threadpool(
            road_recommender.get_recommendations, 
            user_input=input_data, 
            artifacts=road_artifacts
        )
        
        # 2. Format IDs (e.g., "R005") to reduce payload size
        formatted_ids = format_ids(raw_ids, "R")
        
        return formatted_ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/trail", tags=["Content-Based"], response_model=List[str])
async def recommend_trail(prefs: TrailInput):
    if not trail_artifacts: raise HTTPException(status_code=503, detail="Engine not ready")
    try:
        input_data = {k: v for k, v in prefs.model_dump().items() if v is not None}
        
        # 1. Get Raw IDs from ML Engine
        raw_ids = await run_in_threadpool(
            trail_recommender.get_recommendations, 
            user_input=input_data, 
            artifacts=trail_artifacts
        )
        
        # 2. Format IDs (e.g., "T012")
        formatted_ids = format_ids(raw_ids, "T")
        
        return formatted_ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interact", tags=["Collaborative Filtering"])
async def user_interaction(payload: UserAction, background_tasks: BackgroundTasks):
    global interaction_counter
    if not cf_engine: raise HTTPException(status_code=503, detail="Engine not ready")
    try:
        is_like = (payload.action_type.lower() == "like")
        
        # 1. Update In-Memory Engine (Instant Feedback)
        recommendations = await run_in_threadpool(
            cf_engine.get_realtime_recommendations, 
            user_id=payload.user_id, 
            new_item_id=payload.shoe_id, 
            new_rating_val=payload.value, 
            is_like=is_like
        )
        
        # 2. Persist to Database (Background Task) - FIX: No more data loss!
        background_tasks.add_task(
            save_interaction_routed,
            user_id=payload.user_id,
            shoe_id=payload.shoe_id,
            action_type=payload.action_type,
            rating=payload.value
        )
        
        # 3. Trigger Continuous Training (CT) if threshold met
        interaction_counter += 1
        if interaction_counter >= REFRESH_THRESHOLD:
            background_tasks.add_task(refresh_global_cf_engine)
            interaction_counter = 0 
            
        return {"status": "success", "data": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/feed/{user_id}", tags=["Hybrid Feed"], response_model=List[str])
async def get_feed(user_id: int):
    if not cf_engine: raise HTTPException(status_code=503, detail="Engine not ready")
    try:
        # Collaborative filtering returns raw IDs as strings/ints
        # We assume the frontend handles these or they match the DB directly
        feed = await run_in_threadpool(cf_engine.get_realtime_recommendations, user_id=user_id)
        
        # Convert all to string just in case
        return [str(item) for item in feed]
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Workers set to 4. Decrease to 1 or 2 if you get OOM errors on Hugging Face.
    uvicorn.run("src.main:app", host="0.0.0.0", port=7860, workers=4)