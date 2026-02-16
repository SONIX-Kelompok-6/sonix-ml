import os
import glob
import pickle
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, BackgroundTasks # Required for non-blocking CT
from fastapi.responses import RedirectResponse, UJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

from .recommender import road_recommender, trail_recommender, collaborative_filtering
from .database import fetch_and_merge_training_data

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("sonix_ml_api")

# --- Global In-Memory Artifacts & State ---
road_artifacts: Dict[str, Any] = {}
trail_artifacts: Dict[str, Any] = {}
cf_engine: Optional[collaborative_filtering.UserCollaborativeRecommender] = None

# CT State: Tracks community activity to trigger global retraining
interaction_counter = 0 
REFRESH_THRESHOLD = 50 # Retrain the global model every 50 new interactions

# --- Utility Functions ---

def get_latest_model_path(base_path: str, prefix: str = 'v_') -> str:
    """Retrieves the newest versioned model folder based on modification time."""
    search_pattern = os.path.join(base_path, f'{prefix}*')
    folders = glob.glob(search_pattern)
    if not folders:
        raise FileNotFoundError(f"Critical: No model version folders found in {base_path}")
    latest_version = max(folders, key=os.path.getmtime)
    logger.info(f"Version Control: Selected latest artifact '{os.path.basename(latest_version)}'")
    return latest_version

def load_cb_artifacts(base_path: str) -> Dict[str, Any]:
    """Loads Content-Based ML artifacts (Autoencoders, Scalers, Metadata) into RAM."""
    try:
        v_path = get_latest_model_path(base_path)
        logger.info(f"Loading artifacts from: {v_path}")
        df_meta = pd.read_pickle(os.path.join(v_path, "shoe_metadata.pkl"))
        with open(os.path.join(v_path, "shoe_features.pkl"), "rb") as f:
            X_features = pickle.load(f)
        with open(os.path.join(v_path, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        encoder = tf.keras.models.load_model(os.path.join(v_path, "shoe_encoder.keras"), compile=False)
        with open(os.path.join(v_path, "kmeans_model.pkl"), "rb") as f:
            kmeans = pickle.load(f)

        return {
            "df_data": df_meta, 
            "X_combined_data": X_features,
            "scaler": scaler,
            "encoder_model": encoder,
            "kmeans_model": kmeans,
            "binary_cols": df_meta.attrs.get('binary_cols', []),
            "continuous_cols": df_meta.attrs.get('continuous_cols', [])
        }
    except Exception as e:
        logger.critical(f"Artifact Loading Failure in {base_path}: {str(e)}")
        raise RuntimeError(f"Failed to initialize ML engine: {str(e)}")

# --- Continuous Training (CT) Background Task ---

async def refresh_global_cf_engine():
    """
    Background worker to rebuild the Collaborative Filtering engine.
    Synchronizes the global community 'knowledge' with the latest Supabase data.
    """
    global cf_engine
    logger.info("CT Process: Syncing global CF engine with latest Supabase data...")
    try:
        # Offload heavy DB fetch and computation to maintain low latency (<100ms)
        interaction_df = await run_in_threadpool(fetch_and_merge_training_data)
        master_metadata = road_artifacts['df_data'] 
        
        # Instantiate a fresh engine with the new data
        new_cf_engine = collaborative_filtering.UserCollaborativeRecommender(
            df_interactions=interaction_df, 
            shoe_metadata=master_metadata
        )
        
        # Hot-swap the engine in RAM
        cf_engine = new_cf_engine
        logger.info("CT Success: Global community matrix has been updated.")
    except Exception as e:
        logger.error(f"CT Failure: Background synchronization failed: {e}")

# --- API Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global road_artifacts, trail_artifacts, cf_engine
    logger.info("--- Starting Sonix-ML Hybrid Engine ---")
    try:
        # Load CB models strictly before accepting traffic
        road_artifacts = load_cb_artifacts("model_artifacts/road")
        trail_artifacts = load_cb_artifacts("model_artifacts/trail")
        
        # Initial Bootstrap of CF data
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
    road_artifacts.clear()
    trail_artifacts.clear()
    cf_engine = None

# --- Application Definition ---

app = FastAPI(
    title="Sonix-ML Hybrid Recommender API",
    version="2.0.0",
    lifespan=lifespan,
    default_response_class=UJSONResponse # Optimization: High-speed JSON serialization
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas ---

class UserAction(BaseModel):
    user_id: int
    shoe_id: str
    action_type: str 
    value: Optional[int] = None 

# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Exposes current CT sync status for monitoring (e.g., in Locust)."""
    return {
        "status": "healthy", 
        "ct_sync_progress": f"{interaction_counter}/{REFRESH_THRESHOLD}"
    }

@app.post("/interact", tags=["Collaborative Filtering"])
async def user_interaction(payload: UserAction, background_tasks: BackgroundTasks):
    global interaction_counter
    if not cf_engine: raise HTTPException(status_code=503, detail="CF engine not ready")
    
    try:
        # 1. Smart Sentiment Logic: Map stars to binary likes
        is_like = True
        if payload.action_type.lower() in ["review", "edit_review"]:
            if payload.value is not None and payload.value < 3:
                is_like = False
        elif payload.action_type.lower() == "dislike":
            is_like = False
        
        # 2. Personal Real-time Update: Immediately inject this user's action
        recommendations = await run_in_threadpool(
            cf_engine.get_realtime_recommendations,
            user_id=payload.user_id,
            new_item_id=payload.shoe_id,
            new_rating_val=payload.value,
            is_like=is_like
        )
        
        # 3. Global Continuous Training Trigger: Track total activity
        interaction_counter += 1
        if interaction_counter >= REFRESH_THRESHOLD:
            logger.info(f"CT Trigger: Threshold {REFRESH_THRESHOLD} reached. Initializing background sync.")
            background_tasks.add_task(refresh_global_cf_engine)
            interaction_counter = 0 # Reset counter post-trigger
            
        return {
            "status": "success", 
            "data": recommendations, 
            "sync_status": f"{interaction_counter}/{REFRESH_THRESHOLD}"
        }
    except Exception as e:
        logger.error(f"Interaction Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process interaction")

# Note: Other endpoints (road/trail/feed) remain optimized with run_in_threadpool

if __name__ == "__main__":
    import uvicorn
    # Local dev run with workers to simulate production behavior
    uvicorn.run("src.main:app", host="0.0.0.0", port=7860, workers=4)