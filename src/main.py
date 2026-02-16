import os
import glob
import pickle
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, BackgroundTasks 
from fastapi.responses import RedirectResponse, UJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

# Project Internal Modules - sonix-ml project
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

# CT State: Triggered every 50 new interactions
interaction_counter = 0 
REFRESH_THRESHOLD = 50 

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
    """Background worker to rebuild the CF engine without blocking traffic."""
    global cf_engine
    logger.info("CT Process: Syncing global CF engine with latest Supabase data...")
    try:
        interaction_df = await run_in_threadpool(fetch_and_merge_training_data)
        master_metadata = road_artifacts['df_data'] 
        
        new_cf_engine = collaborative_filtering.UserCollaborativeRecommender(
            df_interactions=interaction_df, 
            shoe_metadata=master_metadata
        )
        
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
    default_response_class=UJSONResponse # Optimization for Locust performance
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas ---

class RecommendationRequest(BaseModel):
    shoe_name: str

class UserAction(BaseModel):
    user_id: int
    shoe_id: str
    action_type: str 
    value: Optional[int] = None 

# --- API Endpoints ---

@app.get("/")
async def root():
    """Redirects to documentation for easy testing."""
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    """Exposes current CT sync status for monitoring."""
    return {
        "status": "healthy", 
        "ct_sync_progress": f"{interaction_counter}/{REFRESH_THRESHOLD}",
        "engine": "Hybrid (CB + CF)"
    }

@app.post("/recommend/road", tags=["Content-Based"])
async def recommend_road(request: RecommendationRequest):
    """Road Shoe recommendations using Autoencoders."""
    try:
        recs = await run_in_threadpool(
            road_recommender.get_recommendations,
            user_input=request.shoe_name,
            artifacts=road_artifacts
        )
        return {"status": "success", "data": recs}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/recommend/trail", tags=["Content-Based"])
async def recommend_trail(request: RecommendationRequest):
    """Trail Shoe recommendations using Autoencoders."""
    try:
        recs = await run_in_threadpool(
            trail_recommender.get_recommendations,
            user_input=request.shoe_name,
            artifacts=trail_artifacts
        )
        return {"status": "success", "data": recs}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/interact", tags=["Collaborative Filtering"])
async def user_interaction(payload: UserAction, background_tasks: BackgroundTasks):
    global interaction_counter
    if not cf_engine: raise HTTPException(status_code=503, detail="CF engine not ready")
    
    try:
        is_like = True
        if payload.action_type.lower() in ["review", "edit_review"]:
            if payload.value is not None and payload.value < 3:
                is_like = False
        elif payload.action_type.lower() == "dislike":
            is_like = False
        
        recommendations = await run_in_threadpool(
            cf_engine.get_realtime_recommendations,
            user_id=payload.user_id,
            new_item_id=payload.shoe_id,
            new_rating_val=payload.value,
            is_like=is_like
        )
        
        # Continuous Training Trigger every 50 interactions
        interaction_counter += 1
        if interaction_counter >= REFRESH_THRESHOLD:
            logger.info(f"CT Trigger: Threshold {REFRESH_THRESHOLD} reached.")
            background_tasks.add_task(refresh_global_cf_engine)
            interaction_counter = 0 
            
        return {
            "status": "success", 
            "data": recommendations, 
            "sync_status": f"{interaction_counter}/{REFRESH_THRESHOLD}"
        }
    except Exception as e:
        logger.error(f"Interaction Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process interaction")

@app.get("/recommend/feed/{user_id}", tags=["Hybrid Feed"])
async def get_personalized_feed(user_id: int):
    """Generates personalized feed based on community data."""
    if not cf_engine: raise HTTPException(status_code=503, detail="Engine not ready")
    try:
        feed = await run_in_threadpool(cf_engine.get_user_recommendations, user_id=user_id)
        return {"status": "success", "user_id": user_id, "data": feed}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Optimized port for Hugging Face deployment
    uvicorn.run("src.main:app", host="0.0.0.0", port=7860, workers=4)