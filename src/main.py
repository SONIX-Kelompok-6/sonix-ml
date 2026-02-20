"""
Sonix-ML Hybrid Recommender API
-------------------------------
Orchestrates the Content-Based (Deep Autoencoder + K-Means) and 
Collaborative Filtering (Neural CF) recommendation engines.

Built with FastAPI for asynchronous, high-throughput inference operations.
Refactored for optimal cyclomatic complexity, robust exception handling, 
and comprehensive documentation.
"""

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
from src.recommender import road_recommender, trail_recommender, collaborative_filtering
from src.database import fetch_and_merge_training_data, save_interaction_routed 

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
logger = logging.getLogger("sonix_ml_api")

road_artifacts: Dict[str, Any] = {}
trail_artifacts: Dict[str, Any] = {}
cf_engine: Optional[collaborative_filtering.NeuralCollaborativeRecommender] = None

interaction_counter = 0 
REFRESH_THRESHOLD = 50 
NCF_MODEL_PATH = "model_artifacts/collaborative/ncf_model.h5"

# --- Core Utility Functions ---

def get_latest_model_path(base_path: str, prefix: str = 'v_') -> str:
    """
    Retrieves the most recent versioned model directory based on modification time.
    
    Args:
        base_path (str): The root directory containing versioned artifact folders.
        prefix (str, optional): The prefix of the versioned folders. Defaults to 'v_'.

    Raises:
        FileNotFoundError: If no valid artifact directories are found.

    Returns:
        str: Absolute path to the latest model directory.
    """
    search_pattern = os.path.join(base_path, f'{prefix}*')
    folders = glob.glob(search_pattern)
    if not folders: 
        raise FileNotFoundError(f"Critical: No model folders found in {base_path}")
    
    latest_version = max(folders, key=os.path.getmtime)
    logger.info(f"Version Control: Selected latest artifact '{os.path.basename(latest_version)}'")
    return latest_version


def load_cb_artifacts(base_path: str) -> Dict[str, Any]:
    """
    Loads all serialized artifacts (models, scalers, metadata) into memory.
    
    Args:
        base_path (str): The root directory for a specific model type (e.g., 'road' or 'trail').

    Raises:
        RuntimeError: If deserialization or model loading fails.

    Returns:
        Dict[str, Any]: A dictionary containing the loaded ML artifacts.
    """
    try:
        v_path = get_latest_model_path(base_path)
        logger.info(f"Loading from: {v_path}")
        
        df_meta = pd.read_pickle(os.path.join(v_path, "shoe_metadata.pkl"))
        
        with open(os.path.join(v_path, "shoe_features.pkl"), "rb") as f:
            X_features = pickle.load(f)
            
        with open(os.path.join(v_path, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
            
        with open(os.path.join(v_path, "kmeans_model.pkl"), "rb") as f:
            kmeans = pickle.load(f)
            
        encoder = tf.keras.models.load_model(os.path.join(v_path, "shoe_encoder.h5"), compile=False)

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


async def refresh_global_cf_engine() -> None:
    """
    Background task to rebuild the Neural CF engine and sync with the latest data.
    Ensures zero-downtime updates to the real-time community matrix.
    """
    global cf_engine
    logger.info("CT Process: Syncing global Neural CF engine with latest database state...")
    try:
        # Re-initialize engine with fresh weights or updated data structure
        new_cf_engine = collaborative_filtering.NeuralCollaborativeRecommender(
            model_path=NCF_MODEL_PATH, 
            shoe_metadata=road_artifacts.get('df_data', pd.DataFrame())
        )
        cf_engine = new_cf_engine
        logger.info("CT Success: Global community matrix has been updated.")
    except Exception as e:
        logger.error(f"CT Failure: Background synchronization failed: {e}")

# --- API Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the startup and shutdown lifecycle of the FastAPI application.
    Loads all deep learning models into memory before accepting requests.
    """
    global road_artifacts, trail_artifacts, cf_engine
    logger.info("--- Starting Sonix-ML Hybrid Engine ---")
    try:
        road_artifacts = load_cb_artifacts("model_artifacts/road")
        trail_artifacts = load_cb_artifacts("model_artifacts/trail")
        
        cf_engine = collaborative_filtering.NeuralCollaborativeRecommender(
            model_path=NCF_MODEL_PATH, 
            shoe_metadata=road_artifacts['df_data']
        )
        logger.info("--- Sonix-ML API is READY ---")
    except Exception as e:
        logger.critical(f"Fatal Startup Error: {e}")
        raise e
        
    yield 
    
    # Graceful Cleanup
    road_artifacts.clear()
    trail_artifacts.clear()

# --- Application Definition ---

app = FastAPI(
    title="Sonix-ML Hybrid Recommender API",
    version="2.2.0",
    lifespan=lifespan,
    default_response_class=UJSONResponse 
)

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
    """Schema for road running shoe preferences."""
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
    """Schema for trail running shoe preferences."""
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
    """Schema for user interaction tracking (Likes/Ratings)."""
    user_id: int
    shoe_id: str
    action_type: str 
    value: Optional[int] = None 

# --- API Endpoints ---

@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirects root URL to API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    """Returns the operational status of the API and sync progress."""
    return {
        "status": "healthy", 
        "ct_sync_progress": f"{interaction_counter}/{REFRESH_THRESHOLD}"
    }

@app.post("/recommend/road", tags=["Content-Based"], response_model=List[str])
async def recommend_road(prefs: RoadInput):
    """
    Generates content-based road shoe recommendations.
    
    Returns:
        List[str]: A list of recommended shoe IDs.
    """
    if not road_artifacts: 
        raise HTTPException(status_code=503, detail="Road engine not ready")
        
    try:
        input_data = {k: v for k, v in prefs.model_dump().items() if v is not None}
        raw_ids = await run_in_threadpool(
            road_recommender.get_recommendations, 
            user_input=input_data, 
            artifacts=road_artifacts
        )
        return raw_ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/trail", tags=["Content-Based"], response_model=List[str])
async def recommend_trail(prefs: TrailInput):
    """
    Generates content-based trail shoe recommendations.
    
    Returns:
        List[str]: A list of recommended shoe IDs.
    """
    if not trail_artifacts: 
        raise HTTPException(status_code=503, detail="Trail engine not ready")
        
    try:
        input_data = {k: v for k, v in prefs.model_dump().items() if v is not None}
        raw_ids = await run_in_threadpool(
            trail_recommender.get_recommendations, 
            user_input=input_data, 
            artifacts=trail_artifacts
        )
        return raw_ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interact", tags=["Collaborative Filtering"])
async def user_interaction(payload: UserAction, background_tasks: BackgroundTasks):
    """
    Records a user interaction, fetches real-time NCF recommendations, 
    and triggers background synchronization.
    
    Returns:
        Dict: Status and list of recommended shoe IDs.
    """
    global interaction_counter
    if not cf_engine: 
        raise HTTPException(status_code=503, detail="Neural CF engine not ready")
        
    try:
        is_like = (payload.action_type.lower() == "like")
        
        recommendations = await run_in_threadpool(
            cf_engine.get_realtime_recommendations, 
            user_id=payload.user_id, 
            new_item_id=payload.shoe_id, 
            is_like=is_like
        )
        
        background_tasks.add_task(
            save_interaction_routed,
            user_id=payload.user_id,
            shoe_id=payload.shoe_id,
            action_type=payload.action_type,
            rating=payload.value
        )
        
        interaction_counter += 1
        if interaction_counter >= REFRESH_THRESHOLD:
            background_tasks.add_task(refresh_global_cf_engine)
            interaction_counter = 0 
            
        return {"status": "success", "data": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/feed/{user_id}", tags=["Hybrid Feed"], response_model=List[str])
async def get_feed(user_id: int):
    """
    Retrieves the customized, real-time recommendation feed for a specific user.
    
    Returns:
        List[str]: A list of recommended shoe IDs.
    """
    if not cf_engine: 
        raise HTTPException(status_code=503, detail="Neural CF engine not ready")
        
    try:
        feed = await run_in_threadpool(cf_engine.get_realtime_recommendations, user_id=user_id)
        return [str(item) for item in feed]
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=7860, workers=4) 