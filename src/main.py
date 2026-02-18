"""
Sonix-ML Inference API Gateway
------------------------------
The central entry point for the Sonix Recommendation Engine. This FastAPI application 
orchestrates a Hybrid Recommendation System combining Content-Based Filtering (Cold Start) 
and Collaborative Filtering (User-User/Item-Item).

Architecture Overview:
    1. Stateful Service: Loads heavy ML artifacts (Autoencoders, Scalers, Matrices) 
       into Global Memory on startup to ensure sub-millisecond inference latency.
    2. Asynchronous Concurrency: Uses 'run_in_threadpool' to offload CPU-bound 
       matrix operations, keeping the main Event Loop non-blocking.
    3. Continuous Learning (CT): Implements a counter-based trigger system to 
       refresh the Collaborative Filtering matrix in the background without downtime.

Global State Management:
    - road_artifacts (Dict): Holds TensorFlow Encoders & K-Means models for Road shoes.
    - trail_artifacts (Dict): Holds TensorFlow Encoders & K-Means models for Trail shoes.
    - cf_engine (Class): Instance of UserCollaborativeRecommender holding the 
      User-Item Interaction Matrix.

Lifecycle Events:
    - Startup:
        a. Connects to Supabase.
        b. Fetches latest Interaction Data.
        c. Loads the latest versioned Model Artifacts from disk.
        d. Initializes the Collaborative Filtering Engine.
    - Shutdown:
        a. Clears GPU/RAM memory.

Environment Dependencies:
    - SUPABASE_URL & SUPABASE_KEY: For database persistence.
    - MODEL_ARTIFACTS_DIR: Directory structure for versioned models.
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
# Ensure these match your actual folder structure
from .recommender import road_recommender, trail_recommender, collaborative_filtering
from .database import fetch_and_merge_training_data, save_interaction_routed 

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
logger = logging.getLogger("sonix_ml_api")

# Global State Containers
road_artifacts: Dict[str, Any] = {}
trail_artifacts: Dict[str, Any] = {}
cf_engine: Optional[collaborative_filtering.UserCollaborativeRecommender] = None

# CT Logic: Trigger retraining every 50 new interactions
interaction_counter = 0 
REFRESH_THRESHOLD = 50 

# --- Core Utility Functions ---

def get_latest_model_path(base_path: str, prefix: str = 'v_') -> str:
    """
    Scans the artifact directory and selects the most recently created version folder.
    Follows the naming convention: {base_path}/v_{YYYYMMDD_HHMMSS}/
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
    Deserializes machine learning artifacts from disk into memory.
    
    Operations:
    1. Identifies latest version via timestamp.
    2. Loads Pandas Metadata (Product Catalog).
    3. Loads Feature Scalers (MinMax) & Feature Arrays (Numpy).
    4. Loads TensorFlow Keras Model (Autoencoder).
    5. Loads Scikit-Learn Model (K-Means).

    Args:
        base_path (str): Root directory for specific category (e.g., 'model_artifacts/road').

    Returns:
        Dict: A packaged dictionary containing all components needed for inference.
    """
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
    """
    Background Task: Performs a 'Hot Reload' of the Collaborative Filtering Engine.
    
    Triggered when 'interaction_counter' exceeds 'REFRESH_THRESHOLD'.
    Fetching fresh data and rebuilding the matrix happens in isolation; 
    the global pointer 'cf_engine' is only swapped once the new build is ready, 
    ensuring zero downtime for incoming requests.
    """
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

# --- API Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the startup and shutdown sequence of the ML Application.
    Ensures all heavy models are loaded into RAM before the API starts accepting requests.
    """
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
    version="2.2.0", # Clean Version
    lifespan=lifespan,
    default_response_class=UJSONResponse 
)

# --- CORS OPTIMIZATION (Anti-Latency) ---
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
    """
    Generates Content-Based recommendations for Road Running Shoes.
    
    Method:
    1. Vectorizes user preferences (One-Hot Encoding + Scaling).
    2. Projects vector into Learned Latent Space (via Encoder).
    3. Identifies the nearest cluster centroid (K-Means).
    4. Computes Cosine Similarity within the cluster to find top matches.
    """
    if not road_artifacts: raise HTTPException(status_code=503, detail="Engine not ready")
    try:
        input_data = {k: v for k, v in prefs.model_dump().items() if v is not None}
        
        # Offload heavy calculation to threadpool
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
    Generates Content-Based recommendations for Trail Running Shoes.
    Similar to Road, but utilizes trail-specific feature sets (terrain, protection).
    """
    if not trail_artifacts: raise HTTPException(status_code=503, detail="Engine not ready")
    try:
        input_data = {k: v for k, v in prefs.model_dump().items() if v is not None}
        
        # Offload heavy calculation to threadpool
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
    Handles Real-time User Interactions (Like/Rate).
    
    Dual Responsibility:
    1. Immediate Feedback: Returns updated recommendations based on the new action.
    2. Data Persistence: Queues a background task to save to Supabase.
    3. Continuous Training: Increments counter; triggers matrix rebuild if threshold met.
    """
    global interaction_counter
    if not cf_engine: raise HTTPException(status_code=503, detail="Engine not ready")
    try:
        is_like = (payload.action_type.lower() == "like")
        
        # Get Real-time Feedback (Collaborative)
        recommendations = await run_in_threadpool(
            cf_engine.get_realtime_recommendations, 
            user_id=payload.user_id, 
            new_item_id=payload.shoe_id, 
            new_rating_val=payload.value, 
            is_like=is_like
        )
        
        # Persist to DB (Background Task)
        # This prevents the API from waiting on the Database Write
        background_tasks.add_task(
            save_interaction_routed,
            user_id=payload.user_id,
            shoe_id=payload.shoe_id,
            action_type=payload.action_type,
            rating=payload.value
        )
        
        # Check for Continuous Training Trigger
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
    Retrieves the personalized feed for a user based on their historical interactions.
    If the user is new (Cold Start), this will fallback to popular items 
    (logic handled inside cf_engine).
    """
    if not cf_engine: raise HTTPException(status_code=503, detail="Engine not ready")
    try:
        feed = await run_in_threadpool(cf_engine.get_realtime_recommendations, user_id=user_id)
        # Convert to string just in case
        return [str(item) for item in feed]
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=7860, workers=4)