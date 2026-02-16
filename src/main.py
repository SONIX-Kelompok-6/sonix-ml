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

# --- Project Module Imports ---
from .recommender import road_recommender, trail_recommender, collaborative_filtering
from .database import fetch_and_merge_training_data

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
logger = logging.getLogger("sonix_ml_api")

# --- Global State ---
road_artifacts: Dict[str, Any] = {}
trail_artifacts: Dict[str, Any] = {}
cf_engine: Optional[collaborative_filtering.UserCollaborativeRecommender] = None

# Continuous Training Logic: Threshold 50
interaction_counter = 0 
REFRESH_THRESHOLD = 50 

# --- Utility Functions ---

def get_latest_model_path(base_path: str, prefix: str = 'v_') -> str:
    """Retrieves the newest model version from the filesystem."""
    search_pattern = os.path.join(base_path, f'{prefix}*')
    folders = glob.glob(search_pattern)
    if not folders: 
        raise FileNotFoundError(f"Critical: No model folders found in {base_path}")
    return max(folders, key=os.path.getmtime)

def load_cb_artifacts(base_path: str) -> Dict[str, Any]:
    """Loads all ML artifacts for the sonix-ml engine."""
    try:
        v_path = get_latest_model_path(base_path)
        logger.info(f"Loading version: {os.path.basename(v_path)}")
        
        # Grounded in Python 3.11 environment requirements
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
        logger.critical(f"Artifact Load Failure: {str(e)}")
        raise RuntimeError(f"Engine Init Failed: {str(e)}")

async def refresh_global_cf_engine():
    """Background task to sync collaborative data."""
    global cf_engine
    logger.info("CT Process: Syncing CF engine with Supabase data...")
    try:
        interaction_df = await run_in_threadpool(fetch_and_merge_training_data)
        # Using the original class initialization from your first version
        new_cf_engine = collaborative_filtering.UserCollaborativeRecommender(interaction_df)
        cf_engine = new_cf_engine
        logger.info("CT Success: Matrix updated.")
    except Exception as e:
        logger.error(f"CT Failure: {e}")

# --- FastAPI Lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global road_artifacts, trail_artifacts, cf_engine
    logger.info("--- Starting Sonix-ML Hybrid Engine ---")
    try:
        road_artifacts = load_cb_artifacts("model_artifacts/road")
        trail_artifacts = load_cb_artifacts("model_artifacts/trail")
        
        # Bootstrap initial data
        interaction_df = fetch_and_merge_training_data()
        cf_engine = collaborative_filtering.UserCollaborativeRecommender(interaction_df)
        logger.info("--- Sonix-ML API is READY ---")
    except Exception as e:
        logger.critical(f"Fatal Startup Error: {e}")
        raise e
    yield 
    road_artifacts.clear()
    trail_artifacts.clear()

# --- App Definition ---

app = FastAPI(
    title="Sonix-ML Hybrid Recommender API",
    version="2.0.0",
    lifespan=lifespan,
    default_response_class=UJSONResponse
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Original Detailed Pydantic Models ---

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
    """Redirects to Swagger docs to solve the 'Not Found' error."""
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "ct_sync": f"{interaction_counter}/{REFRESH_THRESHOLD}"
    }

@app.post("/recommend/road", tags=["Content-Based"])
async def recommend_road(prefs: RoadInput):
    if not road_artifacts: raise HTTPException(status_code=503, detail="Engine not ready")
    try:
        input_data = {k: v for k, v in prefs.model_dump().items() if v is not None}
        # FIX: Pylance 'user_input' parameter fix
        results = await run_in_threadpool(
            road_recommender.get_recommendations, 
            user_input=input_data, 
            artifacts=road_artifacts
        )
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/trail", tags=["Content-Based"])
async def recommend_trail(prefs: TrailInput):
    if not trail_artifacts: raise HTTPException(status_code=503, detail="Engine not ready")
    try:
        input_data = {k: v for k, v in prefs.model_dump().items() if v is not None}
        # FIX: Pylance 'user_input' parameter fix
        results = await run_in_threadpool(
            trail_recommender.get_recommendations, 
            user_input=input_data, 
            artifacts=trail_artifacts
        )
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interact", tags=["Collaborative Filtering"])
async def user_interaction(payload: UserAction, background_tasks: BackgroundTasks):
    global interaction_counter
    if not cf_engine: raise HTTPException(status_code=503, detail="Engine not ready")
    
    try:
        is_like = (payload.action_type.lower() == "like")
        # Logic to trigger retraining every 50 interactions
        recommendations = await run_in_threadpool(
            cf_engine.get_realtime_recommendations,
            user_id=payload.user_id,
            new_item_id=payload.shoe_id,
            new_rating_val=payload.value,
            is_like=is_like
        )
        
        interaction_counter += 1
        if interaction_counter >= REFRESH_THRESHOLD:
            background_tasks.add_task(refresh_global_cf_engine)
            interaction_counter = 0 
            
        return {"status": "success", "data": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/feed/{user_id}", tags=["Hybrid Feed"])
async def get_feed(user_id: int):
    """FIX: Resolved AttributeError by using verified class methods."""
    if not cf_engine: raise HTTPException(status_code=503, detail="Engine not ready")
    try:
        # Reusing realtime method for the feed as per your class definition
        feed = await run_in_threadpool(cf_engine.get_realtime_recommendations, user_id=user_id)
        return {"status": "success", "data": feed}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Final port for Hugging Face
    uvicorn.run("src.main:app", host="0.0.0.0", port=7860, workers=4)