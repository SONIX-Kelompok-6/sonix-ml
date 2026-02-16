import os
import glob
import pickle
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Local Module Imports ---
# Assumes 'src' is in PYTHONPATH or using relative imports within the package
from .recommender import road_recommender, trail_recommender, collaborative_filtering
from .database import fetch_and_merge_training_data

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("sonix_ml_api")

# --- Global In-Memory Artifacts ---
# These are loaded once during startup to ensure low-latency inference.
road_artifacts: Dict[str, Any] = {}
trail_artifacts: Dict[str, Any] = {}
cf_engine: Optional[collaborative_filtering.UserCollaborativeRecommender] = None

# --- Utility Functions ---

def get_latest_model_path(base_path: str, prefix: str = 'v_') -> str:
    """
    Retrieves the most recent versioned model directory based on filesystem modification time.
    Strategy: automated rollback/rollforward based on the latest folder present.
    """
    search_pattern = os.path.join(base_path, f'{prefix}*')
    folders = glob.glob(search_pattern)
    if not folders:
        raise FileNotFoundError(f"Critical: No model version folders found in {base_path}")
    
    # Return the folder with the latest modification timestamp
    latest_version = max(folders, key=os.path.getmtime)
    logger.info(f"Version Control: Selected latest artifact '{os.path.basename(latest_version)}'")
    return latest_version

def load_cb_artifacts(base_path: str) -> Dict[str, Any]:
    """
    Loads serialized ML artifacts (Scalers, PCA, Autoencoders, Metadata) into memory.
    """
    try:
        v_path = get_latest_model_path(base_path)
        logger.info(f"Loading artifacts from: {v_path}")
        
        # 1. Load Metadata (Pandas DataFrame)
        df_meta = pd.read_pickle(os.path.join(v_path, "shoe_metadata.pkl"))
        
        # 2. Load Pre-computed Feature Matrices
        with open(os.path.join(v_path, "shoe_features.pkl"), "rb") as f:
            X_features = pickle.load(f)
            
        # 3. Load Scikit-Learn Scaler
        with open(os.path.join(v_path, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

        # 4. Load TensorFlow/Keras Autoencoder
        # Note: 'compile=False' prevents warnings if custom metrics are missing
        encoder = tf.keras.models.load_model(os.path.join(v_path, "shoe_encoder.keras"), compile=False)
        
        # 5. Load K-Means Clusterer
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

# --- API Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Orchestrates the application startup and shutdown sequence.
    """
    global road_artifacts, trail_artifacts, cf_engine
    
    logger.info("--- Starting Sonix-ML Hybrid Engine ---")
    
    try:
        # Phase 1: Initialize Content-Based Engines
        # Load both Road and Trail models into RAM strictly before accepting requests.
        road_artifacts = load_cb_artifacts("model_artifacts/road")
        trail_artifacts = load_cb_artifacts("model_artifacts/trail")
        
        # Phase 2: Bootstrap Collaborative Filtering
        logger.info("Synchronizing user interaction data from Supabase...")
        interaction_df = fetch_and_merge_training_data()
        
        # Load master shoe metadata for enrichment of CF recommendations
        master_metadata = road_artifacts['df_data'] 
        
        logger.info("Initializing Collaborative Filtering (Matrix Factorization)...")
        
        # Insert the master metadata into the CF engine for enrichment purposes
        cf_engine = collaborative_filtering.UserCollaborativeRecommender(
            df_interactions=interaction_df, 
            shoe_metadata=master_metadata
        )
        
        logger.info("--- Sonix-ML API is READY to serve requests ---")
        
    except Exception as e:
        logger.critical(f"Fatal Startup Error: {e}")
        # Re-raise to prevent the container from starting in a broken state
        raise e

    yield  # The application runs here
    
    # Phase 3: Graceful Shutdown
    logger.info("Shutting down... Cleaning up memory resources.")
    road_artifacts.clear()
    trail_artifacts.clear()
    cf_engine = None

# --- Application Definition ---

app = FastAPI(
    title="Sonix-ML Hybrid Recommender API",
    description="High-performance recommendation engine for running shoes utilizing Deep Autoencoders and Collaborative Filtering.",
    version="2.0.0",
    lifespan=lifespan
)

# --- Middleware Configuration ---

# CORS (Cross-Origin Resource Sharing)
# Essential for allowing Frontend applications (React, Next.js, Mobile) to communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with specific frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Data Models (Schemas) ---

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
    action_type: str  # e.g., "rate", "like", "view"
    value: Optional[int] = None  # 1-5 for ratings, None for implicit likes

# --- API Endpoints ---

@app.get("/", include_in_schema=False)
async def root_redirect():
    """
    Redirects the root URL to the interactive API documentation (Swagger UI).
    Prevents 404 errors on the home page.
    """
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    """Kubernetes/Docker health probe endpoint."""
    return {"status": "healthy", "version": "2.0.0"}

@app.post("/recommend/road", tags=["Content-Based Recommendation"])
async def recommend_road(prefs: RoadInput):
    """
    Generates personalized ROAD shoe recommendations.
    
    Process:
    1. Validates user biomechanical inputs.
    2. Maps inputs to the latent feature space using the Road Autoencoder.
    3. Performs similarity search within the appropriate K-Means cluster.
    """
    if not road_artifacts: 
        raise HTTPException(status_code=503, detail="Road engine not initialized")
    
    try:
        # Dump Pydantic model to dict, filtering out None values
        input_data = {k: v for k, v in prefs.model_dump().items() if v is not None}
        results = road_recommender.get_recommendations(input_data, road_artifacts)
        return {"status": "success", "category": "road", "data": results}
    except Exception as e:
        logger.error(f"Road Inference Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.post("/recommend/trail", tags=["Content-Based Recommendation"])
async def recommend_trail(prefs: TrailInput):
    """
    Generates personalized TRAIL shoe recommendations.
    Includes logic for terrain specificities (Rock Plate, Waterproofing, Lug Depth).
    """
    if not trail_artifacts: 
        raise HTTPException(status_code=503, detail="Trail engine not initialized")
    
    try:
        input_data = {k: v for k, v in prefs.model_dump().items() if v is not None}
        results = trail_recommender.get_recommendations(input_data, trail_artifacts)
        return {"status": "success", "category": "trail", "data": results}
    except Exception as e:
        logger.error(f"Trail Inference Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.post("/interact", tags=["Collaborative Filtering"])
async def user_interaction(payload: UserAction):
    """
    Records real-time user interactions to update the recommendation engine instantenously.
    Supports 'Cold Start' mitigation by blending collaborative signals with content attributes.
    """
    if not cf_engine: 
        raise HTTPException(status_code=503, detail="Collaborative engine not initialized")
    
    try:
        is_like = (payload.action_type.lower() == "like")
        
        # Perform partial update on the CF model
        recommendations = cf_engine.get_realtime_recommendations(
            user_id=payload.user_id,
            new_item_id=payload.shoe_id,
            new_rating_val=payload.value,
            is_like=is_like,
            n_neighbors=10
        )
        
        return {
            "status": "success",
            "message": "User preference profile updated",
            "collaborative_candidates": recommendations
        }
    except Exception as e:
        logger.error(f"Interaction Processing Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process interaction")

# Tambahkan endpoint ini di main.py

@app.get("/recommend/user/{user_id}", tags=["Collaborative Filtering"])
async def get_user_feed(user_id: int):
    """
    Fetch 'You Might Also Like' recommendations based on user's PAST history.
    Used for populating the Home Feed on page load.
    """
    if not cf_engine:
        raise HTTPException(status_code=503, detail="Collaborative engine not initialized")
    
    try:
        # Panggil fungsi yang SAMA, tapi tanpa parameter interaction (item_id & rating)
        # Engine akan otomatis menggunakan data historis yang ada di memori.
        recommendations = cf_engine.get_realtime_recommendations(
            user_id=user_id,
            new_item_id=None,  # Tidak ada aksi baru
            new_rating_val=None
        )
        
        # Jika user baru (belum ada history), return list kosong []
        return {
            "status": "success",
            "user_id": user_id,
            "data": recommendations
        }
            
    except Exception as e:
        logger.error(f"Feed Retrieval Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch user feed")

# --- Execution Entry Point ---
if __name__ == "__main__":
    import uvicorn
    # Note: Host 0.0.0.0 is required for Docker containers
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)