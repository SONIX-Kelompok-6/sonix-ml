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

# --- Project Imports ---
from .recommender import road_recommender, trail_recommender, collaborative_filtering
from .database import fetch_and_merge_training_data

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
logger = logging.getLogger("sonix_ml_api")

road_artifacts: Dict[str, Any] = {}
trail_artifacts: Dict[str, Any] = {}
cf_engine: Optional[collaborative_filtering.UserCollaborativeRecommender] = None

# CT Logic: Threshold 50 interactions
interaction_counter = 0 
REFRESH_THRESHOLD = 50 

# --- Core Functions ---

def get_latest_model_path(base_path: str, prefix: str = 'v_') -> str:
    search_pattern = os.path.join(base_path, f'{prefix}*')
    folders = glob.glob(search_pattern)
    if not folders: raise FileNotFoundError(f"No models in {base_path}")
    latest_version = max(folders, key=os.path.getmtime)
    logger.info(f"Version Control: Selected {os.path.basename(latest_version)}")
    return latest_version

def load_cb_artifacts(base_path: str) -> Dict[str, Any]:
    try:
        v_path = get_latest_model_path(base_path)
        df_meta = pd.read_pickle(os.path.join(v_path, "shoe_metadata.pkl"))
        with open(os.path.join(v_path, "shoe_features.pkl"), "rb") as f:
            X_features = pickle.load(f)
        with open(os.path.join(v_path, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        encoder = tf.keras.models.load_model(os.path.join(v_path, "shoe_encoder.keras"), compile=False)
        with open(os.path.join(v_path, "kmeans_model.pkl"), "rb") as f:
            kmeans = pickle.load(f)

        return {
            "df_data": df_meta, "X_combined_data": X_features, "scaler": scaler,
            "encoder_model": encoder, "kmeans_model": kmeans,
            "binary_cols": df_meta.attrs.get('binary_cols', []),
            "continuous_cols": df_meta.attrs.get('continuous_cols', [])
        }
    except Exception as e:
        logger.critical(f"Load Failure: {str(e)}")
        raise RuntimeError(f"Engine Init Failed: {str(e)}")

async def refresh_global_cf_engine():
    global cf_engine
    logger.info("CT Process: Syncing CF engine with Supabase...")
    try:
        interaction_df = await run_in_threadpool(fetch_and_merge_training_data)
        cf_engine = collaborative_filtering.UserCollaborativeRecommender(
            df_interactions=interaction_df, 
            shoe_metadata=road_artifacts['df_data']
        )
        logger.info("CT Success: Matrix updated.")
    except Exception as e:
        logger.error(f"CT Failure: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global road_artifacts, trail_artifacts, cf_engine
    logger.info("--- Starting Sonix-ML Engine ---")
    try:
        road_artifacts = load_cb_artifacts("model_artifacts/road")
        trail_artifacts = load_cb_artifacts("model_artifacts/trail")
        interaction_df = fetch_and_merge_training_data()
        cf_engine = collaborative_filtering.UserCollaborativeRecommender(
            df_interactions=interaction_df, 
            shoe_metadata=road_artifacts['df_data']
        )
        logger.info("--- API READY ---")
    except Exception as e:
        logger.critical(f"Startup Fatal: {e}")
        raise e
    yield 
    road_artifacts.clear()
    trail_artifacts.clear()

# --- App & Schemas (Input Lengkap Sesuai Kode Pertama Kamu) ---

app = FastAPI(title="Sonix-ML API", version="2.0.0", lifespan=lifespan, default_response_class=UJSONResponse)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

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

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health():
    return {"status": "healthy", "sync": f"{interaction_counter}/{REFRESH_THRESHOLD}"}

@app.post("/recommend/road", tags=["Content-Based"])
async def recommend_road(prefs: RoadInput):
    if not road_artifacts: raise HTTPException(status_code=503, detail="Road engine not initialized")
    try:
        input_data = {k: v for k, v in prefs.model_dump().items() if v is not None}
        results = await run_in_threadpool(road_recommender.get_recommendations, user_input=input_data, artifacts=road_artifacts)
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/trail", tags=["Content-Based"])
async def recommend_trail(prefs: TrailInput):
    if not trail_artifacts: raise HTTPException(status_code=503, detail="Trail engine not initialized")
    try:
        input_data = {k: v for k, v in prefs.model_dump().items() if v is not None}
        results = await run_in_threadpool(trail_recommender.get_recommendations, user_input=input_data, artifacts=trail_artifacts)
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interact", tags=["Collaborative Filtering"])
async def interact(payload: UserAction, background_tasks: BackgroundTasks):
    global interaction_counter
    if not cf_engine: raise HTTPException(status_code=503, detail="CF engine not ready")
    try:
        is_like = (payload.action_type.lower() == "like")
        recs = await run_in_threadpool(cf_engine.get_realtime_recommendations, user_id=payload.user_id, new_item_id=payload.shoe_id, new_rating_val=payload.value, is_like=is_like)
        
        interaction_counter += 1
        if interaction_counter >= REFRESH_THRESHOLD:
            background_tasks.add_task(refresh_global_cf_engine)
            interaction_counter = 0 
            
        return {"status": "success", "data": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/feed/{user_id}", tags=["Hybrid Feed"])
async def get_feed(user_id: int):
    if not cf_engine: raise HTTPException(status_code=503, detail="Engine not ready")
    try:
        feed = await run_in_threadpool(cf_engine.get_user_recommendations, user_id=user_id)
        return {"status": "success", "data": feed}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Final production port for Hugging Face
    uvicorn.run("src.main:app", host="0.0.0.0", port=7860, workers=4)