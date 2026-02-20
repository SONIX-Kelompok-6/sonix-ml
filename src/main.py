"""
Sonix-ML Hybrid Recommender API
-------------------------------
Orchestrates the Content-Based (Deep Autoencoder + K-Means) and 
Collaborative Filtering (UBCF NearestNeighbors) recommendation engines.

Built with FastAPI for asynchronous, high-throughput inference operations.
Optimized with LRU Caching for sub-millisecond response times on frequent queries.
Includes integrated real-time latency diagnostics.
"""

import os
import glob
import pickle
import logging
import json
import time
import statistics
from functools import lru_cache
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import RedirectResponse, UJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

# --- Project Imports ---
from .recommender import road_recommender, trail_recommender, collaborative_filtering
from .database import fetch_and_merge_training_data 
# Removed save_interaction_routed as BE handles DB writes

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
logger = logging.getLogger("sonix_ml_api")

road_artifacts: Dict[str, Any] = {}
trail_artifacts: Dict[str, Any] = {}
cf_engine: Optional[collaborative_filtering.UserCollaborativeRecommender] = None

interaction_counter = 0 
REFRESH_THRESHOLD = 50 

# --- Diagnostics Buffer ---
MAX_LOGS = 1000 
LATENCY_LOGS: Dict[str, List[float]] = {
    "road": [],
    "trail": [],
    "interact": [],
    "feed": []
}

# --- Core Utility Functions ---

def get_latest_model_path(base_path: str, prefix: str = 'v_') -> str:
    search_pattern = os.path.join(base_path, f'{prefix}*')
    folders = glob.glob(search_pattern)
    if not folders: 
        raise FileNotFoundError(f"Critical: No model folders found in {base_path}")
    
    latest_version = max(folders, key=os.path.getmtime)
    logger.info(f"Version Control: Selected latest artifact '{os.path.basename(latest_version)}'")
    return latest_version


def load_cb_artifacts(base_path: str) -> Dict[str, Any]:
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
            
        # Adhering to strict Deep Learning approach
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
    global cf_engine
    logger.info("CT Process: Syncing global CF engine with latest database state...")
    try:
        interaction_df = await run_in_threadpool(fetch_and_merge_training_data)
        cf_engine = collaborative_filtering.UserCollaborativeRecommender(
            df_interactions=interaction_df, 
            shoe_metadata=road_artifacts.get('df_data', pd.DataFrame())
        )
        logger.info("CT Success: Global community matrix has been updated.")
    except Exception as e:
        logger.error(f"CT Failure: Background synchronization failed: {e}")

# --- Inference Cache Engines ---

@lru_cache(maxsize=1024)
def cached_road_inference(payload_str: str) -> List[str]:
    user_input = json.loads(payload_str)
    return road_recommender.get_recommendations(user_input=user_input, artifacts=road_artifacts)

@lru_cache(maxsize=1024)
def cached_trail_inference(payload_str: str) -> List[str]:
    user_input = json.loads(payload_str)
    return trail_recommender.get_recommendations(user_input=user_input, artifacts=trail_artifacts)

# --- API Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global road_artifacts, trail_artifacts, cf_engine
    logger.info("--- Starting Sonix-ML Hybrid Engine ---")
    try:
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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400, 
)

# --- Latency Tracking Middleware ---

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    process_ms = process_time * 1000
    
    response.headers["X-Process-Time"] = str(process_time)
    print(f"[{request.url.path}] Processed in {process_ms:.2f} ms")
    
    path = request.url.path
    if "/recommend/road" in path:
        LATENCY_LOGS["road"].append(process_ms)
        if len(LATENCY_LOGS["road"]) > MAX_LOGS: LATENCY_LOGS["road"].pop(0)
    elif "/recommend/trail" in path:
        LATENCY_LOGS["trail"].append(process_ms)
        if len(LATENCY_LOGS["trail"]) > MAX_LOGS: LATENCY_LOGS["trail"].pop(0)
    elif "/interact" in path:
        LATENCY_LOGS["interact"].append(process_ms)
        if len(LATENCY_LOGS["interact"]) > MAX_LOGS: LATENCY_LOGS["interact"].pop(0)
    elif "/recommend/feed" in path:
        LATENCY_LOGS["feed"].append(process_ms)
        if len(LATENCY_LOGS["feed"]) > MAX_LOGS: LATENCY_LOGS["feed"].pop(0)
        
    return response

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

@app.get("/report/latency", tags=["Diagnostics"])
async def get_latency_report():
    """Generates a real-time statistical report of server processing times."""
    report = {}
    for endpoint, times in LATENCY_LOGS.items():
        if not times:
            report[endpoint] = "No data yet"
            continue
            
        report[endpoint] = {
            "total_requests": len(times),
            "avg_ms": round(statistics.mean(times), 2),
            "p50_ms": round(statistics.median(times), 2),
            "p95_ms": round(statistics.quantiles(times, n=100)[94] if len(times) > 1 else times[0], 2),
            "max_ms": round(max(times), 2)
        }
        
    return {"status": "success", "internal_latency_report": report}

@app.post("/recommend/road", tags=["Content-Based"], response_model=List[str])
async def recommend_road(prefs: RoadInput):
    if not road_artifacts: 
        raise HTTPException(status_code=503, detail="Road engine not ready")
        
    try:
        input_data = prefs.model_dump(exclude_none=True)
        payload_str = json.dumps(input_data, sort_keys=True)
        
        return await run_in_threadpool(cached_road_inference, payload_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/trail", tags=["Content-Based"], response_model=List[str])
async def recommend_trail(prefs: TrailInput):
    if not trail_artifacts: 
        raise HTTPException(status_code=503, detail="Trail engine not ready")
        
    try:
        input_data = prefs.model_dump(exclude_none=True)
        payload_str = json.dumps(input_data, sort_keys=True)
        
        return await run_in_threadpool(cached_trail_inference, payload_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interact", tags=["Collaborative Filtering"], response_model=List[str])
async def user_interaction(payload: UserAction, background_tasks: BackgroundTasks):
    global interaction_counter
    if not cf_engine: 
        raise HTTPException(status_code=503, detail="CF engine not ready")
        
    try:
        is_like = (payload.action_type.lower() == "like")
        
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

        # Returning strictly a list of IDs per system requirements
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/feed/{user_id}", tags=["Hybrid Feed"], response_model=List[str])
async def get_feed(user_id: int):
    if not cf_engine: 
        raise HTTPException(status_code=503, detail="CF engine not ready")
        
    try:
        feed = await run_in_threadpool(cf_engine.get_realtime_recommendations, user_id=user_id)
        return feed
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=7860, workers=1)