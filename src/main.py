import os
import glob
import pickle
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
# Hapus BackgroundTasks karena kita gak perlu simpan background lagi
from pydantic import BaseModel

# --- Local Imports ---
from recommender import road_recommender, trail_recommender, collaborative_filtering
# Hapus save_interaction_routed, kita cuma butuh fetch
from database import fetch_and_merge_training_data

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Artifacts ---
road_artifacts: Dict[str, Any] = {}
trail_artifacts: Dict[str, Any] = {}
cf_engine: Optional[collaborative_filtering.UserCollaborativeRecommender] = None

# --- Helper: Content-Based Loading ---
def get_latest_model_path(base_path: str, prefix: str = 'v_') -> str:
    search_pattern = os.path.join(base_path, f'{prefix}*')
    folders = glob.glob(search_pattern)
    if not folders:
        raise FileNotFoundError(f"No model folders found in {base_path}")
    return max(folders, key=os.path.getmtime)

def load_cb_artifacts(base_path: str) -> Dict[str, Any]:
    try:
        v_path = get_latest_model_path(base_path)
    
        # Load DataFrame Metadata (Isinya info brand, name, cluster)
        df_meta = pd.read_pickle(os.path.join(v_path, "shoe_metadata.pkl"))
        
        # Load Matrix Features (Numpy Array untuk similarity)
        with open(os.path.join(v_path, "shoe_features.pkl"), "rb") as f:
            X_features = pickle.load(f)
            
        # Load Scaler
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
        logger.error(f"Failed loading CB artifacts: {e}")
        raise RuntimeError(str(e))

# --- Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global road_artifacts, trail_artifacts, cf_engine
    
    logger.info("--- API STARTUP ---")
    try:
        # 1. Load Content-Based Models
        road_artifacts = load_cb_artifacts("model_artifacts/road")
        trail_artifacts = load_cb_artifacts("model_artifacts/trail")
        
        # 2. Load Collaborative Filtering Data & Model
        # Kita tetap perlu LOAD data dari DB saat startup agar model pintar
        logger.info("Fetching training data from Supabase...")
        interaction_df = fetch_and_merge_training_data()
        
        logger.info("Initializing CF Engine...")
        cf_engine = collaborative_filtering.UserCollaborativeRecommender(interaction_df)
        
        logger.info("--- API READY ---")
        
    except Exception as e:
        logger.critical(f"Startup Failed: {e}")
        raise e

    yield
    
    logger.info("--- API SHUTDOWN ---")
    road_artifacts.clear()
    trail_artifacts.clear()
    cf_engine = None

# --- App Init ---
app = FastAPI(title="Sonix-ML Hybrid Recommender", version="2.0", lifespan=lifespan)

# --- Schemas ---
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
    value: Optional[int] = None # 1-5 for rate, None for like

# --- Endpoints ---

# 1. Content-Based Recommendations
@app.post("/predict/road", tags=["Content-Based"])
async def predict_road(prefs: RoadInput):
    if not road_artifacts: raise HTTPException(503, "Road model not ready")
    try:
        results = road_recommender.get_recommendations(prefs.model_dump(), road_artifacts)
        return {"status": "success", "category": "road", "results": results}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict/trail", tags=["Content-Based"])
async def predict_trail(prefs: TrailInput):
    if not trail_artifacts: raise HTTPException(503, "Trail model not ready")
    try:
        results = trail_recommender.get_recommendations(prefs.model_dump(), trail_artifacts)
        return {"status": "success", "category": "trail", "results": results}
    except Exception as e:
        raise HTTPException(500, str(e))

# 2. Real-time Collaborative Filtering
@app.post("/interact", tags=["Real-time Collaborative"])
async def user_interaction(payload: UserAction):
    """
    Menangani aksi User (Like/Rate) UNTUK REKOMENDASI REAL-TIME.
    Catatan: Penyimpanan ke DB dilakukan oleh Backend Utama, bukan API ini.
    API ini hanya mengupdate memori sementara (RAM) agar rekomendasi langsung berubah.
    """
    global cf_engine
    if not cf_engine: raise HTTPException(503, "CF Engine not ready")
    
    try:
        is_like = (payload.action_type.lower() == "like")
        
        # A. Update Memory & Get Recs (Fast)
        # Langkah ini PENTING agar user merasa aplikasinya responsif
        recommendations = cf_engine.get_realtime_recommendations(
            user_id=payload.user_id,
            new_item_id=payload.shoe_id,
            new_rating_val=payload.value,
            is_like=is_like,
            n_neighbors=10
        )
        
        # BAGIAN SAVE DB SUDAH DIHAPUS
        
        return {
            "status": "success",
            "message": "Real-time memory updated",
            "data": {
                "triggered_by": payload.shoe_id,
                "recommendations": recommendations
            }
        }
    except Exception as e:
        logger.error(f"Interaction Error: {e}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)