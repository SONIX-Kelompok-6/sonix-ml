import pickle
import pandas as pd
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager

# Import logic rekomendasimu
from recommender import road_recommender, trail_recommender

# --- 1. DEFINISI ARTIFACTS (Wadah Kosong Dulu) ---
# Kita buat variabel global biar bisa diakses di dalam fungsi endpoint di bawah
road_artifacts = {}
trail_artifacts = {}

# --- 2. FUNGSI LOAD MODEL (Dijalankan Sekali saat Start) ---
# Ini fungsi lifecycle (pengganti @app.on_event("startup") yang lama)
import os
import glob

def get_latest_model_path(base_path):
    """
    Mencari sub-folder dengan prefix 'v_' yang paling baru dibuat 
    di dalam direktori base_path.
    """
    # Mencari semua folder yang namanya mulai dengan 'v_'
    folders = glob.glob(os.path.join(base_path, 'v_*'))
    
    if not folders:
        raise FileNotFoundError(f"Tidak ditemukan folder model 'v_*' di {base_path}")
    
    # Mengurutkan berdasarkan waktu modifikasi folder (terbaru di akhir)
    # atau karena formatmu timestamp string (v_YYYYMMDD_HHMMSS), 
    # diurutkan secara alfabetis pun yang terbaru akan ada di paling bawah.
    latest_folder = max(folders, key=os.path.getmtime)
    
    return latest_folder

@asynccontextmanager
async def lifespan(app: FastAPI):
    # -- LOAD ROAD --
    print("Loading Road Models...")
    # Ganti path ini sesuai lokasi aslimu
    path_road = get_latest_model_path("model_artifacts/road")
    
    # Load metadata
    with open(f"{path_road}/shoe_metadata.pkl", "rb") as f:
        road_meta = pickle.load(f)
    
    # Masukkan semua ke dalam tas 'road_artifacts'
    global road_artifacts
    road_artifacts = {
        "df_data": pd.read_pickle(f"{path_road}/shoe_features.pkl"),
        "encoder_model": tf.keras.models.load_model(f"{path_road}/shoe_encoder.keras"),
        "kmeans_model": pickle.load(open(f"{path_road}/kmeans_model.pkl", "rb")),
        "binary_cols": road_meta['binary_cols'],
        "continuous_cols": road_meta['continuous_cols'],
        "X_combined_data": road_meta['X_combined_data'] # Atau load manual jika ga ada di meta
    }

    # -- LOAD TRAIL --
    print("Loading Trail Models...")
    # Ganti path ini sesuai lokasi aslimu
    path_trail = get_latest_model_path("model_artifacts/trail")
    
    with open(f"{path_trail}/shoe_metadata.pkl", "rb") as f:
        trail_meta = pickle.load(f)

    global trail_artifacts
    trail_artifacts = {
        "df_data": pd.read_pickle(f"{path_trail}/shoe_features.pkl"),
        "encoder_model": tf.keras.models.load_model(f"{path_trail}/shoe_encoder.keras"),
        "kmeans_model": pickle.load(open(f"{path_trail}/kmeans_model.pkl", "rb")),
        "binary_cols": trail_meta['binary_cols'],
        "continuous_cols": trail_meta['continuous_cols'],
        "X_combined_data": trail_meta['X_combined_data']
    }
    
    print("All Models Loaded! API Ready.")
    yield
    # Code setelah yield akan jalan pas aplikasi mati (clean up)
    print("Shutting down...")

# --- 3. INISIALISASI APP ---
app = FastAPI(title="Shoe Recommender API", lifespan=lifespan)

# --- 4. SCHEMA INPUT ---
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

# --- 5. ENDPOINTS ---

@app.post("/predict/road")
def predict_road(prefs: RoadInput):
    # prefs.dict() mengubah input user jadi dictionary
    # road_artifacts dikirim supaya fungsi logic bisa pake model yang udah di-load
    results = road_recommender.get_recommendations(prefs.dict(), road_artifacts)
    return {"category": "road", "results": results}

@app.post("/predict/trail")
def predict_trail(prefs: TrailInput):
    results = trail_recommender.get_recommendations(prefs.dict(), trail_artifacts)
    return {"category": "trail", "results": results}