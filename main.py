from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Shoe Recommender API")

# Class ini akan otomatis membaca JSON dari FE
class PreferencesInput(BaseModel):
    # Field yang ada di JSON Trail kamu:
    terrain: Optional[str] = None          # "Muddy"
    rock_sensitive: Optional[str] = None   # "Yes"
    pace: Optional[str] = None             # "Fast"
    orthotic_usage: Optional[str] = None   # "Yes"
    arch_type: Optional[str] = None        # "Normal" (Ingat fix mappingnya!)
    strike_pattern: Optional[str] = None   # "Mid"
    foot_width: Optional[str] = None       # "Wide"
    season: Optional[str] = None           # "Winter"
    water_resistance: Optional[str] = None # "Waterproof"
    
    # Field tambahan buat Road (biar satu class bisa dipake dua-duanya)
    running_purpose: Optional[str] = None
    cushion_preferences: Optional[str] = None
    stability_need: Optional[str] = None
    
    # Field tambahan kalau ada yang belum ke-cover
    drop_preference: Optional[str] = None

# Import
from recommender import road_recommender, trail_recommender

# Endpoint Road
@app.post("/predict/road")
def predict_road(prefs: PreferencesInput):
    # function name saya ganti jadi get_recommendations biar seragam
    results = road_recommender.get_recommendations(prefs.dict(), road_artifacts)
    return {"category": "road", "results": results}

# Endpoint Trail
@app.post("/predict/trail")
def predict_trail(prefs: PreferencesInput):
    results = trail_recommender.get_recommendations(prefs.dict(), trail_artifacts)
    return {"category": "trail", "results": results}