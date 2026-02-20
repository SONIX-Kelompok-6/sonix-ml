"""
Sonix-ML Feature Configuration Module
-------------------------------------
Defines the global feature sets for Road and Trail shoe categories.

This module acts as the single source of truth for feature column names,
ensuring strict naming consistency between the Supabase database schema, 
the Machine Learning models (Autoencoder & K-Means), and the inference API.
"""

from typing import List

# --- ROAD SHOE FEATURE SET ---
# These features focus on pavement performance, speed (plates), 
# and biomechanical efficiency (pace/strike).
ROAD_FEATURES: List[str] = [
    "brand", 
    "name", 
    "lightweight", 
    "rocker", 
    "removable_insole", 
    "pace_daily_running", 
    "pace_tempo", 
    "pace_competition", 
    "arch_neutral", 
    "arch_stability", 
    "weight_lab_oz", 
    "drop_lab_mm", 
    "strike_heel", 
    "strike_mid", 
    "strike_forefoot", 
    "midsole_softness", 
    "toebox_durability", 
    "heel_durability", 
    "outsole_durability", 
    "breathability_scaled", 
    "width_fit", 
    "toebox_width", 
    "stiffness_scaled", 
    "torsional_rigidity", 
    "heel_stiff", 
    "plate_rock_plate", 
    "plate_carbon_plate", 
    "heel_lab_mm", 
    "forefoot_lab_mm", 
    "season_summer", 
    "season_winter", 
    "season_all"
]

# --- TRAIL SHOE FEATURE SET ---
# These features prioritize traction (lugs), protection (shock absorption), 
# and environmental resistance (waterproof/terrain).
TRAIL_FEATURES: List[str] = [
    "brand", 
    "name", 
    "lightweight", 
    "terrain_light", 
    "terrain_moderate", 
    "terrain_technical", 
    "shock_absorption", 
    "energy_return", 
    "traction_scaled", 
    "arch_neutral", 
    "arch_stability", 
    "weight_lab_oz", 
    "drop_lab_mm", 
    "strike_heel", 
    "strike_mid", 
    "strike_forefoot", 
    "midsole_softness", 
    "toebox_durability", 
    "heel_durability", 
    "outsole_durability", 
    "breathability_scaled", 
    "plate_rock_plate", 
    "plate_carbon_plate", 
    "width_fit", 
    "toebox_width", 
    "stiffness_scaled", 
    "torsional_rigidity", 
    "heel_stiff", 
    "lug_dept_mm", 
    "heel_lab_mm", 
    "forefoot_lab_mm", 
    "season_summer", 
    "season_winter", 
    "season_all", 
    "removable_insole", 
    "waterproof", 
    "water_repellent"
]