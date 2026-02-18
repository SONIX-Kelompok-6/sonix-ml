"""
Sonix-ML Feature Registry
-------------------------
The authoritative "Source of Truth" for feature definitions used across the 
Sonix Recommendation Ecosystem. This module dictates the data contract between 
the Supabase persistence layer and the Machine Learning inference engine.

System Role:
    1. Database Projection: Defines which columns are explicitly fetched via SQL.
    2. Input Dimensionality: Determines the input_shape for the Autoencoder and 
       feature vector size for Cosine Similarity.
    3. Consistency Enforcement: Ensures the Training Pipeline (offline) and 
       Inference API (online) use the exact same feature set and ordering.

Critical Constraints:
    ! IMMUTABILITY: Changing the order or content of these lists invalidates 
      all previously serialized model artifacts (encoders/scalers).
    ! SCHEMA SYNC: Every string here must match a column name in the 
      'shoes_metadata' table in Supabase.
    ! RETRAINING: If you modify these lists, you MUST re-run the training 
      pipeline to generate compatible model versions.

Feature Groups:
    - Road: 32 Features (Focus: Efficiency, Speed, Biomechanics)
    - Trail: 37 Features (Focus: Protection, Traction, Environment)
"""

from typing import List

# --- ROAD SHOE FEATURE SET ---
# Dimensionality: 32 Columns
#
# Semantic Focus:
#   - Biomechanics: Strike zones (heel/mid/fore), drop, stability vs neutral.
#   - Performance: Pace suitability (daily/tempo/comp), energy return mechanisms.
#   - Structure: Carbon plates, rocker geometry, stiffness.
#
# Usage:
#   - Used by src.training.engine to shape the Road Autoencoder (Input: 32 -> Latent: 8).
#   - Used by src.inference.router to validate incoming user preference vectors.
ROAD_FEATURES: List[str] = [
    # Metadata (Non-training)
    "brand", 
    "name", 
    
    # Core Specs
    "lightweight", 
    "rocker", 
    "removable_insole", 
    
    # Pace & Usage (One-Hot Logic)
    "pace_daily_running", 
    "pace_tempo", 
    "pace_competition", 
    
    # Stability Mechanics
    "arch_neutral", 
    "arch_stability", 
    
    # Lab Measurements
    "weight_lab_oz", 
    "drop_lab_mm", 
    
    # Strike Pattern Compatibility
    "strike_heel", 
    "strike_mid", 
    "strike_forefoot", 
    
    # Material Properties
    "midsole_softness", 
    "toebox_durability", 
    "heel_durability", 
    "outsole_durability", 
    "breathability_scaled", 
    
    # Fit Dimensions
    "width_fit", 
    "toebox_width", 
    
    # Stiffness & Propulsion
    "stiffness_scaled", 
    "torsional_rigidity", 
    "heel_stiff", 
    "plate_rock_plate", 
    "plate_carbon_plate", 
    
    # Stack Heights
    "heel_lab_mm", 
    "forefoot_lab_mm", 
    
    # Seasonality
    "season_summer", 
    "season_winter", 
    "season_all"
]

# --- TRAIL SHOE FEATURE SET ---
# Dimensionality: 37 Columns
#
# Semantic Focus:
#   - Terrain Mastery: Technical vs Light trails, lug depth, traction scores.
#   - Protection: Rock plates, toe guards, shock absorption.
#   - Elements: Waterproofing, seasonal suitability.
#
# Usage:
#   - Used by src.training.engine to shape the Trail Autoencoder (Input: 37 -> Latent: 8).
#   - Differentiated from Road by the inclusion of 'lug_dept_mm', 'waterproof', 
#     and granular 'terrain_*' flags.
TRAIL_FEATURES: List[str] = [
    # Metadata (Non-training)
    "brand", 
    "name", 
    
    # Core Specs
    "lightweight", 
    
    # Terrain Suitability
    "terrain_light", 
    "terrain_moderate", 
    "terrain_technical", 
    
    # Dynamic Properties
    "shock_absorption", 
    "energy_return", 
    "traction_scaled", 
    
    # Stability Mechanics
    "arch_neutral", 
    "arch_stability", 
    
    # Lab Measurements
    "weight_lab_oz", 
    "drop_lab_mm", 
    
    # Strike Pattern Compatibility
    "strike_heel", 
    "strike_mid", 
    "strike_forefoot", 
    
    # Durability & Material
    "midsole_softness", 
    "toebox_durability", 
    "heel_durability", 
    "outsole_durability", 
    "breathability_scaled", 
    
    # Protection Tech
    "plate_rock_plate", 
    "plate_carbon_plate", 
    
    # Fit Dimensions
    "width_fit", 
    "toebox_width", 
    
    # Stiffness
    "stiffness_scaled", 
    "torsional_rigidity", 
    "heel_stiff", 
    
    # Trail Specifics
    "lug_dept_mm", 
    "heel_lab_mm", 
    "forefoot_lab_mm", 
    
    # Seasonality
    "season_summer", 
    "season_winter", 
    "season_all", 
    
    # Utility
    "removable_insole", 
    "waterproof", 
    "water_repellent"
]