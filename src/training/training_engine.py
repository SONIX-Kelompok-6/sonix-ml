"""
Sonix-ML Training Engine
------------------------
Orchestrates the training lifecycle for both Road and Trail recommendation models.
This engine implements a hybrid Deep Learning + Clustering approach.

Workflow:
1. Data Ingestion (Supabase)
2. Normalization (MinMaxScaler)
3. Dimensionality Reduction (Deep Autoencoder)
4. Cluster Generation (K-Means)
5. Artifact Serialization (Models, Scalers, and Metadata)
"""

import os
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from src.training.architecture import build_autoencoder
from src.database import fetch_shoes_by_type
from src.config import ROAD_FEATURES, TRAIL_FEATURES

# Configure logging for training visibility
logger = logging.getLogger(__name__)

def run_training(shoe_type: str):
    """
    Executes the full training pipeline for a specific shoe category.
    
    Args:
        shoe_type: 'road' or 'trail'.
    """
    logger.info(f">>> Initializing training sequence for: {shoe_type.upper()}")

    # --- 1. DATA INGESTION ---
    df = fetch_shoes_by_type(shoe_type)
    if df.empty: 
        logger.error(f"Aborting: No source data retrieved for {shoe_type}")
        return

    # Select numerical features based on the global configuration
    target_features = ROAD_FEATURES if shoe_type == 'road' else TRAIL_FEATURES
    numeric_cols = [c for c in target_features if c in df.select_dtypes(include=[np.number]).columns]
    X_raw = df[numeric_cols].values

    # --- 2. DATA PREPROCESSING ---
    # Scaling is mandatory for Neural Networks to ensure all features contribute 
    # equally to the gradient updates during backpropagation.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # --- 3. AUTOENCODER TRAINING ---
    # Build a category-specific Autoencoder based on the input dimension
    autoencoder, encoder = build_autoencoder(input_dim=X_scaled.shape[1])
    
    logger.info(f"Training Deep Autoencoder (Epochs: 300, Batch: 64)...")
    autoencoder.fit(
        X_scaled, X_scaled, 
        epochs=300,        # Optimized based on previous road/trail experiments
        batch_size=64,     
        verbose=0          # Set to 1 for debugging loss curves
    )

    # --- 4. LATENT SPACE PROJECTION ---
    # Compress the high-dimensional scaled features into an 8D latent space.
    # Note: We use the scaled data to maintain weights consistency.
    X_latent = encoder.predict(X_scaled, verbose=0)

    # --- 5. CLUSTERING (K-MEANS) ---
    # Group shoes into 5 semantic clusters within the latent space.
    # K=5 was determined as the optimal 'elbow' point in exploratory notebooks.
    logger.info(f"Generating clusters with K-Means (K=5)...")
    kmeans = KMeans(
        n_clusters=5, 
        random_state=42, 
        n_init=20          # Multiple initializations to avoid local minima
    )
    kmeans.fit(X_latent)

    # --- 6. ARTIFACT SERIALIZATION ---
    # Prepare versioned directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"model_artifacts/{shoe_type}/v_{ts}"
    os.makedirs(save_path, exist_ok=True)

    # A. Encoder Model (TensorFlow Keras format)
    encoder.save(os.path.join(save_path, "shoe_encoder.keras"))
    
    # B. K-Means Model (Pickle format)
    with open(os.path.join(save_path, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(kmeans, f)

    # C. Scaler (MinMaxScaler)
    # CRITICAL: The inference API must use this exact scaler to normalize user inputs.
    with open(os.path.join(save_path, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # D. Processed Features (Numpy Array)
    # Used for real-time Cosine Similarity calculation in the API.
    with open(os.path.join(save_path, "shoe_features.pkl"), "wb") as f:
        pickle.dump(X_scaled, f)

    # E. Metadata & Column Definitions (DataFrame)
    # Encapsulate cluster labels and column groupings for the API.
    df_meta = df.copy()
    df_meta['cluster'] = kmeans.labels_
    
    # Store column groupings as attributes to avoid hardcoding in the main API
    df_meta.attrs['binary_cols'] = [c for c in numeric_cols if df[c].nunique() <= 2]
    df_meta.attrs['continuous_cols'] = [c for c in numeric_cols if df[c].nunique() > 2]
    
    df_meta.to_pickle(os.path.join(save_path, "shoe_metadata.pkl"))

    logger.info(f"Training Complete. 5 artifacts saved to: {save_path}")

if __name__ == "__main__":
    run_training('road')
    run_training('trail')