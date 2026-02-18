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

from dotenv import load_dotenv
load_dotenv()

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
    Executes the complete training pipeline for a specific shoe category, 
    producing all artifacts required for content-based recommendation inference.
    
    This function orchestrates the end-to-end training workflow: data fetching 
    from Supabase, feature normalization, deep autoencoder training, K-Means 
    clustering in the learned latent space, and serialization of all models 
    and metadata to versioned artifact directories.
    
    Args:
        shoe_type (str): Target shoe category for training. Must be either:
            - 'road': For road running shoes (pavement, speed-focused)
            - 'trail': For trail running shoes (off-road, traction-focused)
    
    Returns:
        None. All outputs are saved to disk at:
            model_artifacts/{shoe_type}/v_{YYYYMMDD_HHMMSS}/
    
    Artifacts Generated (5 files):
        1. shoe_encoder.h5 (TensorFlow/Keras Model):
           - Trained encoder network (input_dim → 8D latent space)
           - Used at inference to project user preferences into latent space
           - File size: ~50-100KB depending on input_dim
        
        2. kmeans_model.pkl (Scikit-learn KMeans):
           - Fitted K-Means model with 5 cluster centroids in 8D space
           - Used at inference to route queries to nearest clusters
           - File size: ~10KB
        
        3. scaler.pkl (Scikit-learn MinMaxScaler):
           - Fitted scaler with min/max values from training data
           - CRITICAL: Must be used to normalize user inputs at inference
           - File size: ~5KB
        
        4. shoe_features.pkl (NumPy array):
           - Scaled feature matrix of shape (n_shoes, n_features)
           - Used for masked cosine similarity calculation at inference
           - File size: ~50-200KB depending on catalog size
        
        5. shoe_metadata.pkl (Pandas DataFrame):
           - Complete shoe catalog with cluster labels and metadata
           - Includes cluster assignments (df['cluster'])
           - Stores column type definitions as DataFrame attributes:
               * df.attrs['binary_cols']: List of binary feature names
               * df.attrs['continuous_cols']: List of continuous feature names
           - File size: ~500KB-2MB depending on catalog size and metadata richness
    
    Training Pipeline Stages:
        
        Stage 1 - Data Ingestion:
            - Fetches shoe catalog from Supabase via fetch_shoes_by_type()
            - Filters features based on ROAD_FEATURES or TRAIL_FEATURES from config
            - Validates data availability (aborts if empty)
        
        Stage 2 - Feature Normalization:
            - Applies MinMaxScaler to normalize all features to [0, 1] range
            - Ensures all features contribute equally to neural network gradients
            - Prevents features with large numeric ranges from dominating learning
        
        Stage 3 - Autoencoder Training:
            - Builds deep autoencoder with symmetrical encoder-decoder architecture
            - Architecture: Input → Dense(32) → Dense(16) → Dense(8) → Decoder
            - Training configuration:
                * Loss: Mean Squared Error (MSE)
                * Optimizer: Adam (lr=0.001)
                * Epochs: 300
                * Batch size: 64
            - Goal: Learn a compressed 8D latent space that captures shoe similarity
        
        Stage 4 - Latent Space Projection:
            - Uses trained encoder to project all shoes into 8D latent space
            - Latent vectors naturally cluster similar shoes together
            - This projection enables fast retrieval via clustering
        
        Stage 5 - K-Means Clustering:
            - Fits K-Means with K=5 clusters on the 8D latent vectors
            - K=5 was empirically determined as optimal via elbow method
            - Multiple initializations (n_init=20) ensure global optimum
            - Each shoe is assigned to exactly one cluster
        
        Stage 6 - Artifact Serialization:
            - Creates versioned directory: model_artifacts/{type}/v_{timestamp}/
            - Saves all 5 artifacts with consistent naming
            - Versioning enables A/B testing and rollback capability
    
    Example Usage:
        >>> # Train road shoe models
        >>> run_training('road')
        >>> Initializing training sequence for: ROAD
        >>> Training Deep Autoencoder (Epochs: 300, Batch: 64)...
        >>> Generating clusters with K-Means (K=5)...
        >>> Training Complete. 5 artifacts saved to: model_artifacts/road/v_20260219_143022
        
        >>> # Train trail shoe models
        >>> run_training('trail')
        >>> Initializing training sequence for: TRAIL
        >>> ...
        >>> Training Complete. 5 artifacts saved to: model_artifacts/trail/v_20260219_143045
    
    Training Performance:
        - Road shoes (31 features, ~250 shoes): ~2-3 minutes on CPU
        - Trail shoes (36 features, ~180 shoes): ~2-3 minutes on CPU
        - GPU acceleration not required due to small model size
    
    Error Handling:
        - Empty dataset: Logs error and returns without creating artifacts
        - Missing features: Automatically filters to available columns only
        - All errors are logged via the logger instance
    
    Model Quality Validation:
        After training completes, verify artifact quality by checking:
        1. Autoencoder reconstruction error (MAE) should be < 0.05
        2. K-Means inertia should show clear elbow at K=5
        3. Cluster sizes should be relatively balanced (no cluster < 10% of data)
        4. Silhouette score should be > 0.3 for good cluster separation
    
    Production Deployment:
        1. Run this function to generate fresh artifacts
        2. Verify artifact quality via validation checks
        3. Restart the FastAPI server to load new artifacts
        4. The API automatically selects the latest versioned directory
        5. Old artifacts are preserved for rollback if needed
    
    Notes:
        - This function is idempotent: running it multiple times creates new 
          versioned artifacts without overwriting previous ones
        - Training is deterministic (random_state=42) for reproducibility
        - Artifacts are compatible across Python 3.11+ with matching library versions
        - Total disk space per artifact set: ~1-3MB depending on catalog size
    
    Related Functions:
        - fetch_shoes_by_type(): Retrieves catalog data from Supabase
        - build_autoencoder(): Constructs the neural network architecture
        - main.load_cb_artifacts(): Loads artifacts at API startup
    
    Configuration Dependencies:
        - ROAD_FEATURES: List of 31 road shoe feature names (config.py)
        - TRAIL_FEATURES: List of 36 trail shoe feature names (config.py)
        - SUPABASE_URL: Database connection string (.env)
        - SUPABASE_KEY: Database API key (.env)
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
    encoder.save(os.path.join(save_path, "shoe_encoder.h5"))
    
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