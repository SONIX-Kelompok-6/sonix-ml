"""
Sonix-ML Training Engine Module
-------------------------------
Orchestrates the training lifecycle for recommendation models.
Refactored into a modular pipeline class to ensure Single Responsibility 
and minimized cyclomatic complexity.
"""

import os
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Any

# --- CRITICAL: LOAD ENV VARS FIRST ---
# This must be executed before importing src.database to ensure 
# the Supabase client initializes with valid credentials.
from dotenv import load_dotenv
load_dotenv() 

# --- Project Imports ---
from src.training.architecture import build_autoencoder
from src.database import fetch_shoes_by_type
from src.config import ROAD_FEATURES, TRAIL_FEATURES

# Configure logging to display output in the terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Modular training pipeline isolating ingestion, training, and serialization."""

    def __init__(self, shoe_type: str):
        self.shoe_type = shoe_type
        self.target_features = ROAD_FEATURES if shoe_type == 'road' else TRAIL_FEATURES
        self.n_clusters = 5

    def _ingest_and_scale(self) -> Tuple[np.ndarray, MinMaxScaler, pd.DataFrame, List[str]]:
        logger.info(f"Fetching data from Supabase for category: {self.shoe_type}...")
        df = fetch_shoes_by_type(self.shoe_type)
        
        if df.empty:
            raise ValueError(f"CRITICAL: No source data retrieved for {self.shoe_type}. Check your database connection or table data.")

        # Ensure target features exist in the dataframe
        numeric_cols = [c for c in self.target_features if c in df.columns]
        
        # Data Cleaning: Coerce non-numeric data to NaN, then fill with 0
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        X_raw = df[numeric_cols].values

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_raw)
        
        return X_scaled, scaler, df, numeric_cols

    def _train_models(self, X_scaled: np.ndarray) -> Tuple[Any, KMeans]:
        autoencoder, encoder = build_autoencoder(input_dim=X_scaled.shape[1])
        
        logger.info("Training Deep Autoencoder (Epochs: 50, Batch: 32)...")
        # Reduced epochs for rapid testing; increase to 300 for production
        autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=0)

        X_latent = encoder.predict(X_scaled, verbose=0)

        logger.info(f"Generating clusters with K-Means (K={self.n_clusters})...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_latent)

        return encoder, kmeans

    def _save_artifacts(self, encoder: Any, kmeans: KMeans, scaler: MinMaxScaler, 
                        X_scaled: np.ndarray, df: pd.DataFrame, numeric_cols: List[str]) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure path compatibility across operating systems
        save_path = os.path.join("model_artifacts", self.shoe_type, f"v_{ts}")
        os.makedirs(save_path, exist_ok=True)

        # Save Keras Model
        encoder.save(os.path.join(save_path, "shoe_encoder.h5"))
        
        # Save Pickle Artifacts
        with open(os.path.join(save_path, "kmeans_model.pkl"), "wb") as f:
            pickle.dump(kmeans, f)
        with open(os.path.join(save_path, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(save_path, "shoe_features.pkl"), "wb") as f:
            pickle.dump(X_scaled, f)

        # Save Metadata with Attributes
        df_meta = df.copy()
        df_meta['cluster'] = kmeans.labels_
        df_meta.attrs['binary_cols'] = [c for c in numeric_cols if df[c].nunique() <= 2]
        df_meta.attrs['continuous_cols'] = [c for c in numeric_cols if df[c].nunique() > 2]
        df_meta.to_pickle(os.path.join(save_path, "shoe_metadata.pkl"))

        return save_path

    def run(self) -> None:
        """Executes the full orchestrated training sequence."""
        logger.info(f"STARTING PIPELINE: {self.shoe_type.upper()}")
        try:
            X_scaled, scaler, df, numeric_cols = self._ingest_and_scale()
            encoder, kmeans = self._train_models(X_scaled)
            save_path = self._save_artifacts(encoder, kmeans, scaler, X_scaled, df, numeric_cols)
            logger.info(f"PIPELINE SUCCESS. Artifacts saved to: {save_path}")
        except Exception as e:
            logger.error(f"PIPELINE FAILED for {self.shoe_type}: {str(e)}")
            raise e

def run_training(shoe_type: str) -> None:
    pipeline = TrainingPipeline(shoe_type)
    pipeline.run()

# --- ENTRY POINT ---
if __name__ == "__main__":
    print("--- INITIATING MANUAL TRAINING JOB ---")
    try:
        run_training('road')
        run_training('trail')
        print("--- JOB COMPLETE: Models Generated Successfully ---")
    except Exception as e:
        print(f"--- JOB FAILED: {e} ---")