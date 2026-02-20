"""
Sonix-ML Training Engine Module
-------------------------------
Orchestrates the training lifecycle for both Road and Trail recommendation models.
Refactored into a modular pipeline class to ensure Single Responsibility, 
testability, and minimized cyclomatic complexity.

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
from typing import Tuple, List, Any

from src.training.architecture import build_autoencoder
from src.database import fetch_shoes_by_type
from src.config import ROAD_FEATURES, TRAIL_FEATURES

# Configure logging for training visibility
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """
    Modular training pipeline for Sonix-ML recommendation models.
    Isolates data ingestion, model training, and artifact serialization.
    """

    def __init__(self, shoe_type: str):
        """
        Initializes the training pipeline configuration based on shoe category.
        
        Args:
            shoe_type (str): Category of the shoe to train ('road' or 'trail').
        """
        self.shoe_type = shoe_type
        self.target_features = ROAD_FEATURES if shoe_type == 'road' else TRAIL_FEATURES
        self.n_clusters = 5

    def _ingest_and_scale(self) -> Tuple[np.ndarray, MinMaxScaler, pd.DataFrame, List[str]]:
        """
        Fetches raw data from the database, filters target features, and applies min-max scaling.
        
        Returns:
            Tuple: 
                - X_scaled (np.ndarray): Scaled numerical feature matrix.
                - scaler (MinMaxScaler): Fit scaler instance.
                - df (pd.DataFrame): Raw dataframe for metadata construction.
                - numeric_cols (List[str]): List of active feature columns.
        """
        df = fetch_shoes_by_type(self.shoe_type)
        if df.empty:
            raise ValueError(f"No source data retrieved for {self.shoe_type}")

        numeric_cols = [c for c in self.target_features if c in df.select_dtypes(include=[np.number]).columns]
        X_raw = df[numeric_cols].values

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_raw)
        
        return X_scaled, scaler, df, numeric_cols

    def _train_models(self, X_scaled: np.ndarray) -> Tuple[Any, KMeans]:
        """
        Constructs and trains the Deep Autoencoder and K-Means clustering models.
        
        Args:
            X_scaled (np.ndarray): Scaled numerical feature matrix.
            
        Returns:
            Tuple:
                - encoder (tf.keras.Model): Trained standalone encoder model.
                - kmeans (KMeans): Fit K-Means clustering model.
        """
        autoencoder, encoder = build_autoencoder(input_dim=X_scaled.shape[1])
        
        logger.info("Training Deep Autoencoder (Epochs: 300, Batch: 64)...")
        autoencoder.fit(X_scaled, X_scaled, epochs=300, batch_size=64, verbose=0)

        X_latent = encoder.predict(X_scaled, verbose=0)

        logger.info(f"Generating clusters with K-Means (K={self.n_clusters})...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=20)
        kmeans.fit(X_latent)

        return encoder, kmeans

    def _save_artifacts(self, encoder: Any, kmeans: KMeans, scaler: MinMaxScaler, 
                        X_scaled: np.ndarray, df: pd.DataFrame, numeric_cols: List[str]) -> str:
        """
        Serializes trained models, scalers, and metadata into versioned directories.
        
        Args:
            encoder (tf.keras.Model): Trained standalone encoder model.
            kmeans (KMeans): Fit K-Means clustering model.
            scaler (MinMaxScaler): Fit scaler instance.
            X_scaled (np.ndarray): Scaled numerical feature matrix.
            df (pd.DataFrame): Base dataframe for metadata extraction.
            numeric_cols (List[str]): Active feature columns.
            
        Returns:
            str: Path to the created artifact directory.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"model_artifacts/{self.shoe_type}/v_{ts}"
        os.makedirs(save_path, exist_ok=True)

        # Maintained .h5 extension to ensure compatibility with existing inference API
        encoder.save(os.path.join(save_path, "shoe_encoder.h5"))
        
        with open(os.path.join(save_path, "kmeans_model.pkl"), "wb") as f:
            pickle.dump(kmeans, f)

        with open(os.path.join(save_path, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        with open(os.path.join(save_path, "shoe_features.pkl"), "wb") as f:
            pickle.dump(X_scaled, f)

        df_meta = df.copy()
        df_meta['cluster'] = kmeans.labels_
        df_meta.attrs['binary_cols'] = [c for c in numeric_cols if df[c].nunique() <= 2]
        df_meta.attrs['continuous_cols'] = [c for c in numeric_cols if df[c].nunique() > 2]
        
        df_meta.to_pickle(os.path.join(save_path, "shoe_metadata.pkl"))

        return save_path

    def run(self) -> None:
        """
        Executes the full, orchestrated training sequence.
        """
        logger.info(f">>> Initializing training sequence for: {self.shoe_type.upper()}")
        try:
            X_scaled, scaler, df, numeric_cols = self._ingest_and_scale()
            encoder, kmeans = self._train_models(X_scaled)
            save_path = self._save_artifacts(encoder, kmeans, scaler, X_scaled, df, numeric_cols)
            logger.info(f"Training Complete. Artifacts saved to: {save_path}")
        except Exception as e:
            logger.error(f"Training failed for {self.shoe_type}: {str(e)}")

def run_training(shoe_type: str) -> None:
    """
    Legacy wrapper for module-level execution to maintain backward compatibility.
    
    Args:
        shoe_type (str): Category of the shoe to train.
    """
    pipeline = TrainingPipeline(shoe_type)
    pipeline.run()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    run_training('road')
    run_training('trail')