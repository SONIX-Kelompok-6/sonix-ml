import os
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from src.training.architecture import build_autoencoder
from src.database import fetch_shoes_by_type
from src.config import ROAD_FEATURES, TRAIL_FEATURES

def run_training(shoe_type: str):
    """
    Alur: Ambil Data -> Preprocess -> Train Autoencoder -> KMeans -> Simpan 5 Artifacts
    """
    # 1. Load Data dari Supabase
    df = fetch_shoes_by_type(shoe_type)
    if df.empty: 
        print(f"❌ Aborting: No data found for {shoe_type}")
        return

    # Filter fitur sesuai config (Hanya kolom numerik untuk training)
    features = ROAD_FEATURES if shoe_type == 'road' else TRAIL_FEATURES
    numeric_cols = [c for c in features if c in df.select_dtypes(include=[np.number]).columns]
    X_raw = df[numeric_cols].values

    # 2. Normalisasi (PENTING: Scaler harus disimpan untuk digunakan di API)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_raw)

    # 3. Training Autoencoder
    # Sesuai notebook: Dimensi input mengikuti jumlah fitur, latent space = 8
    autoencoder, encoder = build_autoencoder(input_dim=X_train_scaled.shape[1])
    
    print(f"Training Autoencoder for {shoe_type}...")
    autoencoder.fit(
        X_train_scaled, X_train_scaled, 
        epochs=300,        # Sesuai notebook road
        batch_size=64,     # Sesuai notebook trail
        verbose=0
    )

    # 4. Generate Latent Space (8D Embeddings)
    # PERBAIKAN: Gunakan data yang sudah di-scale agar sesuai dengan bobot model
    X_latent = encoder.predict(X_train_scaled, verbose=0)

    # 5. Pelatihan KMeans
    # Sesuai hasil seleksi terbaik di notebook: Best K=5
    print(f"Clustering with KMeans (K=5) for {shoe_type}...")
    kmeans = KMeans(
        n_clusters=5, 
        random_state=42, 
        n_init=20          # Parameter n_init dari notebook
    )
    kmeans.fit(X_latent)

    # 6. Simpan Hasil (5 Artifacts)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"model_artifacts/{shoe_type}/v_{ts}"
    os.makedirs(save_path, exist_ok=True)

    # A. Simpan model Encoder (.keras)
    encoder.save(f"{save_path}/shoe_encoder.keras")
    
    # B. Simpan model KMeans (.pkl)
    with open(f"{save_path}/kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)

    # C. Simpan Scaler (.pkl) - BARU: Dibutuhkan API untuk memproses input user
    with open(f"{save_path}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # D. Simpan fitur yang sudah di-scale (Numpy Array untuk Cosine Similarity)
    with open(f"{save_path}/shoe_features.pkl", "wb") as f:
        pickle.dump(X_train_scaled, f)

    # E. Simpan Metadata (DataFrame lengkap dengan label cluster)
    df_meta = df.copy()
    df_meta['cluster'] = kmeans.labels_
    
    # Simpan info kolom ke atribut dataframe agar API tahu mana biner mana kontinu
    df_meta.attrs['binary_cols'] = [c for c in numeric_cols if df[c].nunique() <= 2]
    df_meta.attrs['continuous_cols'] = [c for c in numeric_cols if df[c].nunique() > 2]
    
    df_meta.to_pickle(f"{save_path}/shoe_metadata.pkl")

    print(f"✅ All 5 artifacts for {shoe_type} saved successfully at {save_path}")

if __name__ == "__main__":
    run_training('road')
    run_training('trail')