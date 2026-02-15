import os
import pandas as pd
from supabase import create_client, Client
from typing import Optional

# --- KONFIGURASI SUPABASE ---
# Pastikan variabel environment ini diset, atau ganti string di bawah
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://xmsffgwcjeequpqjhvqi.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_dcZpsxnL14BPHFol5Yu-Ug_P1AZSGaF")

# Definisikan variabel dengan tipe Optional[Client] dan default None
supabase: Optional[Client] = None

# Inisialisasi Client
try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    else:
        print("Warning: SUPABASE_URL or SUPABASE_KEY is missing.")
except Exception as e:
    print(f"CRITICAL WARNING: Supabase client failed to initialize. {e}")
    supabase = None

def fetch_and_merge_training_data() -> pd.DataFrame:
    """
    Mengambil data dari tabel 'favorites' (Likes) dan 'reviews' (Bintang),
    lalu menggabungkannya menjadi satu DataFrame Skor Interaksi tunggal.
    
    Logika Penggabungan:
    - Like Action : +1.0
    - Rating 1-5  : Dikonversi ke skala -2.0 s/d +2.0
    - Jika User Like DAN Rate sepatu yang sama -> Skor dijumlahkan.
    
    Returns:
        pd.DataFrame: Columns ['user_id', 'item_id', 'rating']
    """
    if not supabase:
        return pd.DataFrame(columns=['user_id', 'item_id', 'rating'])

    try:
        # --- 1. FETCH FAVORITES (LIKES) ---
        res_fav = supabase.table("favorites").select("user_id, shoe_id").execute()
        
        df_fav = pd.DataFrame()
        if res_fav.data:
            df_fav = pd.DataFrame(res_fav.data)
            df_fav = df_fav.rename(columns={'shoe_id': 'item_id'})
            df_fav['score'] = 1.0  # Base score for Like
            df_fav = df_fav[['user_id', 'item_id', 'score']]

        # --- 2. FETCH RATINGS (STARS) ---
        res_rate = supabase.table("reviews").select("user_id, shoe_id, rating").execute()
        
        df_rate = pd.DataFrame()
        if res_rate.data:
            df_rate = pd.DataFrame(res_rate.data)
            df_rate = df_rate.rename(columns={'shoe_id': 'item_id'})
            
            # Konversi Rating 1-5 ke Skala Simetris (-2 s/d 2)
            # Logika ini harus sinkron dengan collaborative_filtering.py
            def convert_db_rating(r):
                if r == 5: return 2.0
                if r == 4: return 1.0
                if r == 3: return 0.1
                if r == 2: return -1.0
                if r == 1: return -2.0
                return 0.1
            
            df_rate['score'] = df_rate['rating'].apply(convert_db_rating)
            df_rate = df_rate[['user_id', 'item_id', 'score']]

        # --- 3. MERGE & AGGREGATE ---
        frames = []
        if not df_fav.empty: frames.append(df_fav)
        if not df_rate.empty: frames.append(df_rate)
        
        if not frames:
            return pd.DataFrame(columns=['user_id', 'item_id', 'rating'])

        # Gabungkan tumpukan data
        df_combined = pd.concat(frames)

        # Jumlahkan skor jika ada duplikat (User sama, Item sama)
        df_final = df_combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
        
        # Rename kolom 'score' jadi 'rating' agar standar
        df_final = df_final.rename(columns={'score': 'rating'})
        
        print(f"[Database] Successfully loaded {len(df_final)} interactions.")
        return df_final

    except Exception as e:
        print(f"[Database] Error fetching data: {e}")
        return pd.DataFrame(columns=['user_id', 'item_id', 'rating'])

def save_interaction_routed(user_id: int, shoe_id: str, action_type: str, rating: Optional[int]):
    """
    Menyimpan interaksi ke tabel yang benar (Routing) berdasarkan action_type.
    """
    if not supabase: return

    try:
        if action_type == 'like':
            # Simpan ke tabel favorites
            data = {"user_id": user_id, "shoe_id": shoe_id}
            # upsert=True mencegah error jika user like 2x
            supabase.table("favorites").upsert(data, on_conflict="user_id, shoe_id").execute()
            # print(f"[Database] Saved LIKE: {user_id} -> {shoe_id}")
            
        elif action_type == 'rate':
            # Simpan ke tabel ratings
            if rating is None: return 
            
            data = {"user_id": user_id, "shoe_id": shoe_id, "rating": rating}
            supabase.table("reviews").upsert(data, on_conflict="user_id, shoe_id").execute()
            # print(f"[Database] Saved RATE: {user_id} -> {shoe_id} ({rating})")
            
    except Exception as e:
        print(f"[Database] Failed to save interaction: {e}")