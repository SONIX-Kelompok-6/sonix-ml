import random
from locust import HttpUser, task, between

class SonixMlTester(HttpUser):
    # Simulasi waktu tunggu user antara 1-3 detik agar tidak membebani server HF secara instan
    wait_time = between(1, 3)
    
    # Masukkan URL Hugging Face kamu sebagai host default
    host = "https://sonix-rush-sonix-ml-api.hf.space"

    @task(2)
    def test_recommend_road(self):
        """Uji performa model Road Recommender (CC 9)"""
        payload = {
            "pace": "Fast",
            "arch_type": "Normal",
            "strike_pattern": "Mid",
            "foot_width": "Regular",
            "season": "Summer",
            "orthotic_usage": "No",
            "running_purpose": "Race",
            "cushion_preferences": "Firm",
            "stability_need": "Neutral"
        }
        self.client.post("/recommend/road", json=payload, name="/recommend/road")

    @task(2)
    def test_recommend_trail(self):
        """Uji performa model Trail Recommender (CC 10) - Paling Berat"""
        payload = {
            "pace": "Steady",
            "arch_type": "Flat",
            "strike_pattern": "Heel",
            "foot_width": "Wide",
            "season": "Spring & Fall",
            "orthotic_usage": "Yes",
            "terrain": "Mixed",
            "rock_sensitive": "Yes",
            "water_resistance": "Waterproof"
        }
        self.client.post("/recommend/trail", json=payload, name="/recommend/trail")

    @task(2)
    def test_interact(self):
        """Uji logging interaksi user"""
        payload = {
            "pace": "Steady",
            "arch_type": "Flat",
            "strike_pattern": "Heel",
            "foot_width": "Wide",
            "season": "Spring & Fall",
            "orthotic_usage": "Yes",
            "terrain": "Mixed",
            "rock_sensitive": "Yes",
            "water_resistance": "Waterproof"
        }
        self.client.post("/interact", json=payload, name="/interact")

    @task(1)
    def test_recommend_feed(self):
        """Uji Home Feed (Collaborative Filtering/NCF)"""
        self.client.get("/recommend/feed/2", name="/recommend/feed/[id]")