"""
Sonix-ML Performance Testing Suite
----------------------------------
This module defines the load testing scenarios for the Sonix-ML API 
hosted on Hugging Face. It simulates concurrent user behavior to 
measure system latency (P50, P90, P95) across recommendation and 
interaction endpoints.
"""

import random
from locust import HttpUser, task, between

class SonixMlTester(HttpUser):
    """
    Simulates a running shoe enthusiast interacting with the Sonix-ML ecosystem.
    
    Attributes:
        wait_time (callable): Simulates user 'think time' between 1 to 3 seconds.
        host (str): The target production environment on Hugging Face Spaces.
    """
    
    wait_time = between(1, 3)
    host = "https://sonix-rush-sonix-ml-api.hf.space"

    @task(2)
    def test_recommend_road(self):
        """
        Benchmarks the Road Recommendation endpoint.
        
        Tests the heuristic mapping logic (CC: 9) for road running preferences.
        Payload includes pace, arch type, and stability needs.
        """
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
        self.client.post("/recommend/road", json=payload, name="POST /recommend/road")

    @task(2)
    def test_recommend_trail(self):
        """
        Benchmarks the Trail Recommendation endpoint.
        
        This is the most computationally expensive task due to high cyclomatic 
        complexity (CC: 10) in the pre-processing logic. It validates 
        vectorization performance for complex terrain and protection features.
        """
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
        self.client.post("/recommend/trail", json=payload, name="POST /recommend/trail")

    @task(2)
    def test_interact(self):
        """
        Benchmarks the Interaction Logging endpoint.
        
        Simulates real-time user feedback (likes/dislikes) to evaluate 
        write-operation latency and database logging performance.
        """
        payload = {
            "user_id": 2,
            "shoe_id": "R050",
            "action_type": "like",
            "value": 1
        }
        self.client.post("/interact", json=payload, name="POST /interact")

    @task(1)
    def test_recommend_feed(self):
        """
        Benchmarks the Collaborative Filtering Home Feed.
        
        Evaluates the latency of the Neural Collaborative Filtering (NCF) 
        pipeline for generating personalized shoe feeds based on User ID.
        """
        self.client.get("/recommend/feed/2", name="GET /recommend/feed/[id]")