# Professional English Code: locustfile.py
import random
from locust import HttpUser, task, between

class SonixMlTester(HttpUser):
    # Simulate realistic user think time
    wait_time = between(1, 2)

    @task(3)
    def test_interact_endpoint(self):
        """
        Tests the /interact POST endpoint.
        Matches the 'UserAction' pydantic model in main.py.
        """
        payload = {
            "user_id": 8,
            "shoe_id": "R278",
            "action_type": "like",
            "value": 5
        }
        # Correct path from your main.py
        self.client.post("/interact", json=payload)

    @task(1)
    def test_user_feed(self):
        """
        Tests the /recommend/user/{user_id} GET endpoint.
        Used for populating the Home Feed.
        """
        user_id = 8
        self.client.get(f"/recommend/feed/{user_id}")

    @task(1)
    def test_health_check(self):
        """Tests the lightweight health probe."""
        self.client.get("/health")