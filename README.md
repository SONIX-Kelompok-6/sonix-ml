---
title: Sonix ML API
emoji: 👟
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# SONIX RUSH AI — Running Shoes Recommender Engine

**SONIX RUSH AI** is a specialized machine learning inference service designed for the SONIX RUSH application. It implements a **Hybrid Recommendation System** that combines **Content-Based Filtering** (via Deep Autoencoders and K-Means Clustering) with **Collaborative Filtering** (User-Based KNN) to deliver personalized running shoe recommendations.

This engine operates as a standalone inference service and communicates with the core backend and frontend via RESTful APIs.

---

## Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation Guide](#installation-guide)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Continuous Training](#continuous-training)
- [Error Handling](#error-handling)
- [Docker Deployment](#docker-deployment)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## System Overview

The SONIX RUSH AI engine isolates high-computational ML workloads from the primary transactional backend. Running as an independent service ensures that model inference and data processing do not impact the performance of the main application.

| Property | Detail |
|---|---|
| **Integration Method** | REST API |
| **Methodology** | Hybrid — Content-Based + Collaborative (separate pipelines) |
| **Optimization** | In-Memory Micro-Caching with TTL (60s) |
| **Continuous Training** | Background CF refresh every 50 interactions |
| **Deployment** | Docker / Hugging Face Spaces |

---

## Architecture

The system uses two **independent** recommendation pipelines, each serving a different use case.
```
┌──────────────────────────────────────────────────────────────────────┐
│                        SONIX RUSH AI Engine                          │
│                                                                      │
│  ┌─────────────────────────────────┐   ┌────────────────────────┐   │
│  │    Content-Based Pipeline       │   │  Collaborative Pipeline │   │
│  │  POST /recommend/road           │   │  POST /interact        │   │
│  │  POST /recommend/trail          │   │  GET  /recommend/feed  │   │
│  │                                 │   │                        │   │
│  │  1. Map user questionnaire      │   │  1. Receive interaction │   │
│  │     → numerical vector          │   │  2. Real-time inject   │   │
│  │  2. Encode → 8D Latent Space    │   │     into user vector   │   │
│  │     (Deep Autoencoder)          │   │  3. KNN: find similar  │   │
│  │  3. Route to nearest clusters   │   │     users (cosine)     │   │
│  │     (K-Means, top ⌈K/3⌉)       │   │  4. Weighted score     │   │
│  │  4. Masked Cosine Similarity    │   │     aggregation        │   │
│  │  5. Return Top 10 shoes         │   │  5. Return Top 20 shoes│   │
│  └─────────────────────────────────┘   └────────────────────────┘   │
│                                                                      │
│                    FastAPI + Uvicorn (4 workers)                     │
└──────────────────────────────────────────────────────────────────────┘
              ↑                                      ↑
      SONIX RUSH Backend                    SONIX RUSH Frontend
        (REST calls)                          (REST calls)
```

### Content-Based: Training Pipeline (Offline)
```
Supabase → fetch_shoes_by_type() → MinMaxScaler
    → Deep Autoencoder (300 epochs, batch=64)
    → K-Means (K=5, n_init=20) on 8D latent space
    → Save versioned artifacts to model_artifacts/{type}/v_{timestamp}/
```

### Content-Based: Inference Pipeline (Online)
```
User Questionnaire → Heuristic Feature Mapping → Numerical Vector
    → Encoder → 8D Latent Vector
    → K-Means: select top ⌈K/3⌉ nearest clusters
    → Masked Cosine Similarity on candidate pool
    → Top 10 shoes
```

Feature masking ensures similarity is computed **only on features the user explicitly provided**, preventing noise from neutral default values.

### Collaborative Filtering Pipeline (Real-Time)
```
POST /interact → Real-Time Injection into user vector
    → User-Based KNN (cosine, brute force, sparse CSR matrix)
    → Weighted score aggregation from k neighbors
    → Filter seen items → Enrich with shoe metadata
    → Top 20 shoes (TTL-cached 60s per user)
    → Background CT trigger every 50 interactions
```

### Autoencoder Architecture
```
Input (N-dim)
  → Dense(32) + BatchNorm + Dropout(0.3)   ← Encoder
  → Dense(16) + BatchNorm + Dropout(0.3)   ← Encoder
  → Dense(8)  + BatchNorm + Dropout(0.3)   ← Latent Space (saved as shoe_encoder.h5)
  → Dense(16) + BatchNorm + Dropout(0.3)   ← Decoder
  → Dense(32) + BatchNorm + Dropout(0.3)   ← Decoder
  → Dense(N, sigmoid)                       ← Reconstruction Output

Loss: MSE | Optimizer: Adam (lr=0.001) | Metric: MAE
```

---

## Technology Stack

### Core & Backend
| Library | Version | Purpose |
|---|---|---|
| Python | 3.11 | Primary language — strict version required for TensorFlow 2.15.0 |
| FastAPI | 0.129.0 | High-performance async web framework |
| Uvicorn | 0.40.0 | ASGI web server |
| Gunicorn | 25.1.0 | Process manager |
| Pydantic | 2.12.5 | Request/response schema validation |
| ujson | 5.11.0 | High-speed JSON serialization (`UJSONResponse`) |

### Machine Learning & Data Science
| Library | Version | Purpose |
|---|---|---|
| TensorFlow / Keras | 2.15.0 | Deep Autoencoder for latent space projection |
| Scikit-Learn | 1.5.2 | K-Means Clustering and User-Based KNN |
| Pandas | 2.2.2 | Data manipulation and user-item pivot matrix |
| NumPy | 1.26.4 | Numerical computation and vector operations |
| SciPy | 1.13.1 | Sparse CSR matrix for memory-efficient CF |

### Database & Infrastructure
| Library | Version | Purpose |
|---|---|---|
| Supabase | 2.27.3 | PostgreSQL-based Backend-as-a-Service |
| HTTPX | 0.28.1 | Async HTTP client |
| python-dotenv | 1.0.1 | Environment variable management |
| Docker | latest | Containerization |

### Testing & Quality Assurance
| Library | Version | Purpose |
|---|---|---|
| Pytest | 8.0.2 | Unit and integration testing |
| pytest-mock | 3.12.0 | Supabase client mocking in CI |
| pytest-asyncio | 0.23.5 | Async test support |
| Locust | 2.24.0 | Load testing and benchmarking |
| Flake8 | 7.0.0 | Code linting |

---

## Project Structure
```
sonix-ml/
│
├── src/
│   ├── main.py                        # FastAPI app, lifespan, all endpoints
│   ├── config.py                      # ROAD_FEATURES and TRAIL_FEATURES definitions
│   ├── database.py                    # Supabase client, interaction aggregation, upsert logic
│   │
│   ├── recommender/
│   │   ├── content_based.py           # Core pipeline: cluster routing + masked cosine similarity
│   │   ├── road_recommender.py        # Road heuristic mapping + pipeline wrapper
│   │   ├── trail_recommender.py       # Trail heuristic mapping + pipeline wrapper
│   │   └── collaborative_filtering.py # UserCollaborativeRecommender (UBCF + TTL cache)
│   │
│   └── training/
│       ├── architecture.py            # Deep Autoencoder definition (build_autoencoder)
│       └── training_engine.py         # Full training orchestration (run_training)
│
├── model_artifacts/
│   ├── road/
│   │   └── v_YYYYMMDD_HHMMSS/        # Versioned — latest selected automatically on startup
│   │       ├── shoe_encoder.h5
│   │       ├── kmeans_model.pkl
│   │       ├── scaler.pkl
│   │       ├── shoe_features.pkl
│   │       └── shoe_metadata.pkl
│   └── trail/
│       └── v_YYYYMMDD_HHMMSS/        # Same structure as road
│
├── data/
│   ├── road_dataset.csv
│   ├── trail_dataset.csv
│   └── unused-data/                   # Archived previous dataset versions
│
├── notebooks/
│   ├── data-preparation/
│   ├── data-preparation-v2/
│   ├── data-preparation-v3/
│   └── modelling/                     # road_ml.ipynb, trail_ml.ipynb
│
├── tests/
│   ├── conftest.py                    # Auto-mocks Supabase for all tests
│   ├── test_data_processing.py        # Rating conversion + DB error handling
│   └── test_recommender_logic.py      # Heuristic mapping + priority logic
│
├── .github/workflows/
│   ├── ci-ml.yml                      # Continuous Integration (lint + test)
│   ├── cd-ml.yml                      # Continuous Deployment
│   └── ct-ml.yml                      # Continuous Training trigger
│
├── locustfile.py
├── Dockerfile
├── requirements.txt                   # Production dependencies (all pinned)
├── requirements-dev.txt               # Dev/test dependencies
└── README.md
```

---

## Installation Guide

### Prerequisites
- Python **3.11** (strictly required)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/SONIX-Kelompok-6/sonix-ml
cd sonix-ml
```

### 2. Set Up a Virtual Environment

**Windows (PowerShell):**
```powershell
py -3.11 -m venv env
.\env\Scripts\activate
```

**macOS / Linux:**
```bash
python3.11 -m venv env
source env/bin/activate
```

### 3. Install Dependencies

**Production:**
```bash
pip install -r requirements.txt
```

**Development** (includes testing and linting tools):
```bash
pip install -r requirements-dev.txt
```

---

## Configuration

Create a `.env` file in the root directory:
```bash
cp .env.example .env
```

Then populate it with your credentials:
```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-public-key
```

### Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `SUPABASE_URL` | ✅ Yes | Your Supabase project URL |
| `SUPABASE_KEY` | ✅ Yes | Supabase anon/public API key |

> 🔒 Never commit `.env` to version control. Only `.env.example` (with placeholder values) should be committed.

---

## Usage Guide

### 1. Training Pipeline (Offline)

Before starting the API, model artifacts must be generated. This step fetches shoe catalog data from Supabase, trains the Deep Autoencoders, runs K-Means Clustering, and serializes all artifacts with a versioned timestamp.
```bash
python -m src.training.training_engine
```

The engine runs sequentially for both categories:
```
>>> Initializing training sequence for: ROAD
>>> Initializing training sequence for: TRAIL
```

Artifacts are saved to versioned directories:
```
model_artifacts/
├── road/v_YYYYMMDD_HHMMSS/
└── trail/v_YYYYMMDD_HHMMSS/
```

**Generated files per category:**

| File | Description |
|---|---|
| `shoe_encoder.h5` | Encoder model — projects features to 8D latent space |
| `kmeans_model.pkl` | K-Means (K=5) — routes inputs to candidate clusters |
| `scaler.pkl` | MinMaxScaler — must be used for inference normalization |
| `shoe_features.pkl` | Scaled feature matrix — used for cosine similarity at inference |
| `shoe_metadata.pkl` | DataFrame with cluster labels + column type definitions |

> ⚠️ The API will fail to start if artifacts are missing. Always run training before launching the server.
>
> 💡 The API automatically selects the **latest versioned directory** on startup — no manual version management required.

### 2. Starting the API (Online)

Once artifacts are generated, launch the FastAPI server:
```bash
python -m src.main
```

The server starts at `http://0.0.0.0:7860` with **4 Uvicorn workers**. On startup, the server loads both artifact sets and initializes the CF engine from Supabase interaction data.

---

## API Documentation

Interactive documentation is available at runtime:

- **Swagger UI:** `http://127.0.0.1:7860/docs`
- **ReDoc:** `http://127.0.0.1:7860/redoc`

Visiting `/` auto-redirects to `/docs`.

---

### `GET /health`

Returns service health and Continuous Training sync progress.

**Response `200 OK`:**
```json
{
  "status": "healthy",
  "ct_sync_progress": "12/50"
}
```

---

### `POST /recommend/road`

Returns Top 10 road shoe recommendations via the Content-Based pipeline. All fields are optional — unset fields are excluded from similarity calculation via feature masking.

**Request Body:**
```json
{
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
```

**Request Body Fields:**

| Field | Accepted Values |
|---|---|
| `pace` | `"Easy"`, `"Steady"`, `"Fast"` |
| `arch_type` | `"Flat"`, `"Normal"`, `"High"` |
| `strike_pattern` | `"Heel"`, `"Mid"`, `"Forefoot"` |
| `foot_width` | `"Narrow"`, `"Regular"`, `"Wide"` |
| `season` | `"Summer"`, `"Spring & Fall"`, `"Winter"` |
| `orthotic_usage` | `"Yes"`, `"No"` |
| `running_purpose` | `"Daily"`, `"Tempo"`, `"Race"` |
| `cushion_preferences` | `"Soft"`, `"Balanced"`, `"Firm"` |
| `stability_need` | `"Neutral"`, `"Guided"` |

**Response `200 OK`:**
```json
{
  "status": "success",
  "data": [
    {
      "shoe_id": "R045",
      "name": "Nike Vaporfly 3",
      "brand": "Nike",
      "cluster": 2,
      "match_score": 0.97
    }
  ]
}
```

---

### `POST /recommend/trail`

Returns Top 10 trail shoe recommendations. Uses trail-specific heuristics for terrain, traction, lug depth, and water resistance.

**Request Body:**
```json
{
  "pace": "Steady",
  "arch_type": "Normal",
  "strike_pattern": "Mid",
  "foot_width": "Regular",
  "season": "Summer",
  "orthotic_usage": "No",
  "terrain": "Rocky",
  "rock_sensitive": "Yes",
  "water_resistance": "Water Repellent"
}
```

**Request Body Fields:**

| Field | Accepted Values |
|---|---|
| `pace` | `"Easy"`, `"Steady"`, `"Fast"` |
| `arch_type` | `"Flat"`, `"Normal"`, `"High"` |
| `strike_pattern` | `"Heel"`, `"Mid"`, `"Forefoot"` |
| `foot_width` | `"Narrow"`, `"Regular"`, `"Wide"` |
| `season` | `"Summer"`, `"Spring & Fall"`, `"Winter"` |
| `orthotic_usage` | `"Yes"`, `"No"` |
| `terrain` | `"Light"`, `"Mixed"`, `"Rocky"`, `"Muddy"` |
| `rock_sensitive` | `"Yes"`, `"No"` |
| `water_resistance` | `"Waterproof"`, `"Water Repellent"` |

**Response `200 OK`:**
```json
{
  "status": "success",
  "data": [
    {
      "shoe_id": "T012",
      "name": "Hoka Speedgoat 5",
      "brand": "Hoka",
      "cluster": 4,
      "match_score": 0.94
    }
  ]
}
```

---

### `POST /interact`

Records a user interaction (Like or Rating) and immediately returns personalized CF recommendations via **Real-Time Injection** — the new interaction is injected into the user's vector before running KNN, so results reflect the latest signal without waiting for a full rebuild.

Also triggers a **background CT refresh** every 50 interactions (see [Continuous Training](#continuous-training)).

**Request Body:**
```json
{
  "user_id": 8,
  "shoe_id": "R278",
  "action_type": "like",
  "value": null
}
```

**Request Body Fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `user_id` | `integer` | ✅ Yes | Interacting user's ID |
| `shoe_id` | `string` | ✅ Yes | Target shoe ID (`R` prefix = road, `T` prefix = trail) |
| `action_type` | `string` | ✅ Yes | `"like"` or `"rate"` |
| `value` | `integer` | Conditional | Star rating `1–5`. Required when `action_type` is `"rate"` |

**Interaction Score Mapping:**

| Signal | Converted Score |
|---|---|
| Like | `+1.0` |
| Rate 5★ | `+2.0` |
| Rate 4★ | `+1.0` |
| Rate 3★ | `+0.1` (neutral) |
| Rate 2★ | `-1.0` |
| Rate 1★ | `-2.0` |

> If a user has both liked and rated the same shoe, scores are **summed** for higher confidence.

**Response `200 OK`:**
```json
{
  "status": "success",
  "data": [
    {
      "shoe_id": "R145",
      "name": "ASICS Gel-Kayano 31",
      "brand": "ASICS",
      "match_score": 1.84
    }
  ]
}
```

---

### `GET /recommend/feed/{user_id}`

Returns a personalized shoe feed from the CF engine. Served from the **60-second TTL cache** if available, otherwise computed fresh.

**Path Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `user_id` | `integer` | ✅ Yes | Target user's ID |

**Example Request:**
```
GET /recommend/feed/8
```

**Response `200 OK`:**
```json
{
  "status": "success",
  "data": [
    {
      "shoe_id": "R278",
      "name": "Nike Pegasus 41",
      "brand": "Nike",
      "match_score": 2.31
    }
  ]
}
```

**Response `404 Not Found`** — cold-start user with no interaction history.

---

## Continuous Training

Every **50 new interactions** received via `POST /interact`, a non-blocking background task is dispatched that:

1. Fetches fresh interaction data from Supabase (`favorites` + `reviews` tables)
2. Rebuilds `UserCollaborativeRecommender` from the updated data
3. Hot-swaps the global `cf_engine` with zero downtime

The API response is returned immediately — the rebuild happens in the background. Progress is visible in `/health` via `ct_sync_progress`.

---

## Error Handling

All error responses follow a consistent format:
```json
{
  "detail": "Human-readable error message."
}
```

| Status Code | Meaning | Common Cause |
|---|---|---|
| `200 OK` | Success | — |
| `404 Not Found` | Resource not found | Cold-start user with no interaction history |
| `422 Unprocessable Entity` | Validation error | Missing or invalid request fields |
| `500 Internal Server Error` | Unexpected error | Check server logs |
| `503 Service Unavailable` | Engine not ready | Run training pipeline first |

---

## Docker Deployment

This service is containerized and compatible with **Hugging Face Spaces** and standard Docker environments.

### Build and Run
```bash
docker build -t sonix-ml .
docker run -p 7860:7860 --env-file .env sonix-ml
```

The service will be accessible at `http://localhost:7860`.

### Docker Compose (Optional)
```yaml
version: "3.9"
services:
  rush-ai:
    build: .
    ports:
      - "7860:7860"
    env_file:
      - .env
    restart: unless-stopped
```
```bash
docker compose up --build
```

---

## Testing

### Unit & Integration Tests

`conftest.py` automatically mocks the Supabase client across all tests, preventing real network calls during CI.
```bash
# Run all tests
pytest

# With verbose output and coverage report
pytest -v --tb=short --cov=src --cov-report=term-missing
```

**Test coverage:**

| File | What it tests |
|---|---|
| `test_data_processing.py` | Rating-to-score conversion, empty DataFrame handling from DB |
| `test_recommender_logic.py` | Heuristic priority mapping, fallback defaults, multi-source priority |

### Load Testing
```bash
locust -f locustfile.py
```

Open `http://localhost:8089` to configure and launch. Default scenario:

| Endpoint | Weight | Description |
|---|---|---|
| `POST /interact` | 3× | Heaviest endpoint — simulates likes and ratings |
| `GET /recommend/feed/{user_id}` | 1× | Feed retrieval |
| `GET /health` | 1× | Lightweight health probe |

### Load Test Results

> Total requests: **10,420** across all endpoints — **0 failures** (0% error rate).
> Aggregated throughput: **65.5 RPS**.

#### Request Statistics

| Method | Endpoint | # Requests | Median (ms) | Avg (ms) | Min (ms) | Max (ms) | RPS |
|---|---|---|---|---|---|---|---|
| GET | `/health` | 2,052 | 4 | 27.6 | 2 | 3,014 | 14.3 |
| POST | `/interact` | 6,218 | 23 | 46.93 | 10 | 3,173 | 39.1 |
| GET | `/recommend/feed/8` | 2,150 | 7 | 37.36 | 3 | 2,146 | 12.1 |
| | **Aggregated** | **10,420** | **19** | **41.15** | **2** | **3,173** | **65.5** |

#### Response Time Percentiles

| Method | Endpoint | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|
| GET | `/health` | 4 | 13 | 19 | 900 |
| POST | `/interact` | 23 | 41 | 53 | 730 |
| GET | `/recommend/feed/8` | 7 | 23 | 33 | 2,000 |
| | **Aggregated** | **19** | **36** | **46** | **920** |

**Key observations:**
- `POST /interact` handles the highest load (39.1 RPS, 3× weight) with a p95 of **53ms** — well within real-time UX thresholds.
- `GET /recommend/feed/8` achieves a p50 of **7ms** for cached responses thanks to the 60-second TTL cache.
- `GET /health` spikes at p99 (900ms) and max (3,014ms) due to occasional GIL contention under high concurrency — negligible for a health probe.
- Zero failures across 10,420 total requests confirms production-grade stability.

### Code Linting
```bash
flake8 src/
```

---

## Contributing

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes using [Conventional Commits](https://www.conventionalcommits.org/).
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request.

Ensure all code passes `flake8` and `pytest` before submitting.

### Commit Message Convention

| Prefix | Use for |
|---|---|
| `feat:` | New features |
| `fix:` | Bug fixes |
| `docs:` | Documentation changes |
| `refactor:` | Code restructuring without behavior change |
| `test:` | Adding or updating tests |
| `chore:` | Maintenance tasks (deps, CI, config) |

---

## License

This project is licensed under the MIT License.

Copyright (c) 2026 SONIX RUSH

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

*Built with ❤️ for the SONIX RUSH Application.*