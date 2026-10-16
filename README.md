🎬 CineAI v2: Enterprise Two-Stage Hybrid Recommender System

CineAI v2 is a production-grade, two-stage hybrid recommendation engine built on the MovieLens 32M dataset. Inspired by industrial recommender architectures used at YouTube and Netflix, CineAI decouples candidate retrieval from machine learning re-ranking. It delivers personalized movie recommendations with native, zero-latency TreeSHAP Explainable AI (XAI).

📐 System Architecture

graph TD
    User([User Request / History]) --> Retrieval

    subgraph Retrieval [Stage 1: Multi-Vector Candidate Generation ~150 candidates]
        ALS[Implicit ALS Collaborative Factors]
        SVD[TF-IDF + TruncatedSVD Content Embeddings]
        POP[IMDB-Weighted Popularity Baseline]
    end

    Retrieval --> FeatureEng[Feature Engineering Pipeline - 18 Dense Features]

    subgraph Ranking [Stage 2: Dual-Model ML Re-Ranking]
        FeatureEng --> LambdaRank[LightGBM LambdaRANK Model - Top-N Ordering]
        FeatureEng --> LikerModel[LightGBM Binary Model - Calibrated Match %]
        FeatureEng --> SHAP[Native LightGBM TreeSHAP - pred_contrib=True]
    end

    LambdaRank --> Response
    LikerModel --> Response
    SHAP --> Response

    subgraph Serving [Microservice Layer]
        Response[Aggregated Recommendation Payload] --> API[FastAPI Microservice]
        API --> UI[Netflix-Style Web Client + TMDB Posters]
    end


✨ Key Features & Algorithmic Highlights

1. Two-Stage Candidate Pipeline

Stage 1 (Retrieval): Shrinks a catalogue of 200,000+ movies to ~150 high-quality candidates using a union of:

Matrix Factorization: Implicit Alternating Least Squares (ALS, $d=64$) for latent user-item interaction signals.

Content Vector Space: TF-IDF text vectorization on movie titles, genres, and user tags, reduced via TruncatedSVD ($d=64$).

Weighted Popularity: IMDB-style Bayesian average baseline (pop_w) to prevent candidate sparsity.

Stage 2 (Re-ranking): A pairwise LightGBM LambdaRANK model scores candidate pairs across 18 engineered features (e.g., als_z, cont_z, gaff, gshare, year_gap, u_pos, i_mean) to optimize NDCG@10.

2. Dual-Model Scoring System

Ordering Engine (ranker): LightGBM trained with lambdarank objective and NDCG metric to optimize relative item ordering.

Probability Engine (liker): A secondary binary LightGBM model trained on rated items ($\ge 4.0$ stars) to calculate a calibrated probability score, powering the "92% Match" UI badge.

3. Native Zero-Latency Explainable AI (TreeSHAP)

Eliminates runtime shap library dependencies by extracting exact SHAP values directly from LightGBM decision paths using pred_contrib=True.

Aggregates raw feature contributions into 5 human-readable explanation categories:

Your genre taste (gaff, gshare)

Similar to what you liked (cont, cont_z)

People with similar taste (als, als_z)

Community rating (i_mean, i_pos, base_pred, pop_w)

Era & profile fit (year_gap, u_year, u_cnt)

4. Cold-Start Handling

New Items: Applies a Ridge regression bridge to map TF-IDF content vectors directly into ALS collaborative factor space for newly added movies with few ratings.

New Users: Bypasses matrix operations to serve a genre-diverse, exploration-backed cold-start feed without runtime exceptions.

5. Leak-Free Evaluation Protocol

Evaluated strictly using a chronological per-user split (Oldest 75% train, Middle 10% validation, Newest 15% test) to ensure no future data leaks into model training.

📊 Evaluation Metrics

The system is evaluated across two distinct evaluation protocols on unseen test-period ratings:

Protocol

Strategy

Goal

Metric Focus

Protocol A (Taste)

Ranks items the user actually rated later.

Tests accuracy of high-rating predictions.

NDCG@10, HitRate@10, Precision@10

Protocol B (Discovery)

Ranks the entire 200,000+ catalog.

Tests retrieval and diversity across the platform.

Catalog Coverage, Novelty (Bits), Genre Diversity

📁 Repository Structure

.
├── app.py                  # FastAPI REST microservice wrapper
├── cineai.py               # Core algorithmic engine (ALS, SVD, LightGBM, TreeSHAP)
├── cineai_widget.html      # Netflix-style web frontend (Vanilla JS / CSS3)
├── CineAI_v2_colab.ipynb   # Model training & evaluation pipeline (Colab ready)
├── requirements.txt        # Production dependencies
└── README.md               # Documentation


🚀 Quickstart & Setup

1. Prerequisites

Python 3.10+

Virtual environment tool (venv or conda)

2. Installation

# Clone the repository
git clone https://github.com/your-username/CineAI-v2.git
cd CineAI-v2

# Install dependencies
pip install -r requirements.txt


3. Local Model Training

To train the model on the MovieLens 32M dataset and export the serialized artifact (rs_full.pkl), run the training notebook CineAI_v2_colab.ipynb.

4. Running the FastAPI Server

Point the environment variable CINEAI_MODEL to your dumped model artifact and launch Uvicorn:

export CINEAI_MODEL="./rs_full.pkl"
uvicorn app:app --host 0.0.0.0 --port 8000 --reload


The server will start at http://localhost:8000. You can access interactive Swagger API documentation at http://localhost:8000/docs.

5. Launching the Web UI

Open cineai_widget.html in your text editor.

Set your backend URL in JavaScript:

const API_URL = "http://localhost:8000"; // Or your production URL


(Optional) Add a free TMDB API Key (TMDB_API_KEY) to enable real-time movie poster fetching.

Double-click cineai_widget.html to open it in any web browser.

📡 API Specification

POST /recommend

Generates personalized recommendations based on user rating history, genre filters, or natural language prompts.

Request Payload:

{
  "history": [
    {"movieId": 1, "rating": 5.0},
    {"movieId": 257, "rating": 4.5}
  ],
  "genres": ["Sci-Fi", "Action"],
  "query": "slow burn psychological space thriller",
  "top_n": 10
}


Response Payload:

{
  "items": [
    {
      "movieId": 1090,
      "title": "Moon",
      "year": 2009,
      "genres": ["Drama", "Mystery", "Sci-Fi"],
      "score": 1.42,
      "match": 94,
      "mode": "personalized",
      "because": {
        "movieId": 1,
        "title": "The Matrix"
      },
      "reasons": [
        {
          "group": "Similar to what you liked",
          "text": "Close in style to movies you rated highly",
          "impact": 0.52
        }
      ],
      "contributions": [
        {"group": "Your genre taste", "value": 0.31},
        {"group": "Similar to what you liked", "value": 0.52},
        {"group": "People with similar taste", "value": 0.18},
        {"group": "Community rating", "value": 0.22},
        {"group": "Era & profile fit", "value": 0.05}
      ]
    }
  ]
}


🛠️ Deployment

Docker Deployment

FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py cineai.py rs_full.pkl ./

ENV CINEAI_MODEL=/app/rs_full.pkl
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


📄 License

Distributed under the MIT License. See LICENSE for more information.
