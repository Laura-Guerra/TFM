"""Script to generate news-based features for DRL input."""

import pandas as pd
from pathlib import Path
from tfm.src.config.settings import PATH_DATA_PROCESSED
from tfm.src.news.nlp_processor import NLProcessor

# Params
CUTOFF_DATE = "2024-01-01"
WORD2VEC_DIM = 100
LDA_TOPICS = 10
MIN_WORD_COUNT = 3
MARKET = "us"  # Canvia per "eu" si vols Europa
STRATEGY = "concat"  # O "mean"

# Paths
MODEL_DIR = PATH_DATA_PROCESSED / "nlp_models"
INPUT_FILE = PATH_DATA_PROCESSED / "articles_final_nlp.csv"
OUTPUT_FILE = PATH_DATA_PROCESSED / f"articles_nlp_vectors_{MARKET}.csv"

# Load data
df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])
df[f"market_date_{MARKET}"] = pd.to_datetime(df[f"market_date_{MARKET}"])  # assegura format

# Init processor with FinBERT
nlp = NLProcessor(
    word2vec_dim=WORD2VEC_DIM,
    lda_topics=LDA_TOPICS,
    min_word_count=MIN_WORD_COUNT,
    cutoff_date=CUTOFF_DATE,
    use_finbert=True
)

# Ensure model dir exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load or train models
try:
    nlp.load(str(MODEL_DIR))
    print("✅ news models loaded from disk.")
except Exception as e:
    print("⚠️ Could not load models, training from scratch...")
    nlp.train(df)
    nlp.save(str(MODEL_DIR))
    print("✅ news models trained and saved.")

# Transform all data
df_vectors = nlp.transform_dataset_by_day(df, strategy=STRATEGY, market=MARKET)
df_vectors.to_csv(OUTPUT_FILE, index=False)

# Print topics
nlp.print_lda_topics(top_n=8)

print(f"✅ Features saved to {OUTPUT_FILE.name}")
