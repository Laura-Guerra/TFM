"""Script to generate NLP-based features for DRL input."""

import pandas as pd
from pathlib import Path
from tfm.src.config.settings import PATH_DATA_PROCESSED
from tfm.src.nlp.nlp_processor import NLProcessor

# Paths
MODEL_DIR = PATH_DATA_PROCESSED / "nlp_models"
INPUT_FILE = PATH_DATA_PROCESSED / "articles_final_nlp.csv"
OUTPUT_FILE = PATH_DATA_PROCESSED / "articles_nlp_vectors.csv"

# Params
CUTOFF_DATE = "2024-01-01"
WORD2VEC_DIM = 100
LDA_TOPICS = 10
MIN_WORD_COUNT = 3

# Load data
df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

# Init processor
nlp = NLProcessor(
    word2vec_dim=WORD2VEC_DIM,
    lda_topics=LDA_TOPICS,
    min_word_count=MIN_WORD_COUNT,
    cutoff_date=CUTOFF_DATE
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
try:
    nlp.load(str(MODEL_DIR))
    print("✅ NLP models loaded from disk.")
except Exception as e:
    print("⚠️ Could not load models, training from scratch...")
    nlp.train(df)
    nlp.save(str(MODEL_DIR))
    print("✅ NLP models trained and saved.")

# Transform all data
df_vectors = nlp.transform(df)
df_vectors.to_csv(OUTPUT_FILE, index=False)

nlp.print_lda_topics(top_n=8)

print(f"✅ Features saved to {OUTPUT_FILE.name}")
