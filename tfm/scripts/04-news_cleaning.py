"""Script to process cleaned news and prepare NLP-ready text."""

import pandas as pd
from loguru import logger

from tfm.src.config.settings import PATH_DATA_PROCESSED
from tfm.src.NLP.news_processor import NewsProcessor
from tfm.src.NLP.text_processor import TextProcessor
from tfm.src.utils.generate_new_market_date import compute_effective_market_date

# %% Load data
news_df = pd.read_csv(PATH_DATA_PROCESSED / "articles_cleaned.csv")
keywords_df = pd.read_csv(PATH_DATA_PROCESSED / "keywords_cleaned.csv")

# %% Ensure datetime with timezone
news_df["date"] = pd.to_datetime(news_df["date"], utc=True)

# %% Add market-based effective dates
news_df["market_date_us"] = news_df["date"].apply(lambda x: compute_effective_market_date(x, market="us"))
news_df["market_date_eu"] = news_df["date"].apply(lambda x: compute_effective_market_date(x, market="eu"))

# %% Unify text
news_processor = NewsProcessor(
    allowed_types=["subject", "glocations", "organizations", "persons"],
    top_n_keywords=5,
    use_tags=False
)
news_with_text = news_processor.unify_text(news_df, keywords_df)

# %% Clean text for NLP
text_processor = TextProcessor(lang="english", use_lemmatizer=True)
news_final = text_processor.apply(news_with_text, text_column="full_text")

# %% Save output
news_final.to_csv(PATH_DATA_PROCESSED / "articles_final_nlp.csv", index=False)

logger.info("✅ NLP-ready news data saved.")
