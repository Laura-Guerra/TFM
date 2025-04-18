"""Script to process cleaned news and prepare NLP-ready text."""

import pandas as pd
from tfm.src.config.settings import PATH_DATA_PROCESSED
from tfm.src.nlp.news_processor import NewsProcessor
from tfm.src.nlp.text_processor import TextProcessor

# %% Load data
news_df = pd.read_csv(PATH_DATA_PROCESSED / "articles_cleaned.csv")
keywords_df = pd.read_csv(PATH_DATA_PROCESSED / "keywords_cleaned.csv")

# %% Unify text
news_processor = NewsProcessor(
    allowed_types=["subject", "glocations", "organizations", "persons"],
    top_n_keywords=5,
    use_tags=False
)

news_with_text = news_processor.unify_text(news_df, keywords_df)

# %% Clean text for NLP
a = news_with_text[:100]
text_processor = TextProcessor(lang="english", use_lemmatizer=True)
news_final = text_processor.apply(a, text_column="full_text")

# %% Save output
news_final.to_csv(PATH_DATA_PROCESSED / "articles_final_nlp.csv", index=False)

print("Done!")
