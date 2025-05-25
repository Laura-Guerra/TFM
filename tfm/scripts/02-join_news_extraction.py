"""Script to join news in two csv."""

import pandas as pd
import os

from tfm.src.config.settings import PATH_DATA_RAW, PATH_DATA_PROCESSED

# %% Join news
articles_paths = [file for file in os.listdir(PATH_DATA_RAW / "news") if "articles" in file]
keywords_paths = [file for file in os.listdir(PATH_DATA_RAW / "news") if "keywords" in file]


articles = pd.DataFrame()
for path in articles_paths:
    df = pd.read_csv(PATH_DATA_RAW / "news" / path)
    articles = pd.concat([articles, df], ignore_index=True)

keywords = pd.DataFrame()
for path in keywords_paths:
    df = pd.read_csv(PATH_DATA_RAW / "news" / path)
    keywords = pd.concat([keywords, df], ignore_index=True)

# %% Save dataframe
articles.to_csv(PATH_DATA_PROCESSED / "articles.csv", index=False)
keywords.to_csv(PATH_DATA_PROCESSED / "keywords.csv", index=False)
