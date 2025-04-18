"""Script to analyze news data."""

import pandas as pd
import matplotlib.pyplot as plt

from tfm.src.config.settings import PATH_DATA_PROCESSED, NEWS_SECTIONS, NEWS_SUBSECTIONS
from tfm.src.viz.plot_theme import set_custom_theme

set_custom_theme()

# %% Load data
news_raw = pd.read_csv(PATH_DATA_PROCESSED / "articles.csv")
keywords_df = pd.read_csv(PATH_DATA_PROCESSED / "keywords.csv")

# %% Filter news
print("Distribució de tipus de notícies:")
print(news_raw["doc_type"].value_counts())
news_df = news_raw[news_raw["doc_type"] == "article"].copy()
news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
news_df = news_df[news_df["date"].notna()]
news_df.loc[:, "year"] = news_df["date"].dt.year
news_df = news_df[news_df["year"] >= 2013].sort_values(by="date")

# %% Check missing values
print("Missing values per column:")
print(news_df.isnull().sum())

news_with_null_header = news_df[news_df["title"].isnull()]
print(f"Notícies sense títol: {len(news_with_null_header)}")

news_with_null_abstract = news_df[news_df["abstract"].isnull()]
print(f"Notícies sense abstract: {len(news_with_null_abstract)}")

# Clean up nulls
news_df = news_df.dropna(subset=["title", "abstract"])

# %% Check duplicates
duplicates_df = news_df[news_df.duplicated(subset=["title", "abstract"])]
duplicated_title = news_df[news_df.duplicated(subset=["title"])]
duplicated_abstract = news_df[news_df.duplicated(subset=["abstract"])]

news_df = news_df.drop_duplicates(subset=["title", "abstract"], keep="last")

# %% Get sections counts
sections = news_df["section"].value_counts()

# %% Add text length columns
news_df.loc[:,"title_len"] = news_df["title"].fillna("").str.len()
news_df.loc[:,"abstract_len"] = news_df["abstract"].fillna("").str.len()

# %% Plot title and abstract length distributions
plt.figure()
news_df["title_len"].hist(bins=30)
plt.title("Distribució longitud dels títols")
plt.xlabel("Nombre de caràcters")
plt.ylabel("Nombre de notícies")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
news_df["abstract_len"].hist(bins=30)
plt.title("Distribució longitud dels abstracts")
plt.xlabel("Nombre de caràcters")
plt.ylabel("Nombre de notícies")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Distribution of articles per year
year_counts = news_df["year"].value_counts().sort_index()

plt.figure()
year_counts.plot(kind="bar")
plt.title("Nombre de notícies per any")
plt.xlabel("Any")
plt.ylabel("Nombre de notícies")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# %% Top sections and subsections
top_sections = news_df["section"].value_counts().head(10)
top_subsections = news_df["subsection"].value_counts().head(10)

plt.figure()
top_sections.plot(kind="barh")
plt.title("Top 10 seccions")
plt.xlabel("Nombre de notícies")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

plt.figure()
top_subsections.plot(kind="barh")
plt.title("Top 10 subseccions")
plt.xlabel("Nombre de notícies")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# %% Filter categories
cleaned_news = news_df[news_df["section"].isin(NEWS_SECTIONS)].copy()
subsections = cleaned_news["subsection"].value_counts()
cleaned_news = cleaned_news[cleaned_news["subsection"].isin(NEWS_SUBSECTIONS)]

# %% Filter keywords
keywords_df = keywords_df.drop_duplicates(subset=["new_id", "value"])
keywords = keywords_df[keywords_df["new_id"].isin(cleaned_news["new_id"])]
keywords_counts = keywords[["name", "value"]].value_counts()

# %% Save cleaned data
cleaned_news.to_csv(PATH_DATA_PROCESSED / "articles_cleaned.csv", index=False)
keywords.to_csv(PATH_DATA_PROCESSED / "keywords_cleaned.csv", index=False)