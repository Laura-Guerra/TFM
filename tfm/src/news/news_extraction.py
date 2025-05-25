# Create class for news extraction
import uuid

import pandas as pd
import requests

from tfm.models.nyt_schema import NYTParams, NYTResponse
from tfm.src.config.settings import NEWS_API_KEY, NEWS_URL, PATH_DATA_RAW


class NewsExtractor:
    def __init__(self):
        self.news_data = None

    def extract_nyt_news(self, month, year):
        """
        Extract news data from the API.
        """
        params = NYTParams(
            year=year,
            month=month,
            api_key=NEWS_API_KEY
        )
        self._fetch_news(params)
        articles = self._parse_nyt_response(self.news_data)
        self._save_news_data(articles, params)

        return

    def _fetch_news(self, params):
        """
        Fetch news data from the API.
        """
        url = f"{NEWS_URL}{params.year}/{params.month}.json"
        query_params = {
            "api-key": params.api_key
        }
        response = requests.get(url, params=query_params)

        if response.status_code == 200:
            self.news_data=response.json()
        else:
            print(f"Error fetching data: {response.status_code} "
                  f"with date {params.year}-{params.month}")
        return

    def _parse_nyt_response(self, response):
        """
        Parse the response from the NYT API.
        """
        articles = response.get("response", {}).get("docs", [])
        parsed_articles = []
        parsed_keywords = []

        for article in articles:
            id = uuid.uuid4()

            parsed_article = {
                "title": article.get("headline", {}).get("main"),
                "abstract": article.get("snippet") or article.get("abstract"),
                "url": article.get("web_url"),
                "date": article.get("pub_date"),
                "section": article.get("section_name"),
                "subsection": article.get("subsection_name"),
                "organization": article.get("source"),
                "doc_type": article.get("document_type"),
                "material_type": article.get("type_of_material"),
                "new_id": id,
            }
            parsed_articles.append(parsed_article)

            keywords = article.get("keywords", [])
            for keyword in keywords:
                parsed_keyword = {
                    "new_id": id,
                    "name": keyword.get("name"),
                    "value": keyword.get("value"),
                    "rank": keyword.get("rank"),
                }
                parsed_keywords.append(parsed_keyword)

        return NYTResponse(articles=parsed_articles, keywords=parsed_keywords)


    def _save_news_data(self, response: NYTResponse, params):
        """
        Save the news data to a csv file.
        """
        month = params.month
        year = params.year

        articles_df = pd.DataFrame(response.articles)
        keywords_df = pd.DataFrame(response.keywords)

        articles_df.to_csv(f"{PATH_DATA_RAW}/news"
                           f"/articles_{month}_{year}.csv", index=False)
        keywords_df.to_csv(f"{PATH_DATA_RAW}/news"
                           f"/keywords_{month}_{year}.csv", index=False)

