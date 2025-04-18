import pandas as pd


class NewsProcessor:
    def __init__(
        self,
        allowed_types: list[str] = None,
        top_n_keywords: int = None,
        use_tags: bool = False
    ):
        """
        Initialize variables.
        """
        self.allowed_types = allowed_types
        self.top_n_keywords = top_n_keywords
        self.use_tags = use_tags

    def aggregate_keywords(self, keywords_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine keywords per article.
        """
        df = keywords_df.copy()

        if self.allowed_types:
            df = df[df["name"].isin(self.allowed_types)]

        if self.top_n_keywords:
            df = df[df["rank"] <= self.top_n_keywords]

        df = df.sort_values(by=["new_id", "rank"])

        if self.use_tags:
            def tag_and_concat(sub_df):
                return " ".join(f"[{row['name'].upper()}] {row['value']}" for _, row in sub_df.iterrows())

            keywords_agg = (
                df.groupby("new_id")
                .apply(tag_and_concat)
                .reset_index()
                .rename(columns={0: "keywords_str"})
            )
        else:
            keywords_agg = (
                df.groupby("new_id")["value"]
                .apply(lambda vals: " ".join(vals.astype(str)))
                .reset_index()
                .rename(columns={"value": "keywords_str"})
            )

        return keywords_agg

    def unify_text(self, news_df: pd.DataFrame, keywords_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create 'full_text'.
        """
        df = news_df.copy()
        keywords_agg = self.aggregate_keywords(keywords_df)
        df = df.merge(keywords_agg, on="new_id", how="left")

        df["full_text"] = (
            df["title"].fillna("") + ". " +
            df["abstract"].fillna("") + ". " +
            df["keywords_str"].fillna("")
        )

        return df
