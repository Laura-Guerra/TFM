import joblib
import numpy as np
import pandas as pd
from loguru import logger
from gensim.models import Word2Vec
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

from transformers import pipeline

from tfm.src.nlp.text_processor import TextProcessor

class NLProcessor:
    def __init__(
        self,
        word2vec_dim=100,
        lda_topics=10,
        min_word_count=3,
        cutoff_date="2024-01-01",
        use_finbert: bool = False,
        top_n: int = 5
    ):
        self.word2vec_dim = word2vec_dim
        self.lda_topics = lda_topics
        self.min_word_count = min_word_count
        self.cutoff_date = pd.to_datetime(cutoff_date).tz_localize("UTC")
        self.use_finbert = use_finbert
        self.top_n = top_n
        self.text_processor = TextProcessor(lang="english", use_lemmatizer=True)

        self.word2vec_model = None
        self.lda_model = None
        self.vectorizer = None
        self.finbert = None

        if self.use_finbert:
            logger.info("Loading FinBERT sentiment model...")
            try:
                self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
                logger.success("FinBERT loaded successfully.")
            except Exception as e:
                logger.error(f"Could not load FinBERT: {e}")
                self.use_finbert = False

    def train(self, df: pd.DataFrame):
        logger.info("Training NLP models...")
        df = df[df["date"] < self.cutoff_date]

        logger.info(f"Training on {len(df)} documents prior to {self.cutoff_date.date()}")
        token_lists = [text.split() for text in df["clean_text"]]
        raw_texts = df["clean_text"].tolist()

        self.train_word2vec(token_lists)
        self.train_lda(raw_texts)
        logger.success("Finished training NLP models.")

    def train_word2vec(self, texts: list[list[str]]):
        logger.debug("Training Word2Vec...")
        self.word2vec_model = Word2Vec(
            sentences=texts,
            vector_size=self.word2vec_dim,
            window=5,
            min_count=self.min_word_count,
            workers=4
        )
        logger.debug("Word2Vec training completed.")

    def train_lda(self, raw_texts: list[str]):
        logger.debug("Training LDA...")
        self.vectorizer = CountVectorizer(min_df=5)
        doc_term_matrix = self.vectorizer.fit_transform(raw_texts)

        self.lda_model = LatentDirichletAllocation(
            n_components=self.lda_topics,
            learning_method='online',
            random_state=42
        )
        self.lda_model.fit(doc_term_matrix)
        logger.debug("LDA training completed.")

    def get_doc_vector(self, tokens: list[str]) -> np.ndarray:
        if not self.word2vec_model:
            logger.error("Word2Vec model not trained.")
            raise ValueError("Word2Vec model not trained.")
        vectors = [self.word2vec_model.wv[w] for w in tokens if w in self.word2vec_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.word2vec_dim)

    def get_topic_vector(self, text: str) -> np.ndarray:
        if not self.lda_model or not self.vectorizer:
            logger.error("LDA model or vectorizer not trained.")
            raise ValueError("LDA model not trained.")
        doc_vec = self.vectorizer.transform([text])
        return self.lda_model.transform(doc_vec)[0]

    def get_sentiment_score(self, text: str) -> float:
        try:
            text = str(text).strip()
            if not text:
                return 0.0

            if not self.use_finbert and not self.finbert:
                return TextBlob(text).sentiment.polarity

            result = self.finbert(text)[0]
            label = result["label"].lower()
            score = result["score"]

            if label == "positive":
                return score
            elif label == "negative":
                return -score
            else:  # neutral
                return 0.0

        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return 0.0

    def transform(self, df: pd.DataFrame) -> pd.DataFrame: # TODO eliminar funció
        logger.info(f"Transforming {len(df)} documents into NLP feature vectors...")

        df = df.reset_index(drop=True)
        token_lists = [text.split() for text in df["clean_text"]]

        # Word2Vec
        doc_vectors = np.array([self.get_doc_vector(toks) for toks in token_lists])
        w2v_df = pd.DataFrame(doc_vectors, columns=[f"w2v_{i}" for i in range(self.word2vec_dim)])

        # LDA
        topic_vectors = np.array([self.get_topic_vector(txt) for txt in df["clean_text"]])
        lda_df = pd.DataFrame(topic_vectors, columns=[f"lda_{i}" for i in range(self.lda_topics)])

        # Sentiment
        sentiment_scores = df["full_text"].apply(self.get_sentiment_score)
        sentiment_scores.name = "sentiment"

        # Selecció de les columnes base
        base_df = df[["date", "full_text"]].copy()

        # Concatenar tot
        final_df = pd.concat([base_df, w2v_df, lda_df, sentiment_scores], axis=1)

        logger.success("Transformation completed.")
        return final_df

    def _get_top_n_articles_as_text(self, df: pd.DataFrame) -> str:
        """
        Selects the top-N most sentimentally extreme articles from a DataFrame
        and concatenates their full_text into a single string.

        Args:
            df: DataFrame containing at least 'full_text' and 'sentiment' columns.

        Returns:
            A single string with the concatenated full_texts of the top-N articles.
        """
        df = df.copy()
        if "sentiment" not in df.columns:
            df["sentiment"] = df["full_text"].apply(self.get_sentiment_score)

        df["abs_sentiment"] = df["sentiment"].abs()
        top_articles = df.sort_values("abs_sentiment", ascending=False).head(self.top_n)
        return " ".join(top_articles["full_text"].tolist())

    def transform_day_with_top_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the most relevant articles from a single day by:
        - Selecting top-N articles by absolute sentiment
        - Concatenating their text
        - Computing NLP features (Word2Vec, LDA, sentiment)

        Args:
            df: DataFrame with all articles for a given date.

        Returns:
            DataFrame with a single row: NLP features for that day.
        """
        logger.info(f"Transforming {len(df)} documents → top {self.top_n} most relevant")

        df = df.reset_index(drop=True)
        concatenated_text = self._get_top_n_articles_as_text(df)

        clean_text = self.text_processor.process_text(concatenated_text)
        tokens = clean_text.split()

        w2v_vec = self.get_doc_vector(tokens)
        lda_vec = self.get_topic_vector(clean_text)
        sentiment = self.get_sentiment_score(concatenated_text)
        date = df["date"].iloc[0] if "date" in df.columns else pd.NaT

        data = {
            "date": date,
            "full_text": concatenated_text,
            "sentiment": sentiment,
            **{f"w2v_{i}": w2v_vec[i] for i in range(len(w2v_vec))},
            **{f"lda_{i}": lda_vec[i] for i in range(len(lda_vec))}
        }

        logger.success("Day-level NLP transformation completed.")
        return pd.DataFrame([data])

    def _transform_day_mean_of_top_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Selects top-N articles by absolute sentiment, computes NLP vectors for each,
        and returns the mean of all vectors.

        Args:
            df: DataFrame of news from one day.

        Returns:
            DataFrame with a single row of averaged NLP features.
        """
        df = df.copy()
        df["sentiment"] = df["full_text"].apply(self.get_sentiment_score)
        df["abs_sentiment"] = df["sentiment"].abs()
        top_df = df.sort_values("abs_sentiment", ascending=False).head(self.top_n)

        text_processor = self.text_processor
        tokens_list = [text_processor.process_text(text).split() for text in top_df["full_text"]]

        w2v_vectors = np.array([self.get_doc_vector(tokens) for tokens in tokens_list])
        lda_vectors = np.array([self.get_topic_vector(" ".join(tokens)) for tokens in tokens_list])
        sentiments = top_df["sentiment"].to_numpy()

        w2v_mean = w2v_vectors.mean(axis=0)
        lda_mean = lda_vectors.mean(axis=0)
        sentiment_mean = sentiments.mean()
        date = top_df["date"].iloc[0] if "date" in top_df.columns else pd.NaT

        data = {
            "date": date,
            "full_text": " ".join(top_df["full_text"].tolist()),
            "sentiment": sentiment_mean,
            **{f"w2v_{i}": w2v_mean[i] for i in range(len(w2v_mean))},
            **{f"lda_{i}": lda_mean[i] for i in range(len(lda_mean))}
        }

        return pd.DataFrame([data])

    def transform_dataset_by_day(
            self,
            df: pd.DataFrame,
            strategy: str = "concat"  # o "mean"
    ) -> pd.DataFrame:
        """
        Transforms a news dataset grouped by date, using one of two strategies:
        - "concat": top-N articles are concatenated and a single NLP vector is computed.
        - "mean": top-N articles are processed individually and their vectors averaged.

        Args:
            df: DataFrame with 'date' and 'full_text' columns.
            strategy: Aggregation strategy, either "concat" or "mean".

        Returns:
            DataFrame with one row per day and NLP vectors.
        """
        assert strategy in ["concat", "mean"], "strategy must be 'concat' or 'mean'"

        logger.info(f"Starting NLP transformation by day using strategy: {strategy}")
        results = []

        for date, group in df.groupby("date"):
            try:
                if strategy == "concat":
                    day_vector = self.transform_day_with_top_articles(group)
                else:  # "mean"
                    day_vector = self._transform_day_mean_of_top_articles(group)
                results.append(day_vector)
            except Exception as e:
                logger.warning(f"Skipping date {date} due to error: {e}")

        final_df = pd.concat(results, ignore_index=True)
        logger.success(f"Transformation completed for {len(final_df)} days.")
        return final_df


    def save(self, path: str):
        logger.info(f"Saving NLP models to {path}")
        if self.word2vec_model:
            self.word2vec_model.save(f"{path}/word2vec.model")
        if self.lda_model:
            joblib.dump(self.lda_model, f"{path}/lda_model.pkl")
        if self.vectorizer:
            joblib.dump(self.vectorizer, f"{path}/vectorizer.pkl")
        logger.success("Models saved successfully.")

    def load(self, path: str):
        logger.info(f"Loading NLP models from {path}")
        self.word2vec_model = Word2Vec.load(f"{path}/word2vec.model")
        self.lda_model = joblib.load(f"{path}/lda_model.pkl")
        self.vectorizer = joblib.load(f"{path}/vectorizer.pkl")
        logger.success("Models loaded successfully.")

    def print_lda_topics(self, top_n: int = 10):
        """
        Print top N words for each LDA topic.
        """
        if not self.lda_model or not self.vectorizer:
            logger.warning("LDA model or vectorizer not available.")
            return

        logger.info(f"🧠 Top {top_n} words per topic:")
        words = self.vectorizer.get_feature_names_out()

        for idx, topic in enumerate(self.lda_model.components_):
            top_words = [words[i] for i in topic.argsort()[:-top_n - 1:-1]]
            logger.info(f"🟢 Topic {idx}: {' | '.join(top_words)}")

    def get_topic_keywords(self, top_n: int = 10) -> dict[int, list[str]]:
        """
        Return a dictionary with topic index and top N keywords per topic.

        Returns:
            dict: {topic_index: [word1, word2, ...]}
        """
        if not self.lda_model or not self.vectorizer:
            logger.warning("LDA model or vectorizer not available.")
            return {}

        words = self.vectorizer.get_feature_names_out()
        topic_keywords = {}

        for idx, topic in enumerate(self.lda_model.components_):
            top_words = [words[i] for i in topic.argsort()[:-top_n - 1:-1]]
            topic_keywords[idx] = top_words

        return topic_keywords

