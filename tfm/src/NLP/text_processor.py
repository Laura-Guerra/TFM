"""Class to clean and tokenize news text."""

import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextProcessor:
    def __init__(self, lang: str = "english", use_lemmatizer: bool = True):
        """
        Initialize text processor.
        """
        self.stop_words = set(stopwords.words(lang))
        self.use_lemmatizer = use_lemmatizer
        self.lemmatizer = WordNetLemmatizer() if use_lemmatizer else None

    def _normalize_text(self, text: str) -> str:
        """Normalize text."""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    def _tokenize(self, text: str) -> list[str]:
        """Split into tokens."""
        return text.split()

    def _remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove stopwords."""
        return [t for t in tokens if t not in self.stop_words]

    def _lemmatize(self, tokens: list[str]) -> list[str]:
        """Lemmatize tokens."""
        if not self.use_lemmatizer:
            return tokens
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def process_text(self, text: str) -> str:
        """Full text processing pipeline."""
        if pd.isna(text):
            return ""

        text = self._normalize_text(text)
        tokens = self._tokenize(text)
        tokens = self._remove_stopwords(tokens)
        tokens = self._lemmatize(tokens)

        return " ".join(tokens)

    def apply(self, df: pd.DataFrame, text_column: str = "full_text") -> pd.DataFrame:
        """
        Apply processing to a column in a DataFrame.
        """
        df = df.copy()
        df["clean_text"] = df[text_column].apply(self.process_text)
        return df
