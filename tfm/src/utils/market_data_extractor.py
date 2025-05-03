import yfinance as yf
import pandas as pd
import ta
from pathlib import Path
from loguru import logger

class MarketDataExtractor:
    def __init__(self, tickers: list[str], start: str = "2014-01-01", end: str = "2025-04-30"):
        """
        Extracts market data and computes indicators.
        """
        self.tickers = tickers
        self.start = start
        self.end = end

    def fetch_data(self, ticker: str) -> pd.DataFrame:
        """
        Downloads historical market data for a single ticker.
        """
        logger.debug(f"Downloading data for {ticker}")

        start_download = (pd.to_datetime(self.start) - pd.DateOffset(days=60)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start_download, end=self.end)

        if df.empty:
            logger.warning(f"No data found for ticker: {ticker}")
            return pd.DataFrame()

        df = df.reset_index()
        df.columns = df.columns.get_level_values(0)
        df.rename(columns={"Date": "date"}, inplace=True)
        df["ticker"] = ticker
        return df

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes technical indicators on a market DataFrame.
        """
        df["sma_20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["rsi_14"] = ta.momentum.rsi(df["Close"], window=14)
        df["macd"] = ta.trend.macd(df["Close"])
        return df

    def process_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Processes and filters the market data for a single ticker.
        """
        df = self.fetch_data(ticker)
        if df.empty:
            return pd.DataFrame()

        df = self.compute_indicators(df)
        df = df[(df["date"] >= pd.to_datetime(self.start)) & (df["date"] <= pd.to_datetime(self.end))]
        df = df[["date", "ticker", "Open", "High", "Low", "Close", "Volume", "sma_20", "rsi_14", "macd"]]
        df.dropna(inplace=True)
        return df

    def build_dataset(self) -> pd.DataFrame:
        """
        Builds the complete dataset for all tickers.
        """
        results = []
        for ticker in self.tickers:
            logger.info(f"Processing ticker: {ticker}")
            df = self.process_ticker(ticker)
            if not df.empty:
                results.append(df)

        if not results:
            raise ValueError("No market data retrieved.")

        df_final = pd.concat(results).sort_values(by=["date", "ticker"]).reset_index(drop=True)
        logger.success(f"Built dataset with {len(df_final)} rows and {len(self.tickers)} tickers.")
        return df_final

    def save_to_csv(self, df: pd.DataFrame, output_path: Path):
        """
        Saves the final DataFrame to CSV.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved market data to {output_path}")
