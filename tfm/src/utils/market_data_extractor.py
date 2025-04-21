import yfinance as yf
import pandas as pd
import ta
from pathlib import Path
from loguru import logger


class MarketDataExtractor:
    def __init__(self, tickers: list[str], start: str = "2010-01-01", end: str = "2025-01-01"):
        """
        Initializes the engineer with asset tickers and date range.

        Args:
            tickers: List of tickers to fetch data for.
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
        """
        self.tickers = tickers
        self.start = start
        self.end = end

    def fetch_data(self, ticker: str) -> pd.DataFrame:
        """
        Downloads historical data for a given ticker using yfinance.

        Args:
            ticker: Stock ticker.

        Returns:
            DataFrame with historical price data.
        """
        logger.debug(f"Downloading data for {ticker}")
        df = yf.download(ticker, start=self.start, end=self.end)
        if df.empty:
            logger.warning(f"No data found for ticker: {ticker}")
            return pd.DataFrame()

        df = df.reset_index()
        df["date"] = pd.to_datetime(df["Date"]).dt.tz_localize("UTC")
        df.drop(columns=["Date"], inplace=True)
        df["ticker"] = ticker
        return df

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes SMA, RSI, MACD indicators on a price DataFrame.

        Args:
            df: Price DataFrame with at least 'Close'.

        Returns:
            DataFrame with indicators added.
        """
        df["sma_20"] = ta.trend.sma_indicator(df["Close"].squeeze(), window=20)
        df["rsi_14"] = ta.momentum.rsi(df["Close"].squeeze(), window=14)
        df["macd"] = ta.trend.macd(df["Close"].squeeze())
        return df

    def process_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Downloads and processes all data for one ticker.

        Args:
            ticker: Stock ticker.

        Returns:
            Final cleaned DataFrame with indicators.
        """
        df = self.fetch_data(ticker)
        if df.empty:
            return pd.DataFrame()
        df = self.compute_indicators(df)
        df = df[["date", "ticker", "Open", "High", "Low", "Close", "Volume", "sma_20", "rsi_14", "macd"]]
        df.dropna(inplace=True)
        return df

    def build_dataset(self) -> pd.DataFrame:
        """
        Processes all tickers and combines their data.

        Returns:
            Combined DataFrame with all tickers and indicators.
        """
        results = []
        for ticker in self.tickers:
            logger.info(f"Processing ticker: {ticker}")
            processed = self.process_ticker(ticker)
            if not processed.empty:
                results.append(processed)

        if not results:
            raise ValueError("No market data retrieved for any ticker.")

        df_final = pd.concat(results).sort_values(by=["date", "ticker"]).reset_index(drop=True)
        logger.success(f"Built dataset for {len(results)} tickers.")
        return df_final

    def save_to_csv(self, df: pd.DataFrame, output_path: Path):
        """
        Saves a DataFrame to CSV.

        Args:
            df: DataFrame to save.
            output_path: Path to CSV file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved market features to {output_path}")
