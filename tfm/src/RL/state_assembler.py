import pandas as pd
from sklearn.preprocessing import StandardScaler
from loguru import logger

class StateAssembler:
    def __init__(self, market_cols_to_normalize=None):
        """
        Initializes the state assembler.

        Args:
            market_cols_to_normalize: List of market columns to normalize (default = standard set)
        """
        self.scaler = StandardScaler()
        self.market_cols = market_cols_to_normalize or [
            "sma_20", "rsi_14", "macd", "Volumne"
        ]
        self.fitted = False

    def fit_scaler(self, market_df: pd.DataFrame):
        """
        Fit scaler only to technical indicators.
        """
        valid_cols = [col for col in self.market_cols if col in market_df.columns]
        self.scaler.fit(market_df[valid_cols])
        self.fitted = True
        logger.info(f"Scaler fitted on technical indicators: {valid_cols}")

    def transform_market(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize only the technical indicators in the market data.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit_scaler() first.")

        df = market_df.copy()
        valid_cols = [col for col in self.market_cols if col in df.columns]
        df[valid_cols] = self.scaler.transform(df[valid_cols])
        logger.info(f"Technical indicators normalized: {valid_cols}")
        return df

    def assemble_state(self, market_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize technical indicators and merge with NLP features on date.
        """
        logger.info("Merging normalized technical indicators with news features...")

        self.fit_scaler(market_df)
        market_norm = self.transform_market(market_df)

        combined_df = pd.merge(market_norm, news_df, on="date", how="left")
        combined_df = combined_df.sort_values(by="date").reset_index(drop=True)

        logger.success(f"State assembled: {combined_df.shape[0]} rows, {combined_df.shape[1]} features.")
        return combined_df
