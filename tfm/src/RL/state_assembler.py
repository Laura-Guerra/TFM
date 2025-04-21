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
            "Open", "High", "Low", "Close", "Volume", "sma_20", "rsi_14", "macd"
        ]
        self.fitted = False

    def fit_scaler(self, market_df: pd.DataFrame):
        """
        Fits the scaler on the specified columns from the market DataFrame.
        """
        valid_cols = [col for col in self.market_cols if col in market_df.columns]
        self.scaler.fit(market_df[valid_cols])
        self.fitted = True
        logger.info(f"Scaler fitted on columns: {valid_cols}")

    def transform_market(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies normalization to the specified columns of the market DataFrame.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit_scaler() first.")

        df = market_df.copy()
        valid_cols = [col for col in self.market_cols if col in df.columns]

        df[valid_cols] = self.scaler.transform(df[valid_cols])
        logger.info(f"Market data normalized for columns: {valid_cols}")
        return df

    def assemble_state(self, news_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes market data and merges it with NLP data by date.

        Args:
            news_df: DataFrame with NLP features (must include 'date')
            market_df: DataFrame with market features (must include 'date')

        Returns:
            Combined DataFrame with state vectors
        """
        logger.info("Preparing state features by merging NLP and market data...")

        self.fit_scaler(market_df)
        market_norm = self.transform_market(market_df)

        # Merge
        combined_df = pd.merge( market_norm, news_df, on="date", how="left")
        logger.success(f"Final state DataFrame with {len(combined_df)} rows and {combined_df.shape[1]} features.")
        return combined_df
