from tfm.src.config.settings import PATH_DATA_PROCESSED
from tfm.src.utils.market_data_extractor import MarketDataExtractor

tickers = ["SPY", "GLD"]
market_extractor = MarketDataExtractor(tickers)

df_market = market_extractor.build_dataset()
market_extractor.save_to_csv(df_market, PATH_DATA_PROCESSED / "market_features.csv")
