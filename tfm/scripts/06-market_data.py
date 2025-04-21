from tfm.src.config.settings import PATH_DATA_PROCESSED
from tfm.src.utils.market_data_extractor import MarketDataExtractor

tickers = ["SPY"]
engineer = MarketDataExtractor(tickers)

df_market = engineer.build_dataset()
engineer.save_to_csv(df_market, PATH_DATA_PROCESSED / "market_features.csv")
