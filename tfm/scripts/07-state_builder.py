"""Script to assemble the state features for the rl agent."""
import pandas as pd

from tfm.src.rl.state_assembler import StateAssembler
from tfm.src.config.settings import PATH_DATA_PROCESSED

# %% Load data
df_nlp = pd.read_csv(PATH_DATA_PROCESSED / "articles_nlp_vectors_us.csv")
df_market = pd.read_csv(PATH_DATA_PROCESSED / "market_features.csv")

# %% Assemble
assembler = StateAssembler()
df_state = assembler.assemble_state(news_df=df_nlp, market_df=df_market)

# %% Save
df_state.to_csv(PATH_DATA_PROCESSED / "state_features.csv", index=False)
