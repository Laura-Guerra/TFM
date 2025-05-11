import pandas as pd
from datetime import date
from stable_baselines3.common.monitor import Monitor

from tfm.src.rl.stock_env import StockEnvironment
from tfm.src.rl.agents.dqn_agent import DQNAgent
from tfm.src.config.settings import (
    PATH_DATA_PROCESSED,
    PATH_DATA_LOGS,
    PATH_DATA_MODELS,
)

# %% 1. Carrega i separa les dades
df = pd.read_csv(PATH_DATA_PROCESSED / "state_features.csv", parse_dates=["date"]).sort_values(["date", "ticker"])

cut_val  = pd.Timestamp(date(2021, 12, 31))
cut_test = pd.Timestamp(date(2023, 12, 31))

df_train = df[df["date"] <= cut_val].copy()
df_val   = df[(df["date"] > cut_val) & (df["date"] <= cut_test)].copy()
df_test  = df[df["date"] > cut_test].copy()

assert not df_val.empty and not df_test.empty, "⚠️ Val o test està buit!"

# %% 2. Entorn de test
test_env_raw = StockEnvironment(df_test, 50_000, False, do_save_history=True)
test_env = Monitor(test_env_raw)

# %% 3. Defineix carpeta del model entrenat el 05-05
run_name = "dqn_spy_2025-05-11"
model_dir = PATH_DATA_MODELS / run_name
log_dir = PATH_DATA_LOGS / run_name
log_dir.mkdir(parents=True, exist_ok=True)

# %% 4. Crea l’agent i carrega el model
agent = DQNAgent(env=None, eval_env=test_env, model_dir=model_dir, log_dir=log_dir)
agent.load(model_dir / "dqn_final.zip")

# %% 5. Avaluació
agent.evaluate(n_episodes=20)
