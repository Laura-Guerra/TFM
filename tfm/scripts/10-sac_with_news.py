import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from stable_baselines3.common.monitor import Monitor

from tfm.src.rl.stock_env import StockEnvironment
from tfm.src.rl.agents.sac_agent import SACAgent  # Assegura't que existeixi aquest fitxer
from tfm.src.config.settings import (
    PATH_DATA_PROCESSED,
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

# %% 2. Entorns — continuous_actions=True per a SAC
model_name = "sac"
train_env = Monitor(StockEnvironment(df_train, 50_000, continuous_actions=True, model_name=model_name))
val_env   = Monitor(StockEnvironment(df_val,   50_000, continuous_actions=True, model_name=model_name))

# %% 3. Carpetes
now = datetime.today().strftime("%Y-%m-%d_%H-%M")
run_path = PATH_DATA_MODELS / f"{model_name}_with_news" / now

# %% 4. Cerca d'hiperparàmetres
agent_tune = SACAgent(train_env, val_env)
best_params = agent_tune.optimize_hyperparameters(n_trials=20, n_eval_episodes=10)

with (run_path / "best_params.json").open("w") as f:
    json.dump(best_params, f, indent=2)

# %% 5. Entrenament final (train + val)
df_train_full = pd.concat([df_train, df_val]).sort_values("date")

full_train_env_raw = StockEnvironment(df_train_full, 50_000, continuous_actions=True, is_train=True, model_name=model_name)
test_env_raw       = StockEnvironment(df_test, 50_000, continuous_actions=True, do_save_history=True, model_name=model_name)
full_env = Monitor(full_train_env_raw)
test_env = Monitor(test_env_raw)

agent = SACAgent(full_env, test_env, params=best_params)
agent.train(total_timesteps=500_000)
agent.save("sac_final")

print(f"✅ Model final desat a {run_path}")

# %% 7. Avaluació
agent.evaluate(n_episodes=20)
