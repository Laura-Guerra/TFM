# train_dqn_with_optuna.py
# ══════════════════════════════════════════════════════════════
import json
from datetime import date
from typing import cast
from pathlib import Path

import pandas as pd
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from tfm.src.rl.stock_env import StockEnvironment
from tfm.src.rl.agents.dqn_agent import DQNAgent
from tfm.src.config.settings import (
    PATH_DATA_PROCESSED,
    PATH_DATA_LOGS,
    PATH_DATA_MODELS,
)

# %% 1. Carrega i separa les dades
df = pd.read_csv(PATH_DATA_PROCESSED / "state_features.csv", parse_dates=["date"])
df = df[df["ticker"] == "SPY"].reset_index(drop=True)

cut_val  = pd.Timestamp(date(2021, 12, 31))   # train ≤ 31‑12‑2021
cut_test = pd.Timestamp(date(2023, 12, 31))   # val ≤ 31‑12‑2023

df_train = df[df["date"] <= cut_val].copy()
df_val   = df[(df["date"] > cut_val) & (df["date"] <= cut_test)].copy()
df_test  = df[df["date"] > cut_test].copy()

assert not df_val.empty and not df_test.empty, "⚠️ Val o test està buit!"

# %% 2. Entorns
MAX_STEPS = 1_000
train_env = Monitor(TimeLimit(StockEnvironment(df_train, 50_000, False), MAX_STEPS))
val_env   = Monitor(TimeLimit(StockEnvironment(df_val,   50_000, False), MAX_STEPS))
test_env  = Monitor(TimeLimit(StockEnvironment(df_test,  50_000, False), MAX_STEPS))

# %% 3. Carpetes
run_name  = "dqn_spy_split2023"
log_dir   = PATH_DATA_LOGS   / run_name
model_dir = PATH_DATA_MODELS / run_name
log_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

# %% 4. Busca hiperparàmetres
agent_tune = DQNAgent(train_env, val_env, model_dir, log_dir)
best_params = agent_tune.optimize_hyperparameters(n_trials=30, n_eval_episodes=5)

with (model_dir / "best_params.json").open("w") as f:
    json.dump(best_params, f, indent=2)

# %% 5. Entrenament final (train + val)
df_train_full = pd.concat([df_train, df_val]).sort_values("date")
full_env = Monitor(StockEnvironment(df_train_full, 50_000, False))

# ► activa registre d’episodis
inner_env = cast(StockEnvironment, full_env.unwrapped)
inner_env.episode_id = 1
inner_env.do_save_history = True

agent = DQNAgent(full_env, test_env, model_dir, log_dir, params=best_params)
agent.train(total_timesteps=500_000)
agent.save("dqn_final")

# ─── 6. Desa historial complet  -------------------------------

print(f"✅ Model final desat a {model_dir/'dqn_final.zip'}")
print(f"📊 TensorBoard: tensorboard --logdir {log_dir}")
print(f"📝 Hiperparàmetres: {model_dir/'best_params.json'}")
