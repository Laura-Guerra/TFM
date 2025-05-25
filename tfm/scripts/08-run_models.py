import csv
import json
import os

import pandas as pd
from datetime import date, datetime

from codecarbon import EmissionsTracker
from stable_baselines3.common.monitor import Monitor
from tfm.src.rl.stock_env import StockEnvironment
from tfm.src.rl.agents.ppo_agent import PPOAgent
from tfm.src.rl.agents.sac_agent import SACAgent
from tfm.src.rl.agents.dqn_agent import DQNAgent
from tfm.src.config.settings import PATH_DATA_PROCESSED, PATH_DATA_MODELS, PATH_DATA_MODELS_2

# --- Paràmetres generals ---
INITIAL_BALANCE = 50_000
TOTAL_TIMESTEPS = 250_000
N_TRIALS = 20
N_EVAL_EPISODES = 15
EVAL_EPISODES_FINAL = 50
MODELS = [
    ("sac", SACAgent, True,  True),
    ("ppo", PPOAgent, True, False),# PPO amb notícies, accions discretes
    ("dqn", DQNAgent, True,  False),
    ("sac", SACAgent, False, True),
    ("ppo", PPOAgent, False, False), # PPO sense notícies, accions discretes
    ("dqn", DQNAgent, False, False),
]

date_str=datetime.today().strftime("%Y-%m-%d_%H-%M")

def clean_news_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[[col for col in df.columns if not any(key in col for key in ['sentiment', 'w2v_', 'lda_'])]]

def log_result(now, model_name, with_news, is_continuous, mean_reward, emissions):
    row = {
        "timestamp": now,
        "model": model_name,
        "with_news": with_news,
        "continuous": is_continuous,
        "mean_reward": mean_reward,
        "emissions": emissions,
    }
    log_path = "model_results_log.csv"
    file_exists = os.path.isfile(log_path)

    with open(log_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()  # Escriu capçalera només si el fitxer és nou
        writer.writerow(row)

def run_model(model_name, AgentClass, with_news, is_continuous):
    try:
        print(f"\n🚀 Entrenant {model_name.upper()} {'amb' if with_news else 'sense'} notícies...")

        # --- Carrega i prepara dades ---
        df = pd.read_csv(PATH_DATA_PROCESSED / "state_features.csv", parse_dates=["date"])
        df = df.sort_values(["date", "ticker"]).copy()
        if not with_news:
            df = clean_news_columns(df)

        cut_val = pd.Timestamp(date(2021, 12, 31))
        cut_test = pd.Timestamp(date(2023, 12, 31))
        df_train = df[df["date"] <= cut_val].copy()
        df_val   = df[(df["date"] > cut_val) & (df["date"] <= cut_test)].copy()
        df_test  = df[df["date"] > cut_test].copy()

        assert not df_val.empty and not df_test.empty, "⚠️ Val o test està buit!"

        # --- Entrenament ---
        run_name = f"{model_name}/{'with' if with_news else 'without'}/iteration_{iteration}"
        run_path = PATH_DATA_MODELS / run_name
        run_path.mkdir(parents=True, exist_ok=True)
        tracker = EmissionsTracker(
            project_name=f"{model_name}_{'with' if with_news else 'without'}_news",
            output_dir=PATH_DATA_MODELS / run_name,
            output_file="emissions.csv",
            log_level="error"
        )
        tracker.start()

        train_env = Monitor(StockEnvironment(df_train, INITIAL_BALANCE, continuous_actions=is_continuous, model_name=model_name))
        val_env   = Monitor(StockEnvironment(df_val,   INITIAL_BALANCE, continuous_actions=is_continuous, model_name=model_name))

        # Ruta per carregar/guardar best_params.json segons model i configuració
        best_params_path = PATH_DATA_MODELS / model_name / ('with' if with_news else 'without') / "best_params.json"

        # Si ja existeixen paràmetres òptims, els carrega
        if best_params_path.exists():
            print(f"🔁 Carregant hiperparàmetres des de {best_params_path}")
            with open(best_params_path, "r") as f:
                params = json.load(f)
        else:
            print(f"🔍 Cercant hiperparàmetres per {model_name.upper()} {'amb' if with_news else 'sense'} notícies")
            agent_tune = AgentClass(train_env, val_env, path=best_params_path.parent)
            params = agent_tune.optimize_hyperparameters(n_trials=N_TRIALS, n_eval_episodes=N_EVAL_EPISODES)

            with open(best_params_path, "w") as f:
                json.dump(params, f, indent=2)
            print(f"✅ Paràmetres guardats a {best_params_path}")

        # Entrenament final
        df_train_full = pd.concat([df_train, df_val]).sort_values("date")
        train_raw = StockEnvironment(df_train_full, INITIAL_BALANCE, continuous_actions=is_continuous, is_train=True, model_name=model_name)
        test_raw  = StockEnvironment(df_test,       INITIAL_BALANCE, continuous_actions=is_continuous, do_save_history=True, model_name=model_name)
        agent = AgentClass(Monitor(train_raw), Monitor(test_raw), params=params, path=run_path)
        agent.train(total_timesteps=TOTAL_TIMESTEPS)
        agent.save("final_model")

        # Avaluació
        print(f"✅ {model_name.upper()} {'amb' if with_news else 'sense'} notícies completat.")
        emissions = tracker.stop()
        print(f"🌱 Emissions estimades: {emissions:.4f} kg de CO₂")
        mean_reward = agent.evaluate(n_episodes=EVAL_EPISODES_FINAL)
        log_result(iteration, model_name, with_news, is_continuous, mean_reward, emissions)

    except Exception as e:
        print(f"❌ ERROR a {model_name.upper()} {'amb' if with_news else 'sense'} notícies → {e}")

# --- Executa tots els models ---
for i in range(10):
    iteration = i+1
    for index, (model_name, AgentClass, with_news, is_continuous) in enumerate(MODELS):
        run_model(model_name, AgentClass, with_news, is_continuous)
