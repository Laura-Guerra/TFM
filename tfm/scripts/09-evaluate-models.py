import os
import json
import pandas as pd
from pathlib import Path
from stable_baselines3 import DQN, PPO, SAC
from tfm.src.rl.stock_env import StockEnvironment
from tfm.src.config.settings import PATH_DATA_PROCESSED, PATH_DATA_MODELS, PATH_DATA_RESULTS
from tfm.src.rl.agents.ppo_agent import PPOAgent
from tfm.src.rl.agents.sac_agent import SACAgent
from tfm.src.rl.agents.dqn_agent import DQNAgent

# === Paràmetres ===
INITIAL_BALANCE = 50_000
ITERATIONS = range(30, 41)
MODELS = [
    ("sac", SACAgent, SAC, True),
    ("ppo", PPOAgent, PPO, False),
    ("dqn", DQNAgent, DQN, False),
]
VARIANTS = ["with", "without"]

# === Càrrega de dades de test ===
df = pd.read_csv(PATH_DATA_PROCESSED / "state_features.csv", parse_dates=["date"])
df = df.sort_values(["date", "ticker"])
df_test = df[df["date"] > pd.Timestamp("2023-12-31")].copy()
assert not df_test.empty, "⚠️ Dataset de test buit!"

def clean_news_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[[col for col in df.columns if not any(k in col for k in ['sentiment', 'w2v_', 'lda_'])]]

# === Avaluació ===
for model_name, AgentClass, ModelLoader, is_continuous in MODELS:
    for with_news in VARIANTS:
        use_news = with_news == "with"
        for iteration in ITERATIONS:
            print(f"\n🚀 Avaluant {model_name.upper()} | {with_news} | iteració {iteration}")

            model_dir = PATH_DATA_MODELS / model_name / with_news / f"iteration_{iteration}"
            model_path = model_dir / "trained_modelfinal_model.zip"
            if not model_path.exists():
                print(f"⛔ No trobat: {model_path}")
                continue

            # Prepara dades d'entrada
            df_eval = df_test.copy()
            if not use_news:
                df_eval = clean_news_columns(df_eval)

            # Configura entorn de test
            env = StockEnvironment(
                df_eval,
                initial_balance=INITIAL_BALANCE,
                continuous_actions=is_continuous,
                model_name=model_name,
                do_save_history=True,
                is_eval=True
            )

            try:
                model = ModelLoader.load(model_path, env=env)
                obs, _ = env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, _, _ = env.step(action)

                # Desa resultats
                output_dir = PATH_DATA_RESULTS / "evaluations" / model_name / with_news / f"iteration_{iteration}"
                output_dir.mkdir(parents=True, exist_ok=True)
                env.save_history(output_dir / "test_episode.csv")

            except Exception as e:
                print(f"❌ Error amb {model_name} {with_news} iteració {iteration}: {e}")
