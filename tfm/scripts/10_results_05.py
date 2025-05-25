import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from tfm.src.config.settings import PATH_DATA_RESULTS

# Substitueix per la teva ruta real
AGENTS = ["dqn", "ppo", "sac"]
VARIANTS = ["with", "without"]
BASE_PATH = PATH_DATA_RESULTS / "evaluations"

# Aplica tema si el tens definit
from tfm.src.viz.plot_theme import set_custom_theme
set_custom_theme()

for agent in AGENTS:
    for variant in VARIANTS:
        path = BASE_PATH / agent / variant
        all_trajectories = []

        for file in sorted(path.glob("iteration_*/test_episode.csv")):
            try:
                df = pd.read_csv(file)
                df["shares_held_vector"] = df["shares_held_vector"].apply(eval)

                # Filtra: exclou si no s'ha comprat cap acció en tot l'episodi
                total_shares = pd.DataFrame(df["shares_held_vector"].tolist()).sum().sum()
                if total_shares == 0:
                    continue  # salta aquest fitxer

                df["iteration"] = file.parent.name
                all_trajectories.append(df)
            except Exception as e:
                print(f"⚠️ Error carregant {file}: {e}")

        if not all_trajectories:
            continue

        df_all = pd.concat(all_trajectories, ignore_index=True)

        # === Gràfica 1: evolució del net worth ===
        net_worth_all = df_all.groupby(["iteration", "step"])["net_worth"].mean().reset_index()
        final_values = net_worth_all.groupby("iteration").last().reset_index()
        best_iter = final_values.loc[final_values["net_worth"].idxmax(), "iteration"]
        worst_iter = final_values.loc[final_values["net_worth"].idxmin(), "iteration"]

        mean_net_worth = df_all.groupby("step")["net_worth"].mean().reset_index()
        best_net_worth = df_all[df_all["iteration"] == best_iter][["step", "net_worth"]]
        worst_net_worth = df_all[df_all["iteration"] == worst_iter][["step", "net_worth"]]

        plt.figure(figsize=(10, 5))
        sns.lineplot(data=mean_net_worth, x="step", y="net_worth", label="Mitjana")
        sns.lineplot(data=best_net_worth, x="step", y="net_worth", label=f"Millor iteració")
        sns.lineplot(data=worst_net_worth, x="step", y="net_worth", label=f"Pitjor iteració")
        plt.xlabel("Pas temporal")
        plt.ylabel("Valor net ($)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{agent}_{variant}_networth_evolution.png")
        plt.show()

        # === Gràfica 2: accions mitjanes ===
        shares_df = pd.DataFrame(df_all["shares_held_vector"].tolist(), columns=["GLD", "SPY", "XLE"])
        shares_df["step"] = df_all["step"]
        mean_shares = shares_df.groupby("step").mean().reset_index()

        plt.figure(figsize=(10, 5))
        for col in ["GLD", "SPY", "XLE"]:
            sns.lineplot(data=mean_shares, x="step", y=col, label=col)
        plt.xlabel("Pas temporal")
        plt.ylabel("Mitjana d'accions")
        plt.tight_layout()
        plt.savefig(f"{agent}_{variant}_mean_shares.png")
        plt.show()
