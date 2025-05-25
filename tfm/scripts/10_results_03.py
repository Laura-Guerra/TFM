import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from tfm.src.config.settings import PATH_DATA_RESULTS
from tfm.src.viz.plot_theme import set_custom_theme

# Aplica tema personalitzat
set_custom_theme()

USE_NEWS = False
AGENTS = ["dqn", "ppo", "sac"]
VARIANT = "with" if USE_NEWS else "without"

BASE_PATH = PATH_DATA_RESULTS / "evaluations"
OUTPUT_DIR = PATH_DATA_RESULTS / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig_net, axes_net = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
fig_shares, axes_shares = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

for idx, agent in enumerate(AGENTS):
    path = BASE_PATH / agent / VARIANT
    all_trajectories = []
    discarded_count = 0

    for file in sorted(path.glob("iteration_*/test_episode.csv")):
        try:
            df = pd.read_csv(file)
            df["shares_held_vector"] = df["shares_held_vector"].apply(eval)

            # Filtra: descarta si no s'ha comprat cap acció
            total_shares = pd.DataFrame(df["shares_held_vector"].tolist()).sum().sum()
            if total_shares == 0:
                discarded_count += 1
                continue

            df["iteration"] = file.parent.name
            all_trajectories.append(df)
        except Exception as e:
            print(f"⚠️ Error carregant {file}: {e}")

    print(f"🔍 {agent.upper()} ({VARIANT}): {discarded_count} episodis descartats per inactivitat")

    if not all_trajectories:
        continue

    df_all = pd.concat(all_trajectories, ignore_index=True)

    # Valor net
    net_worth_all = df_all.groupby(["iteration", "step"])["net_worth"].mean().reset_index()
    final_values = net_worth_all.groupby("iteration").last().reset_index()
    best_iter = final_values.loc[final_values["net_worth"].idxmax(), "iteration"]
    worst_iter = final_values.loc[final_values["net_worth"].idxmin(), "iteration"]

    mean_net_worth = df_all.groupby("step")["net_worth"].mean().reset_index()
    best_net_worth = df_all[df_all["iteration"] == best_iter][["step", "net_worth"]]
    worst_net_worth = df_all[df_all["iteration"] == worst_iter][["step", "net_worth"]]

    # Accions mitjanes
    shares_df = pd.DataFrame(df_all["shares_held_vector"].tolist(), columns=["GLD", "SPY", "XLE"])
    shares_df["step"] = df_all["step"]
    mean_shares = shares_df.groupby("step").mean().reset_index()

    # Gràfica de valor net
    ax_net = axes_net[idx]
    sns.lineplot(data=mean_net_worth, x="step", y="net_worth", label="Mitjana", ax=ax_net)
    sns.lineplot(data=best_net_worth, x="step", y="net_worth", label="Millor iteració", ax=ax_net)
    sns.lineplot(data=worst_net_worth, x="step", y="net_worth", label="Pitjor iteració", ax=ax_net)
    ax_net.set_title(f"{agent.upper()}")
    ax_net.set_xlabel("Pas temporal")
    ax_net.set_ylabel("Valor net ($)")
    if idx == len(AGENTS) - 1:
        ax_net.legend()
    else:
        ax_net.get_legend().remove()

    # Gràfica d'accions
    ax_shares = axes_shares[idx]
    for col in ["GLD", "SPY", "XLE"]:
        sns.lineplot(data=mean_shares, x="step", y=col, label=col, ax=ax_shares)
    ax_shares.set_title(f"{agent.upper()}")
    ax_shares.set_xlabel("Pas temporal")
    ax_shares.set_ylabel("Accions mitjanes")
    if idx == len(AGENTS) - 1:
        ax_shares.legend(title="Actiu")
    else:
        ax_shares.get_legend().remove()

# Desa figures
variant_label = "amb" if USE_NEWS else "sense"
fig_net.tight_layout()
fig_net.subplots_adjust(top=0.85)
fig_net.savefig(OUTPUT_DIR / f"net_worth_by_agent_{VARIANT}.png")
plt.close(fig_net)

fig_shares.tight_layout()
fig_shares.subplots_adjust(top=0.85)
fig_shares.savefig(OUTPUT_DIR / f"shares_by_asset_{VARIANT}.png")
plt.close(fig_shares)
