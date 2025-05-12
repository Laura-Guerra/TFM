import json
from datetime import date
from pathlib import Path
import gymnasium
import numpy as np
import pandas as pd
from gymnasium import spaces
from loguru import logger
from itertools import product
from tfm.src.config.settings import PATH_DATA_RESULTS


class StockEnvironment(gymnasium.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, initial_balance=50000.0, continuous_actions=False,
                 model_name="dqn", do_save_history=False, episode_length=200, is_train=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.assets = sorted(self.df["ticker"].unique())
        self.n_assets = len(self.assets)
        self.initial_balance = initial_balance
        self.continuous_actions = continuous_actions
        self.episode_length = episode_length
        self.model_name = model_name
        self.do_save_history = do_save_history
        self.is_train = is_train
        self.episode_id = 1
        self.today = date.today().strftime("%Y-%m-%d")
        self.min_reward = np.inf
        self.max_reward = -np.inf
        self.best_episode = None
        self.worse_episode = None

        if self.continuous_actions:
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)
        else:
            self.action_set = self._valid_action_set()
            self.action_space = spaces.Discrete(len(self.action_set))

        self.feature_columns = [col for col in self.df.columns if col not in ["date", "ticker", "Open"]]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_assets * (len(self.feature_columns) + 2),),
            dtype=np.float32
        )

        self.reset()

    def _valid_action_set(self):
        all_actions = list(product(range(7), repeat=self.n_assets))

        def compra_proportion(a):
            val = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 0.0, 5: 0.0, 6: 0.0}
            return val[a]

        valid_actions = [
            action for action in all_actions
            if sum(compra_proportion(a) for a in action) <= 1.0
        ]
        return valid_actions

    def reset(self, seed=None, options=None):
        self.max_steps = len(self.df) // self.n_assets - 1
        self.current_step = np.random.randint(1, self.max_steps - self.episode_length)
        self.start_step = self.current_step
        self.balance = self.initial_balance
        self.shares_held = [0] * self.n_assets
        self.previous_net_worth = self.initial_balance
        self.steps_without_positions = 0
        self.asset_usage_counter = [0] * self.n_assets

        self.history = {
            "step": [self.start_step],
            "date": [self.df.iloc[self.start_step * self.n_assets]["date"]],
            "balance": [self.initial_balance],
            "net_worth": [self.initial_balance],
            "reward": [0],
            "action_vector": [[]],
            "shares_held_vector": [self.shares_held.copy()]  # 🔁 afegim això
        }

        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for i in range(self.n_assets):
            row = self.df.iloc[self.current_step * self.n_assets + i]
            features = row[self.feature_columns].values
            obs.extend(np.concatenate([features, [self.balance], [self.previous_net_worth]]))
        return np.array(obs, dtype=np.float32)

    def decode_action(self, action_idx):
        return list(self.action_set[action_idx])

    def step(self, action):
        if self.continuous_actions:
            # --- ACCIONS CONTÍNUES ---
            actions = np.clip(action, -1, 1)  # assegura accions dins [-1, 1]

            # Calcular proporcions de compra
            buy_props = np.clip(actions, 0, 1)
            total_buy_prop = np.sum(buy_props)
            if total_buy_prop > 1.0:
                buy_props = buy_props / total_buy_prop  # normalitza per evitar sobreassignació

            # Compres
            for i in range(self.n_assets):
                row = self.df.iloc[self.current_step * self.n_assets + i]
                price = row["Open"]
                prop = buy_props[i]
                if prop > 0:
                    cash = self.balance * prop
                    self._buy_fixed_cash(i, price, cash)

            # Vendes
            for i in range(self.n_assets):
                row = self.df.iloc[self.current_step * self.n_assets + i]
                price = row["Open"]
                sell_prop = -min(actions[i], 0.0)  # només si < 0
                if sell_prop > 0:
                    self._sell(i, price, sell_prop)

            actions_record = actions.tolist()

        else:
            # --- ACCIONS DISCRETES ---
            actions = self.decode_action(int(action))

            def get_buy_proportion(act):
                return {1: 0.25, 2: 0.5, 3: 0.75}.get(act, 0.0)

            # Calcula el cash a utilitzar per cada acció de compra
            cash_allocations = [self.balance * get_buy_proportion(act) for act in actions]

            for i in range(self.n_assets):
                row = self.df.iloc[self.current_step * self.n_assets + i]
                price = row["Open"]
                act = actions[i]

                if act in [1, 2, 3]:  # buy
                    self._buy_fixed_cash(i, price, cash_allocations[i])

                elif act in [4, 5, 6]:  # sell
                    proportion = {4: 0.25, 5: 0.5, 6: 1.0}[act]
                    self._sell(i, price, proportion)

            actions_record = actions

        # ✅ Post-acció
        self.current_step += 1
        net_worth = self.balance
        for i in range(self.n_assets):
            close_price = self.df.iloc[self.current_step * self.n_assets + i]["Close"]
            net_worth += self.shares_held[i] * close_price

        if sum(self.shares_held) == 0:
            self.steps_without_positions += 1
        else:
            self.steps_without_positions = 0

        for i, s in enumerate(self.shares_held):
            if s > 0:
                self.asset_usage_counter[i] += 1

        reward = self._compute_reward(net_worth)
        self.previous_net_worth = net_worth

        self.history["step"].append(self.current_step)
        self.history["date"].append(self.df.iloc[self.current_step * self.n_assets]["date"])
        self.history["balance"].append(self.balance)
        self.history["net_worth"].append(net_worth)
        self.history["reward"].append(reward)
        self.history["shares_held_vector"].append(self.shares_held.copy())
        self.history["action_vector"].append(actions_record)

        terminated = self.current_step >= self.start_step + self.episode_length
        truncated = net_worth <= self.initial_balance * 0.15

        if terminated or truncated:
            if self.episode_id % 10 == 0 and self.is_train:
                logger.info(
                    f"Episode {self.episode_id} finished at {self.current_step} after {self.current_step - self.start_step} steps")
            if self.do_save_history:
                if net_worth < self.min_reward:
                    self.min_reward = net_worth
                    self.worse_episode = self.episode_id
                if net_worth > self.max_reward:
                    self.max_reward = net_worth
                    self.best_episode = self.episode_id
            self.episode_id += 1

        return self._get_obs(), reward, terminated, truncated, {}

    def _buy_fixed_cash(self, i, price, cash):
        if self.balance <= 0 or cash <= 0:
            return
        shares = int(cash // price)
        cost = shares * price
        if shares > 0 and cost <= self.balance:
            self.balance -= cost
            self.shares_held[i] += shares

    def _compute_reward(self, net_worth):
        # Premi base pel canvi absolut del patrimoni net
        reward = net_worth - self.previous_net_worth

        # Penalització per ocasions perdudes
        missed_penalty = 0.0
        penalty_scale = 100.0  # Penalització proporcional al canvi de preu si es perd una oportunitat clara
        threshold = 0.01  # Canvi del 1%

        if self.current_step > 1 and self.history["action_vector"]:
            try:
                last_actions = self.history["action_vector"][-1]

                for i in range(self.n_assets):
                    prev_price = self.df.iloc[(self.current_step - 1) * self.n_assets + i]["Close"]
                    curr_price = self.df.iloc[self.current_step * self.n_assets + i]["Close"]
                    price_change = (curr_price - prev_price) / (prev_price + 1e-8)
                    action = last_actions[i]

                    # Penalització per no comprar quan el preu ha caigut ≥ 1% i es pot fer
                    if price_change <= -threshold and self.balance > 0.25 * curr_price and action not in [1, 2, 3]:
                        missed_penalty += abs(price_change) * penalty_scale

                    # Penalització per no vendre quan el preu ha pujat ≥ 1% i es tenen accions
                    if price_change >= threshold and self.shares_held[i] > 0 and action not in [4, 5, 6]:
                        missed_penalty += abs(price_change) * penalty_scale
            except IndexError:
                pass  # Per seguretat, en cas que no hi hagi accions encara

        reward -= missed_penalty
        return reward

    def _sell(self, i, price, proportion):
        shares = int(self.shares_held[i] * proportion)
        if shares > 0:
            self.shares_held[i] -= shares
            self.balance += shares * price

    def render(self, mode="human"):
        logger.info(
            f"Step {self.current_step} | Balance: {self.balance:.2f} | Net Worth: {self.previous_net_worth:.2f} | Holdings: {self.shares_held}")

    def _compute_metrics(self, risk_free_rate: float = 0.02) -> dict:
        df = pd.DataFrame(self.history)
        net_worth = df["net_worth"].values
        returns = np.diff(net_worth) / net_worth[:-1]
        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
        sharpe = (returns.mean() - daily_rf) / (returns.std() + 1e-8) * np.sqrt(252)
        sortino = (returns.mean() - daily_rf) / (returns[returns < 0].std() + 1e-8) * np.sqrt(252)
        cum_max = np.maximum.accumulate(net_worth)
        drawdown = (net_worth - cum_max) / cum_max
        max_drawdown = drawdown.min()
        volatility = returns.std() * np.sqrt(252)
        cumulative_return = net_worth[-1] / net_worth[0] - 1
        return {
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown": max_drawdown,
            "Volatility": volatility,
            "Cumulative Return": cumulative_return,
        }

    def save_history(self, path: str):
        df_history = pd.DataFrame(self.history)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df_history.to_csv(path, index=False)
        logger.info(f"✅ History saved to {path}")
        metrics = self._compute_metrics()
        metrics_path = path.with_suffix(".json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"📊 Metrics saved to {metrics_path}")