import gym
import numpy as np
import pandas as pd
from gym import spaces
from loguru import logger


class MarketEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        continuous_actions: bool = False,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.continuous_actions = continuous_actions

        # Asset detection
        self.asset_list = self.df["ticker"].unique().tolist()
        self.num_assets = len(self.asset_list)

        # Define action space
        if self.continuous_actions:
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_assets,), dtype=np.float32)
        else:
            self.action_space = spaces.MultiDiscrete([7] * self.num_assets)

        # Observation = features of d-1 + balance + net worth
        self.feature_columns = [col for col in self.df.columns if col not in ["date", "ticker", "Open"]]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_columns) * self.num_assets + 2,),
            dtype=np.float32
        )

        # Initialize history logging
        self.history = {
            "step": [],
            "date": [],
            "balance": [],
            "net_worth": [],
            "shares_held": []
        }

        self.reset()

    def reset(self):
        self.current_step = 1  # Start at 1 to have a valid d-1
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.num_assets, dtype=np.float32)
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance
        return self._get_obs()

    def _get_obs(self):
        rows = []
        for asset in self.asset_list:
            asset_df = self.df[self.df["ticker"] == asset].reset_index(drop=True)
            row_d_minus_1 = asset_df.iloc[self.current_step - 1]
            features = row_d_minus_1[self.feature_columns].values
            rows.append(features)

        rows = np.concatenate(rows)
        obs = np.concatenate([
            rows,
            [self.balance],
            [self.net_worth]
        ])
        return obs.astype(np.float32)

    def step(self, actions):

        if self.continuous_actions:
            actions = np.clip(actions, -1, 1)
        else:
            actions = np.array(actions)

        for i, action in enumerate(actions):
            asset = self.asset_list[i]
            asset_df = self.df[self.df["ticker"] == asset].reset_index(drop=True)

            if self.current_step >= len(asset_df):
                continue

            row_d = asset_df.iloc[self.current_step]
            execution_price = row_d["Open"]

            if self.continuous_actions:
                self._continuous_action_logic(i, execution_price, action)
            else:
                self._discrete_action_logic(i, execution_price, action)

        self.current_step += 1

        # Update net worth
        self.net_worth = self.balance
        for i, asset in enumerate(self.asset_list):
            asset_df = self.df[self.df["ticker"] == asset].reset_index(drop=True)
            if self.current_step < len(asset_df):
                price = asset_df.iloc[self.current_step]["Close"]
                self.net_worth += self.shares_held[i] * price

        reward = self.net_worth - self.previous_net_worth
        self.previous_net_worth = self.net_worth

        # Log the current state
        self.history["step"].append(self.current_step)
        self.history["date"].append(self.df["date"].iloc[self.current_step])
        self.history["balance"].append(self.balance)
        self.history["net_worth"].append(self.net_worth)
        self.history["shares_held"].append(self.shares_held.copy())

        done = self.current_step >= min(len(self.df[self.df["ticker"] == asset]) for asset in self.asset_list) - 2
        truncated = self.net_worth <= self.initial_balance * 0.10

        return self._get_obs(), reward, done, truncated, {}

    def _continuous_action_logic(self, asset_idx: int, price: float, action_value: float):
        if action_value > 0:  # Buy
            amount_to_invest = self.balance * action_value
            shares = int(amount_to_invest // price)
            if shares > 0:
                self.balance -= shares * price
                self.shares_held[asset_idx] += shares
        elif action_value < 0:  # Sell
            shares_to_sell = int(self.shares_held[asset_idx] * (-action_value))
            if shares_to_sell > 0:
                self.shares_held[asset_idx] -= shares_to_sell
                self.balance += shares_to_sell * price

    def _discrete_action_logic(self, asset_idx: int, price: float, action: int):
        if action == 1:  # Buy 25%
            self._buy_asset(asset_idx, price, 0.25)
        elif action == 2:  # Buy 50%
            self._buy_asset(asset_idx, price, 0.5)
        elif action == 3:  # Buy 100%
            self._buy_asset(asset_idx, price, 1.0)
        elif action == 4:  # Sell 25%
            self._sell_asset(asset_idx, price, 0.25)
        elif action == 5:  # Sell 50%
            self._sell_asset(asset_idx, price, 0.5)
        elif action == 6:  # Sell 100%
            self._sell_asset(asset_idx, price, 1.0)

    def _buy_asset(self, asset_idx: int, price: float, proportion: float):
        available_cash = self.balance * proportion
        shares = int(available_cash // price)
        if shares > 0:
            self.shares_held[asset_idx] += shares
            self.balance -= shares * price

    def _sell_asset(self, asset_idx: int, price: float, proportion: float):
        shares_to_sell = int(self.shares_held[asset_idx] * proportion)
        if shares_to_sell > 0:
            self.shares_held[asset_idx] -= shares_to_sell
            self.balance += shares_to_sell * price

    def render(self, mode="human"):
        logger.info(f"Step {self.current_step} | Balance: {self.balance:.2f} | Net Worth: {self.net_worth:.2f} | Holdings: {self.shares_held}")

    def save_history(self, path: str):
        """
        Save the history of the simulation to a CSV file for further analysis.
        """
        df_history = pd.DataFrame(self.history)
        df_history.to_csv(path, index=False)
        logger.info(f"✅ History saved to {path}")
