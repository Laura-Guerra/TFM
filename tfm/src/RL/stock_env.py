import gym
import numpy as np
import pandas as pd
from gym import spaces
from loguru import logger


class MarketEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance

        # Detectar actius
        self.asset_list = self.df["ticker"].unique().tolist()
        self.num_assets = len(self.asset_list)

        # Accions possibles per actiu: 0=Hold, 1=Buy25%, 2=Buy50%, 3=Buy100%, 4=Sell25%, 5=Sell50%, 6=Sell100%
        self.action_space = spaces.MultiDiscrete([7] * self.num_assets)

        # Observació: tot l'estat rellevant
        self.feature_columns = [col for col in self.df.columns if col not in ["date", "ticker", "Open"]]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_columns) * self.num_assets + 2,),  # features + balance + net worth
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.num_assets, dtype=np.float32)
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance
        return self._get_obs()

    def _get_obs(self):
        rows = []
        for asset in self.asset_list:
            asset_row = self.df[(self.df["ticker"] == asset)].iloc[self.current_step]
            features = asset_row[self.feature_columns].values
            rows.append(features)

        rows = np.concatenate(rows)

        obs = np.concatenate([
            rows,
            [self.balance],
            [self.net_worth]
        ])
        return obs.astype(np.float32)

    def step(self, actions: np.ndarray):
        """
        Args:
            actions: array d'accions per actiu (valor entre 0 i 6)
        """
        total_reward = 0.0

        for i, action in enumerate(actions):
            asset = self.asset_list[i]
            asset_df = self.df[self.df["ticker"] == asset].reset_index(drop=True)

            if self.current_step >= len(asset_df):
                continue

            row = asset_df.iloc[self.current_step]
            execution_price = row["Open"]

            # Buy actions
            if action == 1:
                self._buy_asset(i, execution_price, 0.25)
            elif action == 2:
                self._buy_asset(i, execution_price, 0.5)
            elif action == 3:
                self._buy_asset(i, execution_price, 1.0)

            # Sell actions
            elif action == 4:
                self._sell_asset(i, execution_price, 0.25)
            elif action == 5:
                self._sell_asset(i, execution_price, 0.5)
            elif action == 6:
                self._sell_asset(i, execution_price, 1.0)

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

        done = self.current_step >= min([len(self.df[self.df["ticker"] == asset]) for asset in self.asset_list]) - 2
        truncated = self.net_worth <= self.initial_balance * 0.25

        return self._get_obs(), reward, done, truncated, {}

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
