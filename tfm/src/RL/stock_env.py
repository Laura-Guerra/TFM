import gym
import numpy as np
import pandas as pd
from gym import spaces
from loguru import logger


class MarketEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0

        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.previous_net_worth = initial_balance

        # 7 discrete actions: hold, buy/sell (25%, 50%, 100%)
        self.action_space = spaces.Discrete(7)

        # Observation = features from df + balance + shares + current price
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.df.columns) - 1 + 3,),  # exclude date + 3 (balance, shares, price)
            dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance
        return self._get_obs()

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        obs = row.drop(["date"]).values.astype(np.float32)
        current_price = row["Close"]
        return np.concatenate([
            obs,
            [self.balance],
            [self.shares_held],
            [current_price]
        ])

    def step(self, action: int):
        row = self.df.iloc[self.current_step]
        current_price = row["Close"]

        if action == 1:  # Buy 25%
            amount = 0.25
            shares = int((self.balance * amount) // current_price)
            self.balance -= shares * current_price
            self.shares_held += shares

        elif action == 2:  # Buy 50%
            amount = 0.50
            shares = int((self.balance * amount) // current_price)
            self.balance -= shares * current_price
            self.shares_held += shares

        elif action == 3:  # Buy 100%
            shares = int(self.balance // current_price)
            self.balance -= shares * current_price
            self.shares_held += shares

        elif action == 4:  # Sell 25%
            shares = int(self.shares_held * 0.25)
            self.balance += shares * current_price
            self.shares_held -= shares

        elif action == 5:  # Sell 50%
            shares = int(self.shares_held * 0.50)
            self.balance += shares * current_price
            self.shares_held -= shares

        elif action == 6:  # Sell 100%
            self.balance += self.shares_held * current_price
            self.shares_held = 0

        # Reward
        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth - self.previous_net_worth
        self.previous_net_worth = self.net_worth

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = self.net_worth <= self.initial_balance * 0.25

        return self._get_obs(), reward, done, truncated, {}

    def render(self, mode="human"):
        logger.info(f"Step {self.current_step} | Balance: {self.balance:.2f} | Shares: {self.shares_held} | Net Worth: {self.net_worth:.2f}")
