import gymnasium
import numpy as np
import pandas as pd
from gymnasium import spaces
from loguru import logger


class StockEnvironment(gymnasium.Env):
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

        self.max_steps = len(self.df) - 2

        # Define action space para un solo activo
        if self.continuous_actions:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(7)  # 0: hold, 1-3: buy, 4-6: sell

        # Observation = features of d-1 + balance + net worth
        self.feature_columns = [col for col in self.df.columns if col not in ["date", "ticker", "Open"]]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_columns) + 2,),  # features + balance + net worth
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

    def reset(self, seed=None, options=None):
        self.current_step = 1
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance

        return self._get_obs(), {}

    def _get_obs(self):
        row_d_minus_1 = self.df.iloc[self.current_step - 1]
        features = row_d_minus_1[self.feature_columns].values
        obs = np.concatenate([
            features,
            [self.balance],
            [self.net_worth]
        ])
        return obs.astype(np.float32)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()  # serveix tant per shape () com (1,)
        elif isinstance(action, (list, tuple)):
            action = action[0]

        if self.continuous_actions:
            action = np.clip(action, -1, 1)[0]
        else:
            action = int(action)

        if self.current_step >= len(self.df):
            done = True
            return self._get_obs(), 0.0, done, {}

        row_d = self.df.iloc[self.current_step]
        execution_price = row_d["Open"]

        if self.continuous_actions:
            self._continuous_action_logic(execution_price, action)
        else:
            self._discrete_action_logic(execution_price, action)

        self.current_step += 1

        # Net worth calculation
        next_close_price = self.df.iloc[self.current_step]["Close"]
        self.net_worth = self.balance + self.shares_held * next_close_price

        reward = self.net_worth - self.previous_net_worth
        self.previous_net_worth = self.net_worth

        # Log the current state
        self.history["step"].append(self.current_step)
        self.history["date"].append(self.df["date"].iloc[self.current_step])
        self.history["balance"].append(self.balance)
        self.history["net_worth"].append(self.net_worth)
        self.history["shares_held"].append(self.shares_held)

        terminated = self.current_step >= self.max_steps
        truncated  = bool(self.net_worth <= self.initial_balance * 0.10)

        return self._get_obs(), reward, terminated, truncated, {}


    def _continuous_action_logic(self, price: float, action_value: float):
            if action_value > 0:  # Buy
                amount_to_invest = self.balance * action_value
                shares = int(amount_to_invest // price)
                if shares > 0:
                    self.balance -= shares * price
                    self.shares_held += shares
            elif action_value < 0:  # Sell
                shares_to_sell = int(self.shares_held * (-action_value))
                if shares_to_sell > 0:
                    self.shares_held -= shares_to_sell
                    self.balance += shares_to_sell * price

    def _discrete_action_logic(self, price: float, action: int):
        if action == 1:  # Buy 25%
            self._buy_asset(price, 0.25)
        elif action == 2:  # Buy 50%
            self._buy_asset(price, 0.5)
        elif action == 3:  # Buy 100%
            self._buy_asset(price, 1.0)
        elif action == 4:  # Sell 25%
            self._sell_asset(price, 0.25)
        elif action == 5:  # Sell 50%
            self._sell_asset(price, 0.5)
        elif action == 6:  # Sell 100%
            self._sell_asset(price, 1.0)
        # 0: Hold (no acción)

    def _buy_asset(self, price: float, proportion: float):
        available_cash = self.balance * proportion
        shares = int(available_cash // price)
        if shares > 0:
            self.shares_held += shares
            self.balance -= shares * price

    def _sell_asset(self, price: float, proportion: float):
        shares_to_sell = int(self.shares_held * proportion)
        if shares_to_sell > 0:
            self.shares_held -= shares_to_sell
            self.balance += shares_to_sell * price

    def render(self, mode="human"):
        logger.info(f"Step {self.current_step} | Balance: {self.balance:.2f} | Net Worth: {self.net_worth:.2f} | Holdings: {self.shares_held:.2f}")

    def compute_metrics(self, risk_free_rate: float = 0.0) -> dict:
        """
        Compute financial metrics from the logged history.
        """
        df = pd.DataFrame(self.history)
        net_worth = df["net_worth"].values
        returns = np.diff(net_worth) / net_worth[:-1]

        sharpe = (returns.mean() - risk_free_rate) / (returns.std() + 1e-8) * np.sqrt(252)
        sortino = (returns.mean() - risk_free_rate) / (returns[returns < 0].std() + 1e-8) * np.sqrt(252)
        cum_max = np.maximum.accumulate(net_worth)
        drawdown = (net_worth - cum_max) / cum_max
        max_drawdown = drawdown.min()
        volatility = returns.std() * np.sqrt(252)

        return {
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown": max_drawdown,
            "Volatility": volatility
        }

    def save_history(self, path: str):
        """
        Save the history of the simulation and computed metrics to a CSV and JSON file.
        """
        df_history = pd.DataFrame(self.history)
        df_history.to_csv(path, index=False)
        logger.info(f"✅ History saved to {path}")

        # Save metrics to separate file
        metrics = self.compute_metrics()
        metrics_path = path.replace(".csv", "_metrics.json")
        pd.Series(metrics).to_json(metrics_path, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")
