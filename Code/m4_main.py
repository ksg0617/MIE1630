import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import Env, spaces
from stable_baselines3 import PPO

# Load the M4 Monthly dataset
m4_train = pd.read_csv('D:/MIE1630/Data/Monthly-train.csv')

# Preprocess the data
time_series_data = []

for _, row in m4_train.iterrows():
    ts_values = row.dropna().values[1:]  # Exclude the ID
    time_series_data.append(ts_values.astype(float))

class HoltWintersEnvMulti(gym.Env):
    def __init__(self, time_series_data, window_size=24, forecast_horizon=1):
        super(HoltWintersEnvMulti, self).__init__()
        self.time_series_data = time_series_data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.num_series = len(time_series_data)
        self.current_series = 0
        self.current_step = 0
        self.max_steps = None  # Will be set per series

        # Action space: Continuous values for alpha, beta, gamma
        self.action_space = spaces.Box(low=0.01, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space: Will be determined dynamically
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Select a random time series
        self.current_series = np.random.randint(0, self.num_series)
        self.series_data = self.time_series_data[self.current_series]
        self.series_length = len(self.series_data)
        self.max_steps = self.series_length - self.window_size - self.forecast_horizon
        self.current_step = 0

        # Check if the series is long enough
        if self.max_steps <= 0:
            return self.reset(seed=seed, options=options)

        # Update observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size,), dtype=np.float32)

        state = self.series_data[self.current_step:self.current_step + self.window_size]
        return state.astype(np.float32), {}

    def step(self, action):
        alpha, beta, gamma = action

        start = self.current_step
        end = start + self.window_size
        train_data = self.series_data[start:end]

        # Fit the Holt-Winters model
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(
            train_data,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )
        try:
            model_fit = model.fit(
                smoothing_level=alpha,
                smoothing_slope=beta,
                smoothing_seasonal=gamma,
                optimized=False
                # Remove 'initialization_method' argument
            )
        except ValueError:
            # Penalize invalid parameter combinations
            reward = -100.0
            terminated = True
            truncated = False
            next_state = np.zeros(self.window_size)
            return next_state.astype(np.float32), reward, terminated, truncated, {}


        # Forecast
        forecast = model_fit.forecast(self.forecast_horizon)
        actual_end = end + self.forecast_horizon
        if actual_end > len(self.series_data):
            actual_end = len(self.series_data)
        actual = self.series_data[end:actual_end]

        # Handle cases where actual is shorter than forecast horizon
        if len(actual) < self.forecast_horizon:
            last_value = actual[-1] if len(actual) > 0 else train_data[-1]
            actual = np.pad(actual, (0, self.forecast_horizon - len(actual)), 'edge')

        # Calculate reward
        mse = np.mean((forecast - actual) ** 2)
        reward = -mse  # Negative MSE as reward

        # Prepare for next step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        if not terminated:
            next_state = self.series_data[self.current_step:self.current_step + self.window_size]
        else:
            next_state = np.zeros(self.window_size)

        return next_state.astype(np.float32), reward, terminated, truncated, {}

    def render(self):
        pass  # Rendering is not implemented

# Instantiate the environment
env = HoltWintersEnvMulti(time_series_data, window_size=24, forecast_horizon=1)

# Check the environment
from stable_baselines3.common.env_checker import check_env
check_env(env)

# Create the RL agent
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the model
model.save('ppo_holt_winters_m4')

# Evaluate the agent on a sample time series
sample_series = time_series_data[0]
env = HoltWintersEnvMulti([sample_series], window_size=24, forecast_horizon=1)

state, _ = env.reset()
rewards = []
alphas, betas, gammas = [], [], []
forecasts = []
actuals = []

while True:
    action, _ = model.predict(state)
    state, reward, terminated, truncated, _ = env.step(action)
    rewards.append(reward)
    alphas.append(action[0])
    betas.append(action[1])
    gammas.append(action[2])

    # Collect forecasts and actuals for plotting
    forecast_step = env.current_step + env.window_size - 1
    if forecast_step < len(env.series_data):
        forecasts.append(env.series_data[forecast_step])
    else:
        forecasts.append(np.nan)

    actual_step = env.current_step + env.window_size
    if actual_step < len(env.series_data):
        actuals.append(env.series_data[actual_step])
    else:
        actuals.append(np.nan)

    if terminated or truncated:
        break

print(f"Total Reward: {sum(rewards)}")

# Plot hyperparameters
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(alphas, label='Alpha')
plt.plot(betas, label='Beta')
plt.plot(gammas, label='Gamma')
plt.title('Hyperparameters Over Time')
plt.legend()
plt.show()

# Plot forecasts vs actuals
plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual')
plt.plot(forecasts, label='Forecast')
plt.title('Actual vs Forecasted Values')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()
