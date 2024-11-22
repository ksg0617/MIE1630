from bayes_opt import BayesianOptimization
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import pandas as pd
from data_loader import load_passenger_data
import matplotlib.pyplot as plt

# Define the objective function
def holt_winters_mse(alpha, beta, gamma):
    """
    Compute Mean Squared Error (MSE) for Holt-Winters given hyperparameters.
    """
    ts_data, test_ts_data = load_passenger_data()
    window_size = 48
    forecast_horizon = 12

    train_data = ts_data[:-window_size]
    val_data =ts_data[-window_size:-window_size + forecast_horizon]
    try:
        # Fit the Holt-Winters model with given parameters
        model_hw = ExponentialSmoothing(
            train_data,
            trend='mul',
            seasonal='mul',
            seasonal_periods=12
        )
        model_fit = model_hw.fit(
            smoothing_level=alpha,
            smoothing_trend=beta,
            smoothing_seasonal=gamma,
            optimized=False
        )
        forecast = model_fit.forecast(forecast_horizon)
        mse = np.mean((forecast - val_data) ** 2)
        return -mse  # Negate MSE because Bayesian Optimization maximizes by default
    except Exception as e:
        print(f"Failed for alpha={alpha}, beta={beta}, gamma={gamma}: {e}")
        return -np.inf  # Return a large negative value for failed attempts

np.random.seed(98)


# Define the parameter bounds
pbounds = {
    'alpha': (0.01, 1.0),
    'beta': (0.01, 1.0),
    'gamma': (0.01, 1.0)
}

# Perform Bayesian Optimization
optimizer = BayesianOptimization(
    f=holt_winters_mse,
    pbounds=pbounds,
    random_state=99,
    verbose=2
)

optimizer.maximize(
    init_points=20,  # Number of random explorations
    n_iter=100        # Number of optimization steps
)

# Extract the best hyperparameters
best_params = optimizer.max['params']
best_alpha, best_beta, best_gamma = best_params['alpha'], best_params['beta'], best_params['gamma']
print(f"Best parameters: alpha={best_alpha}, beta={best_beta}, gamma={best_gamma}")

train_ts_data, test_ts_data = load_passenger_data()

# Evaluate the model with the best hyperparameters
model_hw_best = ExponentialSmoothing(
    train_ts_data,
    trend='mul',
    seasonal='mul',
    seasonal_periods=12
)
model_fit_best = model_hw_best.fit(
    smoothing_level=best_alpha,
    smoothing_trend=best_beta,
    smoothing_seasonal=best_gamma,
    optimized=False
)
forecast_best = model_fit_best.forecast(len(test_ts_data))
mse_best = np.mean((forecast_best - test_ts_data) ** 2)
print(f"Test MSE with Bayesian Optimization: {mse_best}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train_ts_data.index, train_ts_data.values, label='Training Data')
plt.plot(test_ts_data.index, test_ts_data.values, label='Actual Test Data')
plt.plot(test_ts_data.index, forecast_best.values, label='Bayesian Optimization Forecast')
plt.title('Actual vs Forecasted Passengers (Bayesian Optimization)')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()
