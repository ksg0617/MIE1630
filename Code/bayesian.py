from bayes_opt import BayesianOptimization
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import pandas as pd
from data_loader import load_data
import matplotlib.pyplot as plt

# Optionally scale the data
from sklearn.preprocessing import MinMaxScaler


np.random.seed(98)
# Define the objective function
# data_file_path = "D:/MIE1630/Data/AirQualityUCI.csv"
# data_file_path = "D:/MIE1630/Data/electricity.csv"
data_file_path = "D:/MIE1630/Data/traffic.csv"

def holt_winters_mse(alpha, beta, gamma):
    """
    Compute Mean Squared Error (MSE) for Holt-Winters given hyperparameters.
    """


    ts_data, test_ts_data= load_data(data_file_path)
    ts_data = ts_data.dropna()

    forecast_horizon = 48

    train_data = ts_data[:-forecast_horizon]
    val_data =ts_data[-forecast_horizon:]

    try:
        # Fit the Holt-Winters model with given parameters
        model_hw = ExponentialSmoothing(
            train_data,
            trend='add',
            seasonal='add',
            seasonal_periods=24
        )
        model_fit = model_hw.fit(
            smoothing_level=alpha,
            smoothing_trend=beta,
            smoothing_seasonal=gamma,
            optimized=False
        )
        forecast = model_fit.forecast(forecast_horizon)
        mse = np.sqrt(np.mean((forecast - val_data) ** 2))
        return -mse  # Negate MSE because Bayesian Optimization maximizes by default
    except Exception as e:
        
        print(f"Failed for alpha={alpha}, beta={beta}, gamma={gamma}: {e}")
        return -1e10  # Return a large negative value for failed attempts




# Define the parameter bounds
pbounds = {'alpha': (0.01, 0.99), 'beta': (0.01, 0.99), 'gamma': (0.01, 0.99)}


# Perform Bayesian Optimization
optimizer = BayesianOptimization(
    f=holt_winters_mse,
    pbounds=pbounds,
    random_state=98
)

optimizer.maximize(
    init_points=10,  # Number of random explorations
    n_iter=50        # Number of optimization steps
)

# Extract the best hyperparameters
best_params = optimizer.max['params']
best_alpha, best_beta, best_gamma = best_params['alpha'], best_params['beta'], best_params['gamma']
print(f"Best parameters: alpha={best_alpha}, beta={best_beta}, gamma={best_gamma}")

train_ts_data, test_ts_data = load_data(data_file_path)

# Evaluate the model with the best hyperparameters
model_hw_best = ExponentialSmoothing(
    train_ts_data,
    trend='add',
    seasonal='add',
    seasonal_periods=24
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

# Specify the number of last data points to display
n_last_points = 50  # Adjust this number based on how many data points you want to see

# Slice the last n data points from the training data
train_last = train_ts_data[-n_last_points:]

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train_last.index, train_last.values, label='Training Data')
plt.plot(test_ts_data.index, test_ts_data.values, label='Actual Test Data')
plt.plot(test_ts_data.index, forecast_best.values, label='Bayesian Optimization Forecast')
plt.title('Actual vs Forecasted (Bayesian Optimization)')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()

if data_file_path == "D:/MIE1630/Data/AirQualityUCI.csv":
    plt.savefig("D:/MIE1630/Results/Bayesian/" + "air_quality.png")
elif data_file_path == "D:/MIE1630/Data/electricity.csv":
    plt.savefig("D:/MIE1630/Results/Bayesian/" + "electricity.png")
elif data_file_path == "D:/MIE1630/Data/traffic.csv":
    plt.savefig("D:/MIE1630/Results/Bayesian/" + "traffic.png")
plt.show()
