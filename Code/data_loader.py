import pandas as pd
import numpy as np
import h5py

def load_nycbike_data():
    f = h5py.File('D:/MIE1630/Data/nyc-bike.h5', 'r')
    raw_data = f['raw_data']
    # Select a single node (e.g., node 0)
    node_id = 0  # Adjust to the desired node index
    time_series = raw_data[:, node_id]  # All timesteps for the selected node
    time_series = time_series[:, 0]
    # Create a date index for the training data
    start_date = pd.to_datetime('2016-04-01')
    freq = '30min'  # Monthly Start frequency
    train_index = pd.date_range(start=start_date, periods=len(time_series), freq=freq)
    train_ts = pd.Series(time_series, index=train_index)
    train_data = train_ts.iloc[:-12]
    test_data = train_ts.iloc[-12:]
    return train_data, test_data

def load_passenger_data():
    # Load the dataset directly from a CSV file
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    data = pd.read_csv(url)

    # Parse the 'Month' column to datetime
    data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')

    # Set 'Month' as the index
    data.set_index('Month', inplace=True)

    # Extract the time series data
    ts_data = data['Passengers']

    ts_data.index = pd.to_datetime(ts_data.index)
    ts_data = ts_data.asfreq('MS') # frequency here is monthly start

    # Split the data into training and testing
    train_ts_data = ts_data.iloc[:132]  # First 132 data points
    test_ts_data = ts_data.iloc[132:]   # Last 12 data points
    return train_ts_data, test_ts_data

def load_m4_monthly_data(series_id=None):

    # Load the training data
    train_df = pd.read_csv('D:\MIE1630\Data\Monthly-train.csv', index_col=0)
    
    # Load the test data
    test_df = pd.read_csv('D:\MIE1630\Data\Monthly-test.csv', index_col=0)

    # If no specific series ID is provided, select the first one
    if series_id is None:
        series_id = train_df.index[0]

    # Ensure the series exists in both training and testing data
    if series_id not in train_df.index or series_id not in test_df.index:
        raise ValueError(f"Series ID '{series_id}' not found in both training and testing data.")

    # Extract the training and testing data for the selected series
    train_ts = train_df.loc[series_id].dropna()
    test_ts = test_df.loc[series_id].dropna()

    # Create a date index for the training data
    # Assume the start date is January 1900
    start_date = pd.to_datetime('1900-01-01')
    freq = 'MS'  # Monthly Start frequency

    train_index = pd.date_range(start=start_date, periods=len(train_ts), freq=freq)
    train_ts = pd.Series(train_ts.values, index=train_index)

    # Create a date index for the test data, starting after the training data
    test_start_date = train_index[-1] + pd.DateOffset(months=1)
    test_index = pd.date_range(start=test_start_date, periods=len(test_ts), freq=freq)
    test_ts = pd.Series(test_ts.values, index=test_index)

    return train_ts, test_ts
