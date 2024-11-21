import pandas as pd

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