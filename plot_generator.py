from io import BytesIO
import base64
import yfinance as yf
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def get_data(symbol,start = "",end = "",period="", interval=""):
    """Get historical stock prices for the given symbol"""
    #stock_data = yf.download(tickers=symbol, start="2019-01-01", end="2023-04-27")
    if start != "" and end != "":
      stock_data = yf.download(tickers=symbol, start=start, end=end)
    if start != "" and end != "" and interval != "":
      stock_data = yf.download(tickers=symbol, start=start, end=end,interval=interval)
    if period != "" and interval != "":
      stock_data = yf.download(tickers=symbol, period=period, interval=interval)
    stock_data.reset_index(drop=True, inplace=True)
    return stock_data
def generate_plot(currency_pairs, periods):
    data_plot = []
    charts_params = [{'5m':'59d'},{'15m':'59d'},{'1h':'729d'},{'1d':'30000d'}]
    # Initialize the SGDRegressor model with appropriate parameters
    sgd_regressor = SGDRegressor(loss='squared_epsilon_insensitive', max_iter=5000, tol=1e-3, random_state=142)
    for c in currency_pairs:
        print(c)
        for p in periods:
            print(p)
            period = [item[p] for item in charts_params if p in item]
            print(period[0])
            data = get_data(c,"","",period[0],p)
            # Normalize the data
            scaler = StandardScaler() #MinMaxScaler()
            scaler.fit(data)
            data[['Close']] = scaler.fit_transform(data[['Close']])

            # Extract the 'Close' prices as the target variable (y)
            y = data['Close'].values

            # Create a feature matrix (X) with time indices
            X = np.arange(len(y)).reshape(-1, 1)

            # Fit the model to the historical data
            sgd_regressor.fit(X, y)

            # Specify the number of future periods to forecast
            num_periods = 80  # Adjust as needed
                        
            # Generate future time indices for forecasting
            future_time_indices = np.arange(len(X), len(X) + num_periods).reshape(-1, 1)
            # Use the trained model to make predictions for future time indices
            future_predictions = sgd_regressor.predict(future_time_indices)
            future_predictions = scaler.fit_transform(future_predictions.reshape(-1, 1))
            
            # Plot the historical data and forecasted values
            plt.figure(figsize=(10, 6))
            plt.plot(X[-50:], y[-50:], label='Historical Data', color='blue')
            plt.plot(future_time_indices, future_predictions, label='Forecasted', color='red', linestyle='--')
            plt.title(f"{c} Stock Price Forecast Chart[{p}]")
            plt.xlabel("Time Index")
            plt.ylabel("Scaled Closing Price")  # Remember, it's scaled
            plt.legend()
            plt.grid(True)    
            # Replace this with your actual data and plotting logic
            # For this example, we'll create a simple plot
            
            #plt.figure()
            #plt.plot([1, 2, 3, 4, 5])
            #plt.title('Sample Plot')

            # Save the plot to a BytesIO object
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            img_data = base64.b64encode(img_buffer.read()).decode('utf-8')
            data_plot.append({
                'plot_data': img_data,
                'currency_pairs': c,
                'periods': p
            })
    # Return the plot data and other relevant information as a dictionary
    return data_plot