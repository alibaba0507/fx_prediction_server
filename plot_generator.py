from io import BytesIO
import base64
import yfinance as yf
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import linregress
#import plotly.graph_objects as go

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
 
def support(df1, l, n1, n2): #n1 n2 before and after candle l
    for i in range(l-n1+1, l+1):
        if(df1.Low[i]>df1.Low[i-1]):
            return 0
    for i in range(l+1,l+n2+1):
        if(df1.Low[i]<df1.Low[i-1]):
            return 0
    return 1

#support(df,46,3,2)

def resistance(df1, l, n1, n2): #n1 n2 before and after candle l
    for i in range(l-n1+1, l+1):
        if(df1.High[i]<df1.High[i-1]):
            return 0
    for i in range(l+1,l+n2+1):
        if(df1.High[i]>df1.High[i-1]):
            return 0
    return 1
#resistance(df, 30, 3, 5)

def generate_supp_ress_plot_v1(currency_pairs):
    data_plot = []
    for c in currency_pairs:
        df = get_data(c,"","","50d","1d")
        ss = []
        rr = []
        n1=2
        n2=2
        maxRange =  len(df) - 1 if len(df) - 1 < 200 else 200
        for row in range(3, maxRange): #len(df)-n2
            if support(df, row, n1, n2):
                ss.append((row,df.Low[row]))
            if resistance(df, row, n1, n2):
                rr.append((row,df.High[row]))
        s = 0 if len(df) - 1 < 80 else 80
        e = len(df) - 1 if len(df) - 1 < 200 else 200
        dfpl = df[s:e]
        fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
            open=dfpl['Open'],
            high=dfpl['High'],
            low=dfpl['Low'],
            close=dfpl['Close'])])

        c=0
        while (1):
            if(c>len(ss)-1 ):
                break
            fig.add_shape(type='line', x0=ss[c][0], y0=ss[c][1],
                x1=e,
                y1=ss[c][1],
                line=dict(color="MediumPurple",width=3)
                )
            c+=1

        c=0
        while (1):
            if(c>len(rr)-1 ):
                break
            fig.add_shape(type='line', x0=rr[c][0], y0=rr[c][1],
                x1=e,
                y1=rr[c][1],
                line=dict(color="RoyalBlue",width=1)
                )
            c+=1
        # Save the figure as an image
        img_buffer = BytesIO()
        fig.write_image(img_buffer, format="png")

        # Encode the image to base64
        img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')    
        data_plot.append({
                    'plot_data': img_data,
                    'currency_pairs': f"{c} S/R Lines",
                    'periods': "1d"
                })
    return data_plot

def generate_supp_ress_plot(currency_pairs):
    data_plot = []
    for c in currency_pairs:
        df = get_data(c,"","","50d","1d")
        ss = []
        rr = []
        n1=2
        n2=2
        maxRange =  len(df) - 2 if len(df) - 2 < 200 else 200
        for row in range(3, maxRange): #len(df)-n2
            if support(df, row, n1, n2):
                ss.append((row,df.Low[row]))
            if resistance(df, row, n1, n2):
                rr.append((row,df.High[row]))
        s = 0 if len(df) - 2 < 80 else 80
        e = len(df) - 2 if len(df) - 2 < 200 else 200
        dfpl = df[s:e]
        # Assuming dfpl is a DataFrame with 'Open', 'High', 'Low', and 'Close' columns
        # Also assuming ss and rr are lists of tuples with (x0, y0) coordinates

        # Create a new figure
        fig, ax = plt.subplots()

        # Create candlestick chart
        ax.plot(dfpl.index, dfpl['Open'], label='Open', color='green', marker='o', linestyle='-')
        ax.plot(dfpl.index, dfpl['Close'], label='Close', color='red', marker='o', linestyle='-')
        ax.vlines(dfpl.index, dfpl['Low'], dfpl['High'], color='black', linewidth=1)

        # Add lines from ss
        for s in ss:
            ax.axhline(y=s[1], color='MediumPurple', linewidth=3)

        # Add lines from rr
        for r in rr:
            ax.axhline(y=r[1], color='RoyalBlue', linewidth=1)

        # Customize the appearance of the chart as needed
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Candlestick Chart with Lines')
        ax.legend()
        # Save the plot to a BytesIO object
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.read()).decode('utf-8')
        data_plot.append({
                    'plot_data': img_data,
                    'currency_pairs': f"{c} S/R Lines",
                    'periods': "1d"
                })
    return data_plot
    
def pivotid(df1, l, n1, n2): #n1 n2 before and after candle l
    if l-n1 < 0 or l+n2 >= len(df1):
        return 0

    pividlow=1
    pividhigh=1
    for i in range(l-n1, l+n2+1):
        if(df1.low[l]>df1.low[i]):
            pividlow=0
        if(df1.high[l]<df1.high[i]):
            pividhigh=0
    if pividlow and pividhigh:
        return 3
    elif pividlow:
        return 1
    elif pividhigh:
        return 2
    else:
        return 0

def pointpos(x):
    if x['pivot']==1:
        return x['low']-1e-3
    elif x['pivot']==2:
        return x['high']+1e-3
    else:
        return np.nan
def triangle_plot(dfpl,xxmin,xxmax,minim,maxim):
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dfpl.index, dfpl['open'], color='r', marker='o', linestyle='-', label='Open')
    ax.plot(dfpl.index, dfpl['close'], color='g', marker='o', linestyle='-', label='Close')
    ax.plot(dfpl.index, dfpl['high'], color='b', marker='o', linestyle='-', label='High')
    ax.plot(dfpl.index, dfpl['low'], color='y', marker='o', linestyle='-', label='Low')

    # Add pivot points as markers
    pivot_points = dfpl[dfpl['pointpos'] == 1]
    ax.scatter(pivot_points.index, pivot_points['low'], color='MediumPurple', label='Pivot', s=40)
    if all(arr.size > 0 for arr in [xxmin, xxmax, minim, maxim]):
        slmin, intercmin, rmin, pmin, semin = linregress(xxmin, minim)
        slmax, intercmax, rmax, pmax, semax = linregress(xxmax, maxim)

        #print(rmin, rmax)
        # Fitting slopes to meet highest or lowest candle point in time slice
        xxmin = np.append(xxmin, xxmin[-1] + 15)
        ax.plot(xxmin, slmin * xxmin + intercmin, linestyle='--', label='Min Slope')
        xxmax = np.append(xxmax, xxmax[-1] + 15)
        ax.plot(xxmax, slmax * xxmax + intercmax, linestyle='--', label='Max Slope')

    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title('Candlestick Chart with Pivot Points')
    ax.legend()

    # Save the plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close(fig)  # Close the figure to release resources

    return img_data        
def generate_trianlges_plot(currency_pairs, periods,shift):
    data_plot = []
    charts_params = [{'5m':'59d'},{'15m':'59d'},{'1h':'729d'},{'1d':'30000d'}]
    # Ensure 'shift' is an integer
    if isinstance(shift, list):
        shift = shift[0]
    # Convert 'shift_str' to an integer
    try:
        shift = int(shift)
    except ValueError:
        # Handle the case where 'shift_str' cannot be converted to an integer
        print("Error: 'shift_str' cannot be converted to an integer.")
        # You can assign a default value or take appropriate action here
    for c in currency_pairs:
        #print(c)
        for p in periods:
            period = [item[p] for item in charts_params if p in item]
            df = get_data(c,"","",period[0],p)
            df.columns=['open', 'high', 'low', 'close', 'Adj Close','volume']
            df.reset_index(drop=True, inplace=True)
            df.isna().sum()
            
            df['pivot'] = df.apply(lambda x: pivotid(df, x.name,2,2), axis=1)
            df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)
            #print(df.head(10))
            backcandles = 5 if shift < 5 else shift
            candleid = len(df) - 1
            maxim = np.array([])
            minim = np.array([])
            xxmin = np.array([])
            xxmax = np.array([])
            print(backcandles , shift , candleid)
            for i in range(candleid - backcandles, candleid + 1):
                print(df.iloc[i].pivot)
                if df.iloc[i].pivot == 1:
                    minim = np.append(minim, df.iloc[i].low)
                    xxmin = np.append(xxmin, i)  # could be i instead df.iloc[i].name
                if df.iloc[i].pivot == 2:
                    maxim = np.append(maxim, df.iloc[i].high)
                    xxmax = np.append(xxmax, i)  # df.iloc[i].name

            

            dfpl = df[candleid - backcandles - 10:candleid + backcandles + 10]
            print("xxmin:", xxmin)
            print("minim:", minim)
            # Call the function to generate the plot
            img_data = triangle_plot(dfpl,xxmin,xxmax,minim,maxim)
            data_plot.append({
                        'plot_data': img_data,
                        'currency_pairs': f"{c} Triangle Patterns",
                        'periods': "1d"
                    })
    return data_plot
def generate_plot(currency_pairs, periods, shift, loop):
    data_plot = []
    charts_params = [{'5m':'59d'},{'15m':'59d'},{'1h':'729d'},{'1d':'30000d'}]
    # Initialize the SGDRegressor model with appropriate parameters
    sgd_regressor = SGDRegressor(loss='squared_epsilon_insensitive', max_iter=5000, tol=1e-3, random_state=142)
    for c in currency_pairs:
        #print(c)
        for p in periods:
            #print(p)
            period = [item[p] for item in charts_params if p in item]
            #print(period[0])
            data = get_data(c,"","",period[0],p)
            # Normalize the data
            scaler = StandardScaler() #MinMaxScaler()
            scaler.fit(data)
            data[['Close']] = scaler.fit_transform(data[['Close']])

            # Extract the 'Close' prices as the target variable (y)
            y = data['Close'].values
            y_shifted = y
            # Number of times to fit the model
            t = int(loop[0])   # Change this to the number of times you want to fit the model
            n = int(shift[0])   # Change this to the number of elements to shift by
            for i in range(t):
                # Create a feature matrix (X) with time indices
                X = np.arange(len(y_shifted)).reshape(-1, 1)

                # Fit the model to the historical data
                sgd_regressor.fit(X, y_shifted)
                
                # Specify the number of future periods to forecast
                num_periods = 50  # Adjust as needed
                # Generate future time indices for forecasting
                future_time_indices = np.arange(len(X), len(X) + num_periods).reshape(-1, 1)
                # Use the trained model to make predictions for future time indices
                future_predictions = sgd_regressor.predict(future_time_indices)
                future_predictions = scaler.fit_transform(future_predictions.reshape(-1, 1))
                # Plot the historical data and forecasted values
                plt.figure(figsize=(10, 6))
                plt.plot(X[-50:], y_shifted[-50:], label='Historical Data', color='blue')
                plt.plot(future_time_indices, future_predictions, label='Forecasted', color='red', linestyle='--')
                plt.title(f"{c} Forecast Chart[{p}] shift[{i * n:}]")
                plt.xlabel("Time Index")
                plt.ylabel("Scaled Closing Price")  # Remember, it's scaled
                plt.legend()
                plt.grid(True)   
                # Save the plot to a BytesIO object
                img_buffer = BytesIO()
                plt.savefig(img_buffer, format='png')
                img_buffer.seek(0)
                img_data = base64.b64encode(img_buffer.read()).decode('utf-8')
                data_plot.append({
                    'plot_data': img_data,
                    'currency_pairs': f"{c} shift[{i * n:}]",
                    'periods': p
                })
                # Calculate the end index for the slice
                end_index = len(y) - (i * n)
                # Shift y back by n elements
                y_shifted = y[:end_index]
                #print(len(y_shifted))
                #print(y_shifted[-2:])
    # Return the plot data and other relevant information as a dictionary
    return data_plot
def generate_plot_v1(currency_pairs, periods):
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