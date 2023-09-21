from io import BytesIO
import base64
import yfinance as yf
import numpy as np
#from sklearn.linear_model import SGDRegressor
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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
        ax.set_title(f"Candlestick {c} Chart with Lines")
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
        
