import yfinance as yf
import plotly.graph_objects as go

# Fetching sufficient data to calculate the 100-day moving average
tickerSymbol = 'AAPL'
tickerData = yf.Ticker(tickerSymbol)
# Assuming '6mo' is enough to cover the 100-day period, adjust if necessary
tickerDf = tickerData.history(period='12mo')

# Calculate the 5-day and 100-day Moving Averages of the Close prices
tickerDf['5d_MA'] = tickerDf['Close'].rolling(window=5).mean()
tickerDf['100d_MA'] = tickerDf['Close'].rolling(window=100).mean()

# Trim the DataFrame to the last 20 days for the plot
last_20_days_df = tickerDf.tail(20)

# Creating a candlestick chart
fig = go.Figure(data=[go.Candlestick(x=last_20_days_df.index,
                                     open=last_20_days_df['Open'],
                                     high=last_20_days_df['High'],
                                     low=last_20_days_df['Low'],
                                     close=last_20_days_df['Close'],
                                     increasing_line_color='green', decreasing_line_color='red',
                                     name="Candlestick")])

# Adding the 5-day Moving Average line
fig.add_trace(go.Scatter(x=last_20_days_df.index, y=last_20_days_df['5d_MA'],
                         mode='lines', name='5-Day MA',
                         line=dict(color='blue', width=2)))

# Adding the 100-day Moving Average line
fig.add_trace(go.Scatter(x=last_20_days_df.index, y=last_20_days_df['100d_MA'],
                         mode='lines', name='100-Day MA',
                         line=dict(color='orange', width=2)))

fig.update_layout(title='AAPL Candlestick Chart with 5-Day and 100-Day Moving Averages for the Last 20 Days',
                  xaxis_title='Date',
                  yaxis_title='Price in USD',
                  xaxis_rangeslider_visible=False)  # Hide the range slider

fig.show()