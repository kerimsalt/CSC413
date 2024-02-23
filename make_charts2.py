import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
from PIL import Image
import numpy as np


# Fetching sufficient data to calculate the 100-day moving average
tickerSymbol = 'AAPL'
tickerData = yf.Ticker(tickerSymbol)
# Assuming '6mo' is enough to cover the 100-day period, adjust if necessary
tickerDf = tickerData.history(period='25y')

# Calculate the 5-day and 100-day Moving Averages of the Close prices
tickerDf['5d_MA'] = tickerDf['Close'].rolling(window=5).mean()
tickerDf['100d_MA'] = tickerDf['Close'].rolling(window=100).mean()

# Trim the DataFrame to the last 20 days for the plot
last_20_days_df = tickerDf.tail(20)

print("length of data")
print(len(tickerDf))

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
fig.write_image("fig1.png")

def png_to_numpy_colored(image_path):
    # Open the image
    img = Image.open(image_path)

    # Resize the image to match the size of PneumoniaMNIST dataset
    img_resized = img.resize((64, 64))

    # Convert the image to a numpy array
    img_array = np.array(img_resized)

    # Normalize the pixel values (0-255) to (0-1)
    img_array = img_array / 255.0

    return img_array


def make_background_black(image_matrix):
    # Create a copy of the image matrix to avoid modifying the original
    modified_image = image_matrix.copy()

    # If the image has RGBA channels, set the alpha channel to 1 (fully opaque)
    if modified_image.shape[2] == 4:
        modified_image[:, :, 3][modified_image[:, :, 3] != 0] = 1

    # Identify pixels where RGB values are all close to 1 (white background)
    white_pixels = np.all(modified_image[:, :, :3] > 0.9, axis=-1)

    # Set the RGB values of white pixels to 0 (black background)
    modified_image[:, :, :3][white_pixels] = 0

    return modified_image

# C:\Users\kerim\PycharmProjects\CSC413\fig1.png
image_path = 'C:/Users/kerim/PycharmProjects/CSC413/fig1.png'
image_matrix = png_to_numpy_colored(image_path)
image_matrix = make_background_black(image_matrix)

print("img matrix shape")
print(image_matrix.shape)  # Output shape should be (28, 28, 3) for RGB images
plt.imshow(image_matrix)
plt.show()

print("matrix itself")
print(image_matrix)