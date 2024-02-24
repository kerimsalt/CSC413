import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import random


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


slices = []
for i in range(0, len(tickerDf) - 20, 1):
    slice_ = tickerDf.iloc[i:i+20]
    slices.append(slice_)


# Got rid of the first 99 slices because all of them had NaN values for their 100 day moving average
slices = slices[99:]
print("1")
print(slices[1])

print("slices 1 Open")
print(slices[1]['Open'][0])

data_with_labels = []
for i in range(len(slices) - 1):
    # labels = {buy = 1, hold = 0, sell = -1}
    # curr_data_with_label = [data, label]
    curr_data_with_label = [slices[i]]  # First insert the data
    close_curr_day = slices[i]['Close'][-1]
    close_next_day = slices[i + 1]['Close'][0]
    threshold = 1.005
    if close_curr_day > close_next_day * threshold:
        curr_data_with_label.append(-1)  # Sell
    elif close_curr_day < close_next_day * threshold:
        curr_data_with_label.append(1)  # Buy
    else:
        curr_data_with_label.append(0)
    data_with_labels.append(curr_data_with_label)


curr_slice = slices[0]

# Plot a candlestick chart
fig = go.Figure(data=[go.Candlestick(x=curr_slice.index,
                                     open=curr_slice['Open'],
                                     high=curr_slice['High'],
                                     low=curr_slice['Low'],
                                     close=curr_slice['Close'],
                                     increasing_line_color='green', decreasing_line_color='red',
                                     name="Candlestick")])
# Adding the 5-day Moving Average line
fig.add_trace(go.Scatter(x=curr_slice.index, y=curr_slice['5d_MA'],
                         mode='lines', name='5-Day MA',
                         line=dict(color='blue', width=2)))

# Adding the 100-day Moving Average line
fig.add_trace(go.Scatter(x=curr_slice.index, y=curr_slice['100d_MA'],
                         mode='lines', name='100-Day MA',
                         line=dict(color='orange', width=2)))

fig.update_layout(title='AAPL Candlestick Chart with 5-Day and 100-Day Moving Averages for the Last 20 Days',
                  xaxis_title='Date',
                  yaxis_title='Price in USD',
                  xaxis_rangeslider_visible=False)  # Hide the range slider

fig.show()
fig.write_image('visual_data/fig1.png')


def png_to_numpy_colored(image_path):
    # Open the image
    img = Image.open(image_path)

    # Resize the image to match the size of PneumoniaMNIST dataset
    img_resized = img.resize((128, 128))

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
data_folder = 'C:/Users/kerim/PycharmProjects/CSC413/visual_data'
image_name = '/fig1'
image_path = data_folder + image_name + '.png'
image_matrix = png_to_numpy_colored(image_path)
image_matrix = make_background_black(image_matrix)

print("img matrix shape")
print(image_matrix.shape)  # Output shape should be (28, 28, 3) for RGB images
plt.imshow(image_matrix)
plt.show()

print("matrix itself")
print(image_matrix)

print("last day")
print(tickerDf.tail(1))

print("len slices")
print(len(slices))

print("data")
print(len(data_with_labels))

print("test")
print("curr")
print(slices[1098]['Close'][-1])
print("next")
print(slices[1099]['Close'][0])

print("label")
print(data_with_labels[1098][1])


# TODO: Shuffles the sample
# random.shuffle(data_with_labels)
