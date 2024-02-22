import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches stock data for the given ticker symbol from Yahoo Finance.

    Args:
    - ticker (str): Ticker symbol of the stock (e.g., 'NVDA' for Nvidia).
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    - pandas.DataFrame: DataFrame containing the stock price data.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data


def save_to_csv(dataframe, filename):
    """
    Saves DataFrame to a CSV file.

    Args:
    - dataframe (pandas.DataFrame): DataFrame to be saved.
    - filename (str): Name of the CSV file.
    """
    dataframe.to_csv(filename)


if __name__ == "__main__":
    ticker_symbol = 'NVDA'
    start_date = '1999-01-01'
    end_date = '2024-01-01'

    stock_data = fetch_stock_data(ticker_symbol, start_date, end_date)
    if not stock_data.empty:
        csv_filename = f"{ticker_symbol}_stock_data_{start_date}_{end_date}.csv"
        save_to_csv(stock_data, csv_filename)
        print(f"Stock data saved to {csv_filename}")
    else:
        print("Failed to fetch stock data. Please check the ticker symbol or date range.")
