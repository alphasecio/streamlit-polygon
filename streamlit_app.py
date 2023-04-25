import os, streamlit as st
from tradier_python import TradierAPI

# Authenticate with the Tradier API
access_token = os.environ["TRADIER_TOKEN"]
tradier = TradierAPI(access_token, "sandbox")

# Define a function to get stock quotes
def get_stock_quote(symbol):
    quotes = tradier.get_quotes(symbol)
    for quote in quotes:
        if quote.symbol == symbol:
            return quote.last
    return None

# Set up the Streamlit app
st.title("Stocks App")
symbol = st.text_input("Enter a stock symbol", "AAPL")
if st.button("Get Quote"):
    quote = get_stock_quote(symbol)
    st.success(f"The last price for {symbol} is {quote}")
