import os, streamlit as st
from polygon import RESTClient

# Set up the Streamlit app
st.subheader("Stocks App")
symbol = st.text_input("Enter a stock symbol", "AAPL")
with st.sidebar:
    polygon_api_key = st.text_input("Polygon API Key", type="password")

# Authenticate with the Polygon API
client = RESTClient(polygon_api_key)

col1, col2 = st.columns(2)

if col1.button("Get Details"):
    if not polygon_api_key.strip() or not symbol.strip():
        st.error("Please provide the missing fields.")
    else:
        try:
            details = client.get_ticker_details(symbol)
            st.success(f"Ticker: {details.ticker}\n\n"
                f"Company Address: {details.address}\n\n"
                f"Market Cap: {details.market_cap}")
        except Exception as e:
            st.exception(f"Exception: {e}") 

if col2.button("Get Quote"):
    if not polygon_api_key.strip() or not symbol.strip():
        st.error("Please provide the missing fields.")
    else:
        try:
            aggs = client.get_previous_close_agg(symbol)
            for agg in aggs:
                st.success(f"Ticker: {agg.ticker}\n\n"
                    f"Close: {agg.close}\n\n"
                    f"High: {agg.high}\n\n"
                    f"Low: {agg.low}\n\n"
                    f"Open: {agg.open}\n\n"
                    f"Volume: {agg.volume}")
        except Exception as e:
            st.exception(f"Exception: {e}") 
