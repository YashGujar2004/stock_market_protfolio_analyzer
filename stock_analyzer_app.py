import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Stock Portfolio Analyzer",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API FUNCTIONS ---
# Use Streamlit's caching to avoid re-fetching data on every interaction.
# TTL (time-to-live) of 600 seconds (10 minutes) means data is cached for 10 minutes.
@st.cache_data(ttl=600)
def fetch_stock_price(ticker, api_key):
    """Fetches the latest stock price from Alpha Vantage."""
    if not api_key or not ticker:
        return None
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        if "Global Quote" in data and "05. price" in data["Global Quote"]:
            return float(data["Global Quote"]["05. price"])
        elif "Note" in data:
            st.warning(f"API rate limit reached. Please wait a moment before refreshing. Ticker: {ticker}")
        else:
            st.error(f"Could not retrieve price for {ticker}. The ticker might be invalid or data is unavailable.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error while fetching data for {ticker}: {e}")
        return None
    except ValueError:
        st.error(f"Failed to parse data for {ticker}. The API response might have changed.")
        return None


@st.cache_data(ttl=3600)
def search_symbols(keywords, api_key):
    """Searches for stock symbols based on keywords."""
    if not api_key or not keywords:
        return []
    url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={keywords}&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "bestMatches" in data:
            return data["bestMatches"]
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Network error during symbol search: {e}")
        return []


# --- SESSION STATE INITIALIZATION ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Ticker", "Quantity", "Purchase Price"])
if 'api_key' not in st.session_state:
    st.session_state.api_key = "2QNV3TPTC7ZT8M7D"
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'search_term' not in st.session_state:
    st.session_state.search_term = ""


# --- UI: SIDEBAR ---
with st.sidebar:
    st.title("_Configuration_")
    st.session_state.api_key = st.text_input(
        "Enter your Alpha Vantage API Key",
        type="password",
        value=st.session_state.api_key,
        help="Get your free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)."
    )
    if st.session_state.api_key:
        st.success("API Key saved for this session.")
    else:
        st.warning("API Key is required to fetch live data.")

    st.markdown("---")
    st.title("_Add a New Stock_")
    
    # --- MODIFIED SEARCH LOGIC ---
    # Use columns for a more compact layout
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        st.session_state.search_term = st.text_input("Search for a company or ticker:", value=st.session_state.search_term)
    
    with search_col2:
        st.write(" ") # Vertically align button
        if st.button("Search"):
            if st.session_state.search_term and st.session_state.api_key:
                with st.spinner("Searching..."):
                     st.session_state.search_results = search_symbols(st.session_state.search_term, st.session_state.api_key)
            else:
                st.session_state.search_results = []

    # Display search results in a selectbox if they exist
    if st.session_state.search_results:
        choices = [f"{match['1. symbol']} - {match['2. name']}" for match in st.session_state.search_results]
        selected_stock = st.selectbox("Select a stock:", choices, index=0)
        ticker_input = selected_stock.split(' - ')[0]
    else:
        # Fallback to using the search term directly if no results
        ticker_input = st.session_state.search_term.upper()

    # Add stock form
    with st.form("add_stock_form", clear_on_submit=True):
        ticker = st.text_input("Ticker Symbol", value=ticker_input)
        quantity = st.number_input("Quantity", min_value=0.0001, format="%.4f")
        purchase_price = st.number_input("Purchase Price", min_value=0.01, format="%.2f")
        
        submitted = st.form_submit_button("Add Stock")
        if submitted:
            if not ticker or quantity <= 0 or purchase_price <= 0:
                st.error("Please fill out all fields with valid values.")
            else:
                new_stock = pd.DataFrame([{
                    "Ticker": ticker.upper(),
                    "Quantity": quantity,
                    "Purchase Price": purchase_price
                }])
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_stock], ignore_index=True)
                st.success(f"Added {ticker.upper()} to your portfolio!")
                # Clear search state after adding
                st.session_state.search_results = []
                st.session_state.search_term = ""
                st.rerun() # Rerun to reflect changes immediately

# --- MAIN PAGE ---
st.title("💹 Stock Portfolio Analyzer")
st.markdown("Track your investments and visualize your portfolio's performance in real-time.")

if st.session_state.portfolio.empty:
    st.info("Your portfolio is empty. Add a stock using the sidebar to get started.")
else:
    # --- PORTFOLIO DATA PROCESSING ---
    portfolio_df = st.session_state.portfolio.copy()
    
    # Fetch current prices for all unique tickers
    unique_tickers = portfolio_df["Ticker"].unique()
    price_data = {}
    
    if st.session_state.api_key:
        price_fetch_progress = st.progress(0, text="Fetching live prices...")
        for i, ticker in enumerate(unique_tickers):
            # Introduce a small delay to respect API rate limits
            time.sleep(1) 
            price_data[ticker] = fetch_stock_price(ticker, st.session_state.api_key)
            price_fetch_progress.progress((i + 1) / len(unique_tickers))
        price_fetch_progress.empty()
    else:
        st.warning("Please enter your API key to see live price data.")


    portfolio_df["Current Price"] = portfolio_df["Ticker"].map(price_data).fillna(0)
    portfolio_df["Current Value"] = portfolio_df["Quantity"] * portfolio_df["Current Price"]
    portfolio_df["Purchase Value"] = portfolio_df["Quantity"] * portfolio_df["Purchase Price"]
    portfolio_df["P/L"] = portfolio_df["Current Value"] - portfolio_df["Purchase Value"]
    
    # --- PORTFOLIO SUMMARY METRICS ---
    total_current_value = portfolio_df["Current Value"].sum()
    total_purchase_value = portfolio_df["Purchase Value"].sum()
    total_pl = portfolio_df["P/L"].sum()
    pl_percentage = (total_pl / total_purchase_value * 100) if total_purchase_value != 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Portfolio Value", f"${total_current_value:,.2f}")
    col2.metric("Total Profit/Loss", f"${total_pl:,.2f}", f"{pl_percentage:,.2f}%")
    
    # Refresh button
    if col3.button("Refresh Prices"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # --- VISUALIZATION & DATA TABLE ---
    col_chart, col_table = st.columns([0.4, 0.6])

    with col_chart:
        st.subheader("Portfolio Distribution")
        if not portfolio_df.empty and portfolio_df["Current Value"].sum() > 0:
            fig = px.pie(
                portfolio_df,
                values='Current Value',
                names='Ticker',
                title='Portfolio Value by Stock',
                hole=.3,
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig.update_traces(textinfo='percent+label', pull=[0.05]*len(portfolio_df))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add stocks with live prices to see the chart.")

    with col_table:
        st.subheader("My Portfolio")
        
        # Function to color P/L column
        def color_pl(val):
            color = 'red' if val < 0 else 'green' if val > 0 else 'gray'
            return f'color: {color}'

        styled_df = portfolio_df.style.format({
            "Purchase Price": "${:,.2f}",
            "Current Price": "${:,.2f}",
            "Current Value": "${:,.2f}",
            "P/L": "${:,.2f}",
            "Quantity": "{:,.4f}"
        }).applymap(color_pl, subset=['P/L'])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # --- DELETING STOCKS ---
        st.markdown("---")
        st.subheader("Manage Portfolio")
        if not portfolio_df.empty:
            stocks_to_delete = st.multiselect(
                "Select stocks to delete from your portfolio:",
                options=portfolio_df.index, # Use index for easier deletion
                format_func=lambda x: portfolio_df.loc[x, "Ticker"]
            )
            if st.button("Delete Selected Stocks"):
                st.session_state.portfolio = st.session_state.portfolio.drop(stocks_to_delete)
                st.rerun()

