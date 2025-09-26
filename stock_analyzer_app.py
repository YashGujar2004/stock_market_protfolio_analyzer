import streamlit as st
import pandas as pd
# import requests
import plotly.express as px
import plotly.graph_objects as go
import time
import yfinance as yf
import pandas_ta as ta
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Enhanced Stock Portfolio Analyzer",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=600)
def fetch_stock_price(ticker, api_key=None):
    """Fetches the latest stock price using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return None
    except Exception as e:
        st.error(f"Error fetching price for {ticker}: {e}")
        return None

def generate_sell_signals(data, ticker, purchase_date, purchase_price):
    """Generate simple sell signals based on technical indicators and time"""
    if data is None or data.empty:
        return {"signal": "HOLD", "confidence": 0, "reason": "No data"}

    latest = data.iloc[-1]
    current_price = latest['Close']
    rsi = latest.get('RSI', 50)

    days_held = (datetime.now().date() - purchase_date).days

    return_pct = (current_price - purchase_price) / purchase_price * 100

    signals = []
    confidence = 0

    if not pd.isna(rsi):
        if rsi > 70:
            confidence += 30
        elif rsi < 30:
            confidence -= 20

    if return_pct > 20:
        confidence += 25
    elif return_pct < -10:
        confidence += 40

    if days_held > 365 and return_pct > 10:
        confidence += 15
    elif days_held > 30 and return_pct > 5:
        confidence += 10

    if confidence >= 50:
        signal = "STRONG SELL"
    elif confidence >= 30:
        signal = "SELL"
    elif confidence <= -10:
        signal = "BUY"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "reasons": signals,
        "days_held": days_held,
        "current_price": current_price
    }

def calculate_portfolio_metrics(portfolio_df):
    """Calculate portfolio performance metrics"""
    if portfolio_df.empty:
        return {}

    metrics = {}

    metrics['total_invested'] = portfolio_df['Purchase_Value'].sum()
    metrics['current_value'] = portfolio_df['Current_Value'].sum()
    metrics['total_pl'] = metrics['current_value'] - metrics['total_invested']

    returns = portfolio_df['P_L_Percent'].dropna()
    if not returns.empty:
        metrics['avg_return'] = returns.mean()

        if not portfolio_df['P_L_Percent'].isna().all():
            best_idx = portfolio_df['P_L_Percent'].idxmax()
            worst_idx = portfolio_df['P_L_Percent'].idxmin()
            metrics['best_stock'] = portfolio_df.loc[best_idx, 'Ticker']
            metrics['worst_stock'] = portfolio_df.loc[worst_idx, 'Ticker']

        excess_return = metrics['avg_return'] - 2

    return metrics

def create_performance_chart(portfolio_df):
    """Create portfolio performance visualization"""
    fig = go.Figure()

    colors = ['red' if x < 0 else 'green' for x in portfolio_df['P_L_Percent']]

    fig.add_trace(go.Bar(
        x=portfolio_df['Ticker'],
        y=portfolio_df['P_L_Percent'],
        text=[f"{x:.1f}%" for x in portfolio_df['P_L_Percent']],
        textposition='auto',
        marker_color=colors,
        name='Return %'
    ))

    fig.update_layout(
        title='Individual Stock Performance (%)',
        xaxis_title='Stock Ticker',
        yaxis_title='Return %',
        showlegend=False,
        height=400
    )

    return fig

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Ticker", "Quantity", "Purchase_Price", "Purchase_Date"])

with st.sidebar:
    st.title("_📈Enhanced Portfolio Analyzer_")

    st.markdown("---")
    st.title("_Add New Stock_")

    with st.form("add_stock_form", clear_on_submit=True):
        ticker = st.text_input("Ticker Symbol (e.g., AAPL)")
        quantity = st.number_input("Quantity", min_value=0.0001, format="%.4f", value=1.0)
        purchase_price = st.number_input("Purchase Price ($)", min_value=0.01, format="%.2f", value=100.0)
        purchase_date = st.date_input("Purchase Date", datetime.now().date())

        submitted = st.form_submit_button("Add Stock")
        if submitted:
            if not ticker or quantity <= 0 or purchase_price <= 0:
                st.error("Please fill out all fields with valid values.")
            else:
                new_stock = pd.DataFrame([{
                    "Ticker": ticker.upper(),
                    "Quantity": quantity,
                    "Purchase_Price": purchase_price,
                    "Purchase_Date": purchase_date
                }])
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_stock], ignore_index=True)
                st.success(f"Added {ticker.upper()} to your portfolio!")
                st.rerun()

st.title("💹 Enhanced Stock Portfolio Analyzer")
st.markdown("**Real-time Portfolio Tracking with Intelligent Sell Signal Predictions**")

if st.session_state.portfolio.empty:
    st.info("👈 Your portfolio is empty. Add stocks using the sidebar to get started.")
else:
    portfolio_df = st.session_state.portfolio.copy()

    unique_tickers = portfolio_df["Ticker"].unique()
    price_data = {}
    stock_signals = {}

    with st.spinner("🔄 Fetching live data and analyzing sell signals..."):
        progress_bar = st.progress(0)

        for i, ticker in enumerate(unique_tickers):
            try:
                stock = yf.Ticker(ticker)
                hist_data = stock.history(period="6mo")

                if not hist_data.empty:
                    price_data[ticker] = hist_data['Close'].iloc[-1]

                    hist_data['RSI'] = ta.rsi(hist_data['Close'], length=14)
                    hist_data['SMA_20'] = ta.sma(hist_data['Close'], length=20)

                    ticker_signals = []
                    for idx, row in portfolio_df[portfolio_df['Ticker'] == ticker].iterrows():
                        signal_data = generate_sell_signals(
                            hist_data, ticker, row['Purchase_Date'], row['Purchase_Price']
                        )
                        ticker_signals.append(signal_data)
                    stock_signals[ticker] = ticker_signals
                else:
                    price_data[ticker] = 0

            except Exception as e:
                price_data[ticker] = 0
                st.warning(f"Could not analyze {ticker}: {str(e)}")

            progress_bar.progress((i + 1) / len(unique_tickers))
            time.sleep(0.2)

        progress_bar.empty()

    portfolio_df["Current_Price"] = portfolio_df["Ticker"].map(price_data).fillna(0)
    portfolio_df["Current_Value"] = portfolio_df["Quantity"] * portfolio_df["Current_Price"]
    portfolio_df["Purchase_Value"] = portfolio_df["Quantity"] * portfolio_df["Purchase_Price"]
    portfolio_df["P_L"] = portfolio_df["Current_Value"] - portfolio_df["Purchase_Value"]
    portfolio_df["P_L_Percent"] = (portfolio_df["P_L"] / portfolio_df["Purchase_Value"] * 100).fillna(0)
    portfolio_df["Days_Held"] = portfolio_df["Purchase_Date"].apply(
        lambda x: (datetime.now().date() - x).days
    )

    metrics = calculate_portfolio_metrics(portfolio_df)

    st.subheader("📊 Portfolio Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Value", f"${metrics.get('current_value', 0):,.2f}")

    with col2:
        st.metric("Total P/L", f"${metrics.get('total_pl', 0):,.2f}",
                f"{metrics.get('total_return_pct', 0):.2f}%")

    with col3:
        st.metric("Best Performer", metrics.get('best_stock', 'N/A'))

    with col4:
        st.metric("Worst Performer", metrics.get('worst_stock', 'N/A'))

    st.markdown("### 📈 Performance & Risk Metrics")
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

    risk_col1.metric("Avg Return", f"{metrics.get('avg_return', 0):.2f}%")
    risk_col4.metric("Holdings", f"{len(portfolio_df)}")

    st.markdown("---")
    st.subheader("🎯 Intelligent Sell Signal Analysis")

    signal_summary = {"STRONG SELL": 0, "SELL": 0, "HOLD": 0, "BUY": 0}

    for ticker in unique_tickers:
        if ticker in stock_signals and price_data.get(ticker, 0) > 0:
            st.markdown(f"**📈 {ticker} Analysis**")

            ticker_positions = portfolio_df[portfolio_df['Ticker'] == ticker]

            for idx, (pos_idx, position) in enumerate(ticker_positions.iterrows()):
                if idx < len(stock_signals[ticker]):
                    signal_info = stock_signals[ticker][idx]
                    signal_summary[signal_info['signal']] += 1

                    col1, col2, col3, col4, col5 = st.columns(5)

                    signal_emoji = {"STRONG SELL": "🔴", "SELL": "🟠", "HOLD": "🟡", "BUY": "🟢"}
                    emoji = signal_emoji.get(signal_info['signal'], "⚪")

                    col1.markdown(f"{emoji} **{signal_info['signal']}**")
                    col4.write(f"Days Held: {signal_info['days_held']}")

                    if signal_info.get('reasons'):
                        st.caption(f"💡 {', '.join(signal_info['reasons'])}")

            st.markdown("---")

    st.markdown("### 📋 Signal Summary")
    summary_cols = st.columns(4)

    for i, (signal_type, count) in enumerate(signal_summary.items()):
        if count > 0:
            color = {"STRONG SELL": "🔴", "SELL": "🟠", "HOLD": "🟡", "BUY": "🟢"}[signal_type]
            summary_cols[i].metric(f"{color} {signal_type}", count)

    st.markdown("---")
    st.subheader("📊 Portfolio Analysis Charts")

    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        if portfolio_df['Current_Value'].sum() > 0:
            fig_pie = px.pie(
                portfolio_df, 
                values='Current_Value', 
                names='Ticker',
                title='Portfolio Distribution by Value',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        perf_chart = create_performance_chart(portfolio_df)
        st.plotly_chart(perf_chart, use_container_width=True)

    st.subheader("📋 Detailed Portfolio Holdings")

    display_df = portfolio_df[[
        'Ticker', 'Quantity', 'Purchase_Price', 'Current_Price', 
        'P_L', 'P_L_Percent', 'Days_Held'
    ]].copy()

    display_df.columns = ['Ticker', 'Qty', 'Buy Price', 'Current', 'P/L ($)', 'P/L (%)', 'Days']

    def color_pl(val):
        if pd.isna(val):
            return ''
        color = 'red' if val < 0 else 'green' if val > 0 else 'gray'
        return f'color: {color}'

    styled_df = display_df.style.format({
        'Buy Price': '${:,.2f}',
        'Current': '${:,.2f}',
        'P/L ($)': '${:,.2f}',
        'P/L (%)': '{:.2f}%',
        'Qty': '{:.4f}'
    }).applymap(color_pl, subset=['P/L ($)', 'P/L (%)'])

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("⚙️ Portfolio Management")

    manage_col1, manage_col2 = st.columns(2)

    with manage_col1:
        st.write("**Remove Holdings**")
        if not portfolio_df.empty:
            stocks_to_delete = st.multiselect(
                "Select holdings to remove:",
                options=portfolio_df.index,
                format_func=lambda x: f"{portfolio_df.loc[x, 'Ticker']} ({portfolio_df.loc[x, 'Quantity']:.4f} shares)"
            )
            if st.button("🗑️ Remove Selected"):
                if stocks_to_delete:
                    st.session_state.portfolio = st.session_state.portfolio.drop(stocks_to_delete)
                    st.success("Selected holdings removed!")
                    st.rerun()

    with manage_col2:
        st.write("**Export Data**")
        if st.button("📥 Download Portfolio"):
            csv_data = portfolio_df.to_csv(index=False)
            st.download_button(
                "💾 Download CSV",
                data=csv_data,
                file_name=f"enhanced_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
