"""
crypto-analysis/dashboard.py
─────────────────────────────
The entire app in one file:
    1. Fetch   — get data from Alpha Vantage API
    2. Clean   — flatten JSON into a tidy table
    3. Calculate — add returns, volatility, drawdown
    4. Display — Streamlit dashboard with 6 charts

Run from the repo root:
    streamlit run crypto-analysis/dashboard.py

Requires a .env file in the repo root:
    ALPHA_VANTAGE_KEY=your_key_here
"""

import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

COINS = ["BTC", "ETH", "SOL", "XRP"]

COLOURS = {
    "BTC": "#F7931A",
    "ETH": "#627EEA",
    "SOL": "#9945FF",
    "XRP": "#00AAE4",
}

THEME    = "plotly_dark"
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "crypto.csv")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FETCH
# Calls Alpha Vantage and returns raw JSON for one coin.
# ─────────────────────────────────────────────────────────────────────────────

def fetch_coin(symbol: str) -> dict:
    api_key = os.getenv("ALPHA_VANTAGE_KEY")
    if not api_key:
        raise EnvironmentError(
            "ALPHA_VANTAGE_KEY not found. "
            "Add it to a .env file in the repo root."
        )

    r = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function":   "DIGITAL_CURRENCY_DAILY",
            "symbol":     symbol,
            "market":     "USD",
            "apikey":     api_key,
            "outputsize": "full",
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    # Alpha Vantage always returns HTTP 200 — errors are hidden inside the JSON
    if "Error Message" in data:
        raise ValueError(f"API error for {symbol}: {data['Error Message']}")
    if "Note" in data or "Information" in data:
        raise RuntimeError(
            "Rate limit reached. Free tier = 25 requests/day. "
            "Wait and try again tomorrow."
        )

    return data


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CLEAN
# Flattens the raw nested JSON into a tidy flat DataFrame.
#
# Raw JSON looks like:
#   { "Time Series (Digital Currency Daily)": {
#       "2024-01-15": {
#           "1a. open (USD)": "42000.00",
#           "4a. close (USD)": "43000.00", ...
#       }, ...
#   }}
#
# Clean output:
#   date | symbol | open | high | low | close | volume
# ─────────────────────────────────────────────────────────────────────────────

def clean_coin(raw: dict, symbol: str) -> pd.DataFrame:
    key = "Time Series (Digital Currency Daily)"
    if key not in raw:
        raise KeyError(
            f"Expected key not found for {symbol}. "
            f"Keys in response: {list(raw.keys())}"
        )

    rows = []
    for date_str, v in raw[key].items():
        rows.append({
            "date":   date_str,
            "symbol": symbol,
            # .get() with a fallback handles both old and new AV field names
            "open":   float(v.get("1a. open (USD)",  v.get("1. open",  0))),
            "high":   float(v.get("2a. high (USD)",  v.get("2. high",  0))),
            "low":    float(v.get("3a. low (USD)",   v.get("3. low",   0))),
            "close":  float(v.get("4a. close (USD)", v.get("4. close", 0))),
            "volume": float(v.get("5. volume", 0)),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CALCULATE
# Adds four analytical columns to the combined DataFrame.
# All maths is done per-coin using groupby so coins never bleed into each other.
# ─────────────────────────────────────────────────────────────────────────────

def calculate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Daily return: % change in close price day-over-day
    # e.g. 0.05 = price went up 5% today
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()

    # Volatility: rolling 30-day std dev of returns, annualised
    # Multiply by √365 to convert daily → yearly (standard finance convention)
    df["volatility"] = (
        df.groupby("symbol")["daily_return"]
        .transform(lambda x: x.rolling(30).std() * np.sqrt(365))
    )

    # Normalised price: rebase each coin to 100 at its first date
    # Lets us compare BTC ($40k) and XRP ($0.50) on the same chart fairly
    def add_norm(group):
        group = group.copy()
        group["price_norm"] = (group["close"] / group["close"].iloc[0]) * 100
        return group

    df = df.groupby("symbol", group_keys=False).apply(add_norm)

    # Drawdown: how far below the running all-time high is the current price?
    # 0.0 = at peak right now. -0.5 = 50% below peak.
    def add_drawdown(group):
        group = group.copy()
        peak  = group["close"].cummax()
        group["drawdown"] = (group["close"] - peak) / peak
        return group

    df = df.groupby("symbol", group_keys=False).apply(add_drawdown)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FETCH + SAVE  (called on first run or when user clicks Refresh)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_save() -> pd.DataFrame:
    """
    Fetch all coins from the API, clean, calculate, and save to CSV.
    Shows a progress bar in the Streamlit UI while running.
    """
    frames   = []
    progress = st.progress(0, text="Starting fetch...")

    for i, symbol in enumerate(COINS):
        progress.progress(i / len(COINS), text=f"Fetching {symbol}...")
        raw = fetch_coin(symbol)
        df  = clean_coin(raw, symbol)
        frames.append(df)
        # Small sleep to be polite to the free API tier
        if i < len(COINS) - 1:
            time.sleep(1)

    progress.progress(0.95, text="Calculating metrics...")
    combined = pd.concat(frames, ignore_index=True)
    combined = calculate(combined)

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    combined.to_csv(CSV_PATH, index=False)
    progress.empty()

    return combined


@st.cache_data(show_spinner=False)
def load_csv() -> pd.DataFrame:
    """
    Load from saved CSV. Cached so Streamlit doesn't re-read the file
    on every single interaction — it only re-reads when the cache is cleared.
    """
    return pd.read_csv(CSV_PATH, parse_dates=["date"])


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + STYLES
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Crypto Dashboard",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  { font-family: 'DM Mono', monospace; }

.stApp {
    background: #08080f;
    background-image:
        radial-gradient(ellipse at 15% 15%, rgba(247,147,26,0.05) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 80%, rgba(98,126,234,0.05) 0%, transparent 55%);
}

/* Page title */
.title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(120deg, #F7931A, #FFD700, #627EEA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.1rem;
}
.subtitle {
    font-size: 0.75rem;
    color: #44445a;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.8rem;
}

/* KPI cards */
.kpi {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
}
.kpi-label { font-size: 0.65rem; color: #44445a;
             letter-spacing: 0.12em; text-transform: uppercase; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 1.4rem;
             font-weight: 700; color: #e8e8ff; }
.kpi-sub   { font-size: 0.72rem; color: #666680; margin-top: 0.15rem; }
.up   { color: #26a69a; }
.down { color: #ef5350; }

/* Section headers */
.section {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem; font-weight: 700;
    color: #ccccdd; letter-spacing: 0.06em; text-transform: uppercase;
    border-left: 3px solid #F7931A; padding-left: 0.6rem;
    margin: 1.5rem 0 0.6rem;
}

/* Explanation notes below charts */
.note {
    background: rgba(247,147,26,0.04);
    border: 1px solid rgba(247,147,26,0.12);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.78rem;
    color: #888899;
    line-height: 1.6;
    margin-top: 0.5rem;
}

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🪙 Crypto Dashboard")
    st.caption("BTC · ETH · SOL · XRP")
    st.divider()

    # Data refresh
    st.markdown("**Data**")
    if st.button("🔄 Fetch fresh data", use_container_width=True,
                 help="Free tier = 25 requests/day. Only click when you need new data."):
        st.cache_data.clear()
        try:
            fetch_and_save()
        except Exception as e:
            st.error(str(e))
        st.rerun()

    if os.path.exists(CSV_PATH):
        mtime = os.path.getmtime(CSV_PATH)
        ts    = pd.Timestamp(mtime, unit="s").strftime("%d %b %Y  %H:%M")
        st.caption(f"Last updated: {ts}")
    else:
        st.caption("No data yet — click Fetch above")

    st.divider()

    # Coin selector
    st.markdown("**Coins**")
    selected = [c for c in COINS if st.checkbox(c, value=True, key=f"cb_{c}")]
    if not selected:
        st.warning("Select at least one coin")
        selected = ["BTC"]

    st.divider()

    # Date range
    st.markdown("**Date range**")
    period = st.selectbox(
        "", ["All time", "3 years", "2 years", "1 year", "6 months"],
        index=2, label_visibility="collapsed",
    )

    st.divider()

    # Portfolio weights
    st.markdown("**Portfolio weights**")
    st.caption("Sliders don't need to sum to 100 — they're auto-normalised")

    raw_w = {c: st.slider(c, 0, 100, 25, 5, key=f"w_{c}") for c in COINS}
    total_w = sum(raw_w.values()) or 1
    alloc   = {c: raw_w[c] / total_w for c in COINS}

    initial = st.number_input(
        "Starting investment ($)", 100, 1_000_000, 10_000, 1000
    )

    st.divider()
    st.caption("Source: Alpha Vantage")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(CSV_PATH):
    st.markdown('<div class="title">🪙 Crypto Dashboard</div>', unsafe_allow_html=True)
    st.info("No saved data found. Fetching from API for the first time (~30 seconds)...")
    try:
        fetch_and_save()
        st.rerun()
    except Exception as e:
        st.error(str(e))
        st.stop()

try:
    df_all = load_csv()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# FILTER BY COIN + DATE
# ─────────────────────────────────────────────────────────────────────────────

offsets = {
    "All time": None,
    "3 years":  pd.DateOffset(years=3),
    "2 years":  pd.DateOffset(years=2),
    "1 year":   pd.DateOffset(years=1),
    "6 months": pd.DateOffset(months=6),
}

df = df_all[df_all["symbol"].isin(selected)].copy()
if offsets[period]:
    df = df[df["date"] >= pd.Timestamp.now() - offsets[period]]

date_from = df["date"].min().strftime("%b %Y")
date_to   = df["date"].max().strftime("%b %Y")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="title">🪙 Crypto Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">'
    f'{" · ".join(selected)}&nbsp;&nbsp;·&nbsp;&nbsp;{date_from} → {date_to}'
    f'</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS — current price + key stats per coin
# ─────────────────────────────────────────────────────────────────────────────

latest = df.sort_values("date").groupby("symbol").tail(1).set_index("symbol")
cols   = st.columns(len(selected))

for i, coin in enumerate(selected):
    if coin not in latest.index:
        continue
    row   = latest.loc[coin]
    price = row["close"]
    ret   = row["daily_return"]
    dd    = row["drawdown"]
    pnorm = row["price_norm"] - 100      # % gain/loss since start of period
    col   = COLOURS.get(coin, "#888")

    with cols[i]:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label" style="color:{col}">● {coin}</div>
            <div class="kpi-value">${price:,.2f}</div>
            <div class="kpi-sub {'up' if ret >= 0 else 'down'}">
                {'+'if ret>=0 else ''}{ret:.2%} today
            </div>
            <div class="kpi-sub {'up' if pnorm >= 0 else 'down'}">
                {'+'if pnorm>=0 else ''}{pnorm:.1f}% since {date_from}
            </div>
            <div class="kpi-sub down">{dd:.1%} from peak</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — PRICE TREND
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section">📈 Price Trend</div>', unsafe_allow_html=True)

normalise = st.toggle(
    "Normalise to 100",
    value=True,
    help="Rebase all coins to 100 at start of period — fair comparison regardless of USD price",
)

fig = go.Figure()
for coin in selected:
    s = df[df["symbol"] == coin]
    y = s["price_norm"] if normalise else s["close"]
    fig.add_trace(go.Scatter(
        x=s["date"], y=y, name=coin, mode="lines",
        line=dict(color=COLOURS.get(coin), width=2),
    ))

fig.update_layout(
    template=THEME, hovermode="x unified",
    yaxis_title="Price (rebased to 100)" if normalise else "Price (USD)",
    xaxis_title="", margin=dict(t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    '<div class="note">💡 <b>Normalised:</b> all coins start at 100. '
    '200 = doubled. 50 = halved. This lets you compare BTC ($40k) and XRP ($0.50) '
    'on the same scale. Toggle off to see raw USD prices.</div>',
    unsafe_allow_html=True,
)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section">🎲 Volatility</div>', unsafe_allow_html=True)

fig = go.Figure()
for coin in selected:
    s = df[df["symbol"] == coin].dropna(subset=["volatility"])
    fig.add_trace(go.Scatter(
        x=s["date"], y=s["volatility"], name=coin, mode="lines",
        line=dict(color=COLOURS.get(coin), width=2),
    ))

fig.update_layout(
    template=THEME, hovermode="x unified",
    yaxis_title="Annualised Volatility", yaxis_tickformat=".0%",
    xaxis_title="", margin=dict(t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    '<div class="note">💡 Rolling 30-day volatility, scaled to a yearly number. '
    'Spikes = periods of panic or euphoria. '
    'Higher = wilder price swings. Lower = calmer market.</div>',
    unsafe_allow_html=True,
)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — DRAWDOWN
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section">📉 Drawdown from Peak</div>', unsafe_allow_html=True)

fig = go.Figure()
for coin in selected:
    s = df[df["symbol"] == coin].dropna(subset=["drawdown"])
    fig.add_trace(go.Scatter(
        x=s["date"], y=s["drawdown"], name=coin, mode="lines",
        line=dict(color=COLOURS.get(coin), width=1.5),
        fill="tozeroy", opacity=0.8,
    ))

fig.update_layout(
    template=THEME, hovermode="x unified",
    yaxis_title="Drawdown", yaxis_tickformat=".0%",
    xaxis_title="", margin=dict(t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# Worst drawdown summary
worst = df.groupby("symbol")["drawdown"].min()
dd_cols = st.columns(len(selected))
for i, coin in enumerate(selected):
    if coin not in worst.index:
        continue
    with dd_cols[i]:
        st.markdown(
            f'<span style="color:{COLOURS.get(coin)}">● {coin}</span> '
            f'<span style="color:#ef5350"> worst: {worst[coin]:.1%}</span>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="note">💡 0% = at all-time high right now. '
    '-80% = 80% below the peak. The shaded area shows how long each downturn lasted '
    'and how deep it went.</div>',
    unsafe_allow_html=True,
)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — RISK VS RETURN
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section">🎯 Risk vs Return</div>', unsafe_allow_html=True)

fig = go.Figure()
for coin in selected:
    s          = df[df["symbol"] == coin].dropna(subset=["daily_return"])
    ann_return = s["daily_return"].mean() * 365
    ann_vol    = s["daily_return"].std()  * np.sqrt(365)
    fig.add_trace(go.Scatter(
        x=[ann_vol], y=[ann_return],
        mode="markers+text", name=coin,
        text=[coin], textposition="top center",
        marker=dict(
            color=COLOURS.get(coin), size=18,
            line=dict(width=2, color="white"),
        ),
        hovertemplate=(
            f"<b>{coin}</b><br>"
            f"Return: {ann_return:.1%}<br>"
            f"Risk: {ann_vol:.1%}<extra></extra>"
        ),
    ))

fig.update_layout(
    template=THEME, showlegend=False,
    xaxis_title="Risk — Annualised Volatility", xaxis_tickformat=".0%",
    yaxis_title="Return — Annualised",          yaxis_tickformat=".0%",
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    '<div class="note">💡 Top-left = ideal (high return, low risk). '
    'Bottom-right = worst (low return, high risk). '
    'Each dot is one coin. The further top-left, the better the risk-adjusted return.</div>',
    unsafe_allow_html=True,
)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 — CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section">🔗 Correlation</div>', unsafe_allow_html=True)

if len(selected) < 2:
    st.info("Select at least 2 coins in the sidebar to see correlation.")
else:
    pivot = df.pivot_table(index="date", columns="symbol", values="daily_return")
    corr  = pivot.corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdYlGn", zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text:.2f}",
        textfont=dict(size=16),
        hovertemplate="<b>%{x} vs %{y}</b><br>Correlation: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        template=THEME, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        '<div class="note">💡 1.0 = always move together. 0.0 = independent. '
        'In crypto, most coins are highly correlated because they all react to the same '
        'macro news — Fed decisions, regulation, market sentiment.</div>',
        unsafe_allow_html=True,
    )

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# CHART 6 — PORTFOLIO SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section">💼 Portfolio Simulator</div>', unsafe_allow_html=True)
st.caption(f"Starting with ${initial:,.0f} · weights set in the sidebar")

# Build a price table: one column per coin, one row per date
# Only keep dates where ALL selected coins have data
pivot = (
    df.pivot_table(index="date", columns="symbol", values="close")
    [selected]
    .dropna()
)

# Normalise each coin to 1.0 at the start
norm = pivot / pivot.iloc[0]

# Dollar value of each coin's slice of the portfolio
fig = go.Figure()

for coin in selected:
    weight     = alloc.get(coin, 0)
    coin_value = norm[coin] * (weight * initial)
    fig.add_trace(go.Scatter(
        x=norm.index, y=coin_value,
        name=f"{coin} ({weight:.0%})", mode="lines",
        line=dict(color=COLOURS.get(coin), width=1.5, dash="dot"),
        opacity=0.6,
    ))

# Total portfolio value line
total       = sum(norm[c] * (alloc.get(c, 0) * initial) for c in selected)
final_value = total.iloc[-1]
total_ret   = (final_value / initial) - 1

fig.add_trace(go.Scatter(
    x=norm.index, y=total,
    name="Total", mode="lines",
    line=dict(color="white", width=3),
))
fig.add_hline(
    y=initial, line_dash="dash", line_color="grey", opacity=0.4,
    annotation_text=f"Start ${initial:,.0f}",
    annotation_position="bottom right",
)
fig.update_layout(
    template=THEME, hovermode="x unified",
    yaxis_title="Portfolio Value (USD)", yaxis_tickprefix="$",
    xaxis_title="", margin=dict(t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# Final value summary per coin
summary_cols = st.columns(len(selected) + 1)
for i, coin in enumerate(selected):
    w         = alloc.get(coin, 0)
    start_val = w * initial
    end_val   = norm[coin].iloc[-1] * start_val
    gain      = end_val - start_val
    col       = COLOURS.get(coin, "#888")
    with summary_cols[i]:
        st.markdown(
            f'<div style="text-align:center">'
            f'<div style="color:{col};font-size:0.7rem">● {coin} ({w:.0%})</div>'
            f'<div style="font-family:Syne,sans-serif;font-size:1.1rem;color:#eeeeff">'
            f'${end_val:,.0f}</div>'
            f'<div style="font-size:0.72rem;'
            f'color:{"#26a69a" if gain>=0 else "#ef5350"}">'
            f'{"+"if gain>=0 else ""}{gain:,.0f} ({(end_val/start_val-1):+.1%})</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

with summary_cols[-1]:
    st.markdown(
        f'<div style="text-align:center">'
        f'<div style="color:#fff;font-size:0.7rem">● TOTAL</div>'
        f'<div style="font-family:Syne,sans-serif;font-size:1.1rem;color:#fff">'
        f'${final_value:,.0f}</div>'
        f'<div style="font-size:0.72rem;'
        f'color:{"#26a69a" if total_ret>=0 else "#ef5350"}">'
        f'{total_ret:+.1%}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="note">⚠️ This is historical backtesting only. '
    'Past performance does not predict future results. Not financial advice.</div>',
    unsafe_allow_html=True,
)