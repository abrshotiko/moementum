import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import time

# --- Page setup ---
st.set_page_config(page_title="MOEmentum", page_icon="🚀", layout="wide")

COMMODITY_TICKERS = {
    "WTI Crude Oil": "CL=F", "Brent Crude Oil": "BZ=F", "Natural Gas": "NG=F",
    "RBOB Gasoline": "RB=F", "Heating Oil": "HO=F", "Gold": "GC=F",
    "Silver": "SI=F", "Copper": "HG=F", "Platinum": "PL=F",
    "Palladium": "PA=F", "Corn": "ZC=F", "Soybeans": "ZS=F",
    "Wheat": "ZW=F", "KC Wheat": "KE=F", "Oats": "ZO=F",
    "Rough Rice": "ZR=F", "Coffee": "KC=F", "Sugar": "SB=F",
    "Cocoa": "CC=F", "Cotton": "CT=F", "Orange Juice": "OJ=F",
    "Lumber": "LBS=F", "Live Cattle": "LE=F", "Feeder Cattle": "GF=F",
    "Lean Hogs": "HE=F",
}

# ---------- Safe data fetching & normalization ----------

def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with DatetimeIndex and ['Close','Volume'] if available."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # If MultiIndex columns (e.g., multiple tickers), take first level
    if isinstance(out.columns, pd.MultiIndex):
        # Try to select ('Close', first) and ('Volume', first)
        try:
            first_symbol = out.columns.levels[1][0] if out.columns.nlevels > 1 else None
            cols = {}
            if ("Close", first_symbol) in out.columns:
                cols["Close"] = out[("Close", first_symbol)]
            elif "Close" in out.columns.get_level_values(0):
                cols["Close"] = out["Close"].iloc[:, 0]
            if ("Volume", first_symbol) in out.columns:
                cols["Volume"] = out[("Volume", first_symbol)]
            elif "Volume" in out.columns.get_level_values(0):
                cols["Volume"] = out["Volume"].iloc[:, 0]
            out = pd.DataFrame(cols)
        except Exception:
            # Fallback: flatten columns and try standard names
            out.columns = ["_".join([str(x) for x in c if x != ""]) for c in out.columns]
            close_like = [c for c in out.columns if c.lower().endswith("close")]
            vol_like = [c for c in out.columns if c.lower().endswith("volume")]
            cols = {}
            if close_like:
                cols["Close"] = out[close_like[0]]
            if vol_like:
                cols["Volume"] = out[vol_like[0]]
            out = pd.DataFrame(cols)

    # Standard single-index case: ensure Close exists (Adj Close fallback)
    if "Close" not in out.columns and "Adj Close" in out.columns:
        out["Close"] = out["Adj Close"]

    # Keep only what we need
    keep = [c for c in ["Close", "Volume"] if c in out.columns]
    out = out[keep]

    # Clean types
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Close"])
    out = out.sort_index()
    return out

@st.cache_data(ttl=1800, show_spinner=False)
def load_history(symbol: str, period: str, retries: int = 3, sleep: float = 0.8) -> pd.DataFrame:
    """
    Robust fetch:
    1) Try Ticker().history (stable for futures)
    2) Fallback to yf.download
    Retries to dodge transient empty responses / 429s.
    """
    symbol = symbol.strip()
    last_exc = None
    for _ in range(retries):
        try:
            df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
            df = _normalize_history(df)
            if not df.empty:
                return df
        except Exception as e:
            last_exc = e
        time.sleep(sleep)

    for _ in range(retries):
        try:
            df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True, threads=False)
            df = _normalize_history(df)
            if not df.empty:
                return df
        except Exception as e:
            last_exc = e
        time.sleep(sleep)

    # Return empty with a hint column for debugging in UI if needed
    return pd.DataFrame()

def first_col(obj):
    """If Yahoo gives multiple contracts/cols, pick the first; else pass through."""
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return pd.Series(dtype=float)
        return obj.iloc[:, 0]
    return obj

# ---------- App ----------

def main():
    st.title("Moementum")

    top1, top2 = st.columns([3, 1])
    with top1:
        commodity = st.selectbox("Commodity", list(COMMODITY_TICKERS))
    with top2:
        period = st.select_slider("Lookback", ["1y", "2y", "3y", "4y", "5y"], value="3y")

    symbol = COMMODITY_TICKERS[commodity]
    history = load_history(symbol, period)

    # Guard: no data
    if history is None or history.empty:
        st.error(
            f"No price data returned for **{commodity} ({symbol})**.\n"
            "Try a different lookback, check the symbol, or retry later."
        )
        st.stop()

    # Safe extraction
    close_series = None
    volume_series = None

    if "Close" in history.columns:
        close_series = first_col(history[["Close"]] if isinstance(history["Close"], pd.Series) else history["Close"])
    else:
        # Shouldn't happen after normalization, but guard anyway
        st.error("Close price column missing after fetch. Try another symbol/period.")
        st.stop()

    if "Volume" in history.columns:
        volume_series = first_col(history[["Volume"]] if isinstance(history["Volume"], pd.Series) else history["Volume"])
    else:
        volume_series = pd.Series(index=history.index, dtype=float)

    if close_series is None or close_series.empty:
        st.error(f"No close prices found for **{commodity} ({symbol})** in the selected period.")
        st.stop()

    # Latest close (safe)
    last_close_val = float(close_series.dropna().iloc[-1])

    # Header & snapshot
    st.subheader(f"{commodity} ({symbol})")
    st.metric("Last Close", f"${last_close_val:,.2f}")
    last_idx = close_series.dropna().index[-1]
    try:
        st.caption(f"Last updated {pd.to_datetime(last_idx):%Y-%m-%d}")
    except Exception:
        st.caption(f"Last updated: {last_idx}")

    # Chart data
    chart_data = pd.DataFrame({
        "Date": close_series.index,
        "Close": close_series.values,
        "Volume": (volume_series if volume_series is not None else pd.Series(index=close_series.index, dtype=float)).fillna(0).values,
    }).dropna(subset=["Close"])

    price_vol_chart = alt.layer(
        alt.Chart(chart_data).mark_line(strokeWidth=2.2).encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Close:Q", axis=alt.Axis(title="Price ($)"))
        ),
        alt.Chart(chart_data).mark_bar(opacity=0.45).encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Volume:Q", axis=alt.Axis(title="Volume", orient="right"))
        ),
    ).resolve_scale(y="independent").properties(height=400)

    st.altair_chart(price_vol_chart, use_container_width=True)

    # ======= MOEMENTUM SECTION =======
    st.subheader("Momentum Windows")
    c1, c2 = st.columns(2)
    with c1:
        long_term_days = st.slider("Long-Term Window (trading days)", 63, 252, 126, step=5)
    with c2:
        short_term_days = st.slider("Short-Term Window (trading days)", 3, 20, 5, step=1)

    df = pd.DataFrame(index=close_series.index)
    df["Close"] = close_series.astype(float)
    df["LogReturn"] = np.log1p(df["Close"].pct_change())

    # Shift to avoid using today's info in today's signal
    df["TSMOM_long"]  = df["LogReturn"].shift(1).rolling(long_term_days).sum()
    df["TSMOM_short"] = df["LogReturn"].shift(1).rolling(short_term_days).sum()
    df["TSMOM_long_smooth"]  = df["TSMOM_long"].ewm(span=5, adjust=False).mean()
    df["TSMOM_short_smooth"] = df["TSMOM_short"].ewm(span=3, adjust=False).mean()

    # Signals chart
    df_reset = df.reset_index().rename(columns={"index": "Date"})
    df_sig = pd.DataFrame({
        "Date": df_reset["Date"],
        "Long_Momentum": df_reset["TSMOM_long_smooth"],
        "Short_Momentum": df_reset["TSMOM_short_smooth"],
    })

    df_long = df_sig[["Date", "Long_Momentum"]].rename(columns={"Long_Momentum": "Value"}).assign(Signal="Long-Term Momentum")
    df_short = df_sig[["Date", "Short_Momentum"]].rename(columns={"Short_Momentum": "Value"}).assign(Signal="Short-Term Momentum")
    df_plot = pd.concat([df_long, df_short], ignore_index=True).dropna(subset=["Value"])

    sig_chart = (
        alt.Chart(df_plot)
        .mark_line()
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %y", labelAngle=-30)),
            y=alt.Y("Value:Q", title="Smoothed Momentum Signals"),
            color=alt.Color("Signal:N", title="Signal Type"),
        )
        .properties(height=280)
    )
    st.altair_chart(sig_chart, use_container_width=True)

    # ======= Combined Momentum & Sizing =======
    st.subheader("Combined Momentum & Position")
    with st.expander("Settings", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            long_weight = st.slider("Long-term weight", 0.0, 3.0, 1.0, 0.1)
        with c2:
            short_weight = st.slider("Short-term weight", 0.0, 3.0, 1.0, 0.1)
        with c3:
            vol_lookback = st.slider("Vol seed lookback (days)", 10, 252, 20, 1)
        with c4:
            target_vol_pct = st.number_input("Target annualized vol (%)", 0.0, 100.0, 10.0, 0.5)
        lambda_ = st.slider("EWMA decay λ (0.80–0.995) !Higher smoother as older days are remembered more", 0.80, 0.995, 0.94, 0.005)
        max_leverage = st.slider("Max Leverage", 0.5, 10.0, 5.0, 0.5)

    df["Score_long"]  = long_weight  * np.sign(df["TSMOM_long_smooth"])
    df["Score_short"] = short_weight * (-np.sign(df["TSMOM_short_smooth"]))
    df["CombinedScore"] = df["Score_long"] + df["Score_short"]
    df["NetDirection"] = np.sign(df["CombinedScore"]).astype(float)

    # EWMA vol targeting
    df["DailyReturn"] = df["Close"].pct_change()
    alpha = 1.0 - lambda_
    sq_ret_lag = df["DailyReturn"].pow(2).shift(1)
    seed_var = df["DailyReturn"].rolling(vol_lookback).var().shift(1)
    ewma_var = sq_ret_lag.ewm(alpha=alpha, adjust=False).mean()
    df["EWMA_Var"] = ewma_var.where(~ewma_var.isna(), seed_var)
    df["ModelVol"] = np.sqrt(df["EWMA_Var"]) * np.sqrt(252)

    # Position sizing (guard inf/NaN)
    position_scaler = (target_vol_pct / 100.0) / df["ModelVol"]
    position_scaler = position_scaler.replace([np.inf, -np.inf], np.nan).clip(upper=max_leverage).fillna(0.0)
    df["SizedPosition"] = df["NetDirection"] * position_scaler

    # ======= PnL (%) zero-centered =======
    st.subheader("PnL (%, zero-centered)")
    pos = df["SizedPosition"].shift(1).fillna(0)
    pnl = pos * df["DailyReturn"]
    cum_pnl = (1 + pnl.fillna(0)).cumprod() - 1
    cum_pnl_pct = (cum_pnl * 100).fillna(0)

    pnl_df = pd.DataFrame({"Date": df.index, "PnL_pct": cum_pnl_pct})
    if len(pnl_df):
        max_abs = float(np.nanmax(np.abs(pnl_df["PnL_pct"].values))) or 1.0
    else:
        max_abs = 1.0
    y_dom = [-max_abs, max_abs]

    pnl_chart = (
        alt.Chart(pnl_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %y", labelAngle=-30)),
            y=alt.Y("PnL_pct:Q", title="Cumulative PnL (%)", scale=alt.Scale(domain=y_dom)),
        )
        .properties(height=300)
    )
    zero_rule = alt.Chart(pnl_df).mark_rule().encode(y=alt.datum(0))
    st.altair_chart(pnl_chart + zero_rule, use_container_width=True)

if __name__ == "__main__":
    main()
