import altair as alt            # streamlit plotting
import numpy as np            
import pandas as pd
import streamlit as st         
import yfinance as yf    



#page setup streamlit requirement 
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

@st.cache_data(ttl=1800)  # cache for 30 minutes so it doesnt fetch on every load
def load_history(symbol: str, period: str):
    return yf.download(symbol, period=period, interval="1d", progress=False)    #pull daily data for symbol no extra messages

def first_col(obj):# if yahoo for some reason decides to give me data of multiple contracts side by side like different month maturities take the first colum
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj  

def main():
    st.title("Moementum")  #title


    top1, top2 = st.columns([3, 1]) #layout two columns 
    with top1:
        commodity = st.selectbox("Commodity", list(COMMODITY_TICKERS)) #name dropdown 
    with top2:
        period = st.select_slider("Lookback", ["1y", "2y", "3y", "4y", "5y"], value="3y") #lookback slider

    history = load_history(COMMODITY_TICKERS[commodity], period)

    close_series = first_col(history["Close"]) #getting closing price 
    volume_series = first_col(history["Volume"]) if "Volume" in history.columns else pd.Series(index=history.index, dtype=float)#getting volume 
    last_close_val = float(close_series.iloc[-1])  #latest close 

    # section header and snapshot metrics
    st.subheader(f"{commodity} ({COMMODITY_TICKERS[commodity]})")
    st.metric("Last Close", f"${last_close_val:,.2f}")  # headline price
    st.caption(f"Last updated {history.index[-1]:%Y-%m-%d}")  # last bar date

    #making a dataframe so charting is easier
    chart_data = pd.DataFrame({
        "Date": history.index,
        "Close": close_series.values,
        "Volume": volume_series.fillna(0).values,  # guard against NaN in volume
    })

    price_vol_chart = alt.layer(#charting price and volume
        alt.Chart(chart_data).mark_line(strokeWidth=2.2).encode(  #plotting price
            x="Date:T",
            y=alt.Y("Close:Q", axis=alt.Axis(title="Price ($)"))
        ),
        alt.Chart(chart_data).mark_bar(opacity=0.45).encode( #plotting volume 
            x="Date:T",
            y=alt.Y("Volume:Q", axis=alt.Axis(title="Volume", orient="right"))
        ),
    ).resolve_scale(y="independent").properties(height=400)  #adjusting height and y axes scale

    st.altair_chart(price_vol_chart, use_container_width=True)  #render horizzontal page


    st.subheader("Momentum Windows") ##MOEMENTUM SECTION STARTS HERE
    c1, c2 = st.columns(2)  # two sliders side-by-side
    with c1:
        long_term_days = st.slider("Long-Term Window (trading days)", 63, 252, 126, step=5) #long term momentum window slide in trading days 
    with c2:
        short_term_days = st.slider("Short-Term Window (trading days)", 3, 20, 5, step=1) #shortS term momentum window slide in trading days 

    df = pd.DataFrame(index=history.index) #working with original df
    df["Close"] = close_series #closing prices 
    df["LogReturn"] = np.log1p(df["Close"].pct_change())  # r_t = log(P_t / P_{t-1})

    # shift(1) so you dont use todays signal in today momentum
    df["TSMOM_long"]  = df["LogReturn"].shift(1).rolling(long_term_days).sum()   #slow momentum leg
    df["TSMOM_short"] = df["LogReturn"].shift(1).rolling(short_term_days).sum()  #fast leg mean reversion
    df["TSMOM_long_smooth"]  = df["TSMOM_long"].ewm(span=5, adjust=False).mean() #just smoothing with emwa
    df["TSMOM_short_smooth"] = df["TSMOM_short"].ewm(span=3, adjust=False).mean()


    # Signals chart: show both smoothed momentum traces over time
    df_reset = df.reset_index().rename(columns={"index": "Date"})  # to have a 'Date' column for Altair
    # Prepare a long-form dataframe for Altair
    df_sig = pd.DataFrame({
        "Date": df_reset["Date"],
        "Long_Momentum": df_reset["TSMOM_long_smooth"],
        "Short_Momentum": df_reset["TSMOM_short_smooth"],
    })

    df_long = df_sig[["Date", "Long_Momentum"]].rename(
        columns={"Long_Momentum": "Value"}
    ).assign(Signal="Long-Term Momentum")

    df_short = df_sig[["Date", "Short_Momentum"]].rename(
        columns={"Short_Momentum": "Value"}
    ).assign(Signal="Short-Term Momentum")

    df_plot = pd.concat([df_long, df_short])

    sig_chart = (
        alt.Chart(df_plot)
        .mark_line()
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %y", labelAngle=-30)),
            y=alt.Y("Value:Q", title="Smoothed Momentum Signals"),
            color=alt.Color("Signal:N", title="Signal Type")  # auto legend
        )
        .properties(height=280)
    )

    st.altair_chart(sig_chart, use_container_width=True)

    # =========================
    # Combined Momentum & Sizing
    # =========================
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
        max_leverage = st.slider("Max Leverage", 0.5, 10.0, 5.0, 0.5)  # EDIT: define once, used for sizing and PnL

    # Direction from your signals
    df["Score_long"]  = long_weight  * np.sign(df["TSMOM_long_smooth"])
    df["Score_short"] = short_weight * (-np.sign(df["TSMOM_short_smooth"]))
    df["CombinedScore"] = df["Score_long"] + df["Score_short"]
    df["NetDirection"] = np.sign(df["CombinedScore"]).astype(float)  # -1,0,+1

    # ===== EWMA Vol targeting (RiskMetrics-style) =====
    df["DailyReturn"] = df["Close"].pct_change()
    alpha = 1.0 - lambda_
    sq_ret_lag = df["DailyReturn"].pow(2).shift(1)
    seed_var = df["DailyReturn"].rolling(vol_lookback).var().shift(1)

    ewma_var = sq_ret_lag.ewm(alpha=alpha, adjust=False).mean()
    df["EWMA_Var"] = ewma_var.where(~ewma_var.isna(), seed_var)
    df["ModelVol"] = np.sqrt(df["EWMA_Var"]) * np.sqrt(252)

    # Position sizing
    # EDIT: use the same max_leverage everywhere and avoid recomputing later
    position_scaler = (target_vol_pct / 100.0) / df["ModelVol"]
    position_scaler = (
        position_scaler.replace([np.inf, -np.inf], np.nan)
        .clip(upper=max_leverage)
        .fillna(0.0)
    )
    df["SizedPosition"] = df["NetDirection"] * position_scaler  # EDIT: canonical position series reused below

    # =========================
    # PnL (%) centered at zero
    # =========================
    st.subheader("PnL (%, zero-centered)")  # EDIT: title reflects new chart

    # EDIT: cumulative PnL in percent, zero in middle
    pos = df["SizedPosition"].shift(1).fillna(0)  # Use previous day’s position for PnL (no look-ahead)
    pnl = pos * df["DailyReturn"]                 # daily strategy return
    cum_pnl = (1 + pnl.fillna(0)).cumprod() - 1   # cumulative return as decimal
    cum_pnl_pct = cum_pnl * 100                   # convert to percent

    pnl_df = pd.DataFrame({
        "Date": df.index,
        "PnL_pct": cum_pnl_pct
    })

    max_abs = float(np.nanmax(np.abs(cum_pnl_pct.values))) if len(cum_pnl_pct) else 1.0
    y_dom = [-max_abs, max_abs]  # symmetric domain so 0 sits in the middle

    pnl_chart = (
        alt.Chart(pnl_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %y", labelAngle=-30)),
            y=alt.Y("PnL_pct:Q", title="Cumulative PnL (%)", scale=alt.Scale(domain=y_dom))
        )
        .properties(height=300)
    )

    # EDIT: add a horizontal zero line for clarity
    zero_rule = alt.Chart(pnl_df).mark_rule().encode(y=alt.datum(0))

    st.altair_chart(pnl_chart + zero_rule, use_container_width=True)


if __name__ == "__main__": #standart ghibberish 
    main()
