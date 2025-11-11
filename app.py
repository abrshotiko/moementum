import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf
import time

st.set_page_config(page_title="MOEmentum", page_icon="🚀", layout="wide")

TICKS = {
    "WTI Crude Oil":"CL=F","Brent Crude Oil":"BZ=F","Natural Gas":"NG=F","RBOB Gasoline":"RB=F",
    "Heating Oil":"HO=F","Gold":"GC=F","Silver":"SI=F","Copper":"HG=F","Platinum":"PL=F",
    "Palladium":"PA=F","Corn":"ZC=F","Soybeans":"ZS=F","Wheat":"ZW=F","KC Wheat":"KE=F",
    "Oats":"ZO=F","Rough Rice":"ZR=F","Coffee":"KC=F","Sugar":"SB=F","Cocoa":"CC=F",
    "Cotton":"CT=F","Orange Juice":"OJ=F","Lumber":"LBS=F","Live Cattle":"LE=F",
    "Feeder Cattle":"GF=F","Lean Hogs":"HE=F"
}

def clean(df):
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        try:
            c = x.columns.levels[1][0]
            out = {}
            if ("Close",c) in x: out["Close"] = x[("Close",c)]
            if ("Volume",c) in x: out["Volume"] = x[("Volume",c)]
            x = pd.DataFrame(out)
        except:
            cols = ["_".join([str(j) for j in i if j]) for i in x.columns]
            x.columns = cols
            close_cols = [i for i in cols if "close" in i.lower()]
            vol_cols = [i for i in cols if "volume" in i.lower()]
            out = {}
            if close_cols: out["Close"] = x[close_cols[0]]
            if vol_cols: out["Volume"] = x[vol_cols[0]]
            x = pd.DataFrame(out)
    if "Close" not in x and "Adj Close" in x:
        x["Close"] = x["Adj Close"]
    keep = [i for i in ["Close","Volume"] if i in x]
    x = x[keep].copy()
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["Close"])
    return x.sort_index()

@st.cache_data(ttl=1800)
def fetch(sym, per):
    for _ in range(2):
        try:
            d = yf.Ticker(sym).history(period=per, interval="1d", auto_adjust=True)
            d = clean(d)
            if not d.empty:
                return d
        except:
            pass
        time.sleep(0.3)
    try:
        d = yf.download(sym, period=per, interval="1d", progress=False, auto_adjust=True, threads=False)
        d = clean(d)
        return d
    except:
        return pd.DataFrame()

def first(o):
    if isinstance(o, pd.DataFrame):
        if o.shape[1] == 0:
            return pd.Series(dtype=float)
        return o.iloc[:,0]
    return o

def main():
    st.title("Moementum")

    c1, c2 = st.columns([3,1])
    with c1:
        name = st.selectbox("Commodity", list(TICKS))
    with c2:
        per = st.select_slider("Lookback", ["1y","2y","3y","4y","5y"], "3y")

    sym = TICKS[name]
    his = fetch(sym, per)
    if his.empty:
        st.error("No data for this commodity.")
        st.stop()

    close = first(his["Close"])
    vol = his["Volume"] if "Volume" in his else pd.Series(index=his.index)
    if close.empty:
        st.error("Missing close prices.")
        st.stop()

    last = float(close.dropna().iloc[-1])

    st.subheader(f"{name} ({sym})")
    st.metric("Last Close", f"${last:,.2f}")

    try:
        st.caption(f"Last updated {pd.to_datetime(close.index[-1]):%Y-%m-%d}")
    except:
        st.caption(str(close.index[-1]))

    dfc = pd.DataFrame({
        "Date": close.index,
        "Close": close.values,
        "Volume": vol.fillna(0).values
    })

    pchart = alt.layer(
        alt.Chart(dfc).mark_line().encode(x="Date:T", y="Close:Q"),
        alt.Chart(dfc).mark_bar(opacity=0.4).encode(x="Date:T", y="Volume:Q")
    ).resolve_scale(y="independent").properties(height=350)

    st.altair_chart(pchart, use_container_width=True)

    st.subheader("Momentum Windows")
    k1, k2 = st.columns(2)
    with k1:
        L = st.slider("Long Window", 63, 252, 126, 5)
    with k2:
        S = st.slider("Short Window", 3, 20, 5, 1)

    df = pd.DataFrame(index=close.index)
    df["Close"] = close.astype(float)
    df["lr"] = np.log1p(df["Close"].pct_change())
    df["L"] = df["lr"].shift(1).rolling(L).sum()
    df["S"] = df["lr"].shift(1).rolling(S).sum()
    df["Ls"] = df["L"].ewm(span=5).mean()
    df["Ss"] = df["S"].ewm(span=3).mean()

    d2 = df.reset_index().rename(columns={"index":"Date"})
    d_long = pd.DataFrame({"Date": d2["Date"], "Value": d2["Ls"], "Signal":"Long"})
    d_short = pd.DataFrame({"Date": d2["Date"], "Value": d2["Ss"], "Signal":"Short"})
    sig = pd.concat([d_long, d_short]).dropna()

    st.altair_chart(
        alt.Chart(sig).mark_line().encode(
            x=alt.X("Date:T", axis=alt.Axis(labelAngle=-30)),
            y="Value:Q", color="Signal:N"
        ).properties(height=250),
        use_container_width=True
    )

    st.subheader("Combined Momentum & Position")

    with st.expander("Settings", True):
        a,b,c,d = st.columns(4)
        with a: wL = st.slider("Long weight",0.0,3.0,1.0,0.1)
        with b: wS = st.slider("Short weight",0.0,3.0,1.0,0.1)
        with c: vb = st.slider("Vol Lookback",10,252,20,1)
        with d: tv = st.number_input("Target vol %",0.0,100.0,10.0,0.5)
        lam = st.slider("Lambda",0.80,0.995,0.94,0.005)
        lev = st.slider("Max Leverage",0.5,10.0,5.0,0.5)

    df["Score"] = wL*np.sign(df["Ls"]) - wS*np.sign(df["Ss"])
    df["Dir"] = np.sign(df["Score"])
    df["r"] = df["Close"].pct_change()
    a = 1 - lam
    var = df["r"].pow(2).shift(1).ewm(alpha=a).mean()
    seed = df["r"].rolling(vb).var().shift(1)
    df["vol"] = np.sqrt(var.where(~var.isna(), seed)) * np.sqrt(252)

    sc = (tv/100) / df["vol"]
    sc = sc.replace([np.inf,-np.inf], np.nan).clip(upper=lev).fillna(0)
    df["pos"] = df["Dir"] * sc

    st.subheader("PnL (%)")
    pnl = df["pos"].shift(1).fillna(0) * df["r"]
    cp = (1 + pnl.fillna(0)).cumprod() - 1
    pnl_df = pd.DataFrame({"Date": df.index, "p": cp*100})

    if len(pnl_df):
        m = float(np.nanmax(np.abs(pnl_df["p"])))
    else:
        m = 1.0

    c_plot = alt.Chart(pnl_df).mark_line().encode(
        x=alt.X("Date:T", axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("p:Q", scale=alt.Scale(domain=[-m,m]))
    ).properties(height=280)

    zero = alt.Chart(pnl_df).mark_rule().encode(y=alt.datum(0))

    st.altair_chart(c_plot + zero, use_container_width=True)

if __name__ == "__main__":
    main()
