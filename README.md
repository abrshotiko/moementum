Moementum 

🚀 Visit the link to explore the app! 🚀
https://moementum.streamlit.app/

A lightweight Streamlit app for exploring momentum signals in commodity futures and how they translate into positions and PnL.

This was built mainly as a sandbox to experiment with:

momentum at different horizons

volatility targeting

simple, clean backtests

and a UI that makes the logic easy to inspect

Data is pulled from Yahoo Finance using yfinance.

What it does
Pick a commodity

Choose from a set of liquid futures (energy, metals, ags, livestock) and a lookback window between 1 and 5 years.

For each selection, the app shows:

a price and volume chart

the most recent close

the last available date

Momentum signals

The app computes two momentum measures:

Long-term momentum (roughly a few months)

Short-term momentum (roughly a few days to a week)

Both are based on log returns, shifted to avoid lookahead bias, and smoothed using simple EWMAs.

The window lengths are adjustable so it’s easy to see how the signals change across regimes.

Direction and position sizing

The long and short momentum signals are combined into a simple score:

long momentum pushes exposure higher

short momentum can reduce or fade exposure

Position size is then scaled using an EWMA volatility estimate to target a chosen annualized volatility, with a hard cap on leverage.

Nothing is hidden — all parameters are visible and adjustable in the UI.

PnL view

The app shows a basic cumulative PnL (%), computed using:

lagged positions

daily returns

no transaction costs (intentionally omitted to focus on signal behavior)