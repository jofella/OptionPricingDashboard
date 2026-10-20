import os

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@st.cache_data
def load_dax_index():
    df = pd.read_csv(os.path.join(DATA_DIR, "DAX_index.csv"), parse_dates=["Date"])
    return df.set_index("Date")["Close"]


@st.cache_data
def load_dax_sample(start="2023-02-01", end="2024-01-31"):
    """The 255 DAX closes of the hedging tutorial (course file DAX_sample.csv), dated via DAX_index.csv.

    Falls back to slicing DAX_index.csv if the course file is not present."""
    s = load_dax_index()
    path = os.path.join(DATA_DIR, "DAX_sample.csv")
    if not os.path.exists(path):
        return s.loc[start:end]
    sample = np.loadtxt(path)
    after = s.loc[start:]
    hits = np.where(np.isclose(after.values, sample[0]))[0]
    if hits.size and hits[0] + len(sample) <= len(after):
        dates = after.index[hits[0]: hits[0] + len(sample)]
    else:
        dates = pd.bdate_range(start, periods=len(sample))
    return pd.Series(sample, index=dates, name="Close")


def sp500_options_path():
    return os.path.join(DATA_DIR, "options_SP500.csv")


@st.cache_data
def load_sp500_options():
    """Course data set (columns Strike, Midpoint, Underlying, TTM, Interest) if it was placed in data/."""
    path = sp500_options_path()
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def get_log_returns(prices):
    return np.diff(np.log(np.asarray(prices, dtype=float)))
