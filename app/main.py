import streamlit as st

st.set_page_config(page_title="CompFinLearn", layout="wide", page_icon="🧮")

# Anchor target for back-to-top link
st.markdown('<div id="top"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧮 CompFinLearn")
    st.caption("Computational Finance · CAU Kiel")
    st.divider()

pages = {
    "": [
        st.Page("pages/home.py", title="Home", icon="🏠", default=True),
    ],
    "Ch. 1 — Binomial Trees": [
        st.Page("pages/1_Binomial_Trees.py", title="CRR Model", icon="🌳", url_path="binomial-trees"),
    ],
    "Ch. 2 — Continuous Models": [
        st.Page("pages/2_1_Black_Scholes.py", title="Black-Scholes Pricing", icon="📐", url_path="black-scholes"),
        st.Page("pages/2_2_Delta_Hedging.py", title="Delta Hedging", icon="🛡️", url_path="delta-hedging"),
        st.Page("pages/2_3_Heston_Model.py", title="Heston Model", icon="🌪️", url_path="heston-model"),
    ],
    "Ch. 3 — Integral Transforms": [
        st.Page("pages/3_1_Laplace_FFT.py", title="Laplace & FFT Pricing", icon="〰️", url_path="laplace-fft"),
        st.Page("pages/3_2_Calibration.py", title="Implied Vol & Calibration", icon="🎯", url_path="calibration"),
    ],
    "Ch. 4 — Monte Carlo": [
        st.Page("pages/4_1_MC_Basics.py", title="Monte Carlo Basics", icon="🎲", url_path="monte-carlo-basics"),
        st.Page("pages/4_2_Random_Numbers.py", title="Random Number Generation", icon="🔢", url_path="random-numbers"),
        st.Page("pages/4_3_Variance_Reduction.py", title="Variance Reduction", icon="📉", url_path="variance-reduction"),
        st.Page("pages/4_4_Sensitivities.py", title="MC Sensitivities", icon="🧭", url_path="mc-sensitivities"),
        st.Page("pages/4_5_SDE_Simulation.py", title="Simulating SDEs", icon="〽️", url_path="sde-simulation"),
        st.Page("pages/4_6_Longstaff_Schwartz.py", title="Longstaff-Schwartz", icon="⏱️", url_path="longstaff-schwartz"),
    ],
    "Ch. 5 — Finite Differences": [
        st.Page("pages/5_Finite_Differences.py", title="Finite Differences", icon="🔲", url_path="finite-differences"),
    ],
    "Ch. 6 — Machine Learning": [
        st.Page("pages/6_Deep_Hedging.py", title="Deep Hedging", icon="🧠", url_path="deep-hedging"),
    ],
}

pg = st.navigation(pages)
pg.run()

st.markdown("""
<style>
.btt-link {
    position: fixed;
    bottom: 2.5rem;
    right: 2rem;
    z-index: 999999;
    background: #262730;
    color: #fafafa !important;
    border: 1px solid #888;
    border-radius: 50%;
    width: 2.8rem;
    height: 2.8rem;
    font-size: 1.4rem;
    line-height: 2.8rem;
    text-align: center;
    text-decoration: none !important;
    opacity: 0.85;
    display: block;
}
.btt-link:hover { opacity: 1; border-color: #fff; }
</style>
<a class="btt-link" href="#top" title="Back to top">↑</a>
""", unsafe_allow_html=True)
