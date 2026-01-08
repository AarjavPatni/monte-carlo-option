"""Interactive Monte Carlo simulation for European call options"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Page configuration
st.set_page_config(page_title="Monte Carlo Option Pricing", layout="wide")

st.title("Monte Carlo European Call Option Simulator")
st.markdown("Interactive simulation using Geometric Brownian Motion")

# Sidebar parameters
st.sidebar.header("Simulation Parameters")

S0 = st.sidebar.slider("S₀: current stock price ($)", 50, 200, 100, 5)
K = st.sidebar.slider("K: strike price ($)", 50, 200, 105, 5)
T = st.sidebar.slider("T: time to expiration (years)", 0.1, 5.0, 1.0, 0.1)
r = st.sidebar.slider("r: risk-free rate (%)", 0.0, 15.0, 5.0, 0.5) / 100
sigma = st.sidebar.slider("σ: volatility (%)", 5.0, 100.0, 20.0, 5.0) / 100
N = st.sidebar.slider("N: number of simulations", 1000, 20000, 10000, 1000)

st.sidebar.markdown("---")
show_bs = st.sidebar.checkbox("Show Black-Scholes Comparison", value=True)


def black_scholes_call(S, K, T, r, sigma):
    """analytical solution for european call option"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def simulate_gbm_paths(S0, r, sigma, T, N, steps=252):
    """Simulate stock price paths using GBM"""
    dt = T / steps

    paths = np.zeros((N, steps + 1))
    paths[:, 0] = S0

    # generate all random shocks at once for speed
    Z = np.random.standard_normal((N, steps))

    # GBM: dS = S(r*dt + sigma*sqrt(dt)*Z)
    for t in range(1, steps + 1):
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z[:, t - 1]
        paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion)

    return paths


@st.cache_data
def monte_carlo_european_call(S0, K, T, r, sigma, N):
    """Price European call option via Monte Carlo"""
    paths = simulate_gbm_paths(S0, r, sigma, T, N)

    final_prices = paths[:, -1]

    # payoff for call option: max(S_T - K, 0)
    payoffs = np.maximum(final_prices - K, 0)

    # discount back to present value
    discount_factor = np.exp(-r * T)
    option_price = discount_factor * np.mean(payoffs)

    # standard error and confidence interval
    se = discount_factor * np.std(payoffs) / np.sqrt(N)
    ci = (option_price - 1.96 * se, option_price + 1.96 * se)

    return option_price, se, ci, paths, payoffs


def create_plots(paths, payoffs, S0, K, T, option_price, N, r, bs_price=None):
    """generate visualization plots"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: sample paths
    n_sample = min(100, N)
    sample_idx = np.random.choice(N, n_sample, replace=False)
    time = np.linspace(0, T, paths.shape[1])

    for i in sample_idx:
        axes[0].plot(time, paths[i, :], alpha=0.3, linewidth=0.5)

    axes[0].axhline(
        y=K, color="r", linestyle="--", linewidth=2, label=f"Strike (K=${K})"
    )
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Stock Price ($)")
    axes[0].set_title(f"Stock Price Paths (showing {n_sample} of {N})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: distribution of final prices
    final_prices = paths[:, -1]
    axes[1].hist(final_prices, bins=50, alpha=0.7, color="blue", edgecolor="black")
    axes[1].axvline(
        x=K, color="r", linestyle="--", linewidth=2, label=f"Strike (K=${K})"
    )

    # shade in/out of money regions
    axes[1].axvspan(
        K, final_prices.max(), alpha=0.2, color="green", label="In the money"
    )
    axes[1].axvspan(
        final_prices.min(), K, alpha=0.2, color="red", label="Out of the money"
    )

    axes[1].set_xlabel("Final Stock Price ($)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Distribution at Expiration (T={T}yr)")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    # Plot 3: convergence
    discount = np.exp(-r * T)
    discounted = discount * payoffs
    cumsum = np.cumsum(discounted)
    n_sims = np.arange(1, N + 1)
    running_avg = cumsum / n_sims

    axes[2].plot(n_sims, running_avg, linewidth=1.5, color="blue", label="Monte Carlo")
    axes[2].axhline(
        y=option_price,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"MC final: ${option_price:.2f}",
    )

    if bs_price is not None:
        axes[2].axhline(
            y=bs_price,
            color="green",
            linestyle=":",
            linewidth=2.5,
            label=f"Black-Scholes: ${bs_price:.2f}",
        )

    axes[2].set_xlabel("number of simulations")
    axes[2].set_ylabel("estimated price ($)")
    axes[2].set_title("convergence of MC estimate")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_xscale("log")

    plt.tight_layout()
    return fig


# Run simulation
with st.spinner("Running simulation..."):
    price, se, ci, paths, payoffs = monte_carlo_european_call(S0, K, T, r, sigma, N)

# Display results
st.subheader("Results")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Option Price", f"${price:.2f}")
col2.metric("Std Error", f"${se:.2f}")
col3.metric("95% CI Lower", f"${ci[0]:.2f}")
col4.metric("95% CI Upper", f"${ci[1]:.2f}")

# additional statistics
final_prices = paths[:, -1]
itm = np.sum(final_prices > K)
st.write(f"**Paths In-The-Money:** {itm:,} ({itm / N * 100:.1f}%)")
st.write(f"**Avg Final Price:** ${np.mean(final_prices):.2f}")

# black-scholes comparison
bs_price = None
if show_bs:
    bs_price = black_scholes_call(S0, K, T, r, sigma)
    error_abs = abs(price - bs_price)
    error_pct = error_abs / bs_price * 100

    st.markdown("---")
    st.subheader("Method Comparison")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Monte Carlo", f"${price:.2f}")
    col2.metric("Black-Scholes", f"${bs_price:.2f}")
    col3.metric("Absolute Error", f"${error_abs:.2f}")
    col4.metric("Relative Error", f"{error_pct:.2f}%")

    within_ci = ci[0] <= bs_price <= ci[1]
    if within_ci:
        st.success(
            f"Black-Scholes price (${bs_price:.2f}) is within the 95% confidence interval"
        )
    else:
        st.warning(
            f"Black-Scholes price (${bs_price:.2f}) is outside the confidence interval - try increasing N"
        )

# plots
st.subheader("Visualizations")
fig = create_plots(paths, payoffs, S0, K, T, price, N, r, bs_price=bs_price)
st.pyplot(fig)

# educational content
if show_bs:
    with st.expander("Understanding Monte Carlo vs Black-Scholes"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### Monte Carlo Simulation
            **How It Works:**
            - Simulates thousands of random stock price paths
            - Calculates payoff for each path
            - Averages and discounts to present value

            **Advantages:**
            - Works for any option type (American, exotic, etc.)
            - Easy to add complexity (dividends, jumps)
            - Intuitive and flexible

            **Disadvantages:**
            - Slower (needs many simulations)
            - Has sampling error (random variation)
            - Accuracy depends on N
            """)

        with col2:
            st.markdown("""
            ### Black-Scholes Formula
            **How It Works:**
            - Analytical solution from stochastic calculus
            - Closed-form mathematical equation
            - Instant calculation

            **Advantages:**
            - Exact theoretical price
            - No sampling error
            - Very fast (no simulation needed)

            **Disadvantages:**
            - Only works for European options
            - Restrictive assumptions (constant volatility, no dividends, log-normal prices)
            """)

        st.info("""
        **Why Compare Both?**

        Monte Carlo should converge to Black-Scholes as N → ∞ for European options.
        Comparing them demonstrates simulation accuracy and quantifies sampling error.
        The relative error shows how close our MC estimate is to the theoretical value.
        """)
