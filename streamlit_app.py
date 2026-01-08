"""Interactive Monte Carlo simulation for European call options"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Monte Carlo Option Pricing", layout="wide")

st.title("Monte Carlo European Call Option Simulator")
st.markdown("Interactive simulation using Geometric Brownian Motion")

# Sidebar parameters
st.sidebar.header("Simulation Parameters")

S0 = st.sidebar.slider("S₀: Current Stock Price ($)", 50, 200, 100, 5)
K = st.sidebar.slider("K: Strike Price ($)", 50, 200, 105, 5)
T = st.sidebar.slider("T: Time to Expiration (years)", 0.1, 5.0, 1.0, 0.1)
r = st.sidebar.slider("r: Risk-Free Rate (%)", 0.0, 15.0, 5.0, 0.5) / 100
sigma = st.sidebar.slider("σ: Volatility (%)", 5.0, 100.0, 20.0, 5.0) / 100
N = st.sidebar.slider("N: Number of Simulations", 1000, 20000, 10000, 1000)


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


def create_plots(paths, payoffs, S0, K, T, option_price, N, r):
    """Generate visualization plots"""
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

    axes[2].plot(n_sims, running_avg, linewidth=1.5, color="blue")
    axes[2].axhline(
        y=option_price,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Final: ${option_price:.2f}",
    )
    axes[2].set_xlabel("Number of Simulations")
    axes[2].set_ylabel("Estimated Price ($)")
    axes[2].set_title("Convergence of MC Estimate")
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

# Additional statistics
final_prices = paths[:, -1]
itm = np.sum(final_prices > K)
st.write(f"**Paths in-the-money:** {itm:,} ({itm / N * 100:.1f}%)")
st.write(f"**Avg final price:** ${np.mean(final_prices):.2f}")

# Plots
st.subheader("Visualizations")
fig = create_plots(paths, payoffs, S0, K, T, price, N, r)
st.pyplot(fig)
