# Monte Carlo European Call Option Simulation

## What is this?

This project simulates the pricing of a **European call option** using **Monte Carlo methods**.

### European Call Option
A financial contract that gives you the **right** (but not obligation) to **buy** a stock at a predetermined price (strike price) on a specific expiration date.

### Monte Carlo Simulation
A computational technique that:
1. Simulates thousands of possible future stock price paths
2. Calculates the payoff for each path
3. Averages the payoffs and discounts them to present value
4. Estimates the fair price of the option today

## Setup

### Option 1: Using direnv (Recommended - Auto-activates)

If you have `direnv` installed, the environment will automatically activate when you `cd` into this directory.

```bash
# Just allow direnv once
direnv allow

# Dependencies are already installed!
# Now just run the script
python monte_carlo_option.py
```

### Option 2: Manual Setup

#### 1. Create Virtual Environment (using uv)
```bash
uv venv
```

#### 2. Activate Virtual Environment
```bash
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
uv pip install -r requirements.txt
```

## Usage

### Run the Simulation
```bash
python monte_carlo_option.py
```

This will:
- Run 10,000 Monte Carlo simulations
- Print the estimated option price with confidence interval
- Display 3 visualization plots showing the simulation results

### Modify Parameters

Edit the configuration section at the top of `monte_carlo_option.py`:

```python
# CONFIGURATION - Edit these parameters
S0 = 100        # Current stock price ($)
K = 105         # Strike price ($)
T = 1.0         # Time to expiration (years)
r = 0.05        # Risk-free rate (5%)
sigma = 0.20    # Volatility (20%)
N = 10000       # Number of simulations
```

## Understanding the Output

### Console Output
- **Estimated Option Price**: The fair value of the option today
- **Standard Error**: Measure of estimation uncertainty
- **95% Confidence Interval**: Range where true price likely falls

### Visualizations
1. **Stock Price Paths**: Shows sample trajectories of possible stock prices over time
2. **Final Price Distribution**: Histogram showing distribution of stock prices at expiration
3. **Convergence Plot**: How the price estimate stabilizes as more simulations are added

## How It Works

The simulation uses **Geometric Brownian Motion (GBM)** to model stock price evolution:

```
S(t+dt) = S(t) × exp((r - 0.5σ²)dt + σ√dt × Z)
```

Where:
- S(t) = stock price at time t
- r = risk-free rate
- σ = volatility
- Z = random normal variable
- dt = time step

For each simulated path:
1. Generate random stock price path to expiration
2. Calculate payoff: max(Final Price - Strike Price, 0)
3. Average all payoffs
4. Discount to present value: Average Payoff × e^(-rT)

## Dependencies

- **NumPy**: Numerical computations and random number generation
- **Matplotlib**: Data visualization

## Learning Resources

- **Payoff**: How much profit you make = max(Stock Price - Strike Price, 0)
- **Discount**: Converting future money to today's value using risk-free rate
- **Volatility**: How much the stock price fluctuates (higher = more uncertainty)
- **Risk-free Rate**: Return from a guaranteed investment (like Treasury bonds)
