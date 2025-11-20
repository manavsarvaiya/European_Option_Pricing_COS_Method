# European Option Pricing with COS Method

## Overview
This project implements the innovative COS Method (Fourier-Cosine Series Expansion) for efficient pricing of European call and put options. Developed by Fang and Oosterlee, this advanced numerical technique delivers exceptional computational performance while maintaining high accuracy across various financial models.

## Table of Contents
- [Understanding European Options](#Understanding-European-Options)
- [Mathematical Foundation](#mathematical-foundation)
- [COS Method Deep Dive](#COS-Method-Deep-Dive)
- [Project Architecture](#Project-Architecture)
- [Implementation Guide](#Implementation-Guide)
- [Performance Analysis](#Performance-Analysis)
- [License](#license)

## Understanding European Options
**European options** represent financial contracts granting holders the right—without obligation—to purchase (Call) or sell (Put) underlying assets at predetermined strike prices upon expiration.

While the Black-Scholes model provides analytical solutions under specific assumptions, the COS method extends pricing capabilities to complex stochastic processes where traditional methods face limitations.

## Mathematical Foundation
European option valuation follows the risk-neutral pricing formula:
`\[
    V(S_0, t) = e^{-r(T-t)} \mathbb{E} [ (S_T - K)^+ ]
\]`
where:
- \( S_0 \) Current underlying asset price
- \( K \) Option strike price
- \( T \) Time to expiration
- \( r \) Risk-free interest rate
- \( S_T \) Asset price at maturity

The COS method revolutionizes this computation through Fourier-cosine series expansions, bypassing direct integration challenges.

## COS Method Deep Dive
The COS technique approximates option payoff functions using Fourier cosine expansions:
\[
    f(x) \approx \sum_{n=0}^{N-1} A_n \cos \left( \frac{n \pi (x - a)}{b - a} \right)
\]
where:
- \( a, b \) defines integration boundaries
- \( A_n \) derived from the characteristic function \( \varphi(u) \),
- \( N \) determines approximation accuracy

The option price is then computed using:
\[
    V(S_0, t) = e^{-r(T-t)} \sum_{n=0}^{N-1} A_n \text{Re} \left[ \varphi \left( \frac{n \pi}{b - a} \right) \right]
\]
This approach achieves remarkable efficiency by leveraging Fourier series properties and closed-form characteristic functions.

## Project Architecture
```
European-Option-Pricing-COS-Method/
│
├── 📓 Pricing of European Call and Put options with the COS method.ipynb
├── 🌐 app.py
└── 📖 README.md
└── 📦 requirements.txt
```

## Implementation Guide
### Running the Jupyter Notebook
To execute the COS method for European option pricing, run:
```bash
jupyter notebook "European_Option_Pricing_COS_Method.ipynb"
```

### Running the Streamlit App
To launch the interactive web application, run:
```bash
streamlit run app.py
```

### Example Usage
Import the COS method and compute an option price:
```python


S0 = 100    # Initial stock price
K = 100     # Strike price
T = 1       # Time to maturity
r = 0.05    # Risk-free rate
sigma = 0.2 # Volatility
N = 64      # Number of expansion terms

price = cos_method(S0, K, T, r, sigma, N)
print(f"Option Price: {price}")
```

## Performance Analysis

The COS method demonstrates superior computational efficiency with exponential convergence rates. Key advantages include:

### Speed Benefits:

- 10-100x faster than Monte Carlo simulations
- Minimal function evaluations required
- Optimal for batch pricing operations

### Model Compatibility:

- Black-Scholes (geometric Brownian motion)
- Heston stochastic volatility model
- Variance Gamma and Lévy processes
- Merton jump-diffusion model

### Accuracy Metrics:

- Consistent across moneyness levels
- Robust numerical stability

