import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import scipy.stats as stats
import pandas as pd

# COS Method Core Functions
def computePayoffCoefficients(optionType, a, b, k):
    """Calculate H_k coefficients for call/put payoff functions"""
    if optionType.lower() in ['c', '1']:  # Call options
        c, d = 0.0, b
        coeffs = computeChiPsi(a, b, c, d, k)
        Chi_k, Psi_k = coeffs['chi'], coeffs['psi']
        
        if a < b and b < 0.0:
            H_k = np.zeros([len(k), 1])
        else:
            H_k = 2.0/(b - a) * (Chi_k - Psi_k)
            
    elif optionType.lower() in ['p', '-1']:  # Put options
        c, d = a, 0.0
        coeffs = computeChiPsi(a, b, c, d, k)
        Chi_k, Psi_k = coeffs['chi'], coeffs['psi']
        H_k = 2.0/(b - a) * (-Chi_k + Psi_k)
        
    return H_k

def computeChiPsi(a, b, c, d, k):
    """Compute Chi_k and Psi_k coefficients for cosine expansion"""
    # Psi_k calculation - integral of cosine terms
    psi = (np.sin(k * np.pi * (d - a) / (b - a)) - 
           np.sin(k * np.pi * (c - a) / (b - a)))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    
    # Chi_k calculation - integral of exponential-cosine terms
    chi = 1.0 / (1.0 + (k * np.pi / (b - a))**2)
    term1 = (np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d) - 
             np.cos(k * np.pi * (c - a) / (b - a)) * np.exp(c))
    term2 = (k * np.pi / (b - a) * np.sin(k * np.pi * (d - a) / (b - a)) - 
             k * np.pi / (b - a) * np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c))
    
    chi = chi * (term1 + term2)
    
    return {"chi": chi, "psi": psi}

def cosOptionPricing(cf, optionType, S0, r, maturity, strikes, N, L):
    """
    Price European options using the COS method
    
    Parameters:
    cf - Characteristic function of log-price process
    optionType - 'c' for call, 'p' for put
    S0 - Current asset price
    r - Risk-free interest rate
    maturity - Time to expiration
    strikes - List of strike prices
    N - Number of cosine expansion terms
    L - Truncation domain size
    """
    strikes = np.array(strikes).reshape([len(strikes), 1])
    i = 1j
    
    # Define truncation domain for integration
    x0 = np.log(S0/strikes)
    a = -L * np.sqrt(maturity)
    b = L * np.sqrt(maturity)
    
    # Generate frequency components
    k = np.linspace(0, N-1, N).reshape([N, 1])
    u = k * np.pi / (b - a)
    
    # Calculate payoff coefficients
    H_k = computePayoffCoefficients(optionType, a, b, k)
    
    # Compute option prices using matrix operations
    mat = np.exp(i * np.outer((x0 - a), u))
    temp = cf(u) * H_k
    temp[0] = 0.5 * temp[0]  # Adjust first coefficient
    
    # Final price calculation with discounting
    price = np.exp(-r * maturity) * strikes * np.real(mat @ temp)
    return price

def blackScholesPrice(optionType, S0, strikes, sigma, maturity, r):
    """Calculate analytical Black-Scholes prices for validation"""
    optionType = optionType.lower()
    strikes = np.array(strikes).reshape([len(strikes), 1])
    
    # Black-Scholes parameters
    d1 = (np.log(S0/strikes) + (r + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    
    if optionType in ['c', '1']:  # Call option pricing
        price = stats.norm.cdf(d1) * S0 - stats.norm.cdf(d2) * strikes * np.exp(-r * maturity)
    elif optionType in ['p', '-1']:  # Put option pricing
        price = stats.norm.cdf(-d2) * strikes * np.exp(-r * maturity) - stats.norm.cdf(-d1) * S0
        
    return price

# Streamlit Application Interface
st.set_page_config(
    page_title="European Option Pricing with COS Method",
    page_icon="📊",
    layout="wide"
)

# Application Header
st.title("European Option Pricing with COS Method")
st.markdown("""
This interactive application demonstrates the powerful COS (Fourier-Cosine) method for pricing European options. 
The technique combines mathematical elegance with computational efficiency to deliver accurate option valuations.
""")

# Parameter Input Section
st.header("🎯 Pricing Parameters")

# Create three columns for organized input layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Option Characteristics")
    option_type = st.radio(
        "Option Type",
        ["Call", "Put"],
        help="Select call option for upside potential or put option for downside protection"
    )
    initial_price = st.number_input(
        "Initial Stock Price (S₀)",
        value=100.0,
        min_value=0.1,
        step=1.0,
        help="Current price of the underlying asset"
    )
    risk_free_rate = st.number_input(
        "Risk-free Rate (r)",
        value=0.1,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.3f",
        help="Annualized risk-free interest rate"
    )

with col2:
    st.subheader("Market Parameters")
    volatility = st.number_input(
        "Volatility (σ)",
        value=0.25,
        min_value=0.01,
        max_value=2.0,
        step=0.01,
        format="%.3f",
        help="Annualized volatility of underlying asset returns"
    )
    time_to_maturity = st.number_input(
        "Time to Maturity (τ) in years",
        value=0.1,
        min_value=0.01,
        max_value=10.0,
        step=0.1,
        format="%.2f",
        help="Time until option expiration date"
    )
    expansion_terms = st.number_input(
        "Number of Expansion Terms (N)",
        value=128,
        min_value=8,
        max_value=512,
        step=32,
        help="Number of terms in cosine series expansion"
    )

with col3:
    st.subheader("Numerical Parameters")
    truncation_size = st.number_input(
        "Truncation Domain Size (L)",
        value=10,
        min_value=4,
        max_value=20,
        step=1,
        help="Controls integration domain: [-L√τ, L√τ]"
    )
    strike_input = st.text_input(
        "Strike Prices (comma-separated)",
        "80, 90, 100, 110, 120",
        help="Enter multiple strike prices to analyze volatility smile"
    )

# Process strike prices input
try:
    strike_prices = [float(k.strip()) for k in strike_input.split(',')]
except ValueError:
    st.error("❌ Please enter valid strike prices as comma-separated numbers")
    strike_prices = [80.0, 90.0, 100.0, 110.0, 120.0]

# Pricing Execution Section
st.header("🚀 Compute Option Prices")

if st.button("Launch Pricing Analysis", type="primary", key="pricing_button"):
    
    # Display computational progress
    with st.spinner('Computing option prices using COS method...'):
        
        # Define Black-Scholes characteristic function
        def characteristic_function(u):
            return np.exp(
                (risk_free_rate - 0.5 * volatility**2) * 1j * u * time_to_maturity 
                - 0.5 * volatility**2 * u**2 * time_to_maturity
            )
        
        # COS method pricing with timing
        computation_start = time.time()
        cos_prices = cosOptionPricing(
            characteristic_function, 
            option_type[0].lower(), 
            initial_price, 
            risk_free_rate, 
            time_to_maturity, 
            strike_prices, 
            expansion_terms, 
            truncation_size
        )
        computation_end = time.time()
        computation_time = computation_end - computation_start
        
        # Benchmark with Black-Scholes model
        bs_prices = blackScholesPrice(
            option_type[0].lower(), 
            initial_price, 
            strike_prices, 
            volatility, 
            time_to_maturity, 
            risk_free_rate
        )
    
    # Results Visualization
    st.header("📈 Pricing Results Analysis")
    
    # Create interactive price comparison chart
    fig = go.Figure()
    
    # Add COS method prices
    fig.add_trace(go.Scatter(
        x=strike_prices, 
        y=cos_prices.flatten(), 
        mode='lines+markers',
        name='COS Method',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Add Black-Scholes benchmark
    fig.add_trace(go.Scatter(
        x=strike_prices, 
        y=bs_prices.flatten(), 
        mode='lines+markers',
        name='Black-Scholes',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Add vertical reference lines for strikes
    for strike in strike_prices:
        fig.add_shape(
            type="line",
            x0=strike, y0=0, 
            x1=strike, y1=max(cos_prices.max(), bs_prices.max()) * 1.1,
            line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dot"),
        )
    
    # Chart layout configuration
    fig.update_layout(
        title=f"{option_type} Option Price Comparison: COS Method vs Black-Scholes",
        xaxis_title="Strike Price (K)",
        yaxis_title="Option Premium",
        legend_title="Pricing Methodology",
        hovermode="x unified",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance Metrics
    st.subheader("⏱️ Performance Metrics")
    col_perf1, col_perf2, col_perf3 = st.columns(3)
    
    with col_perf1:
        st.metric(
            "Computation Time", 
            f"{computation_time:.4f} seconds",
            help="Time to price all options using COS method"
        )
    
    with col_perf2:
        avg_error = np.mean(np.abs(cos_prices - bs_prices))
        st.metric(
            "Average Absolute Error", 
            f"{avg_error:.2E}",
            help="Mean absolute difference from Black-Scholes benchmark"
        )
    
    with col_perf3:
        max_error = np.max(np.abs(cos_prices - bs_prices))
        st.metric(
            "Maximum Absolute Error", 
            f"{max_error:.2E}",
            help="Largest absolute difference from Black-Scholes benchmark"
        )
    
    # Detailed Results Table
    st.subheader("📊 Detailed Pricing Results")
    
    results_data = pd.DataFrame({
        "Strike Price": strike_prices,
        "COS Price": cos_prices.flatten(),
        "Black-Scholes Price": bs_prices.flatten(),
        "Absolute Error": np.abs(cos_prices - bs_prices).flatten(),
        "Relative Error (%)": (np.abs(cos_prices - bs_prices) / bs_prices * 100).flatten()
    })
    
    # Format the results table
    styled_results = results_data.style.format({
        "Strike Price": "{:.2f}",
        "COS Price": "{:.6f}",
        "Black-Scholes Price": "{:.6f}",
        "Absolute Error": "{:.2E}",
        "Relative Error (%)": "{:.4f}"
    }).background_gradient(subset=['Absolute Error'], cmap='YlOrRd')
    
    st.dataframe(styled_results, use_container_width=True)

# Educational Content Section
st.header("🎓 Understanding the COS Method")

st.markdown("""
### Mathematical Foundation

The COS method represents a significant advancement in computational finance, leveraging Fourier-cosine series expansions to achieve exceptional pricing efficiency. The core insight involves reconstructing the probability density function through carefully chosen cosine basis functions.

**Key Mathematical Components:**

1. **Characteristic Function Integration**
   - Transforms the pricing problem to Fourier space
   - Enables efficient handling of complex stochastic processes

2. **Truncation Domain Optimization**
   - Integration range: `[a, b] = [-L√τ, L√τ]`
   - Balances computational speed with numerical accuracy

3. **Cosine Series Expansion**
   - Approximates density functions with exponential convergence
   - `N` terms provide `O(N^(-4))` error reduction for smooth densities

### Computational Advantages

| Feature | Benefit |
|---------|---------|
| **Exponential Convergence** | Fewer terms needed for high precision |
| **Vectorization Friendly** | Simultaneous pricing across multiple strikes |
| **Model Flexibility** | Works with any known characteristic function |
| **Numerical Stability** | Avoids integration/differentiation issues |

### Practical Applications

The COS method excels in real-world scenarios including:
- **Risk Management**: Fast portfolio valuation and Greeks calculation
- **Model Calibration**: Efficient parameter estimation for complex models
- **Exotic Options**: Extension to barrier, digital, and other path-dependent options
- **Multi-Asset Options**: Scalable to higher-dimensional problems

### Interpretation Guidelines

- **Calculation Time**: Typically under 1ms per option demonstrates method efficiency
- **Absolute Errors**: Values below `1E-04` indicate excellent numerical precision  
- **Relative Errors**: Below `0.01%` validate method accuracy across moneyness
- **Volatility Smile**: Multiple strikes reveal implied volatility patterns
""")

# Footer
st.markdown("---")
st.markdown(
    "**COS Method Implementation** • "
    "A computational finance demonstration combining mathematical rigor with practical application"
)