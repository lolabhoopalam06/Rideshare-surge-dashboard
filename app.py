import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Rideshare Surge Pricing Simulator",
    page_icon="🚖",
    layout="wide",
)

st.title("🚖 Real-Time Surge Pricing Equilibrium Simulator")
st.markdown(
    """
This interactive simulator implements the **first-order ordinary differential equation (ODE)** model 
from the project report to determine the optimal surge multiplier ($m^*$) that maximizes total completed rides.
"""
)

# ---------------------------------------------------------
# Sidebar: Parameter Inputs & Scenario Presets
# ---------------------------------------------------------
st.sidebar.header("🕹️ Market Controls")

# Scenario Presets for quick Viva demonstrations
scenario = st.sidebar.selectbox(
    "Load Operational Scenario",
    [
        "Baseline (Indiranagar Peak)",
        "Severe Weather Event (Rain Spike)",
        "High Driver Supply (Off-Peak)",
        "Custom Configuration",
    ],
)

# Default baseline values from Report
if scenario == "Baseline (Indiranagar Peak)":
    default_D0, default_S0, default_Smax = 1000, 150, 500
    default_alpha, default_beta = 0.40, 0.50
elif scenario == "Severe Weather Event (Rain Spike)":
    default_D0, default_S0, default_Smax = 1800, 100, 600
    default_alpha, default_beta = 0.25, 0.65  # Less price sensitive, high driver entry incentive
elif scenario == "High Driver Supply (Off-Peak)":
    default_D0, default_S0, default_Smax = 600, 300, 500
    default_alpha, default_beta = 0.60, 0.35
else:
    default_D0, default_S0, default_Smax = 1000, 150, 500
    default_alpha, default_beta = 0.40, 0.50

st.sidebar.subheader("System Parameters")
D0 = st.sidebar.number_input(
    "Base Unconstrained Demand (D₀)",
    value=default_D0,
    step=50,
    help="Initial trip requests per hour",
)
S0 = st.sidebar.number_input(
    "Base Active Driver Supply (S₀)",
    value=default_S0,
    step=10,
    help="Active drivers at standard pricing (m=1.0)",
)
Smax = st.sidebar.number_input(
    "Fleet Upper Bound (S_max)",
    value=default_Smax,
    step=50,
    help="Maximum regional driver capacity",
)

st.sidebar.subheader("Elasticity Coefficients")
alpha = st.sidebar.slider(
    "Passenger Price Elasticity (α)",
    min_value=0.10,
    max_value=1.00,
    value=default_alpha,
    step=0.05,
    help="Higher α = faster passenger drop-off as price increases",
)
beta = st.sidebar.slider(
    "Driver Labor Elasticity (β)",
    min_value=0.10,
    max_value=1.00,
    value=default_beta,
    step=0.05,
    help="Higher β = faster driver entry in response to higher fares",
)

# ---------------------------------------------------------
# Core Mathematical Modeling & Calculations
# ---------------------------------------------------------
# Define ODE system from Report Section 6.3
def rideshare_ode(t, y, m, alpha, beta, Smax):
    D, S = y
    dDdt = -alpha * (m - 1.0) * D
    dSdt = beta * m * (Smax - S)
    return [dDdt, dSdt]


# Multiplier Range Evaluation (t = 1 hour horizon)
m_range = np.linspace(1.0, 4.0, 250)
D_eval = D0 * np.exp(-alpha * (m_range - 1.0))
S_eval = Smax - (Smax - S0) * np.exp(-beta * m_range)
C_eval = np.minimum(D_eval, S_eval)

# Locate Global Maximum
opt_idx = np.argmax(C_eval)
m_star = m_range[opt_idx]
max_C = C_eval[opt_idx]

# Baseline Performance at m = 1.00
base_C = min(D0, S_eval[0])
efficiency_gain = ((max_C - base_C) / base_C) * 100.0

# ---------------------------------------------------------
# Dashboard KPI Cards
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Optimal Surge Multiplier (m*)", f"{m_star:.2f}x")
col2.metric("Max Completed Rides", f"{max_C:.1f} / hr")
col3.metric("Baseline Completed Rides (m=1.0)", f"{base_C:.1f} / hr")
col4.metric(
    "Platform Efficiency Gain",
    f"{efficiency_gain:+.1f}%",
    delta_color="normal",
)

st.markdown("---")

# ---------------------------------------------------------
# Interactive Visualizations
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "📈 Market Bottleneck Optimization",
        "⏱️ Dynamic Time Trajectories",
        "📊 Parameter Sensitivity Analysis",
    ]
)

with tab1:
    st.subheader(
        "Figure 1: Total Completed Rides vs. Surge Pricing Multiplier (m)"
    )

    fig_opt = go.Figure()

    # Demand Curve
    fig_opt.add_trace(
        go.Scatter(
            x=m_range,
            y=D_eval,
            mode="lines",
            name="Passenger Demand D(m)",
            line=dict(color="crimson", dash="dash", width=2),
        )
    )

    # Supply Curve
    fig_opt.add_trace(
        go.Scatter(
            x=m_range,
            y=S_eval,
            mode="lines",
            name="Driver Supply S(m)",
            line=dict(color="royalblue", dash="dash", width=2),
        )
    )

    # Completed Rides Bottleneck Curve
    fig_opt.add_trace(
        go.Scatter(
            x=m_range,
            y=C_eval,
            mode="lines",
            name="Completed Rides C(m) = min(D, S)",
            line=dict(color="forestgreen", width=4),
        )
    )

    # Dynamic Equilibrium Point
    fig_opt.add_trace(
        go.Scatter(
            x=[m_star],
            y=[max_C],
            mode="markers+text",
            name="Optimal Equilibrium (m*)",
            marker=dict(color="gold", size=14, line=dict(color="black", width=2)),
            text=[f"m* = {m_star:.2f}"],
            textposition="top center",
        )
    )

    fig_opt.update_layout(
        xaxis_title="Surge Pricing Multiplier (m)",
        yaxis_title="Rides / Drivers per Hour",
        hovermode="x unified",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    st.plotly_chart(fig_opt, use_container_width=True)

    st.info(
        f"**Interpretation:** At **m* = {m_star:.2f}**, passenger cancellation rates and driver entry rates reach dynamic equilibrium. "
        f"Pricing below {m_star:.2f} creates a **supply deficit**, while pricing above creates **passenger churn**."
    )

with tab2:
    st.subheader(
        "Figure 2: Time Evolution Trajectories Over Peak Window (t = 1 Hour)"
    )

    # Solve IVP for optimal surge vs base fare
    t_span = (0, 1)
    t_eval = np.linspace(0, 1, 100)

    sol_opt = solve_ivp(
        rideshare_ode,
        t_span,
        [D0, S0],
        args=(m_star, alpha, beta, Smax),
        t_eval=t_eval,
    )
    sol_base = solve_ivp(
        rideshare_ode,
        t_span,
        [D0, S0],
        args=(1.0, alpha, beta, Smax),
        t_eval=t_eval,
    )

    fig_time = go.Figure()

    # Optimal Trajectories
    fig_time.add_trace(
        go.Scatter(
            x=sol_opt.t,
            y=sol_opt.y[0],
            name=f"Demand D(t) at m* = {m_star:.2f}",
            line=dict(color="crimson", width=3),
        )
    )
    fig_time.add_trace(
        go.Scatter(
            x=sol_opt.t,
            y=sol_opt.y[1],
            name=f"Supply S(t) at m* = {m_star:.2f}",
            line=dict(color="royalblue", width=3),
        )
    )

    # Baseline Trajectories (opacity fixed by moving to trace level)
    fig_time.add_trace(
        go.Scatter(
            x=sol_base.t,
            y=sol_base.y[0],
            name="Demand D(t) at Base Pricing (m=1.0)",
            line=dict(color="crimson", dash="dot"),
            opacity=0.5,
        )
    )
    fig_time.add_trace(
        go.Scatter(
            x=sol_base.t,
            y=sol_base.y[1],
            name="Supply S(t) at Base Pricing (m=1.0)",
            line=dict(color="royalblue", dash="dot"),
            opacity=0.5,
        )
    )

    fig_time.update_layout(
        xaxis_title="Time Horizon (Hours)",
        yaxis_title="Active Requests / Drivers",
        hovermode="x unified",
        height=500,
    )

    st.plotly_chart(fig_time, use_container_width=True)

with tab3:
    st.subheader("Data Sweep Export")
    st.markdown(
        "Download simulated dynamic sweep data for inclusion in project reports or appendices."
    )

    export_df = pd.DataFrame(
        {
            "Surge Multiplier (m)": m_range,
            "Demand D(1)": D_eval,
            "Supply S(1)": S_eval,
            "Completed Rides C(m)": C_eval,
            "System Constraint": np.where(
                D_eval < S_eval, "Demand Constrained", "Supply Constrained"
            ),
        }
    )

    st.dataframe(export_df.style.highlight_max(axis=0, subset=["Completed Rides C(m)"]), use_container_width=True)

    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Data Table (CSV)",
        data=csv_data,
        file_name="surge_pricing_simulation_results.csv",
        mime="text/csv",
    )
