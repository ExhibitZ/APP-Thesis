import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from regression_utils import (perform_linear_regression, perform_polynomial_regression,
    mean_absolute_percentage_error, calculate_dynamic_k, solve_epam,
    perform_multivariate_regression, solve_non_autonomous_epam,
    compute_rolling_k, forecast_rolling_k, compute_adjusted_gap, solve_epam_adjusted_gap)
from scipy.interpolate import interp1d as _interp1d
from groq import Groq
from theme import THEMES, get_css

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Market Dynamics Analyzer", page_icon="📈")

# --- Session State ---
if 'analysis_active' not in st.session_state:
    st.session_state.analysis_active = False
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = "dark"

# --- Theme Definitions ---
t = THEMES[st.session_state.theme_mode]

# --- Custom CSS (Theme-Aware) ---
st.markdown(get_css(st.session_state.theme_mode), unsafe_allow_html=True)

# --- Helper Functions ---
def get_plot_layout(title_text, x_title, y_title):
    """Return a standard Plotly layout dict matching the active theme."""
    return dict(
        title=dict(text=title_text, font=dict(color=t['heading'], size=16)),
        xaxis=dict(title=x_title, title_font=dict(color=t['text_muted'], size=16), tickfont=dict(color=t['text_muted']), gridcolor=t['grid'], zeroline=False),
        yaxis=dict(title=y_title, title_font=dict(color=t['text_muted'], size=16), tickfont=dict(color=t['text_muted']), gridcolor=t['grid'], zeroline=False),
        paper_bgcolor=t['plot_paper'],
        plot_bgcolor=t['plot_bg'],
        legend=dict(font=dict(color=t['text']), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family='Inter, sans-serif'),
    )

def plot_regression(X_plot, y_plot, y_pred_linear, y_pred_poly, title, x_label, y_label, degree):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_plot.flatten(), y=y_plot, mode='markers', name='Actual Data', marker=dict(color=t['marker'], opacity=0.8, size=7, line=dict(width=1, color=t['accent']))))
    
    # Sort for plotting lines
    sort_idx = np.argsort(X_plot.flatten())
    X_sorted = X_plot.flatten()[sort_idx]
    y_lin_sorted = y_pred_linear[sort_idx]
    y_poly_sorted = y_pred_poly[sort_idx]
    
    fig.add_trace(go.Scatter(x=X_sorted, y=y_lin_sorted, mode='lines', name='Linear Trend', line=dict(color=t['line_red'], dash='dash', width=2)))
    fig.add_trace(go.Scatter(x=X_sorted, y=y_poly_sorted, mode='lines', name=f'Curve (Deg {degree})', line=dict(color=t['line_blue'], width=2.5)))
    
    fig.update_layout(**get_plot_layout(title, x_label, y_label))
    return fig

@st.cache_data
def get_ai_analysis(api_key, context_text):
    if not api_key:
        return "Please configure your Groq API Key in the Setup tab."
    try:
        client = Groq(api_key=api_key)
        system_prompt = """You are a Senior Market Economist and Quantitative Analyst specializing in dynamic price modeling.
Your task is to interpret Excess Price Adjustment Model (EPAM) simulation results and translate them into a clear, 
concise, and actionable market intelligence report.
Your analysis MUST be grounded in the specific numerical data provided — never give generic statements.
Always cite the exact numbers (MAPE %, K values, price levels) from the context when making claims.
Write in professional but accessible English so that a business executive without a math background can understand."""

        prompt = f"""Below are the quantitative results from a Market Dynamics Simulation using the Excess Price Adjustment Model (EPAM).
The EPAM is a first-order Ordinary Differential Equation: dP/dt = K × (Qd - Qs), where price adjusts proportionally
to the gap between Demand Quantity (Qd) and Supply Quantity (Qs), governed by adjustment speed K.

═══════════════════════════════════════════
 SIMULATION DATA & RESULTS
═══════════════════════════════════════════
{context_text}
═══════════════════════════════════════════

Using ONLY the data above, write a structured market intelligence report with the following EXACT sections:

## 🏦 Executive Summary
In 3–4 sentences, state the overall market condition right now. Is the market in surplus or deficit? 
Are prices trending up or down? Which model performed best (lowest MAPE) and what does that imply?

## 📈 Market Dynamics & Price Behavior
Explain HOW prices have been moving based on the actual price series data. Use the first and last actual price values 
to describe the price range. Interpret what the supply-demand gap (positive or negative) means for buyers and sellers.
Explain whether the market is converging toward equilibrium or diverging.

## ⚙️ Adjustment Speed (K) — Market Reactivity
For each model (Linear, Polynomial, Multivariate), state the Fixed K value and the average Rolling K value.
Interpret what these K values mean:
- If K > 0: prices rise when demand exceeds supply (normal market)
- If K < 0: prices move inversely (administered/regulated pricing)
- Higher |K|: market reacts faster to imbalances
- Rolling K vs Fixed K: does market reactivity change over time or remain stable?

## 🔮 Price Forecast & Scenarios
For each model's forecast (Linear Fixed K, Linear Rolling K, Polynomial Fixed K, etc.), state the forecasted 
final price. Compare across models: do they agree or diverge? What is the consensus price direction?
Quantify the uncertainty range (e.g., "forecasts range from X to Y").

## 🎯 Model Performance Assessment
Present the MAPE results in a clear ranking from best to worst. Explain what MAPE means in plain language 
(e.g., "a 5% MAPE means the model's price prediction was off by about 5% on average").
Discuss whether Fixed K or Rolling K performs better, and WHY that might be the case for this market.

## 💡 Strategic Recommendations
Provide 3–5 concrete, actionable recommendations for business decision-makers based on these findings:
- For buyers/procurement: when to buy, lock in prices, or wait
- For sellers/producers: pricing strategy, inventory decisions
- For investors: market timing signals based on K trajectory
- For policymakers: any stability concerns based on price divergence

## ⚠️ Caveats & Limitations
Briefly note any limitations of this EPAM model for this specific dataset (e.g., data quality, model fit, 
forecast horizon reliability).

IMPORTANT RULES:
- Every section MUST reference specific numbers from the data (prices, MAPE %, K values)
- Do NOT use vague phrases like "the market shows signs of..." without backing it with a number
- Use markdown formatting: bold key numbers, use bullet points for lists
- Total length: 600–900 words
"""
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2048,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error communicating with AI: {str(e)}"

# --- Top Bar ---
def toggle_theme():
    st.session_state.theme_mode = "light" if st.session_state.theme_mode == "dark" else "dark"

@st.dialog("How to Use")
def show_info_dialog():
    st.markdown("""
**1. Setup & Data**  
Upload a CSV with your market data, then map columns to Supply Qty, Demand Qty, Supply Price, and Demand Price.

**2. Configure** *(optional)*  
Tune imputation, polynomial degree, price bridging, Rolling Window period, and forecast steps.

**3. Run Simulation**  
Click **Start Simulation** to compute regressions and run two EPAM simulations per model:
- **Fixed K**: constant adjustment speed from initial data
- **Rolling K + Adjusted Gap**: time-varying K via rolling OLS with mean-bias removed gap

**4. Results**  
View demand/supply curves, K trajectories, and EPAM price simulations for each model. A MAPE summary table is shown at the bottom.

**5. AI Report**  
Switch to the Explanation tab for an AI-generated market narrative (requires a Groq API key).
    """)

tbar_left, tbar_right = st.columns([10, 1])
with tbar_left:
    st.markdown('<p class="top-bar-title">📈 Market Dynamics Analyzer</p>', unsafe_allow_html=True)
with tbar_right:
    st.markdown('<span id="top-right-buttons"></span>', unsafe_allow_html=True)
    toggle_icon = "☀️" if st.session_state.theme_mode == "dark" else "🌙"
    ic1, ic2 = st.columns(2)
    with ic1:
        if st.button("ℹ️", key="info_btn", help="How to use this app"):
            show_info_dialog()
    with ic2:
        st.button(toggle_icon, on_click=toggle_theme, key="theme_toggle", help="Toggle light/dark mode")

# --- Main Layout ---
nav_options = ["Data Setup", "Simulation Results", "Explanation"]
if 'nav_selection' not in st.session_state:
    st.session_state.nav_selection = nav_options[0]

# Navigation Bar
selection = st.radio("Navigation", nav_options, horizontal=True, label_visibility="collapsed", key="nav_selection")

# Globals for analysis results
analysis_results = {}
uploaded_file = None # Initialize to avoid NameError

if selection == nav_options[0]:
    with st.container():
        def start_analysis():
            st.session_state.analysis_active = True
            
            # Snapshot Configuration
            config = {
                's_qty': st.session_state.col_s_qty,
                'd_qty': st.session_state.col_d_qty,
                's_price': st.session_state.col_s_price,
                'd_price': st.session_state.col_d_price,
                'use_impute': st.session_state.use_impute,
                'treat_zero': st.session_state.treat_zero,
                'poly_deg': st.session_state.poly_deg,
                'use_bridge': st.session_state.use_bridging,
                'bridge_master': st.session_state.get('bridging_master', None),
                'bridge_deg': st.session_state.get('bridging_degree', 1),
                'rolling_window': st.session_state.rolling_k_window,
                'fc_hor': st.session_state.forecast_horizon,
                'user_key': st.session_state.get('user_groq_key', None)
            }
            st.session_state.run_config = config
            
            st.session_state.nav_selection = nav_options[1]

        if st.session_state.get('uploader') is None:
            # Big Box Layout
            st.markdown(f"""
            <style>
                section[data-testid="stFileUploaderDropzone"] {{
                    min-height: 400px !important;
                    display: flex !important;
                    flex-direction: column !important;
                    align-items: center !important;
                    justify-content: center !important;
                    border: 2px solid {t['input_border']} !important;
                    background: {t['input_bg']} !important;
                }}
                section[data-testid="stFileUploaderDropzone"] svg {{
                    width: 80px;
                    height: 80px;
                    fill: {t['text_muted']} !important;
                    color: {t['text_muted']} !important;
                }}
            </style>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload Market Data (CSV)", type=['csv'], help="Upload a CSV file containing your Price and Quantity data.", key="uploader")
        else:
            # Small Box Layout
            top_col1, top_col2 = st.columns([3, 1], vertical_alignment="bottom")
            with top_col1:
                uploaded_file = st.file_uploader("Upload Market Data (CSV)", type=['csv'], help="Upload a CSV file containing your Price and Quantity data.", key="uploader")
            with top_col2:
                st.button("Start Simulation", type="primary", on_click=start_analysis, use_container_width=True)
            
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df_raw = df # Persist DataFrame
                
                columns = df.columns.tolist()
                
                # Two column layout for setup
                setup_col1, setup_col2 = st.columns(2)
                
                with setup_col1:
                    supply_qty_col = st.selectbox("Variable 1 (Supply Quantity)", columns, index=0 if len(columns)>0 else 0, help="Column containing the quantity supplied.", key="col_s_qty")
                    demand_qty_col = st.selectbox("Variable 2 (Demand Quantity)", columns, index=1 if len(columns)>1 else 0, help="Column containing the quantity demanded.", key="col_d_qty")
                    supply_price_col = st.selectbox("Variable 3 (Supply Price)", columns, index=2 if len(columns)>2 else 0, help="Column containing the price for supply.", key="col_s_price")
                    demand_price_col = st.selectbox("Variable 4 (Demand Price)", columns, index=3 if len(columns)>3 else 0, help="Column containing the price for demand.", key="col_d_price")

                with setup_col2:
                    with st.container(border=True):
                        st.markdown("**Parameter Settings**")
                        
                        param_col1, param_col2 = st.columns(2)
                        with param_col1:
                            st.markdown("*Data Cleaning*")
                            use_imputation = st.checkbox("Auto-Repair Missing", value=True, help="Automatically fill in missing values.", key="use_impute")
                            treat_zero_as_missing = st.checkbox("Treat 0 as Missing", value=True, key="treat_zero")
                            use_bridging = st.checkbox("Enable Price Matching", value=False, help="Bridges Buy/Sell prices.", key="use_bridging")
                            
                        with param_col2:
                            st.markdown("*Simulation Setup*")
                            rolling_k_window = st.number_input("Rolling Window (Periode K)", min_value=3, max_value=50, value=12, step=1, help="Jumlah periode lookback untuk Rolling Window K.", key="rolling_k_window")
                            if use_bridging:
                                bridging_master = st.selectbox("Master Price", [demand_price_col, supply_price_col], key="bridging_master")
                        
                        st.markdown("---")
                        st.markdown("*Tuning Parameters*")
                        poly_degree = st.slider("Curve Complexity", 2, 5, 2, key="poly_deg")
                        if use_bridging:
                            bridging_degree = st.slider("Matching Complexity", 1, 3, 1, key="bridging_degree")
                        forecast_horizon = st.slider("Forecast Steps", 5, 50, 10, key="forecast_horizon")
                        
                        st.markdown("---")
                        if "GROQ_API_KEY" in st.secrets:
                            st.success("✅ AI Key loaded.")
                        else:
                            groq_api_key = st.text_input("Groq API Key (AI)", type="password", key="user_groq_key")
                    
            except Exception as e:
                st.error(f"Error reading file: {e}")

# --- Analysis Logic ---
if st.session_state.analysis_active and 'df_raw' in st.session_state and 'run_config' in st.session_state:
    
    # Load Snapshot
    df = st.session_state.df_raw.copy()
    conf = st.session_state.run_config
    
    supply_qty_col = conf['s_qty']
    demand_qty_col = conf['d_qty']
    supply_price_col = conf['s_price']
    demand_price_col = conf['d_price']
    
    use_imputation = conf['use_impute']
    treat_zero_as_missing = conf['treat_zero']
    poly_degree = conf['poly_deg']
    
    use_bridging = conf['use_bridge']
    bridging_master = conf['bridge_master']
    bridging_degree = conf['bridge_deg']


    rolling_window = conf['rolling_window']
    forecast_horizon = conf['fc_hor']
    
    # Validation: Ensure columns exist in current dataframe (Handles file switch)
    required_columns = [supply_qty_col, demand_qty_col, supply_price_col, demand_price_col]
    if use_bridging:
        required_columns.append(bridging_master)
        
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.warning(f"Columns mismatch: {missing_cols} not found in current data. Please click 'Run Market Analysis' again to update settings.")
        st.session_state.analysis_active = False
        st.stop()

    # AI Key Logic
    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    else:
        groq_api_key = conf['user_key']


    # 1. Data Preprocessing
    # MICE Imputation Logic
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    try:
        cols_to_use = [supply_qty_col, demand_qty_col, supply_price_col, demand_price_col]

        # --- Strong per-column numeric validation ---
        non_numeric_cols = []
        for col in cols_to_use:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                non_numeric_cols.append(col)

        if non_numeric_cols:
            bad_examples = {}
            for col in non_numeric_cols:
                sample = df[col].dropna().unique()[:3].tolist()
                bad_examples[col] = sample
            
            msg = "### ⚠️ Non-Numeric Column(s) Detected\n\n"
            msg += "The following columns you selected contain **text or non-numeric values** and cannot be used for regression:\n\n"
            for col, examples in bad_examples.items():
                msg += f"- **`{col}`** — e.g. `{', '.join(str(x) for x in examples)}`\n"
            msg += "\n**How to fix this:**\n"
            msg += "- Go back to the **Data Setup** tab and re-select the correct columns.\n"
            msg += "- Make sure Quantity and Price columns only contain numbers — not product names or category labels."
            st.error(msg)
            st.stop()

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if treat_zero_as_missing:
            df[numeric_cols] = df[numeric_cols].replace(0, np.nan)
        
        if use_imputation and df[cols_to_use].isnull().values.any():
            imputer = IterativeImputer(random_state=42)
            df_imputed = df.copy()
            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            df = df_imputed

    except SystemExit:
        raise  # Let st.stop() propagate
    except Exception as e:
        st.error(f"**Data Processing Error:** {e}")
        st.stop()

    # 2. Main Analysis Calculation
    # Prepare Data Arrays
    X_demand_raw = df[[demand_price_col]].values
    X_supply_raw = df[[supply_price_col]].values
    y_demand = df[demand_qty_col].values
    y_supply = df[supply_qty_col].values
    
    if use_bridging:
        # Bridging Logic
        X_bridge = df[[bridging_master]].values
        secondary_price_col = supply_price_col if bridging_master == demand_price_col else demand_price_col
        y_bridge = df[secondary_price_col].values
        
        bridge_model_lin, _, mape_bridge_lin = perform_linear_regression(X_bridge, y_bridge)
        bridge_model_poly, bridge_feat, _, _ = perform_polynomial_regression(X_bridge, y_bridge, degree=bridging_degree)
        
        b_lin_int, b_lin_c = bridge_model_lin.intercept_, bridge_model_lin.coef_[0]
        b_poly_int, b_poly_c = bridge_model_poly.intercept_, bridge_model_poly.coef_[1:]
        
        def bridge_lin_func(p): return b_lin_int + b_lin_c * p
        def bridge_poly_func(p): return b_poly_int + sum(c * (p**i) for i, c in enumerate(b_poly_c, 1))
        
        x_label = bridging_master
        X_plot_shared = df[[bridging_master]].values
    else:
        x_label = "Price"
        X_plot_shared = None

    # Demand Regression (Fit on raw prices to capture true economic relationships)
    lin_model_d, _, mape_lin_d = perform_linear_regression(X_demand_raw, y_demand)
    poly_model_d, poly_feat_d, _, mape_poly_d = perform_polynomial_regression(X_demand_raw, y_demand, degree=poly_degree)

    # Supply Regression (Fit on raw prices)
    lin_model_s, _, mape_lin_s = perform_linear_regression(X_supply_raw, y_supply)
    poly_model_s, poly_feat_s, _, mape_poly_s = perform_polynomial_regression(X_supply_raw, y_supply, degree=poly_degree)

    # --- FAST NUMPY LAMBDAS ---
    # Extract coefficients
    l_d_int, l_d_c = lin_model_d.intercept_, lin_model_d.coef_[0]
    l_s_int, l_s_c = lin_model_s.intercept_, lin_model_s.coef_[0]
    raw_lin_d = lambda p: l_d_int + l_d_c * p
    raw_lin_s = lambda p: l_s_int + l_s_c * p

    p_d_int, p_d_c = poly_model_d.intercept_, poly_model_d.coef_[1:]
    p_s_int, p_s_c = poly_model_s.intercept_, poly_model_s.coef_[1:]
    raw_poly_d = lambda p: p_d_int + sum(c * (p**i) for i, c in enumerate(p_d_c, 1))
    raw_poly_s = lambda p: p_s_int + sum(c * (p**i) for i, c in enumerate(p_s_c, 1))

    # Composed Lambdas
    if use_bridging:
        if bridging_master == demand_price_col:
            lin_d_func = raw_lin_d
            poly_d_func = raw_poly_d
            lin_s_func = lambda p: raw_lin_s(bridge_lin_func(p))
            poly_s_func = lambda p: raw_poly_s(bridge_poly_func(p))
        else:
            lin_d_func = lambda p: raw_lin_d(bridge_lin_func(p))
            poly_d_func = lambda p: raw_poly_d(bridge_poly_func(p))
            lin_s_func = raw_lin_s
            poly_s_func = raw_poly_s
        X_plot_d = X_plot_shared
        X_plot_s = X_plot_shared
    else:
        lin_d_func = raw_lin_d
        poly_d_func = raw_poly_d
        lin_s_func = raw_lin_s
        poly_s_func = raw_poly_s
        X_plot_d = X_demand_raw
        X_plot_s = X_supply_raw

    # Generate Plot arrays (for graphs)
    y_pred_lin_d = np.array([lin_d_func(p[0]) for p in X_plot_d])
    y_pred_lin_s = np.array([lin_s_func(p[0]) for p in X_plot_s])
    y_pred_poly_d = np.array([poly_d_func(p[0]) for p in X_plot_d])
    y_pred_poly_s = np.array([poly_s_func(p[0]) for p in X_plot_s])

    # --- ANALYSIS RESULTS TAB ---
    # Compute prices array (shared across all models)
    prices = df[bridging_master].values if use_bridging else df[demand_price_col].values
    time_steps = np.arange(len(prices))
    
    # === LINEAR MODEL RESULTS ===
    _delta_P = np.diff(prices)
    _last_t = len(prices) - 1
    lin_t_forecast = np.arange(_last_t + 1, _last_t + 1 + forecast_horizon)

    # -- Fixed K (Constant): k0 from raw gap, solve_epam standard --
    if len(prices) >= 2:
        _g0 = lin_d_func(prices[0]) - lin_s_func(prices[0])
        lin_k0 = (prices[1] - prices[0]) / _g0 if abs(_g0) > 1e-6 else 0.0
    else:
        lin_k0 = 0.0
    lin_k_const_func = lambda t, k=lin_k0: k

    lin_P_sim_const = solve_epam(prices[0], lin_k_const_func, lin_d_func, lin_s_func, time_steps)
    lin_mape_const = mean_absolute_percentage_error(prices, lin_P_sim_const)
    _t_in = np.concatenate(([_last_t], lin_t_forecast))
    lin_P_fc_const = solve_epam(lin_P_sim_const[-1], lin_k_const_func, lin_d_func, lin_s_func, _t_in)[1:]

    # -- Rolling K + Adjusted Gap: k_raw from gap_adj, matching notebook Section 7c --
    lin_d_pred = np.array([lin_d_func(p) for p in prices])
    lin_s_pred = np.array([lin_s_func(p) for p in prices])
    _, lin_gap_mean, lin_gap_adj = compute_adjusted_gap(lin_d_pred, lin_s_pred)

    lin_ag_k_dyn = np.array([
        dp / g if abs(g) > 1e-4 else lin_k0
        for dp, g in zip(_delta_P, lin_gap_adj[:-1])
    ])
    lin_ag_k_dyn = np.append(lin_ag_k_dyn, lin_ag_k_dyn[-1] if len(lin_ag_k_dyn) > 0 else 0.0)

    lin_k_roll_vals, _ = compute_rolling_k(lin_ag_k_dyn, time_steps, rolling_window)
    lin_k_roll_func = _interp1d(time_steps, lin_k_roll_vals, kind='previous', fill_value='extrapolate')

    lin_P_sim_roll = solve_epam_adjusted_gap(prices[0], lin_k_roll_func, lin_gap_adj, time_steps)
    lin_mape_roll = mean_absolute_percentage_error(prices, lin_P_sim_roll)

    lin_k_roll_fc_vals = forecast_rolling_k(lin_ag_k_dyn, time_steps, lin_t_forecast, rolling_window)
    _all_t = np.concatenate([time_steps, lin_t_forecast])
    _all_k = np.concatenate([lin_k_roll_vals, lin_k_roll_fc_vals])
    lin_k_roll_fc_func = _interp1d(_all_t, _all_k, kind='previous', fill_value='extrapolate')
    _lin_d_fc = np.array([lin_d_func(prices[-1])] * len(lin_t_forecast))
    _lin_s_fc = np.array([lin_s_func(prices[-1])] * len(lin_t_forecast))
    _lin_gap_fc = (_lin_d_fc - _lin_s_fc) - lin_gap_mean
    lin_P_fc_roll = solve_epam_adjusted_gap(lin_P_sim_roll[-1], lin_k_roll_fc_func, _lin_gap_fc, lin_t_forecast) if len(lin_t_forecast) > 0 else np.array([])

    # === POLYNOMIAL MODEL RESULTS ===
    # -- Fixed K (Constant): k0 from raw gap, solve_epam standard --
    if len(prices) >= 2:
        _g0 = poly_d_func(prices[0]) - poly_s_func(prices[0])
        poly_k0 = (prices[1] - prices[0]) / _g0 if abs(_g0) > 1e-6 else 0.0
    else:
        poly_k0 = 0.0
    poly_k_const_func = lambda t, k=poly_k0: k

    poly_P_sim_const = solve_epam(prices[0], poly_k_const_func, poly_d_func, poly_s_func, time_steps)
    poly_mape_const = mean_absolute_percentage_error(prices, poly_P_sim_const)
    poly_t_forecast = lin_t_forecast
    _t_in = np.concatenate(([_last_t], poly_t_forecast))
    poly_P_fc_const = solve_epam(poly_P_sim_const[-1], poly_k_const_func, poly_d_func, poly_s_func, _t_in)[1:]

    # -- Rolling K + Adjusted Gap --
    poly_d_pred = np.array([poly_d_func(p) for p in prices])
    poly_s_pred = np.array([poly_s_func(p) for p in prices])
    _, poly_gap_mean, poly_gap_adj = compute_adjusted_gap(poly_d_pred, poly_s_pred)

    poly_ag_k_dyn = np.array([
        dp / g if abs(g) > 1e-4 else poly_k0
        for dp, g in zip(_delta_P, poly_gap_adj[:-1])
    ])
    poly_ag_k_dyn = np.append(poly_ag_k_dyn, poly_ag_k_dyn[-1] if len(poly_ag_k_dyn) > 0 else 0.0)

    poly_k_roll_vals, _ = compute_rolling_k(poly_ag_k_dyn, time_steps, rolling_window)
    poly_k_roll_func = _interp1d(time_steps, poly_k_roll_vals, kind='previous', fill_value='extrapolate')

    poly_P_sim_roll = solve_epam_adjusted_gap(prices[0], poly_k_roll_func, poly_gap_adj, time_steps)
    poly_mape_roll = mean_absolute_percentage_error(prices, poly_P_sim_roll)

    poly_k_roll_fc_vals = forecast_rolling_k(poly_ag_k_dyn, time_steps, poly_t_forecast, rolling_window)
    _all_t = np.concatenate([time_steps, poly_t_forecast])
    _all_k = np.concatenate([poly_k_roll_vals, poly_k_roll_fc_vals])
    poly_k_roll_fc_func = _interp1d(_all_t, _all_k, kind='previous', fill_value='extrapolate')
    _poly_d_fc = np.array([poly_d_func(prices[-1])] * len(poly_t_forecast))
    _poly_s_fc = np.array([poly_s_func(prices[-1])] * len(poly_t_forecast))
    _poly_gap_fc = (_poly_d_fc - _poly_s_fc) - poly_gap_mean
    poly_P_fc_roll = solve_epam_adjusted_gap(poly_P_sim_roll[-1], poly_k_roll_fc_func, _poly_gap_fc, poly_t_forecast) if len(poly_t_forecast) > 0 else np.array([])

    # === MULTIVARIATE MODEL RESULTS ===
    df_na = df.copy()
    df_na['Time_Index'] = time_steps
    # Multivariate is always fitted on Raw values to get the relationship with its respective price and time.
    model_d_na, (na_d_intercept, na_d_coefs), mape_d_na = perform_multivariate_regression(df_na, demand_price_col, 'Time_Index', demand_qty_col)
    model_s_na, (na_s_intercept, na_s_coefs), mape_s_na = perform_multivariate_regression(df_na, supply_price_col, 'Time_Index', supply_qty_col)

    raw_na_d = lambda p, t: na_d_intercept + na_d_coefs[0]*p + na_d_coefs[1]*t
    raw_na_s = lambda p, t: na_s_intercept + na_s_coefs[0]*p + na_s_coefs[1]*t
    
    if use_bridging:
        if bridging_master == demand_price_col:
            na_d_func = raw_na_d
            na_s_func = lambda p, t: raw_na_s(bridge_lin_func(p), t)
        else:
            na_d_func = lambda p, t: raw_na_d(bridge_lin_func(p), t)
            na_s_func = raw_na_s
    else:
        na_d_func = raw_na_d
        na_s_func = raw_na_s
        
    y_pred_na_d = np.array([na_d_func(prices[i], time_steps[i]) for i in range(len(prices))])
    y_pred_na_s = np.array([na_s_func(prices[i], time_steps[i]) for i in range(len(prices))])

    # -- Fixed K (Constant): k0 from raw gap, solve_non_autonomous_epam standard --
    if len(prices) >= 2:
        _g0 = na_d_func(prices[0], time_steps[0]) - na_s_func(prices[0], time_steps[0])
        na_k0 = (prices[1] - prices[0]) / _g0 if abs(_g0) > 1e-6 else 0.0
    else:
        na_k0 = 0.0
    na_k_const_func = lambda t, k=na_k0: k

    na_P_sim_const = solve_non_autonomous_epam(prices[0], na_k_const_func, na_d_func, na_s_func, time_steps)
    na_mape_const = mean_absolute_percentage_error(prices, na_P_sim_const) if len(prices) > 0 else 0
    na_t_forecast = lin_t_forecast
    _t_in = np.concatenate(([_last_t], na_t_forecast))
    na_P_fc_const = solve_non_autonomous_epam(na_P_sim_const[-1], na_k_const_func, na_d_func, na_s_func, _t_in)[1:]

    # -- Rolling K + Adjusted Gap (Non-Autonomous): k_raw from gap_adj --
    # Adjusted gap for multivariate (time-aware)
    na_ag_Qd = np.array([na_d_func(prices[i], time_steps[i]) for i in range(len(prices))])
    na_ag_Qs = np.array([na_s_func(prices[i], time_steps[i]) for i in range(len(prices))])
    _, na_gap_mean, na_gap_adj = compute_adjusted_gap(na_ag_Qd, na_ag_Qs)

    na_ag_k_dyn = np.array([
        dp / g if abs(g) > 1e-4 else na_k0
        for dp, g in zip(_delta_P, na_gap_adj[:-1])
    ])
    na_ag_k_dyn = np.append(na_ag_k_dyn, na_ag_k_dyn[-1] if len(na_ag_k_dyn) > 0 else 0.0)

    na_k_roll_vals, _ = compute_rolling_k(na_ag_k_dyn, time_steps, rolling_window)
    na_k_roll_func = _interp1d(time_steps, na_k_roll_vals, kind='previous', fill_value='extrapolate') if len(na_k_roll_vals) > 1 else lambda t: 0.0

    na_P_sim_roll = solve_epam_adjusted_gap(prices[0], na_k_roll_func, na_gap_adj, time_steps)
    na_mape_roll = mean_absolute_percentage_error(prices, na_P_sim_roll) if len(prices) > 0 else 0

    na_k_roll_fc_vals = forecast_rolling_k(na_ag_k_dyn, time_steps, na_t_forecast, rolling_window)
    _all_t = np.concatenate([time_steps, na_t_forecast])
    _all_k = np.concatenate([na_k_roll_vals, na_k_roll_fc_vals])
    na_k_roll_fc_func = _interp1d(_all_t, _all_k, kind='previous', fill_value='extrapolate')
    _na_Qd_fc = np.array([na_d_func(prices[-1], t) for t in na_t_forecast])
    _na_Qs_fc = np.array([na_s_func(prices[-1], t) for t in na_t_forecast])
    _na_gap_fc = (_na_Qd_fc - _na_Qs_fc) - na_gap_mean
    na_P_fc_roll = solve_epam_adjusted_gap(na_P_sim_roll[-1], na_k_roll_fc_func, _na_gap_fc, na_t_forecast) if len(na_t_forecast) > 0 else np.array([])

    # Expose for AI tab
    k_values = lin_ag_k_dyn
    P_forecast = lin_P_fc_const

    # === VISUALIZATION ===
    if selection == nav_options[1]:

        def format_math_expr(coeffs, vars_list):
            parts = []
            for c, v in zip(coeffs, vars_list):
                if abs(c) < 1e-8:
                    continue
                # Use scientific notation for very small coefficients
                ac = abs(c)
                if ac < 0.01:
                    import math
                    exp = math.floor(math.log10(ac))
                    mantissa = ac / (10 ** exp)
                    c_str = f"{mantissa:.2f} \\times 10^{{{exp}}}"
                else:
                    c_str = f"{ac:.4f}"
                if v in ["", "1"]:
                    term = c_str
                else:
                    if c_str == "1.0000":
                        term = v
                    else:
                        term = f"{c_str} \\cdot {v}"
                
                if not parts:
                    parts.append(f"-{term}" if c < 0 else term)
                else:
                    parts.append(f" - {term}" if c < 0 else f" + {term}")
            return " ".join(parts) if parts else "0"

        def render_model_section(title,
                                 X_plot_d, y_d, y_pred_d, mape_d, eq_d,
                                 X_plot_s, y_s, y_pred_s, mape_s, eq_s,
                                 k_const, P_sim_const, P_fc_const, mape_const,
                                 k_roll_vals, k_roll_fc_vals, P_sim_roll, P_fc_roll, mape_roll,
                                 P_actual, time_ax, t_fc, epam_eq,
                                 sd_x_label="Price", d_label="D(P)", s_label="S(P)",
                                 is_3d=False, func_d=None, func_s=None,
                                 section_key="model"):
            with st.container(border=True):
                st.subheader(title)

                # Row 1: Demand and Supply side by side
                c_demand, c_supply = st.columns(2)
                with c_demand:
                    fig_d = go.Figure()
                    
                    if is_3d:
                        P_vals, T_vals = X_plot_d
                        fig_d.add_trace(go.Scatter3d(x=P_vals, y=T_vals, z=y_d, mode='markers', name='Actual',
                                                     marker=dict(color='#FFD166', size=5,
                                                                 line=dict(color='#FF6B35', width=1))))
                        
                        p_grid = np.linspace(min(P_vals), max(P_vals), 20)
                        t_grid = np.linspace(min(T_vals), max(T_vals), 20)
                        P_mesh, T_mesh = np.meshgrid(p_grid, t_grid)
                        Z_mesh = np.array([func_d(p, tm) for p, tm in zip(P_mesh.flatten(), T_mesh.flatten())]).reshape(P_mesh.shape)
                        
                        fig_d.add_trace(go.Surface(x=p_grid, y=t_grid, z=Z_mesh, name='Fitted Plane', opacity=0.75, colorscale='Blues', showscale=False))
                        _axis_style = dict(backgroundcolor='#0D1B2A', gridcolor='rgba(255,255,255,0.12)', zerolinecolor='rgba(255,255,255,0.20)', tickfont=dict(color='#9CA3AF'))
                        fig_d.update_layout(scene=dict(
                            bgcolor='#0D1B2A',
                            xaxis=dict(title='Price', title_font=dict(color='#9CA3AF'), **_axis_style),
                            yaxis=dict(title='Time', title_font=dict(color='#9CA3AF'), **_axis_style),
                            zaxis=dict(title='Qty', title_font=dict(color='#9CA3AF'), **_axis_style),
                        ), margin=dict(l=0, r=0, b=0, t=30), paper_bgcolor='rgba(0,0,0,0)')
                    else:
                        sort_idx_d = np.argsort(X_plot_d.flatten())
                        sorted_x_d = X_plot_d.flatten()[sort_idx_d]
                        sorted_y_pred_d = np.array(y_pred_d)[sort_idx_d]
                        
                        fig_d.add_trace(go.Scatter(x=X_plot_d.flatten(), y=y_d, mode='markers', name='Actual', marker=dict(color=t['marker'], size=6)))
                        fig_d.add_trace(go.Scatter(x=sorted_x_d, y=sorted_y_pred_d, mode='lines', name='Fitted', line=dict(color=t['line_blue'], width=2)))
                        fig_d.update_layout(**get_plot_layout("Demand", sd_x_label, "Qty"))
                        
                    fig_d.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
                    st.plotly_chart(fig_d, use_container_width=True, key=f"{section_key}_demand")
                    st.latex(rf"{d_label} = {eq_d}")
                    st.caption(f"MAPE: **{mape_d:.2%}**")

                with c_supply:
                    fig_s = go.Figure()
                    
                    if is_3d:
                        P_vals, T_vals = X_plot_s
                        fig_s.add_trace(go.Scatter3d(x=P_vals, y=T_vals, z=y_s, mode='markers', name='Actual',
                                                     marker=dict(color='#06D6A0', size=5,
                                                                 line=dict(color='#05A880', width=1))))
                        
                        p_grid = np.linspace(min(P_vals), max(P_vals), 20)
                        t_grid = np.linspace(min(T_vals), max(T_vals), 20)
                        P_mesh, T_mesh = np.meshgrid(p_grid, t_grid)
                        Z_mesh = np.array([func_s(p, tm) for p, tm in zip(P_mesh.flatten(), T_mesh.flatten())]).reshape(P_mesh.shape)
                        
                        fig_s.add_trace(go.Surface(x=p_grid, y=t_grid, z=Z_mesh, name='Fitted Plane', opacity=0.75, colorscale='Reds', showscale=False))
                        _axis_style = dict(backgroundcolor='#0D1B2A', gridcolor='rgba(255,255,255,0.12)', zerolinecolor='rgba(255,255,255,0.20)', tickfont=dict(color='#9CA3AF'))
                        fig_s.update_layout(scene=dict(
                            bgcolor='#0D1B2A',
                            xaxis=dict(title='Price', title_font=dict(color='#9CA3AF'), **_axis_style),
                            yaxis=dict(title='Time', title_font=dict(color='#9CA3AF'), **_axis_style),
                            zaxis=dict(title='Qty', title_font=dict(color='#9CA3AF'), **_axis_style),
                        ), margin=dict(l=0, r=0, b=0, t=30), paper_bgcolor='rgba(0,0,0,0)')
                    else:
                        sort_idx_s = np.argsort(X_plot_s.flatten())
                        sorted_x_s = X_plot_s.flatten()[sort_idx_s]
                        sorted_y_pred_s = np.array(y_pred_s)[sort_idx_s]
                        
                        fig_s.add_trace(go.Scatter(x=X_plot_s.flatten(), y=y_s, mode='markers', name='Actual', marker=dict(color=t['marker'], size=6)))
                        fig_s.add_trace(go.Scatter(x=sorted_x_s, y=sorted_y_pred_s, mode='lines', name='Fitted', line=dict(color=t['line_red'], width=2)))
                        fig_s.update_layout(**get_plot_layout("Supply", sd_x_label, "Qty"))
                        
                    fig_s.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
                    st.plotly_chart(fig_s, use_container_width=True, key=f"{section_key}_supply")
                    st.latex(rf"{s_label} = {eq_s}")
                    st.caption(f"MAPE: **{mape_s:.2%}**")

                # Row 2: K Chart and EPAM Chart (both simulations overlaid)
                c_k, c_epam = st.columns(2)
                with c_k:
                    fig_k = go.Figure()
                    # Fixed K as horizontal line
                    fig_k.add_trace(go.Scatter(
                        x=list(time_ax) + list(t_fc),
                        y=[k_const] * (len(time_ax) + len(t_fc)),
                        mode='lines', name=f'Fixed K = {k_const:.6f}',
                        line=dict(color=t['line_red'], dash='dash', width=2)))
                    # Rolling K historical
                    fig_k.add_trace(go.Scatter(
                        x=list(time_ax), y=list(k_roll_vals),
                        mode='lines', name='Rolling K (Historical)',
                        line=dict(color=t['line_blue'], width=2)))
                    # Rolling K forecast
                    if len(t_fc) > 0 and len(k_roll_fc_vals) > 0:
                        fig_k.add_trace(go.Scatter(
                            x=list(t_fc), y=list(k_roll_fc_vals),
                            mode='lines', name='Rolling K (Forecast)',
                            line=dict(color=t['line_blue'], dash='dot', width=2)))
                    fig_k.update_layout(**get_plot_layout("Adjustment Speed (K)", "Time Step", "K"))
                    fig_k.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_k, use_container_width=True, key=f"{section_key}_k")

                with c_epam:
                    fig_ep = go.Figure()
                    # Actual prices
                    fig_ep.add_trace(go.Scatter(
                        x=list(time_ax), y=list(P_actual),
                        mode='lines+markers', name='Actual',
                        line=dict(color=t['marker'], width=2), marker=dict(size=4)))
                    # Fixed K simulation
                    fig_ep.add_trace(go.Scatter(
                        x=list(time_ax), y=list(P_sim_const),
                        mode='lines', name=f'Fixed K (MAPE={mape_const:.2%})',
                        line=dict(color=t['line_red'], dash='dash', width=2)))
                    # Fixed K forecast
                    if len(t_fc) > 0 and len(P_fc_const) > 0:
                        fig_ep.add_trace(go.Scatter(
                            x=list(t_fc), y=list(P_fc_const),
                            mode='lines', name='Fixed K Forecast',
                            line=dict(color=t['line_red'], dash='dot', width=1.5)))
                    # Rolling K simulation
                    fig_ep.add_trace(go.Scatter(
                        x=list(time_ax), y=list(P_sim_roll),
                        mode='lines', name=f'Rolling K (MAPE={mape_roll:.2%})',
                        line=dict(color=t['line_blue'], width=2)))
                    # Rolling K forecast
                    if len(t_fc) > 0 and len(P_fc_roll) > 0:
                        fig_ep.add_trace(go.Scatter(
                            x=list(t_fc), y=list(P_fc_roll),
                            mode='lines', name='Rolling K Forecast',
                            line=dict(color=t['line_blue'], dash='dot', width=1.5)))
                    fig_ep.update_layout(**get_plot_layout("Price Simulation (EPAM)", "Time Step", "Price"))
                    fig_ep.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_ep, use_container_width=True, key=f"{section_key}_epam")

                # Row 3: Metrics
                st.latex(epam_eq)
                m1, m2 = st.columns(2)
                with m1:
                    st.caption(f"**Fixed K** = `{k_const:.6f}` · MAPE: **{mape_const:.2%}**")
                with m2:
                    st.caption(f"**Rolling K** (w={rolling_window}) · MAPE: **{mape_roll:.2%}**")

        # --- Section 1: Linear ---
        lin_d_coeffs = [lin_model_d.intercept_, lin_model_d.coef_[0]]
        lin_s_coeffs = [lin_model_s.intercept_, lin_model_s.coef_[0]]
        lin_eq_d = format_math_expr(lin_d_coeffs, ["", "P"])
        lin_eq_s = format_math_expr(lin_s_coeffs, ["", "P"])
        
        lin_gap_coeffs = [d - s for d, s in zip(lin_d_coeffs, lin_s_coeffs)]
        lin_gap_eq = format_math_expr(lin_gap_coeffs, ["", "P"])
        lin_epam_eq = rf"\frac{{dP}}{{dt}} = K \cdot \left[ {lin_gap_eq} \right]"

        render_model_section(
            "Linear Regression",
            X_plot_d, y_demand, y_pred_lin_d, mape_lin_d, lin_eq_d,
            X_plot_s, y_supply, y_pred_lin_s, mape_lin_s, lin_eq_s,
            lin_k0, lin_P_sim_const, lin_P_fc_const, lin_mape_const,
            lin_k_roll_vals, lin_k_roll_fc_vals, lin_P_sim_roll, lin_P_fc_roll, lin_mape_roll,
            prices, time_steps, lin_t_forecast, epam_eq=lin_epam_eq, sd_x_label=x_label,
            section_key="lin",
        )

        # --- Section 2: Polynomial ---
        poly_vars = [name.replace("x0", "P").replace("x1", "t").replace(" ", "") for name in poly_feat_d.get_feature_names_out()]
        poly_vars[0] = ""
        poly_d_coeffs = [poly_model_d.intercept_] + list(poly_model_d.coef_[1:])
        poly_s_coeffs = [poly_model_s.intercept_] + list(poly_model_s.coef_[1:])
        
        poly_eq_d = format_math_expr(poly_d_coeffs, poly_vars)
        poly_eq_s = format_math_expr(poly_s_coeffs, poly_vars)
        
        poly_gap_coeffs = [d - s for d, s in zip(poly_d_coeffs, poly_s_coeffs)]
        poly_gap_eq = format_math_expr(poly_gap_coeffs, poly_vars)
        poly_epam_eq = rf"\frac{{dP}}{{dt}} = K \cdot \left[ {poly_gap_eq} \right]"

        render_model_section(
            "Polynomial Regression",
            X_plot_d, y_demand, y_pred_poly_d, mape_poly_d, poly_eq_d,
            X_plot_s, y_supply, y_pred_poly_s, mape_poly_s, poly_eq_s,
            poly_k0, poly_P_sim_const, poly_P_fc_const, poly_mape_const,
            poly_k_roll_vals, poly_k_roll_fc_vals, poly_P_sim_roll, poly_P_fc_roll, poly_mape_roll,
            prices, time_steps, poly_t_forecast, epam_eq=poly_epam_eq, sd_x_label=x_label,
            section_key="poly",
        )

        # --- Section 3: Multivariate ---
        na_vars = ["", "P", "t"]
        na_d_coeffs = [na_d_intercept, na_d_coefs[0], na_d_coefs[1]]
        na_s_coeffs = [na_s_intercept, na_s_coefs[0], na_s_coefs[1]]
        
        na_eq_d = format_math_expr(na_d_coeffs, na_vars)
        na_eq_s = format_math_expr(na_s_coeffs, na_vars)
        
        na_gap_coeffs = [d - s for d, s in zip(na_d_coeffs, na_s_coeffs)]
        na_gap_eq = format_math_expr(na_gap_coeffs, na_vars)
        na_epam_eq = rf"\frac{{dP}}{{dt}} = K \cdot \left[ {na_gap_eq} \right]"

        render_model_section(
            "Multivariate Regression",
            (X_plot_d.flatten(), time_steps), y_demand, y_pred_na_d, mape_d_na, na_eq_d,
            (X_plot_s.flatten(), time_steps), y_supply, y_pred_na_s, mape_s_na, na_eq_s,
            na_k0, na_P_sim_const, na_P_fc_const, na_mape_const,
            na_k_roll_vals, na_k_roll_fc_vals, na_P_sim_roll, na_P_fc_roll, na_mape_roll,
            prices, time_steps, na_t_forecast, epam_eq=na_epam_eq, sd_x_label="Price",
            d_label="D(P,t)", s_label="S(P,t)",
            is_3d=True, func_d=na_d_func, func_s=na_s_func,
            section_key="mv",
        )

        # --- MAPE Summary Table ---
        st.markdown("---")
        st.subheader("📊 Ringkasan MAPE Simulasi")
        mape_data = {
            "Model": ["Linear", "Polynomial", "Multivariate"],
            "Fixed K (In-sample)": [f"{lin_mape_const:.2%}", f"{poly_mape_const:.2%}", f"{na_mape_const:.2%}"],
            "Rolling K (In-sample)": [f"{lin_mape_roll:.2%}", f"{poly_mape_roll:.2%}", f"{na_mape_roll:.2%}"],
        }
        df_mape = pd.DataFrame(mape_data).set_index("Model")
        st.table(df_mape)


    # --- AI REPORT TAB ---
    if selection == nav_options[2]:
        with st.container(border=True):
            st.header("AI Market Report")
            st.markdown("Generate a comprehensive narrative report explaining these findings.")
            
            # Automated Report Generation
        with st.spinner("🤖 Consulting AI Analyst — this may take 15–30 seconds..."):
            # ── Assemble rich, data-specific context for the AI ──
            context = []

            # --- Data Overview ---
            context.append("=== DATA OVERVIEW ===")
            context.append(f"Total observations (time steps): {len(prices)}")
            context.append(f"Actual price range: {prices.min():.4f} → {prices.max():.4f}")
            context.append(f"First actual price: {prices[0]:.4f}")
            context.append(f"Last actual price: {prices[-1]:.4f}")
            price_change_pct = ((prices[-1] - prices[0]) / abs(prices[0])) * 100 if abs(prices[0]) > 1e-9 else 0
            context.append(f"Overall price change over history: {price_change_pct:+.2f}%")
            context.append(f"Forecast horizon: {forecast_horizon} steps ahead")
            context.append(f"Rolling window parameter: {rolling_window} periods")
            context.append(f"Polynomial regression degree: {poly_degree}")

            # --- Regression Fit Quality ---
            context.append("")
            context.append("=== REGRESSION FIT QUALITY (Curve-fitting MAPE) ===")
            context.append(f"Linear Model — Demand curve MAPE: {mape_lin_d:.2%}, Supply curve MAPE: {mape_lin_s:.2%}")
            context.append(f"Polynomial Model — Demand curve MAPE: {mape_poly_d:.2%}, Supply curve MAPE: {mape_poly_s:.2%}")
            context.append(f"Multivariate Model — Demand MAPE: {mape_d_na:.2%}, Supply MAPE: {mape_s_na:.2%}")

            # --- Adjustment Speed K per model ---
            context.append("")
            context.append("=== ADJUSTMENT SPEED K (Market Reactivity) ===")
            context.append(f"LINEAR MODEL — Fixed K: {lin_k0:.6f} | Mean Rolling K: {np.mean(lin_k_roll_vals):.6f} | Std Rolling K: {np.std(lin_k_roll_vals):.6f}")
            context.append(f"POLYNOMIAL MODEL — Fixed K: {poly_k0:.6f} | Mean Rolling K: {np.mean(poly_k_roll_vals):.6f} | Std Rolling K: {np.std(poly_k_roll_vals):.6f}")
            context.append(f"MULTIVARIATE MODEL — Fixed K: {na_k0:.6f} | Mean Rolling K: {np.mean(na_k_roll_vals):.6f} | Std Rolling K: {np.std(na_k_roll_vals):.6f}")

            # --- EPAM Simulation Accuracy ---
            context.append("")
            context.append("=== EPAM SIMULATION ACCURACY (In-sample MAPE) ===")
            context.append(f"Linear — Fixed K MAPE: {lin_mape_const:.2%} | Rolling K MAPE: {lin_mape_roll:.2%}")
            context.append(f"Polynomial — Fixed K MAPE: {poly_mape_const:.2%} | Rolling K MAPE: {poly_mape_roll:.2%}")
            context.append(f"Multivariate — Fixed K MAPE: {na_mape_const:.2%} | Rolling K MAPE: {na_mape_roll:.2%}")

            # Determine best model
            all_mapes = {
                "Linear Fixed K": lin_mape_const, "Linear Rolling K": lin_mape_roll,
                "Polynomial Fixed K": poly_mape_const, "Polynomial Rolling K": poly_mape_roll,
                "Multivariate Fixed K": na_mape_const, "Multivariate Rolling K": na_mape_roll,
            }
            best_model_name = min(all_mapes, key=all_mapes.get)
            context.append(f"→ Best performing model: {best_model_name} with MAPE = {all_mapes[best_model_name]:.2%}")

            # --- Price Forecasts ---
            context.append("")
            context.append("=== PRICE FORECASTS ===")
            forecast_summary = [
                ("Linear Fixed K",       lin_P_fc_const),
                ("Linear Rolling K",     lin_P_fc_roll),
                ("Polynomial Fixed K",   poly_P_fc_const),
                ("Polynomial Rolling K", poly_P_fc_roll),
                ("Multivariate Fixed K", na_P_fc_const),
                ("Multivariate Rolling K", na_P_fc_roll),
            ]
            for fc_name, fc_arr in forecast_summary:
                if fc_arr is not None and len(fc_arr) > 0:
                    fc_direction = "UPWARD ↑" if fc_arr[-1] > prices[-1] else "DOWNWARD ↓"
                    fc_chg = ((fc_arr[-1] - prices[-1]) / abs(prices[-1])) * 100 if abs(prices[-1]) > 1e-9 else 0
                    context.append(f"{fc_name}: {prices[-1]:.4f} → {fc_arr[-1]:.4f} ({fc_chg:+.2f}%, {fc_direction})")

            # --- Supply-Demand Gap at Latest Price ---
            context.append("")
            context.append("=== CURRENT SUPPLY-DEMAND GAP (at last observed price) ===")
            latest_p = prices[-1]
            lin_gap_latest = lin_d_func(latest_p) - lin_s_func(latest_p)
            poly_gap_latest = poly_d_func(latest_p) - poly_s_func(latest_p)
            gap_sign_lin = "EXCESS DEMAND (shortage → upward price pressure)" if lin_gap_latest > 0 else "EXCESS SUPPLY (surplus → downward price pressure)"
            gap_sign_poly = "EXCESS DEMAND (shortage → upward price pressure)" if poly_gap_latest > 0 else "EXCESS SUPPLY (surplus → downward price pressure)"
            context.append(f"Linear model gap (Qd - Qs) at P={latest_p:.4f}: {lin_gap_latest:.4f} → {gap_sign_lin}")
            context.append(f"Polynomial model gap (Qd - Qs) at P={latest_p:.4f}: {poly_gap_latest:.4f} → {gap_sign_poly}")

            full_context = "\n".join(context)

            if groq_api_key:
                report = get_ai_analysis(groq_api_key, full_context)
                st.markdown(report)
            else:
                st.warning("Please provide a Groq API Key in the Setup tab to unlock AI insights.")
                
elif 'df_raw' not in st.session_state:
    # Landing Page Prompt
    if selection != nav_options[0]:
        st.info("👈 Please upload your data in the 'Setup' tab to begin.")
