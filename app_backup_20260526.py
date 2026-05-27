import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from regression_utils import perform_linear_regression, perform_polynomial_regression, mean_absolute_percentage_error, calculate_dynamic_k, solve_epam, perform_multivariate_regression, solve_non_autonomous_epam
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
        prompt = f"""
        You are an expert Economic Analyst. Analyze the following Differential Equation (DE) Market Simulation results.
        
        Context:
        {context_text}
        
        Provide a report for a business stakeholder explaining the DE results (rather than simple curve fitting):
        1. **Executive Summary**: High-level market status based on the DE simulation.
        2. **Market Dynamics**: How prices evolve over time based on the interaction between Supply and Demand.
        3. **Adjustment Speed (K)**: What 'K' tells us about market reactivity and stability in this DE model.
        4. **Forecast**: Future price trajectories derived from the DE simulation.
        5. **Strategic Recommendations**.
        6. **Focus on how it is understood in the context of economics and business**.
        7. **Keep it simple and easily understood by everyone**
        
        Keep it professional, clear, and actionable.
        """
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
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
Expand "Advanced Configuration" to tune imputation, polynomial degree, price bridging, and simulation parameters.

**3. Run Analysis**  
Click **Run Market Analysis** to compute regressions and simulations.

**4. Deep Dive**  
Switch to the Deep Dive tab for detailed curve analysis, EPAM price simulations, and non-autonomous time-aware modeling.

**5. AI Report**  
Switch to the AI Report tab for an AI-generated market narrative (requires a Groq API key).
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
                'epam_mod': 'Both',
                'calc_meth': st.session_state.calc_method,
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
                            calc_method = st.selectbox("Adjustment Speed (K)", ["Dynamic (Time-Varying)", "Initial (Constant)"], key="calc_method")
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


    calc_method = conf['calc_meth']
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
    if calc_method == "Dynamic (Time-Varying)":
        lin_k_values, lin_k_func = calculate_dynamic_k(prices, lin_d_func, lin_s_func)
    else:
        if len(prices) >= 2:
            _P0, _P1 = prices[0], prices[1]
            _gap0 = lin_d_func(_P0) - lin_s_func(_P0)
            _k_init = (_P1 - _P0) / _gap0 if abs(_gap0) > 1e-6 else 0
        else:
            _k_init = 0
        lin_k_values = np.array([_k_init] * len(prices))
        lin_k_func = lambda t: _k_init

    lin_P_sim = solve_epam(prices[0], lin_k_func, lin_d_func, lin_s_func, time_steps)
    lin_mape_sim = mean_absolute_percentage_error(prices, lin_P_sim)
    _last_t = len(prices) - 1
    lin_t_forecast = np.arange(_last_t + 1, _last_t + 1 + forecast_horizon)
    _k_clean = np.nan_to_num(lin_k_values)
    try:
        _sl, _ic = np.polyfit(time_steps, _k_clean, 1)
    except:
        _sl, _ic = 0, 0
    lin_k_future_func = lambda t, s=_sl, i=_ic: s * t + i
    lin_k_future_vals = _sl * lin_t_forecast + _ic
    if len(prices) >= 1:
        _t_in = np.concatenate(([_last_t], lin_t_forecast))
        lin_P_forecast = solve_epam(lin_P_sim[-1], lin_k_future_func, lin_d_func, lin_s_func, _t_in)[1:]
    else:
        lin_P_forecast = np.array([])

    # === POLYNOMIAL MODEL RESULTS ===
    if calc_method == "Dynamic (Time-Varying)":
        poly_k_values, poly_k_func = calculate_dynamic_k(prices, poly_d_func, poly_s_func)
    else:
        if len(prices) >= 2:
            _P0, _P1 = prices[0], prices[1]
            _gap0 = poly_d_func(_P0) - poly_s_func(_P0)
            _k_init = (_P1 - _P0) / _gap0 if abs(_gap0) > 1e-6 else 0
        else:
            _k_init = 0
        poly_k_values = np.array([_k_init] * len(prices))
        poly_k_func = lambda t: _k_init

    poly_P_sim = solve_epam(prices[0], poly_k_func, poly_d_func, poly_s_func, time_steps)
    poly_mape_sim = mean_absolute_percentage_error(prices, poly_P_sim)
    poly_t_forecast = lin_t_forecast
    _k_clean = np.nan_to_num(poly_k_values)
    try:
        _sl, _ic = np.polyfit(time_steps, _k_clean, 1)
    except:
        _sl, _ic = 0, 0
    poly_k_future_func = lambda t, s=_sl, i=_ic: s * t + i
    poly_k_future_vals = _sl * poly_t_forecast + _ic
    if len(prices) >= 1:
        _t_in = np.concatenate(([_last_t], poly_t_forecast))
        poly_P_forecast = solve_epam(poly_P_sim[-1], poly_k_future_func, poly_d_func, poly_s_func, _t_in)[1:]
    else:
        poly_P_forecast = np.array([])

    # === MULTIVARIATE MODEL RESULTS ===
    from scipy.interpolate import interp1d as _interp1d
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

    k_list_na = []
    for i in range(len(prices) - 1):
        p_curr, p_next, t_curr = prices[i], prices[i + 1], time_steps[i]
        d_val = na_d_func(p_curr, t_curr)
        s_val = na_s_func(p_curr, t_curr)
        gap = d_val - s_val
        k_t = (p_next - p_curr) / gap if abs(gap) > 1e-4 else (k_list_na[-1] if k_list_na else 0.0)
        k_list_na.append(k_t)
    if len(prices) > 0:
        k_list_na.append(k_list_na[-1] if k_list_na else 0.0)

    na_k_values = np.array(k_list_na)

    if calc_method == "Dynamic (Time-Varying)":
        na_k_func = _interp1d(time_steps, na_k_values, kind='previous', fill_value="extrapolate") if len(na_k_values) > 1 else lambda t: 0.0
    else:
        _na_k_init = na_k_values[0] if len(na_k_values) > 0 else 0.0
        na_k_values = np.array([_na_k_init] * len(prices))
        na_k_func = lambda t, k=_na_k_init: k

    na_P_sim = solve_non_autonomous_epam(prices[0], na_k_func, na_d_func, na_s_func, time_steps)
    na_mape_sim = mean_absolute_percentage_error(prices, na_P_sim) if len(prices) > 0 else 0

    na_t_forecast = lin_t_forecast
    if calc_method == "Dynamic (Time-Varying)":
        _k_clean_na = np.nan_to_num(k_list_na)
        try:
            _sl_na, _ic_na = np.polyfit(time_steps, _k_clean_na, 1)
        except:
            _sl_na, _ic_na = 0, 0
        na_k_future_func = lambda t, s=_sl_na, i=_ic_na: s * t + i
        na_k_future_vals = _sl_na * na_t_forecast + _ic_na
    else:
        na_k_future_func = na_k_func
        na_k_future_vals = np.array([_na_k_init] * len(na_t_forecast))

    if len(prices) >= 1:
        _t_in = np.concatenate(([_last_t], na_t_forecast))
        na_P_forecast = solve_non_autonomous_epam(na_P_sim[-1], na_k_future_func, na_d_func, na_s_func, _t_in)[1:]
    else:
        na_P_forecast = np.array([])

    # Expose for AI tab
    k_values = lin_k_values
    P_forecast = lin_P_forecast

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
                                 k_vals, k_future, t_fc, k_eq_rhs,
                                 P_actual, P_sim, P_fc, mape_epam,
                                 time_ax, epam_eq, sd_x_label="Price", d_label="D(P)", s_label="S(P)",
                                 k_constant=None, is_3d=False, func_d=None, func_s=None,
                                 section_key="model"):
            with st.container(border=True):
                st.subheader(title)

                # Row 1: Demand and Supply side by side
                c_demand, c_supply = st.columns(2)
                with c_demand:
                    fig_d = go.Figure()
                    
                    if is_3d:
                        P_vals, T_vals = X_plot_d
                        # 3D Scatter for actual points — vivid yellow-orange for high contrast
                        fig_d.add_trace(go.Scatter3d(x=P_vals, y=T_vals, z=y_d, mode='markers', name='Actual',
                                                     marker=dict(color='#FFD166', size=5,
                                                                 line=dict(color='#FF6B35', width=1))))
                        
                        # Generate Mesh Grid for Surface
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
                        # 3D Scatter for actual points — vivid cyan-green for high contrast
                        fig_s.add_trace(go.Scatter3d(x=P_vals, y=T_vals, z=y_s, mode='markers', name='Actual',
                                                     marker=dict(color='#06D6A0', size=5,
                                                                 line=dict(color='#05A880', width=1))))
                        
                        # Generate Mesh Grid for Surface
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

                # Row 2: K and EPAM
                if k_constant is not None:
                    # Constant K: skip K graph, full-width EPAM
                    fig_ep = go.Figure()
                    fig_ep.add_trace(go.Scatter(x=time_ax, y=P_actual, mode='lines+markers', name='Actual', line=dict(color=t['marker'], width=2), marker=dict(size=4)))
                    fig_ep.add_trace(go.Scatter(x=time_ax, y=P_sim, mode='lines', name='Simulated', line=dict(color=t['line_red'], dash='dash', width=2)))
                    fig_ep.add_trace(go.Scatter(x=t_fc, y=P_fc, mode='lines', name='Forecast', line=dict(color=t['line_blue'], dash='dot', width=2)))
                    fig_ep.update_layout(**get_plot_layout("Price Simulation (EPAM)", "Time Step", "Price"))
                    fig_ep.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
                    st.plotly_chart(fig_ep, use_container_width=True, key=f"{section_key}_epam")
                    st.latex(rf"K = {k_constant:.6f} \quad \Rightarrow \quad {epam_eq}")
                    st.caption(f"Simulation MAPE: **{mape_epam:.2%}**")
                else:
                    # Dynamic K: show K graph and EPAM side by side
                    c_k, c_epam = st.columns(2)
                    with c_k:
                        fig_k = go.Figure()
                        fig_k.add_trace(go.Scatter(y=k_vals, mode='lines', name='Historical', line=dict(color=t['line_red'], width=2)))
                        fig_k.add_trace(go.Scatter(x=t_fc, y=k_future, mode='lines', name='Forecast', line=dict(color=t['line_blue'], dash='dot', width=2)))
                        fig_k.update_layout(**get_plot_layout("Adjustment Speed (K)", "Time Step", "K"))
                        fig_k.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
                        st.plotly_chart(fig_k, use_container_width=True, key=f"{section_key}_k")
                        st.latex(rf"K(t) = {k_eq_rhs}")

                    with c_epam:
                        fig_ep = go.Figure()
                        fig_ep.add_trace(go.Scatter(x=time_ax, y=P_actual, mode='lines+markers', name='Actual', line=dict(color=t['marker'], width=2), marker=dict(size=4)))
                        fig_ep.add_trace(go.Scatter(x=time_ax, y=P_sim, mode='lines', name='Simulated', line=dict(color=t['line_red'], dash='dash', width=2)))
                        fig_ep.add_trace(go.Scatter(x=t_fc, y=P_fc, mode='lines', name='Forecast', line=dict(color=t['line_blue'], dash='dot', width=2)))
                        fig_ep.update_layout(**get_plot_layout("Price Simulation (EPAM)", "Time Step", "Price"))
                        fig_ep.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
                        st.plotly_chart(fig_ep, use_container_width=True, key=f"{section_key}_epam")
                        st.latex(epam_eq)
                        st.caption(f"Simulation MAPE: **{mape_epam:.2%}**")

        # --- Section 1: Linear ---
        lin_d_coeffs = [lin_model_d.intercept_, lin_model_d.coef_[0]]
        lin_s_coeffs = [lin_model_s.intercept_, lin_model_s.coef_[0]]
        lin_eq_d = format_math_expr(lin_d_coeffs, ["", "P"])
        lin_eq_s = format_math_expr(lin_s_coeffs, ["", "P"])
        
        lin_gap_coeffs = [d - s for d, s in zip(lin_d_coeffs, lin_s_coeffs)]
        lin_gap_eq = format_math_expr(lin_gap_coeffs, ["", "P"])
        
        _k_clean_lin = np.nan_to_num(lin_k_values)
        try:
            lin_sl, lin_ic = np.polyfit(time_steps, _k_clean_lin, 1)
        except:
            lin_sl, lin_ic = 0, 0
        lin_k_rhs = format_math_expr([lin_ic, lin_sl], ["", "t"])
        
        lin_gap_wrap = f"({lin_gap_eq})" if ("+" in lin_gap_eq or "-" in lin_gap_eq[1:]) else lin_gap_eq
        lin_k_wrap = f"({lin_k_rhs})" if ("+" in lin_k_rhs or "-" in lin_k_rhs[1:]) else lin_k_rhs
        lin_epam_eq = rf"\frac{{dP}}{{dt}} = {lin_k_wrap} \cdot \left[ {lin_gap_eq} \right]"

        _is_constant_k = calc_method != "Dynamic (Time-Varying)"

        render_model_section(
            "Linear Regression",
            X_plot_d, y_demand, y_pred_lin_d, mape_lin_d, lin_eq_d,
            X_plot_s, y_supply, y_pred_lin_s, mape_lin_s, lin_eq_s,
            lin_k_values, lin_k_future_vals, lin_t_forecast, lin_k_rhs,
            prices, lin_P_sim, lin_P_forecast, lin_mape_sim,
            time_steps, epam_eq=lin_epam_eq, sd_x_label=x_label,
            k_constant=lin_k_values[0] if _is_constant_k else None,
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
        
        _k_clean_poly = np.nan_to_num(poly_k_values)
        try:
            poly_sl, poly_ic = np.polyfit(time_steps, _k_clean_poly, 1)
        except:
            poly_sl, poly_ic = 0, 0
        poly_k_rhs = format_math_expr([poly_ic, poly_sl], ["", "t"])
        
        poly_gap_wrap = f"({poly_gap_eq})" if ("+" in poly_gap_eq or "-" in poly_gap_eq[1:]) else poly_gap_eq
        poly_k_wrap = f"({poly_k_rhs})" if ("+" in poly_k_rhs or "-" in poly_k_rhs[1:]) else poly_k_rhs
        poly_epam_eq = rf"\frac{{dP}}{{dt}} = {poly_k_wrap} \cdot \left[ {poly_gap_eq} \right]"

        render_model_section(
            "Polynomial Regression",
            X_plot_d, y_demand, y_pred_poly_d, mape_poly_d, poly_eq_d,
            X_plot_s, y_supply, y_pred_poly_s, mape_poly_s, poly_eq_s,
            poly_k_values, poly_k_future_vals, poly_t_forecast, poly_k_rhs,
            prices, poly_P_sim, poly_P_forecast, poly_mape_sim,
            time_steps, epam_eq=poly_epam_eq, sd_x_label=x_label,
            k_constant=poly_k_values[0] if _is_constant_k else None,
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
        
        _k_clean_na = np.nan_to_num(na_k_values)
        try:
            na_sl, na_ic = np.polyfit(time_steps, _k_clean_na, 1)
        except:
            na_sl, na_ic = 0, 0
        na_k_rhs = format_math_expr([na_ic, na_sl], ["", "t"])
        
        na_gap_wrap = f"({na_gap_eq})" if ("+" in na_gap_eq or "-" in na_gap_eq[1:]) else na_gap_eq
        na_k_wrap = f"({na_k_rhs})" if ("+" in na_k_rhs or "-" in na_k_rhs[1:]) else na_k_rhs
        na_epam_eq = rf"\frac{{dP}}{{dt}} = {na_k_wrap} \cdot \left[ {na_gap_eq} \right]"

        render_model_section(
            "Multivariate Regression",
            (X_plot_d.flatten(), time_steps), y_demand, y_pred_na_d, mape_d_na, na_eq_d,
            (X_plot_s.flatten(), time_steps), y_supply, y_pred_na_s, mape_s_na, na_eq_s,
            na_k_values, na_k_future_vals, na_t_forecast, na_k_rhs,
            prices, na_P_sim, na_P_forecast, na_mape_sim,
            time_steps, epam_eq=na_epam_eq, sd_x_label="Price", d_label="D(P,t)", s_label="S(P,t)",
            k_constant=na_k_values[0] if _is_constant_k else None,
            is_3d=True, func_d=na_d_func, func_s=na_s_func,
            section_key="mv",
        )


    # --- AI REPORT TAB ---
    if selection == nav_options[2]:
        with st.container(border=True):
            st.header("AI Market Report")
            st.markdown("Generate a comprehensive narrative report explaining these findings.")
            
            # Automated Report Generation
        with st.spinner("Consulting AI Analyst..."):
            # Assemble Context
            context = []
            context.append(f"Demand Linear MAPE: {mape_lin_d:.2%}")
            context.append(f"Supply Linear MAPE: {mape_lin_s:.2%}")
            context.append(f"Mean Adjustment Speed (K): {np.mean(k_values):.4f}")
            
            # Trend info
            try:
                p_start, p_end = P_forecast[0], P_forecast[-1]
                trend = "UPWARD" if p_end > p_start else "DOWNWARD"
                context.append(f"Forecast Trend: {trend} from {p_start:.2f} to {p_end:.2f}")
            except:
                pass
            
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
