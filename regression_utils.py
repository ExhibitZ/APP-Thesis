import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_percentage_error

def perform_linear_regression(X, y):
    """
    Performs linear regression.
    X: Independent variable (2D array-like)
    y: Dependent variable (1D array-like)
    Returns: model, predictions, r2_score (not requested but useful), mape
    """
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mape = mean_absolute_percentage_error(y, y_pred)
    return model, y_pred, mape

def perform_polynomial_regression(X, y, degree=2):
    """
    Performs polynomial regression.
    X: Independent variable (2D array-like)
    y: Dependent variable (1D array-like)
    degree: Polynomial degree
    Returns: model, poly_features, predictions, mape
    """
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    mape = mean_absolute_percentage_error(y, y_pred)
    return model, poly_features, y_pred, mape



# --- EPAM & Dynamic K Analysis ---
from scipy.integrate import odeint
from scipy.interpolate import interp1d

def calculate_dynamic_k(prices, quantity_demand_pred_func, quantity_supply_pred_func):
    """
    Calculates dynamic k values over time.
    prices: Array of actual prices (time-series).
    quantity_demand_pred_func: Function that takes Price and returns Demand Qty.
    quantity_supply_pred_func: Function that takes Price and returns Supply Qty.
    Returns: k_values (list), k_func (interpolation function of k over time stats)
    """
    k_values = []
    
    # Iterate through time steps to solve for k_t: (P_next - P_curr) = k_t * (D_curr - S_curr) * dt (dt=1)
    for i in range(len(prices) - 1):
        P_curr = prices[i]
        P_next = prices[i+1]
        
        # Get D and S at current Price
        D_val = quantity_demand_pred_func(P_curr)
        S_val = quantity_supply_pred_func(P_curr)
        
        delta_P = P_next - P_curr
        gap = D_val - S_val
        
        if abs(gap) > 1e-4:
            k_t = delta_P / gap
        else:
            # If gap is zero, k is undefined/unstable. Use previous k or 0.
            k_t = k_values[-1] if k_values else 0.0
            
        k_values.append(k_t)
    
    # Repeat the last k to match dimension of t_indices for interpolation
    if k_values:
        k_values.append(k_values[-1])
    else:
        k_values = [0.0] * len(prices) # Fallback if empty
        
    k_arr = np.array(k_values)
    t_indices = np.arange(len(prices))
    
    # Create Time-Dependent k function
    # We use 'previous' interpolation to respect causality (k at t governs transition to t+1)
    k_func_dynamic = interp1d(t_indices, k_arr, kind='previous', fill_value="extrapolate")
    
    return k_arr, k_func_dynamic

def solve_epam(P0, k_func, quantity_demand_pred_func, quantity_supply_pred_func, t_steps):
    """
    Solves the EPAM ODE with dynamic k.
    dP/dt = k(t) * (D(P) - S(P))
    """
    def epam_ode_dynamic(P, t, k_f, q_d_func, q_s_func):
        # Unwrap P if it comes as an array
        P_val = P[0] if isinstance(P, (list, np.ndarray)) else P
        
        # Get dynamic k for current time t
        # Ensure t doesn't exceed interpolation range essentially handled by extrapolate but good to be safe
        k_t = k_f(t)
        
        # Predict D and S
        D = q_d_func(P_val)
        S = q_s_func(P_val)
        
        return k_t * (D - S)

    # Solve ODE
    P_pred = odeint(epam_ode_dynamic, P0, t_steps, args=(k_func, quantity_demand_pred_func, quantity_supply_pred_func))
    return P_pred.flatten()

def perform_multivariate_regression(df, price_col, time_col, qty_col):
    """
    Performs Multivariate Linear Regression: Q = b0 + b1*P + b2*t
    df: Pandas DataFrame
    price_col: Name of Price column
    time_col: Name of Time column
    qty_col: Name of Quantity column
    Returns: model, coefficients (intercept, [b_price, b_time]), mape
    """
    X = df[[price_col, time_col]].values
    y = df[qty_col].values
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    mape = mean_absolute_percentage_error(y, y_pred)
    
    return model, (model.intercept_, model.coef_), mape

def solve_non_autonomous_epam(P0, k_input, d_func, s_func, t_steps):
    """
    Solves the Non-Autonomous EPAM ODE: dP/dt = k * (D(P,t) - S(P,t))
    D(P,t) and S(P,t) are callable functions (p, t).
    k_input: Can be a scalar (median k) or a callable function k(t).
    """
    def non_autonomous_ode(P, t, k_in, d_f, s_f):
        # Unwrap P
        P_val = P[0] if isinstance(P, (list, np.ndarray)) else P
        
        # Determine k_t
        k_t = k_in(t) if callable(k_in) else k_in
        
        # Predict D and S
        D_qty = d_f(P_val, t)
        S_qty = s_f(P_val, t)
        
        return k_t * (D_qty - S_qty)

    P_pred = odeint(non_autonomous_ode, P0, t_steps, args=(k_input, d_func, s_func))
    return P_pred.flatten()


# --- Rolling Window K & Adjusted Gap ---

def compute_rolling_k(k_raw, t_idx, window_n):
    """Rolling-window OLS *without intercept*: K = beta * t.

    At each step i the window covers indices [max(0, i+1-window_n) .. i].
    """
    k_roll = np.zeros(len(k_raw))
    last_beta = 0.0
    for i in range(len(k_raw)):
        lo = max(0, i + 1 - window_n)
        t_w = t_idx[lo:i+1].astype(float).reshape(-1, 1)
        k_w = k_raw[lo:i+1]
        denom = float(t_w.T @ t_w)
        if abs(denom) > 1e-12:
            last_beta = float(t_w.T @ k_w) / denom
        k_roll[i] = last_beta * float(t_idx[i])
    return k_roll, last_beta


def forecast_rolling_k(k_hist, t_hist, t_fc, window_n):
    """Iteratively forecast K into t_fc using rolling OLS (no intercept)."""
    all_k = list(k_hist)
    all_t = list(t_hist)
    fc_k = []
    for t_new in t_fc:
        lo = max(0, len(all_k) - window_n)
        t_w = np.array(all_t[lo:], dtype=float).reshape(-1, 1)
        k_w = np.array(all_k[lo:])
        denom = float(t_w.T @ t_w)
        beta = float(t_w.T @ k_w) / denom if abs(denom) > 1e-12 else 0.0
        k_new = beta * float(t_new)
        fc_k.append(k_new)
        all_k.append(k_new)
        all_t.append(float(t_new))
    return np.array(fc_k)


def compute_adjusted_gap(q_demand_pred, q_supply_pred):
    """Compute raw gap, mean gap (structural bias), and adjusted gap."""
    gap_raw = q_demand_pred - q_supply_pred
    gap_mean = float(np.mean(gap_raw))
    gap_adj = gap_raw - gap_mean
    return gap_raw, gap_mean, gap_adj


def solve_epam_adjusted_gap(P0, k_input, gap_adj, t_steps):
    """Solve the EPAM ODE with pre-computed adjusted gap.
    dP/dt = k(t) * gap_adj_interp(t).
    """
    gap_interp = interp1d(t_steps, gap_adj, kind='previous',
                          fill_value='extrapolate')

    def ode(P, t):
        kt = k_input(t) if callable(k_input) else k_input
        return kt * gap_interp(t)

    P_pred = odeint(ode, P0, t_steps)
    return P_pred.flatten()
