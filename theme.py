THEMES = {
    "dark": {
        # === Background Layers ===
        "bg":           "#111827",   # Deep navy-grey (easier on eyes than pure black)
        "bg2":          "#1F2937",   # Elevated surface
        "card_bg":      "rgba(31, 41, 55, 0.75)",
        "card_border":  "rgba(99, 179, 237, 0.40)",  # Soft blue border
        "sidebar_bg":   "rgba(17, 24, 39, 0.95)",

        # === Text ===
        "text":         "#F3F4F6",   # Near-white — high contrast on dark navy
        "text_muted":   "#9CA3AF",   # Neutral grey for secondary text
        "heading":      "#FFFFFF",

        # === Accent (CTA, active nav, highlights) ===
        "accent":       "#60A5FA",   # Calm sky-blue — readable, not harsh
        "accent_hover": "#93C5FD",
        "accent_glow":  "rgba(96, 165, 250, 0.30)",

        # === Secondary accent ===
        "secondary":    "#34D399",   # Emerald green

        # === Input Fields ===
        "input_bg":     "rgba(255,255,255,0.06)",
        "input_border": "rgba(96, 165, 250, 0.35)",
        "input_focus":  "rgba(96, 165, 250, 0.50)",

        # === UI Chrome ===
        "grid":         "rgba(255,255,255,0.07)",
        "divider":      "rgba(255,255,255,0.10)",
        "shadow":       "rgba(0,0,0,0.40)",

        # === Plot Canvas ===
        "plot_paper":   "rgba(0,0,0,0)",
        "plot_bg":      "rgba(17, 24, 39, 0.60)",

        # === Chart Data Colors (high contrast, color-blind aware) ===
        "marker":       "#E5E7EB",   # Neutral grey-white for actual data dots
        "line_blue":    "#60A5FA",   # Demand fit line — sky blue
        "line_red":     "#F87171",   # Supply fit line / historical K — coral red
        "line_green":   "#34D399",   # Forecast lines — emerald
        "line_pink":    "#F472B6",   # Simulated price — pink/magenta

        # === Tabs & Cards ===
        "tab_inactive_bg":  "rgba(255,255,255,0.05)",
        "tab_active_bg":    "rgba(96, 165, 250, 0.15)",
        "expander_bg":      "rgba(255,255,255,0.04)",
        "metric_bg":        "rgba(96, 165, 250, 0.10)",
        "success_bg":       "rgba(52, 211, 153, 0.12)",
    },
    "light": {
        # === Background Layers ===
        "bg":           "#F8FAFC",   # Clean off-white
        "bg2":          "#EFF6FF",   # Light blue-tinted surface
        "card_bg":      "rgba(239, 246, 255, 0.90)",  # Soft blue tint — visible on white
        "card_border":  "#60A5FA",   # Strong sky-blue border — clearly visible
        "sidebar_bg":   "#EFF6FF",

        # === Text ===
        "text":         "#0F172A",   # Very dark navy-black — maximum readability
        "text_muted":   "#475569",   # Slate grey
        "heading":      "#0F172A",

        # === Accent ===
        "accent":       "#2563EB",   # Strong blue — accessible AA contrast
        "accent_hover": "#1D4ED8",
        "accent_glow":  "rgba(37, 99, 235, 0.20)",

        # === Secondary accent ===
        "secondary":    "#059669",   # Emerald green

        # === Input Fields ===
        "input_bg":     "#FFFFFF",
        "input_border": "rgba(37, 99, 235, 0.40)",
        "input_focus":  "rgba(37, 99, 235, 0.45)",

        # === UI Chrome ===
        "grid":         "rgba(0,0,0,0.07)",
        "divider":      "rgba(0,0,0,0.10)",
        "shadow":       "rgba(0,0,0,0.10)",

        # === Plot Canvas ===
        "plot_paper":   "rgba(0,0,0,0)",
        "plot_bg":      "rgba(248, 250, 252, 0.85)",

        # === Chart Data Colors (clearly distinct from white bg, AA-compliant) ===
        "marker":       "#1E3A5F",   # Dark navy for actual data dots
        "line_blue":    "#2563EB",   # Demand fit line — strong blue
        "line_red":     "#DC2626",   # Supply fit line / historical K — clear red
        "line_green":   "#059669",   # Forecast lines — emerald
        "line_pink":    "#7C3AED",   # Simulated price — violet/purple

        # === Tabs & Cards ===
        "tab_inactive_bg":  "rgba(0,0,0,0.04)",
        "tab_active_bg":    "rgba(37, 99, 235, 0.10)",
        "expander_bg":      "rgba(0,0,0,0.02)",
        "metric_bg":        "rgba(37, 99, 235, 0.06)",
        "success_bg":       "rgba(5, 150, 105, 0.10)",
    },
}

def get_css(theme_mode):
    t = THEMES[theme_mode]
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ===== Global ===== */
    html, body, .stApp, .stMarkdown, p, span, label, li, h1, h2, h3, h4, h5, h6,
    input, textarea, select, button, .stText, .stCaption,
    div:not([class*="icon"]):not([data-baseweb]) {{ font-family: 'Inter', sans-serif !important; }}
    /* Preserve icon fonts */
    [data-testid="stIconMaterial"], .material-symbols-rounded, [class*="icon"] {{
        font-family: 'Material Symbols Rounded', sans-serif !important;
    }}

    .stApp, .main, [data-testid="stAppViewContainer"] {{
        background: {t['bg']};
        color: {t['text']};
    }}
    header[data-testid="stHeader"] {{
        background: {t['bg']};
        backdrop-filter: blur(12px);
    }}

    /* ===== Typography ===== */
    h1 {{ color: {t['heading']} !important; font-weight: 700 !important; letter-spacing: -0.5px !important; }}
    h2 {{ color: {t['heading']} !important; font-weight: 600 !important; }}
    h3, h4, h5, h6 {{ color: {t['heading']} !important; font-weight: 500 !important; }}

    div, p, span, .stMarkdown, label {{
        color: {t['text']} !important;
    }}
    .stMarkdown, p, span, label, li, .stText {{
        font-size: 15px !important;
        line-height: 1.6 !important;
    }}
    small {{ color: {t['text_muted']} !important; }}
    a {{ color: {t['accent']} !important; }}

    /* ===== Buttons ===== */
    .stButton>button {{
        background: linear-gradient(135deg, {t['accent']}, {t['accent_hover']}) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 12px {t['accent_glow']} !important;
    }}
    .stButton>button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 24px {t['accent_glow']} !important;
    }}
    .stButton>button:active {{
        transform: translateY(0px) !important;
    }}

    /* ===== Inputs ===== */
    div[data-testid="stTextInput"] div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] div[data-baseweb="input"] > div {{
        background-color: {t['input_bg']} !important;
        border: 1px solid {t['input_border']} !important;
        border-radius: 8px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }}
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {{
        color: {t['text']} !important;
        background-color: transparent !important;
        -webkit-text-fill-color: {t['text']} !important;
    }}
    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within > div,
    div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within > div {{
        border-color: {t['accent']} !important;
        box-shadow: 0 0 0 3px {t['input_focus']} !important;
    }}
    
    .stNumberInput button {{
        color: {t['text']} !important;
        background: transparent !important;
        border: none !important;
    }}
    .stNumberInput button:hover {{
        color: {t['accent']} !important;
        background: {t['tab_active_bg']} !important;
    }}

    /* ===== Table (st.table) ===== */
    [data-testid="stTable"] {{
        background: {t['bg2']} !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        border: 1px solid {t['card_border']} !important;
    }}
    [data-testid="stTable"] table {{
        color: {t['text']} !important;
        width: 100% !important;
    }}
    [data-testid="stTable"] th {{
        background: {t['tab_inactive_bg']} !important;
        color: {t['heading']} !important;
        border-bottom: 2px solid {t['divider']} !important;
        padding: 12px 16px !important;
    }}
    [data-testid="stTable"] td {{
        border-bottom: 1px solid {t['divider']} !important;
        color: {t['text']} !important;
        padding: 12px 16px !important;
    }}

    /* ===== Selectbox ===== */
    .stSelectbox > div[data-baseweb="select"] > div {{
        background: {t['input_bg']} !important;
        color: {t['text']} !important;
        border-color: {t['input_border']} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="popover"] {{
        background: {t['bg2']} !important;
        border: 1px solid {t['card_border']} !important;
        border-radius: 10px !important;
        backdrop-filter: blur(16px) !important;
    }}
    div[data-baseweb="menu"] {{ background: transparent !important; }}
    ul[role="listbox"], ul[data-testid="stSelectboxVirtualDropdown"] {{
        background: {t['bg2']} !important;
    }}
    li[role="option"] {{
        background: transparent !important;
        color: {t['text']} !important;
        transition: background 0.2s ease !important;
    }}
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {{
        background: {t['tab_active_bg']} !important;
    }}
    div[role="option"] {{ color: {t['text']} !important; }}

    /* ===== File Uploader ===== */
    section[data-testid="stFileUploaderDropzone"] {{
        background: {t['input_bg']} !important;
        border: 2px dashed {t['input_border']} !important;
        border-radius: 12px !important;
        transition: border-color 0.3s ease !important;
    }}
    section[data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {t['accent']} !important;
    }}
    section[data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        color: {t['text_muted']} !important;
    }}
    section[data-testid="stFileUploaderDropzone"] button {{
        background: linear-gradient(135deg, {t['accent']}, {t['accent_hover']}) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
    }}

    /* ===== Hide Sidebar ===== */
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}
    .stApp {{ /* Remove sidebar gap */ }}

    /* ===== Top Bar ===== */
    .top-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 0;
        margin-bottom: 8px;
    }}
    .top-bar-title {{
        font-size: 24px;
        font-weight: 700;
        background: linear-gradient(135deg, {t['accent']}, {t['secondary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.3;
    }}

    /* ===== Pill-Style Navigation Tabs (Radio) ===== */
    .stRadio > div[role="radiogroup"] {{
        background: {t['tab_inactive_bg']};
        padding: 5px;
        border-radius: 14px;
        border: 1px solid {t['divider']};
        display: flex;
        gap: 4px !important;
    }}
    .stRadio > div[role="radiogroup"] > label {{
        background: transparent !important;
        color: {t['text_muted']} !important;
        padding: 10px 20px !important;
        border-radius: 10px !important;
        border: none !important;
        cursor: pointer;
        transition: all 0.3s ease !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 500 !important;
        font-size: 14px !important;
    }}
    .stRadio > div[role="radiogroup"] > label:hover {{
        background: {t['tab_active_bg']} !important;
        color: {t['text']} !important;
    }}
    .stRadio > div[role="radiogroup"] > label[data-checked="true"] {{
        background: {t['tab_active_bg']} !important;
        color: {t['accent']} !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px {t['shadow']} !important;
    }}
    .stRadio > div[role="radiogroup"] > label > div:first-child {{
        display: none;
    }}

    /* ===== Reset Radios inside Expanders ===== */
    [data-testid="stExpander"] .stRadio > div[role="radiogroup"] {{
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        gap: 10px !important;
    }}
    [data-testid="stExpander"] .stRadio > div[role="radiogroup"] > label {{
        background: transparent !important;
        border: none !important;
        padding: 2px 4px !important;
        justify-content: flex-start;
        color: {t['text']} !important;
        font-weight: 400 !important;
        font-size: 15px !important;
    }}
    [data-testid="stExpander"] .stRadio > div[role="radiogroup"] > label:hover {{
        background: transparent !important;
    }}
    [data-testid="stExpander"] .stRadio > div[role="radiogroup"] > label[data-checked="true"] {{
        background: transparent !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        color: {t['accent']} !important;
    }}
    [data-testid="stExpander"] .stRadio > div[role="radiogroup"] > label > div:first-child {{
        display: flex !important;
    }}

    /* ===== Glassmorphic Containers ===== */
    [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stDecoratedContainer"] {{
        background: {t['card_bg']} !important;
        border: 2px solid {t['card_border']} !important;
        border-radius: 16px !important;
        padding: 24px !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        box-shadow: 0 4px 24px {t['shadow']} !important;
        margin-top: 0px !important;
    }}

    /* ===== Expanders ===== */
    details > summary {{
        background: {t['expander_bg']} !important;
        color: {t['text']} !important;
        border: 1px solid {t['divider']} !important;
        border-left: 3px solid {t['accent']} !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }}
    details > summary:hover {{
        background: {t['tab_active_bg']} !important;
    }}
    div[data-testid="stExpander"] {{
        background: transparent !important;
        color: {t['text']} !important;
    }}

    /* ===== Metrics ===== */
    [data-testid="stMetric"] {{
        background: {t['metric_bg']} !important;
        padding: 16px !important;
        border-radius: 12px !important;
        border: 1px solid {t['card_border']} !important;
    }}
    [data-testid="stMetricLabel"] {{ color: {t['text_muted']} !important; }}
    [data-testid="stMetricValue"] {{ color: {t['accent']} !important; font-weight: 700 !important; }}

    /* ===== Dividers ===== */
    hr {{ border-color: {t['divider']} !important; }}

    /* ===== Slider ===== */
    .stSlider > div > div > div[role="slider"] {{
        background-color: {t['accent']} !important;
    }}
    .stSlider > div > div {{
        color: {t['text']} !important;
    }}

    /* ===== Checkbox ===== */
    .stCheckbox label span {{
        color: {t['text']} !important;
    }}

    /* ===== Spinner ===== */
    .stSpinner > div {{
        border-top-color: {t['accent']} !important;
    }}

    /* ===== Tabs (st.tabs) ===== */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: {t['tab_inactive_bg']};
        border-radius: 10px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px !important;
        color: {t['text_muted']} !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {t['text']} !important;
        background: {t['tab_active_bg']} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {t['tab_active_bg']} !important;
        color: {t['accent']} !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: {t['accent']} !important;
    }}

    /* ===== Success / Warning / Info Alerts ===== */
    .stSuccess, div[data-testid="stNotification"] {{
        background: {t['success_bg']} !important;
        color: {t['text']} !important;
        border-radius: 10px !important;
    }}

    /* ===== Top-bar icon buttons (uniform) ===== */
    div[data-testid="stColumn"]:has(#top-right-buttons) [data-testid="stHorizontalBlock"] {{
        gap: 6px !important;
        align-items: center !important;
        justify-content: flex-end !important;
    }}
    div[data-testid="stColumn"]:has(#top-right-buttons) div[data-testid="stColumn"] {{
        width: auto !important;
        flex: 0 0 auto !important;
        min-width: 0 !important;
        padding: 0 !important;
    }}
    div[data-testid="stColumn"]:has(#top-right-buttons) button {{
        background: {t['bg2']} !important;
        color: {t['accent']} !important;
        border: 1px solid {t['accent']}55 !important;
        box-shadow: none !important;
        font-size: 20px !important;
        padding: 4px !important;
        border-radius: 12px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        min-width: 42px !important;
        max-width: 42px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    div[data-testid="stColumn"]:has(#top-right-buttons) button:hover {{
        background: {t['tab_active_bg']} !important;
        border-color: {t['accent']} !important;
        transform: scale(1.05) !important;
    }}

    /* ===== Info Dialog ===== */
    div[role="dialog"],
    div[role="dialog"] > div,
    div[role="dialog"] [data-testid="stDialogBody"],
    div[role="dialog"] [data-testid="stDialogHeader"] {{
        background: {t['bg2']} !important;
        color: {t['text']} !important;
    }}
    div[role="dialog"] {{
        border: 1px solid {t['card_border']} !important;
        border-radius: 16px !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        overflow: hidden !important;
    }}
    div[role="dialog"] h1,
    div[role="dialog"] h2,
    div[role="dialog"] h3 {{
        color: {t['accent']} !important;
    }}
    div[role="dialog"] p,
    div[role="dialog"] span,
    div[role="dialog"] li,
    div[role="dialog"] strong,
    div[role="dialog"] em {{
        color: {t['text']} !important;
    }}
    /* Dialog close button */
    div[role="dialog"] button[aria-label="Close"] {{
        color: {t['text_muted']} !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        min-width: 0 !important;
        max-width: none !important;
        min-height: 0 !important;
        max-height: none !important;
        padding: 8px !important;
    }}
    div[role="dialog"] button[aria-label="Close"]:hover {{
        color: {t['accent']} !important;
    }}
    /* Dialog backdrop */
    div[data-baseweb="modal"] > div:first-child {{
        background: rgba(0,0,0,0.5) !important;
        backdrop-filter: blur(4px) !important;
    }}

    /* ===== Tooltips (help hover popups) ===== */
    div[data-baseweb="tooltip"],
    div[data-baseweb="tooltip"] > div {{
        background: {t['bg2']} !important;
        color: {t['text']} !important;
        border: 1px solid {t['card_border']} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="tooltip"] div[role="tooltip"],
    div[data-baseweb="tooltip"] div[role="tooltip"] * {{
        background: {t['bg2']} !important;
        color: {t['text']} !important;
    }}
    /* Tooltip arrow */
    div[data-baseweb="tooltip"] div[data-popper-arrow] {{
        background: {t['bg2']} !important;
    }}
    /* Streamlit's own tooltip container */
    [data-testid="stTooltipContent"],
    [data-testid="stTooltipContent"] * {{
        background: {t['bg2']} !important;
        color: {t['text']} !important;
    }}
</style>
"""
