"""ComplaintIQ -- Streamlit dashboard.

Pages / sections:
  - Top stats bar      : totals, SLA at risk, auto-resolved (duplicates), avg breach prob
  - Complaint live feed: filterable list with severity / sentiment badges
  - Customer drill-in  : emotion timeline + risk score for a selected customer
  - India heatmap      : Folium choropleth of complaint density by city
  - SLA tracker        : complaints sorted by days-until-due
  - Root-cause alerts  : systemic clusters detected by Agent 6

Run locally:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

# DLL hygiene (see orchestrator for context).
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_FLAX", "0")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Preload torch FIRST -- before pandas / numpy / chromadb load their own
# MKL/OpenMP runtimes -- so torch's c10.dll wins the Windows DLL init race.
# Without this, Agent 3 (sentence-transformers) triggers a late torch import
# that fails with "WinError 1114 ... c10.dll". See orchestrator.py / api/main.py.
try:
    import torch  # noqa: F401
except Exception:
    pass

import sys
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Make project root importable when run as `streamlit run dashboard/app.py`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import db  # noqa: E402
from dashboard import rbi_report  # noqa: E402
from auth.supabase_auth import sign_in, sign_out, update_last_login  # noqa: E402

st.set_page_config(
    page_title="ComplaintIQ -- Union Bank of India",
    page_icon="UBI",
    layout="wide",
)

## --- Theme tokens ---------------------------------------------------------
# Professional Bank Theme - Clean, Modern, Reputed Brand Look
PAL_BG      = "#f8fafc"   # soft premium gray     (page background)
PAL_SAND    = "#f1f5f9"   # subtle slate gray     (cards / secondary surfaces)
PAL_BLUE    = "#3b82f6"   # primary accent blue   (active tab, highlights)
PAL_INK     = "#0f172a"   # deep slate navy       (primary text)

## --- Categorical palette (distinct hue per value -- judges' brief) --------
# Severity (ordinal alarming -> calm) - Subdued professional colors
SEVERITY_COLORS = {
    "Critical": "#dc2626",     # Urgently red
    "High":     "#ea580c",     # Alert orange
    "Medium":   "#f59e0b",     # Golden warning amber
    "Low":      "#64748b",     # Muted slate gray
}
SEVERITY_ORDER = ["Critical", "High", "Medium", "Low"]

# Category (six distinct hues) - Professional bank-style palette
CATEGORY_COLORS = {
    "General":    "#64748b",     # Slate gray
    "UPI":        "#3b82f6",     # Slate blue
    "Loan":       "#0d9488",     # Teal
    "NetBanking": "#0f172a",     # Deep navy
    "Card":       "#ec4899",     # Rose pink
    "ATM":        "#f59e0b",     # Warm amber
}

# Sentiment (positive -> negative) - Subdued professional
SENTIMENT_COLORS = {
    "Polite":     "#10b981",     # Emerald green - positive
    "Neutral":    "#64748b",     # Slate gray - neutral
    "Frustrated": "#ea580c",     # Orange - frustrated
    "Angry":      "#dc2626",     # Red - angry
}
SENTIMENT_ORDER = ["Angry", "Frustrated", "Neutral", "Polite"]

# Lowercased Roberta sentiment buckets share the same scheme as the LLM ones.
ML_SENTIMENT_COLORS = {
    "negative": SENTIMENT_COLORS["Angry"],
    "neutral":  SENTIMENT_COLORS["Neutral"],
    "positive": SENTIMENT_COLORS["Polite"],
}

# Resolution status (five distinct) - Professional status colors
RESOLUTION_COLORS = {
    "Breached":                  "#dc2626",      # Red
    "Pending":                   "#f59e0b",      # Warm amber
    "Resolved":                  "#10b981",      # Emerald green
    "Auto-Resolved (Standard)":  "#3b82f6",      # Slate blue
    "Auto-Resolved (Dup)":       "#0d9488",      # Teal
}

# Backwards-compat shim for older code paths that referenced PAL_RED.
PAL_RED = SEVERITY_COLORS["Critical"]

SENTIMENT_ICON = {"Angry": "[!!]", "Frustrated": "[!]", "Neutral": "[-]", "Polite": "[+]"}


def risk_color(v: float | int) -> str:
    """Map a 0..100 risk score to one of three severity-aligned shades."""
    if v is None: return SEVERITY_COLORS["Medium"]
    v = float(v)
    if v >= 70: return SEVERITY_COLORS["Critical"]
    if v >= 40: return SEVERITY_COLORS["High"]
    return SEVERITY_COLORS["Low"]


_GLOBAL_CSS = f"""
<style>
    /* hide streamlit deploy button */
    .stAppDeployButton {{display:none;}}
    [data-testid="stToolbar"] {{display:none;}}
    
    /* page-level chrome */
  .stApp {{ background: {PAL_BG}; color: {PAL_INK}; }}
  section[data-testid="stSidebar"] {{
      background-color: #0f172a !important; /* dark navy/slate */
      color: #94a3b8 !important;
      border-right: 1px solid #1e293b;
      padding-top: 20px;
  }}
  
  /* Style sidebar navigation header and radio buttons */
  section[data-testid="stSidebar"] div[role="radiogroup"] {{
      gap: 6px !important;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label {{
      background-color: transparent !important;
      border: none !important;
      border-radius: 6px !important;
      padding: 10px 16px !important;
      color: #94a3b8 !important;
      font-size: 14px !important;
      font-weight: 500 !important;
      transition: all 0.2s ease !important;
      cursor: pointer !important;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
      background-color: #1e293b !important;
      color: #f8fafc !important;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {{
      background-color: #3b82f6 !important; /* slate blue */
      color: #ffffff !important;
      font-weight: 600 !important;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p {{
      color: #ffffff !important;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] p {{
      color: #94a3b8 !important;
      font-size: 14px !important;
      margin: 0 !important;
  }}

  /* Tab styling inside main container */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 8px;
      border-bottom: 2px solid #e2e8f0;
  }}
  .stTabs [data-baseweb="tab"] {{
      background-color: #f8fafc;
      color: #64748b;
      border-radius: 6px 6px 0 0;
      padding: 8px 16px;
      font-weight: 500;
      border: 1px solid #e2e8f0;
      border-bottom: none;
  }}
  .stTabs [aria-selected="true"] {{
      background-color: #ffffff !important;
      color: #3b82f6 !important;
      border-color: #3b82f6 !important;
      font-weight: 600;
  }}

  /* Buttons - Sleek outline secondary style */
  .stButton > button {{
      border: 1px solid #cbd5e1 !important;
      color: #334155 !important;
      background-color: transparent !important;
      border-radius: 6px !important;
      font-weight: 500 !important;
      font-size: 14px !important;
      padding: 6px 14px !important;
      box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
      transition: all 0.2s ease !important;
  }}
  .stButton > button:hover {{
      background-color: #f1f5f9 !important;
      color: #0f172a !important;
      border-color: #94a3b8 !important;
  }}
  .stButton > button[kind="primary"] {{
      background-color: #2563eb !important;
      border-color: #2563eb !important;
      color: #ffffff !important;
  }}
  .stButton > button[kind="primary"]:hover {{
      background-color: #1d4ed8 !important;
      border-color: #1d4ed8 !important;
      color: #ffffff !important;
  }}

  /* Header utilities - Muted Export RBI CSV Button */
  .stDownloadButton > button {{
      border: 1px solid #cbd5e1 !important;
      background-color: transparent !important;
      color: #334155 !important;
      border-radius: 6px !important;
      font-weight: 500 !important;
      font-size: 14px !important;
      padding: 6px 14px !important;
      box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
      transition: all 0.2s ease !important;
      width: 100% !important;
  }}
  .stDownloadButton > button:hover {{
      background-color: #f1f5f9 !important;
      color: #0f172a !important;
      border-color: #94a3b8 !important;
  }}

  /* Header utilities - User Profile Popover as sophisticated dropdown */
  .stPopover > button {{
      border: 1px solid #cbd5e1 !important;
      background-color: transparent !important;
      color: #334155 !important;
      border-radius: 20px !important;
      padding: 6px 16px !important;
      font-size: 14px !important;
      font-weight: 500 !important;
      box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
      width: 100% !important;
  }}
  .stPopover > button:hover {{
      background-color: #f1f5f9 !important;
      border-color: #94a3b8 !important;
      color: #0f172a !important;
  }}

  /* Clean Dataframe / Expander borders */
  .stDataFrame {{
      border: 1px solid #e2e8f0 !important;
      border-radius: 8px !important;
  }}
  .streamlit-expanderHeader {{
      background-color: #ffffff !important;
      color: #1e293b !important;
      border: 1px solid #e2e8f0 !important;
      border-radius: 6px !important;
      font-weight: 500 !important;
  }}
  div[data-testid="stExpander"] {{
      border: none !important;
      margin-bottom: 8px !important;
  }}

  /* Style st.container(border=True) as premium white cards */
  div[data-testid="stContainer"] {{
      background-color: #ffffff !important;
      border: 1px solid #e2e8f0 !important;
      border-radius: 8px !important;
      padding: 16px !important;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03) !important;
      margin-bottom: 16px !important;
  }}

  /* Ensure inline alert button height matches alert row (46px) */
  div[data-testid="column"]:nth-child(2) .stButton > button {{
      height: 46px !important;
      line-height: 46px !important;
      padding-top: 0 !important;
      padding-bottom: 0 !important;
      margin-top: 0px !important;
  }}
</style>
"""

# Approximate lat/long for the cities present in the seed data.
CITY_COORDS = {
    "Mumbai": (19.0760, 72.8777),     "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),  "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),       "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),     "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),     "Nagpur": (21.1458, 79.0882),
    "Indore": (22.7196, 75.8577),     "Bhopal": (23.2599, 77.4126),
    "Patna": (25.5941, 85.1376),      "Vadodara": (22.3072, 73.1812),
    "Nashik": (19.9975, 73.7898),     "Aurangabad": (19.8762, 75.3433),
    "Solapur": (17.6599, 75.9064),    "Thane": (19.2183, 72.9781),
    "Noida": (28.5355, 77.3910),      "Agra": (27.1767, 78.0081),
    "Varanasi": (25.3176, 82.9739),   "Coimbatore": (11.0168, 76.9558),
    "Gurgaon": (28.4595, 77.0266),    "Rajkot": (22.3039, 70.8022),
    "Surat": (21.1702, 72.8311),      "Vizag": (17.6868, 83.2185),
    # Additional cities in the seed dataset (Maharashtra + UP heavy).
    "Akola": (20.7059, 77.0082),       "Aligarh": (27.8974, 78.0880),
    "Allahabad": (25.4358, 81.8463),   "Amravati": (20.9374, 77.7796),
    "Bareilly": (28.3670, 79.4304),    "Bhandara": (21.1700, 79.6500),
    "Buldhana": (20.5292, 76.1842),    "Chandrapur": (19.9615, 79.2961),
    "Dhule": (20.9024, 74.7749),       "Firozabad": (27.1591, 78.3957),
    "Gadchiroli": (20.1809, 80.0024),  "Ghaziabad": (28.6692, 77.4538),
    "Gondia": (21.4624, 80.1961),      "Gorakhpur": (26.7606, 83.3732),
    "Hapur": (28.7305, 77.7782),       "Hingoli": (19.7173, 77.1493),
    "Jalgaon": (21.0077, 75.5626),     "Kolhapur": (16.7050, 74.2433),
    "Latur": (18.4088, 76.5604),       "Mathura": (27.4924, 77.6737),
    "Meerut": (28.9845, 77.7064),      "Moradabad": (28.8389, 78.7768),
    "Nanded": (19.1383, 77.3210),      "Nandurbar": (21.3704, 74.2400),
    "Osmanabad": (18.1862, 76.0454),   "Parbhani": (19.2608, 76.7762),
    "Ratnagiri": (16.9944, 73.3000),   "Sambhal": (28.5818, 78.5660),
    "Sangli": (16.8524, 74.5815),      "Satara": (17.6805, 74.0183),
    "Sindhudurg": (16.3500, 73.5500),  "Wardha": (20.7453, 78.6022),
    "Washim": (20.1110, 77.1331),      "Yavatmal": (20.3897, 78.1204),
}


# --- data loaders (cached) ---------------------------------------------------

@st.cache_data(ttl=20)
def load_complaints() -> pd.DataFrame:
    db.init_db()
    rows = db.list_complaints()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sla_due_date"] = pd.to_datetime(df["sla_due_date"], errors="coerce")
    return df


@st.cache_data(ttl=20)
def load_alerts() -> pd.DataFrame:
    return pd.DataFrame(db.list_root_cause_alerts())


# Choices that match the Classifier agent's vocab.
FB_CATEGORIES = ["UPI", "ATM", "Card", "Loan", "NetBanking", "General"]
FB_SEVERITIES = ["Critical", "High", "Medium", "Low"]
FB_SENTIMENTS = ["Angry", "Frustrated", "Neutral", "Polite"]


# --- header / KPIs ----------------------------------------------------------

def render_kpis(df: pd.DataFrame) -> None:
    total = len(df)
    processed = int(df["processed_at"].notna().sum()) if "processed_at" in df else 0
    status = df["status"].fillna("open")
    at_risk = int(((status == "open") & (df["sla_breach_prob"].fillna(0) >= 0.5)).sum())
    auto_resolved = int(status.isin(("auto_resolved_dup", "auto_resolved_std")).sum())
    avg_breach = float(df.loc[status == "open", "sla_breach_prob"].dropna().mean()) \
                 if (status == "open").any() else 0.0

    cols = st.columns(5)
    
    def _render_kpi_card(col, title, value, trend=None):
        trend_html = f"<div style='margin-top:6px; font-size:12px; color:#64748b; font-weight:400;'>{trend}</div>" if trend else "<div style='margin-top:6px; height:18px;'></div>"
        col.markdown(
            f"<div style='display:flex; flex-direction:column; justify-content:space-between; "
            f"height:120px; background-color:#ffffff; border:1px solid #e2e8f0; "
            f"border-radius:8px; padding:16px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05), "
            f"0 2px 4px -1px rgba(0,0,0,0.03); color:#0f172a;'>"
            f"<div style='font-size:11px; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;'>{title}</div>"
            f"<div style='font-size:28px; font-weight:700; margin-top:12px; line-height:1; color:#0f172a;'>{value}</div>"
            f"{trend_html}"
            f"</div>",
            unsafe_allow_html=True
        )

    _render_kpi_card(cols[0], "Total complaints", f"{total:,}")
    _render_kpi_card(cols[1], "Processed", f"{processed:,}", f"+{total - processed} pending" if processed < total else "all done")
    _render_kpi_card(cols[2], "SLA at risk", f"{at_risk:,}", ">50% breach chance")
    _render_kpi_card(cols[3], "Auto-resolved", f"{auto_resolved:,}", "Duplicates/Templates")
    _render_kpi_card(cols[4], "Avg breach prob", f"{avg_breach:.0%}", "Open rows")


# --- live feed --------------------------------------------------------------

def render_live_feed(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    with st.container(border=True):
        st.subheader("Live Complaint Queue")
        
        # Row 1: Filter Multiselects (giving 33% width each)
        f1, f2, f3 = st.columns([1, 1, 1])
        sev = f1.multiselect("Severity Filters", sorted(df["severity"].dropna().unique()), placeholder="All severities")
        cat = f2.multiselect("Category Filters", sorted(df["category"].dropna().unique()), placeholder="All categories")
        chan = f3.multiselect("Channel Filters", sorted(df["channel"].dropna().unique()), placeholder="All channels")
        
        # Row 2: Sort and Risk checkbox
        f4, f5 = st.columns([1.2, 0.8])
        only_open = f4.checkbox("Show only high-risk complaints (>50% SLA breach probability)", value=False)
        sort_by = f5.selectbox("Sort queue by",
                               ["Priority", "Date", "Risk score", "Breach probability"],
                               index=0)

        view = df.copy()
        if sev:
            view = view[view["severity"].isin(sev)]
        if cat:
            view = view[view["category"].isin(cat)]
        if chan:
            view = view[view["channel"].isin(chan)]
        if only_open:
            view = view[view["sla_breach_prob"].fillna(0) >= 0.5]

        sort_key = {
            "Priority":             ("priority_score", False),
            "Date":                 ("date", False),
            "Risk score":           ("risk_score", False),
            "Breach probability":   ("sla_breach_prob", False),
        }[sort_by]
        if sort_key[0] in view.columns:
            view = view.sort_values(sort_key[0], ascending=sort_key[1],
                                    na_position="last")
        view = view.head(60)

        # Build Category / Sentiment badges
        show = view.copy()

        def _badge(primary, agreement) -> str:
            if pd.isna(primary):
                return "-"
            if agreement == "Needs Review":
                return f"{primary} (*)"
            return str(primary)

        if "category" in show:
            show["Category"] = [
                _badge(p, a) for p, a in zip(show.get("category"),
                                              show.get("category_confidence",
                                                       pd.Series([None] * len(show))))
            ]
        if "sentiment" in show:
            show["Sentiment"] = [
                _badge(p, a) for p, a in zip(show.get("sentiment"),
                                              show.get("sentiment_confidence",
                                                       pd.Series([None] * len(show))))
            ]

        if "date" in show:
            show["date"] = pd.to_datetime(show["date"]).dt.date
        if "sla_due_date" in show:
            show["sla_due_date"] = pd.to_datetime(show["sla_due_date"]).dt.date
        if "sla_breach_prob" in show:
            show["sla_breach_prob"] = show["sla_breach_prob"].apply(
                lambda v: f"{v:.0%}" if pd.notna(v) else "-")
        if "amount_involved" in show:
            show["amount_involved"] = show["amount_involved"].apply(
                lambda v: f"Rs {v:,.0f}" if pd.notna(v) and v > 0 else "-")

        # Final columns
        desired = [
            ("id",              "Complaint ID"),
            ("date",            "Date"),
            ("customer_name",   "Customer"),
            ("channel",         "Channel"),
            ("Category",        "Category"),
            ("severity",        "Severity"),
            ("Sentiment",       "Sentiment"),
            ("priority_score",  "Priority"),
            ("amount_involved", "Amount (INR)"),
            ("sla_due_date",    "SLA Deadline"),
            ("sla_breach_prob", "Breach probability"),
            ("risk_score",      "Risk score"),
            ("duplicate_of",    "Duplicate of"),
        ]
        keep = [(s, t) for s, t in desired if s in show.columns]
        out = show[[s for s, _ in keep]].rename(columns=dict(keep))
        
        # Configure columns configuration to format widths
        col_configs = {
            "Complaint ID": st.column_config.TextColumn("Complaint ID", width="small"),
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", width="small"),
            "Customer": st.column_config.TextColumn("Customer", width="medium"),
            "Channel": st.column_config.TextColumn("Channel", width="small"),
            "Category": st.column_config.TextColumn("Category", width="small"),
            "Severity": st.column_config.TextColumn("Severity", width="small"),
            "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            "Priority": st.column_config.NumberColumn("Priority", format="%d", width="small"),
            "Amount (INR)": st.column_config.TextColumn("Amount (INR)", width="small"),
            "SLA Deadline": st.column_config.DateColumn("SLA Deadline", format="YYYY-MM-DD", width="small"),
            "Breach probability": st.column_config.TextColumn("Breach probability", width="small"),
            "Risk score": st.column_config.NumberColumn("Risk score", format="%d", width="small"),
        }
        
        event = st.dataframe(
            out, 
            use_container_width=True, 
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config=col_configs
        )
        st.caption("(*) next to a Category or Sentiment value means the ML "
                   "second-opinion model disagrees with the LLM. Click on a row to select it.")
        
    selected_row = None
    if event.selection and event.selection.rows:
        idx = event.selection.rows[0]
        selected_row = view.iloc[idx]
        
    return view, selected_row


def render_selected_complaint_workspace(r: pd.Series, df: pd.DataFrame) -> None:
    st.divider()
    st.subheader(f"Workspace Details: Complaint {r['id']}")
    
    col_details, col_actions = st.columns([1, 1], gap="medium")
    
    with col_details:
        with st.container(border=True):
            st.markdown("### Profile & History")
            
            # Meta Card
            severity_col = SEVERITY_COLORS.get(r['severity'], PAL_INK)
            amt_str = f"Rs {r['amount_involved']:,.2f}" if (pd.notna(r.get('amount_involved')) and r['amount_involved'] > 0) else "N/A"
            acct_str = str(r['account_type']).capitalize() if pd.notna(r.get('account_type')) else "N/A"
            st.markdown(
                f"<div style='padding:12px 14px; border-radius:6px; background:#f8fafc; "
                f"border-left:4px solid {severity_col}; margin-bottom:12px; border:1px solid #e2e8f0; color:#1e293b; font-size:13px; line-height:1.4;'>"
                f"<b>Customer Name:</b> {r['customer_name']}<br>"
                f"<b>Channel:</b> {r['channel'].capitalize()} &nbsp;|&nbsp; <b>Language:</b> {r['language'].capitalize()}<br>"
                f"<b>Amount:</b> {amt_str} &nbsp;|&nbsp; <b>Account Type:</b> {acct_str}<br>"
                f"<b>Filed Date:</b> {pd.to_datetime(r['date']).date()}"
                f"</div>",
                unsafe_allow_html=True
            )
            
            st.markdown("**Original Complaint:**")
            st.info(r['complaint_text'])
            
            # Customer history timeline
            st.markdown("**Customer Emotion History:**")
            hist = df[df["customer_name"] == r["customer_name"]].sort_values("date")
            
            if hist["sentiment"].notna().any():
                plot = hist.copy()
                plot["emotion_level"] = plot["sentiment"].map(EMOTION_RANK).fillna(0)
                fig = px.bar(
                    plot, x="date", y="emotion_level",
                    color="severity", color_discrete_map=SEVERITY_COLORS,
                    hover_data=["id", "category", "sentiment", "amount_involved"],
                    height=200,
                )
                fig.update_layout(plot_bgcolor=PAL_BG, paper_bgcolor=PAL_BG,
                                  font_color=PAL_INK, margin=dict(l=10, r=10, t=10, b=10))
                fig.update_yaxes(
                    tickvals=list(EMOTION_RANK.values()),
                    ticktext=list(EMOTION_RANK.keys()),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
            # Risk scores
            st.markdown("**Risk Score Analysis:**")
            overall = int(r["risk_score"]) if (pd.notna(r.get("risk_score")) and r["risk_score"] != "") else 0
            r_color = risk_color(overall)
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(
                    f"<div style='padding:12px;border-radius:6px;background:{r_color};"
                    f"color:#FFFFFF;text-align:center;border:1px solid #e2e8f0;margin-top:10px;'>"
                    f"<div style='font-size:10px;opacity:0.95;font-weight:600;'>Overall Risk</div>"
                    f"<div style='font-size:28px;font-weight:800;line-height:1.1;color:#FFFFFF'>{overall}</div>"
                    f"<div style='font-size:9px;opacity:0.95;'>out of 100</div></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                _render_subscore("RBI Ombudsman escalation risk", r.get("risk_ombudsman"))
                _render_subscore("Customer Churn risk", r.get("risk_churn"))
                _render_subscore("Social-media blow-up risk", r.get("risk_social"))
                
    with col_actions:
        with st.container(border=True):
            st.markdown("### AI & Actions")
            
            # SLA & Prediction card
            due_dt = pd.to_datetime(r.get("sla_due_date")).date() if pd.notna(r.get("sla_due_date")) else None
            breach_prob = float(r.get("sla_breach_prob")) if pd.notna(r.get("sla_breach_prob")) else 0.0
            breach_color = SEVERITY_COLORS["Critical"] if breach_prob >= 0.5 else SEVERITY_COLORS["Low"]
            
            st.markdown(
                f"<div style='padding:12px 14px; border-radius:6px; background:#f8fafc; "
                f"border-left:4px solid {breach_color}; margin-bottom:12px; border:1px solid #e2e8f0; color:#1e293b; font-size:13px; line-height:1.4;'>"
                f"<b>SLA Deadline:</b> {due_dt or 'N/A'}<br>"
                f"<b>SLA Breach Probability:</b> <span style='color:{breach_color}; font-weight:bold;'>{breach_prob:.1%}</span><br>"
                f"<b>Priority Score:</b> {r.get('priority_score', 'N/A')}/100"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # AI Drafted Reply
            st.markdown("**AI-Generated Response Draft:**")
            draft = r.get("draft_response")
            if pd.notna(draft) and draft:
                edited_draft = st.text_area("Review / Edit Response", value=draft, height=200, key=f"draft_{r['id']}")
                st.button("Copy Draft to Clipboard", key=f"btn_copy_{r['id']}", help="Highlight and copy the draft text above.")
            else:
                st.info("No draft response available (duplicate complaint).")
                if r.get("duplicate_of"):
                    st.warning(f"This complaint is a duplicate of: **{r['duplicate_of']}**")
                    
            st.divider()
            
            # Classification Corrections (Feedback)
            st.markdown("**ML Model Classifications Review:**")
            
            f_cols = st.columns(3)
            _feedback_widget(f_cols[0], r, "category", r.get("category"), FB_CATEGORIES)
            _feedback_widget(f_cols[1], r, "severity", r.get("severity"), FB_SEVERITIES)
            _feedback_widget(f_cols[2], r, "sentiment", r.get("sentiment"), FB_SENTIMENTS)


# --- customer drill-in ------------------------------------------------------

EMOTION_RANK = {"Polite": 1, "Neutral": 2, "Frustrated": 3, "Angry": 4}


def render_customer_view(df: pd.DataFrame) -> None:
    st.subheader("Customer emotion timeline & risk score")
    names = sorted(df["customer_name"].dropna().unique())
    if not names:
        st.info("No customer data yet.")
        return
    # Default to the customer with the most complaints (most interesting).
    default = (df.groupby("customer_name").size().sort_values(ascending=False).index[0]
               if not df.empty else names[0])
    pick = st.selectbox("Customer", names, index=names.index(default))
    hist = df[df["customer_name"] == pick].sort_values("date")

    left, right = st.columns([2, 1])
    with left:
        if hist["sentiment"].notna().any():
            plot = hist.copy()
            plot["emotion_level"] = plot["sentiment"].map(EMOTION_RANK).fillna(0)
            fig = px.bar(
                plot, x="date", y="emotion_level",
                color="severity", color_discrete_map=SEVERITY_COLORS,
                hover_data=["id", "category", "sentiment", "amount_involved"],
                title=f"Emotion over time -- {pick}",
            )
            fig.update_layout(plot_bgcolor=PAL_BG, paper_bgcolor=PAL_BG,
                              font_color=PAL_INK)
            fig.update_yaxes(
                tickvals=list(EMOTION_RANK.values()),
                ticktext=list(EMOTION_RANK.keys()),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Customer's complaints not yet classified.")
    with right:
        latest = hist.dropna(subset=["risk_score"]).tail(1)
        if not latest.empty:
            row = latest.iloc[0]
            overall = int(row["risk_score"])
            color = risk_color(overall)
            # All severity hues are deep enough to read white text on, so we
            # use white text + the severity-aligned fill.
            st.markdown(
                f"<div style='padding:18px;border-radius:12px;background:{color};"
                f"color:#FFFFFF;text-align:center;border:1px solid {PAL_INK}22'>"
                f"<div style='font-size:13px;opacity:0.95;font-weight:600;letter-spacing:0.3px'>Overall risk score</div>"
                f"<div style='font-size:48px;font-weight:800;line-height:1.1;color:#FFFFFF'>{overall}</div>"
                f"<div style='font-size:12px;opacity:0.95;color:#FFFFFF'>out of 100</div></div>",
                unsafe_allow_html=True,
            )
            _render_subscore("RBI Ombudsman escalation risk", row.get("risk_ombudsman"),
                             help="Probability customer files formal RBI complaint")
            _render_subscore("Customer churn risk", row.get("risk_churn"),
                             help="Likelihood of leaving the bank")
            _render_subscore("Social-media blow-up risk", row.get("risk_social"),
                             help="Twitter/WhatsApp public-pressure risk")
            st.caption("Weighted: 45% Ombudsman + 30% Churn + 25% Social.")
        st.metric("Complaints filed", len(hist))
        st.metric("Categories touched", hist["category"].nunique())

    st.dataframe(
        hist[["id", "date", "channel", "category", "severity", "sentiment",
              "amount_involved", "sla_due_date", "sla_breach_prob", "risk_score"]]
        .assign(date=lambda d: d["date"].dt.date,
                sla_due_date=lambda d: d["sla_due_date"].dt.date)
        .rename(columns={
            "id": "Complaint ID", "date": "Date", "channel": "Channel",
            "category": "Category", "severity": "Severity", "sentiment": "Sentiment",
            "amount_involved": "Amount (INR)", "sla_due_date": "SLA Deadline",
            "sla_breach_prob": "Breach probability", "risk_score": "Risk score",
        }),
        use_container_width=True, hide_index=True,
    )


def _render_subscore(label: str, value, *, help: str | None = None) -> None:
    """Render a 0-100 sub-score as a label + progress bar."""
    if pd.isna(value):
        return
    v = int(value)
    color = risk_color(v)
    st.markdown(
        f"<div style='margin-top:10px;font-size:12px;color:{PAL_INK}'>"
        f"<div style='display:flex;justify-content:space-between'>"
        f"<span>{label}</span><span><b>{v}/100</b></span></div>"
        f"<div style='background:{PAL_SAND};border-radius:6px;height:10px;margin-top:4px;border:1px solid {PAL_INK}22'>"
        f"<div style='background:{color};height:10px;border-radius:6px;width:{v}%'></div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if help:
        st.caption(help)


# --- India heatmap ----------------------------------------------------------

def render_india_map(df: pd.DataFrame) -> None:
    st.subheader("Complaint hotspots across India")
    by_city_all = df.groupby("location").size().reset_index(name="count")
    by_city = by_city_all[by_city_all["location"].isin(CITY_COORDS)].copy()

    if by_city.empty:
        st.info("No mappable complaints yet.")
        return

    sev_critical = SEVERITY_COLORS["Critical"]
    sev_high     = SEVERITY_COLORS["High"]
    sev_medium   = SEVERITY_COLORS["Medium"]
    sev_low      = SEVERITY_COLORS["Low"]

    # --- Build an India-only Plotly map --------------------------------------
    # Loads the locally-cached India states GeoJSON so the basemap is just
    # India outlined by states. No streamlit-folium plug-in required.
    import json
    from pathlib import Path
    import plotly.graph_objects as go

    data_dir = Path(__file__).resolve().parent.parent / "data"
    geojson_path = data_dir / "india_states.geojson"
    districts_path = data_dir / "india_districts.geojson"
    if not geojson_path.exists():
        st.error("India states GeoJSON missing -- run the install step or "
                 "re-download `data/india_states.geojson`.")
        return
    india_geo = json.loads(geojson_path.read_text(encoding="utf-8"))
    state_names = [f["properties"]["ST_NM"] for f in india_geo["features"]]
    districts_geo = None
    if districts_path.exists():
        try:
            districts_geo = json.loads(districts_path.read_text(encoding="utf-8"))
        except Exception:
            districts_geo = None

    # Volume buckets -- judge-facing labels say "volume" so they don't get
    # confused with the severity scale used elsewhere on the dashboard.
    def _bucket(cnt: int) -> tuple[str, str]:
        if cnt >= 30: return "Very high (30+)", sev_critical
        if cnt >= 15: return "High (15-29)",    sev_high
        if cnt >= 5:  return "Moderate (5-14)", sev_medium
        return "Other (<5)", sev_low
    by_city["lat"] = by_city["location"].map(lambda c: CITY_COORDS[c][0])
    by_city["lon"] = by_city["location"].map(lambda c: CITY_COORDS[c][1])
    by_city[["Volume", "color"]] = by_city["count"].apply(
        lambda c: pd.Series(_bucket(int(c))))

    fig = go.Figure()

    # 1a. District borders -- drawn underneath, in muted slate, so judges see
    #     the administrative grid without it competing with city markers.
    if districts_geo is not None:
        district_ids = [f["properties"].get("district") or f"D{i}"
                        for i, f in enumerate(districts_geo["features"])]
        fig.add_choropleth(
            geojson=districts_geo,
            featureidkey="properties.district",
            locations=district_ids,
            z=[0] * len(district_ids),
            showscale=False,
            colorscale=[[0, PAL_SAND], [1, PAL_SAND]],
            marker_line_color="#9CA3AF",   # muted slate borders
            marker_line_width=0.35,
            hovertemplate="%{location}<extra></extra>",
            name="Districts",
            showlegend=False,
        )

    # 1b. State borders on top, darker, to anchor the eye.
    fig.add_choropleth(
        geojson=india_geo,
        featureidkey="properties.ST_NM",
        locations=state_names,
        z=[1] * len(state_names),
        showscale=False,
        colorscale=[[0, "rgba(232,219,179,0)"], [1, "rgba(232,219,179,0)"]],
        marker_line_color=PAL_INK,
        marker_line_width=1.1,
        hovertemplate="%{location}<extra></extra>",
        name="India",
        showlegend=False,
    )

    # 2. City markers, one trace per volume bucket.
    #    Legend shows only the three "volume" tiers; the residual <5 tier is
    #    plotted in muted slate without a legend entry so we keep coverage
    #    of small cities without cluttering the key.
    tier_order = ["Very high (30+)", "High (15-29)", "Moderate (5-14)"]
    tier_color = {
        "Very high (30+)":  sev_critical,
        "High (15-29)":     sev_high,
        "Moderate (5-14)":  sev_medium,
        "Other (<5)":       sev_low,
    }
    for tier in tier_order + ["Other (<5)"]:
        rows = by_city[by_city["Volume"] == tier]
        if rows.empty:
            continue
        # Tighter sizing so big cities don't swallow neighbours, plus
        # translucency + dark outline to keep overlapping circles legible.
        sizes = (rows["count"] * 0.75 + 7).clip(upper=28)
        fig.add_scattergeo(
            lat=rows["lat"], lon=rows["lon"],
            text=rows["location"], customdata=rows["count"],
            mode="markers",
            marker=dict(
                size=sizes,
                color=tier_color[tier],
                line=dict(color=PAL_INK, width=1.0),
                opacity=0.70,
            ),
            name=tier,
            hovertemplate="<b>%{text}</b> -- %{customdata} complaints<extra></extra>",
            legendgroup="cities",
            showlegend=tier != "Other (<5)",   # mute the <5 tier in the legend
        )

    # 3. Permanent labels on the top 4 cities so judges don't need to hover.
    top4 = by_city.sort_values(["count", "location"],
                                ascending=[False, True]).head(4)
    fig.add_scattergeo(
        lat=top4["lat"], lon=top4["lon"],
        text=top4["location"],
        mode="text",
        textposition="top center",
        textfont=dict(size=13, color=PAL_INK, family="sans-serif"),
        showlegend=False,
        hoverinfo="skip",
    )

    # 3. Crop the geo viewport tightly to India only.
    fig.update_geos(
        visible=False,                # hide the default world basemap
        fitbounds="locations",        # auto-zoom to the choropleth bounds
        projection_type="mercator",
        showcoastlines=False,
        showland=False,
        showocean=False,
        bgcolor=PAL_BG,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=8),
        paper_bgcolor=PAL_BG,
        plot_bgcolor=PAL_BG,
        font_color=PAL_INK,
        height=820,
        legend=dict(
            title_text="<b>Complaint volume</b>",
            orientation="v",
            yanchor="top",  y=0.98,
            xanchor="right", x=0.99,
            bgcolor="rgba(255,253,235,0.92)",
            bordercolor=PAL_INK,
            borderwidth=1,
            font=dict(size=12, color=PAL_INK),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    with st.expander("Per-city counts"):
        st.dataframe(by_city_all.sort_values("count", ascending=False)
                     .rename(columns={"location": "City", "count": "Complaints"}),
                     hide_index=True, use_container_width=True)


# --- SLA tracker ------------------------------------------------------------

def render_sla_tracker_chart(df: pd.DataFrame) -> None:
    today = pd.Timestamp(date.today())
    sla = df.dropna(subset=["sla_due_date"]).copy()
    if sla.empty:
        st.info("No SLA-enriched complaints yet.")
        return
    sla["days_until_due"] = (sla["sla_due_date"] - today).dt.days
    sla = sla.sort_values(["days_until_due", "sla_breach_prob"], ascending=[True, False])

    def _bucket(d):
        if d < 0:    return "Overdue"
        if d <= 1:   return "<1 day"
        if d <= 3:   return "<3 days"
        if d <= 7:   return "<7 days"
        return ">7 days"

    sla["bucket"] = sla["days_until_due"].apply(_bucket)
    bucket_order = ["Overdue", "<1 day", "<3 days", "<7 days", ">7 days"]
    counts = (sla.groupby("bucket").size().reindex(bucket_order, fill_value=0)
                 .rename("count").reset_index())

    # Severity-aligned colour per bucket -- overdue = Critical, >7d = Low.
    bucket_color = {
        "Overdue":  SEVERITY_COLORS["Critical"],
        "<1 day":   SEVERITY_COLORS["High"],
        "<3 days":  SEVERITY_COLORS["Medium"],
        "<7 days":  SEVERITY_COLORS["Low"],
        ">7 days":  SEVERITY_COLORS["Low"],
    }
    
    use_log = st.checkbox("Logarithmic scale (prevent Overdue skew)", value=True, key="sla_log_scale_chk")
    
    fig = px.bar(
        counts, x="bucket", y="count",
        category_orders={"bucket": bucket_order},
        color="bucket", color_discrete_map=bucket_color,
        title="Open complaints by SLA bucket",
        log_y=use_log,
    )
    fig.update_layout(plot_bgcolor=PAL_BG, paper_bgcolor=PAL_BG,
                      font_color=PAL_INK, showlegend=False,
                      margin=dict(l=50, r=20, t=20, b=50),
                      xaxis_title=None, yaxis_title="Complaints (Log Scale)" if use_log else "Complaints")
    fig.update_xaxes(tickangle=0, automargin=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_sla_tracker(df: pd.DataFrame) -> None:
    st.subheader("SLA tracker -- approaching deadline")
    render_sla_tracker_chart(df)
    
    today = pd.Timestamp(date.today())
    sla = df.dropna(subset=["sla_due_date"]).copy()
    if sla.empty:
        return
    sla["days_until_due"] = (sla["sla_due_date"] - today).dt.days
    sla = sla.sort_values(["days_until_due", "sla_breach_prob"], ascending=[True, False])
    
    st.dataframe(
        sla[["id", "customer_name", "category", "severity", "sla_due_date",
             "days_until_due", "sla_breach_prob", "risk_score"]]
        .head(25)
        .assign(sla_due_date=lambda d: d["sla_due_date"].dt.date,
                sla_breach_prob=lambda d: d["sla_breach_prob"].apply(lambda v: f"{v:.0%}"))
        .rename(columns={
            "id": "Complaint ID", "customer_name": "Customer",
            "category": "Category", "severity": "Severity",
            "sla_due_date": "SLA Deadline", "days_until_due": "Days Until Due",
            "sla_breach_prob": "Breach Probability", "risk_score": "Risk Score",
        }),
        use_container_width=True, hide_index=True,
    )


# --- root-cause alerts ------------------------------------------------------

def render_alerts(alerts: pd.DataFrame) -> None:
    st.subheader("Root-Cause Systemic Clusters (Agent 6)")
    if alerts.empty:
        st.info("No systemic clusters detected yet. Run the pipeline first.")
        return
    st.caption(
        "KMeans groups complaints by semantic similarity of the complaint text. "
        "Expand a row to view the detailed summary."
    )

    for _, a in alerts.iterrows():
        cnt = int(a["count"])
        cat = a["category"]
        
        # Header label: Cluster ID | Category | Volume
        header_label = f"⬢ Cluster #{a['cluster_id']}  |  ▪ Category: {cat}  |  ▲ Volume: {cnt} complaints"
        
        with st.expander(header_label):
            st.markdown("**Systemic Issue Summary:**")
            st.write(a["summary"])
            st.caption(f"Cluster detected on: {a.get('created_at', 'N/A')}")


# --- main -------------------------------------------------------------------

def render_live_submit() -> None:
    submission: dict[str, Any] | None = None
    with st.expander("Submit a new complaint (runs all 6 agents in real time)",
                     expanded=False):
        with st.form("live_submit"):
            c1, c2 = st.columns([2, 1])
            with c1:
                text = st.text_area(
                    "Complaint text (English / Hindi / Marathi)", height=110,
                    placeholder="e.g. UPI payment of Rs 5000 failed but amount debited.",
                )
                name = st.text_input("Customer name", placeholder="optional")
            with c2:
                channel = st.selectbox("Channel",
                    ["email", "whatsapp", "twitter", "branch", "bank_portal",
                     "phone_call", "mobile_app"], index=0)
                language = st.selectbox("Language", ["english", "hindi", "marathi"], index=0)
                account_type = st.selectbox(
                    "Account type", ["savings", "current", "credit_card", "loan", "demat"],
                    index=0)
                location = st.text_input("Location", placeholder="e.g. Mumbai")
                amount = st.number_input("Amount involved (INR)", min_value=0.0,
                                         step=100.0, value=0.0)
            submitted = st.form_submit_button("Run pipeline", type="primary",
                                              use_container_width=True)
        if submitted:
            if not text.strip():
                st.error("Please enter a complaint text.")
            else:
                submission = {
                    "complaint_text": text.strip(),
                    "customer_name": name.strip() or "Walk-in",
                    "channel": channel, "language": language,
                    "account_type": account_type, "location": location.strip() or None,
                    "amount_involved": float(amount) if amount > 0 else None,
                }

    # Run the pipeline OUTSIDE the expander so st.status / progress widgets are
    # not nested inside a non-permitted container (Streamlit >=1.41 enforces this).
    if submission is not None:
        _run_live_pipeline(submission)


def _run_live_pipeline(raw: dict[str, Any]) -> None:
    """Stream agent-by-agent status, then show the final result."""
    # Import here so the dashboard does not pay the import cost until the user
    # actually submits a complaint.
    from pipeline.orchestrator import (
        process_one_streaming, ingest_new_complaint, refresh_root_cause,
    )

    new_id = ingest_new_complaint(raw)
    st.info(f"Created complaint **{new_id}** -- now processing...")

    # Plain markdown placeholder + progress bar -- works inside any container
    # and on every Streamlit version (no st.status nesting restrictions).
    status_box = st.empty()
    progress = st.progress(0.0)
    agent_lines: list[str] = []
    step_count = {"i": 0, "total": 6}

    def _render():
        status_box.markdown(
            "<div style='padding:10px 14px;border:1px solid #1F293733;"
            "border-radius:8px;background:#FFFDEB;font-family:monospace;"
            "font-size:13px;color:#1F2937;white-space:pre-wrap'>"
            + "\n".join(agent_lines)
            + "</div>",
            unsafe_allow_html=True,
        )

    def on_step(label, status, payload):
        if status == "started":
            agent_lines.append(f"-> {label}: working...")
        elif status == "skipped":
            agent_lines.append(f"-- {label}: skipped (duplicate)")
            step_count["i"] += 1
        elif status == "done":
            if agent_lines and agent_lines[-1].startswith(f"-> {label}"):
                agent_lines[-1] = f"OK {label}: done"
            else:
                agent_lines.append(f"OK {label}: done")
            step_count["i"] += 1
        _render()
        progress.progress(min(step_count["i"] / step_count["total"], 1.0))

    try:
        result = process_one_streaming(new_id, on_step=on_step)
    except Exception as e:
        agent_lines.append(f"!! Pipeline failed: {type(e).__name__}: {e}")
        _render()
        st.error(f"Error: {e}")
        return

    agent_lines.append("ALL AGENTS FINISHED -- result below")
    _render()
    progress.progress(1.0)

    cls = result["classification"]
    sla = result["sla"]
    dup = result["duplicate"]
    risk = result["risk"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Category", cls["category"])
    c2.metric("Severity", cls["severity"])
    c3.metric("Sentiment", cls["sentiment"])
    c4.metric("Risk score", risk["overall"])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("SLA due", sla["sla_due_date"])
    c6.metric("Breach probability", f"{sla['breach_probability']:.0%}")
    c7.metric("Duplicate?", "Yes -> " + dup["duplicate_of"] if dup["is_duplicate"] else "No")
    c8.metric("Drafted reply?", "Yes" if result["draft_response"] else "Skipped")

    if result["draft_response"]:
        st.markdown("**Drafted bank reply:**")
        st.info(result["draft_response"])
    else:
        st.caption("No draft generated -- complaint was flagged as a duplicate of "
                   f"{dup['duplicate_of']} (similarity {dup['similarity']}).")

    _render_pii_panel(raw)

    if st.button("Re-run Root-Cause clustering (Agent 6)"):
        n = refresh_root_cause()
        st.success(f"Refreshed -- {n} alerts now in the database.")
        load_alerts.clear()

    load_complaints.clear()


def _render_pii_panel(raw: dict[str, Any]) -> None:
    """Show raw-vs-masked text so reviewers can see that customer identifiers
    are stripped before anything is sent to the external LLM."""
    import os
    from agents import pii as _pii

    masking_on = os.getenv("PII_MASKING", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )
    text = raw.get("complaint_text") or ""
    masker = _pii.PIIMasker()
    masked = masker.mask(text, known_values=[raw.get("customer_name")])
    mapping = masker.mapping

    st.markdown("#### 🔒 Data privacy — what actually leaves for the LLM")
    if not masking_on:
        st.warning("PII masking is currently **OFF** (`PII_MASKING=0`) — raw text "
                   "is being sent. Enable it for production.")
    elif mapping:
        st.caption(f"{len(mapping)} identifier(s) masked before the text was sent "
                   "to Groq. The real values stay in the bank's database.")
    else:
        st.caption("No PII identifiers detected in this complaint — nothing to mask.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Raw complaint** _(kept in bank DB)_")
        st.code(text or "(empty)", language=None)
    with col2:
        st.markdown("**Sent to LLM** _(PII-masked)_")
        st.code(masked or "(empty)", language=None)

    if mapping:
        with st.expander(f"Masking detail ({len(mapping)} item(s))"):
            for token, original in mapping.items():
                st.markdown(f"- `{token}`  ⟵  `{original}`")
        st.caption("The same masking is applied to every field sent to the LLM "
                   "(name, account type, etc.), not just the complaint text.")


def render_alert_banners(df: pd.DataFrame, alerts: pd.DataFrame) -> None:
    """Show top-of-page banners only when their conditions are met, styled as premium white cards with left borders."""
    status = df["status"].fillna("open")
    open_df = df[status == "open"]

    # Common styling for alert cards
    def _alert_card_html(label: str, message: str, border_color: str, bg_color: str, text_color: str) -> str:
        return (
            f"<div style='background:{bg_color}; color:{text_color}; padding:12px 16px; "
            f"border-radius:6px; border:1px solid #e2e8f0; border-left:4px solid {border_color}; "
            f"box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05); margin-bottom:8px; font-size:14px; font-weight:500; height:46px; display:flex; align-items:center;'>"
            f"<span style='font-weight:700; color:{border_color}; margin-right:8px;'>{label}</span> &bull; &nbsp; {message}"
            f"</div>"
        )

    # 1. RED: critical SLA breach risk (P > 0.85 on open complaints)
    critical_n = int(((open_df["sla_breach_prob"].fillna(0)) > 0.85).sum())
    if critical_n > 0:
        st.markdown(
            _alert_card_html("CRITICAL SLA RISK", f"{critical_n} open complaints have breach probability above 85% &mdash; immediate action required.", "#b91c1c", "#fef2f2", "#b91c1c"),
            unsafe_allow_html=True
        )

    # 2. AMBER: Systemic issues
    if not alerts.empty:
        big = alerts[alerts["count"] >= 20]
        if not big.empty:
            total_in_clusters = int(big["count"].sum())
            top = big.sort_values("count", ascending=False).head(3)
            preview = "; ".join(
                f"{int(r['count'])} {r['category']}"
                + (f" ({r['location']})" if r.get("location") else "")
                for _, r in top.iterrows()
            )
            st.markdown(
                _alert_card_html("SYSTEMIC ISSUES", f"{len(big)} clusters detected ({total_in_clusters} complaints): {preview}. See Root-cause analysis.", "#b45309", "#fefaf0", "#b45309"),
                unsafe_allow_html=True
            )

    # 3. SLATE: High-risk customers (risk_score > 80)
    high_risk_customers = int(
        df.loc[df["risk_score"].fillna(0) > 80, "customer_name"].nunique()
    )
    if high_risk_customers > 0:
        st.markdown(
            _alert_card_html("ESCALATION RISK", f"{high_risk_customers} customers have risk score above 80 &mdash; proactive outreach recommended.", "#475569", "#f1f5f9", "#475569"),
            unsafe_allow_html=True
        )

    # 4. BLUE: Pending unprocessed ingestion queue (Process button inline)
    unprocessed_count = len(db.list_unprocessed())
    if unprocessed_count > 0:
        c_text, c_btn = st.columns([5, 1], gap="small")
        with c_text:
            st.markdown(
                _alert_card_html("PENDING INTAKE", f"{unprocessed_count} raw complaints are in the queue and need to be ingested.", "#1d4ed8", "#eff6ff", "#1d4ed8"),
                unsafe_allow_html=True
            )
        with c_btn:
            # Inline button aligned inside the alert card row
            if st.button(f"🚀 Ingest {unprocessed_count} Items", key="btn_ingest_inline", use_container_width=True, type="primary"):
                from pipeline.orchestrator import process_all
                with st.spinner("Ingesting complaints..."):
                    process_all()
                st.success("Successfully processed queue!")
                st.rerun()


def render_sidebar(df: pd.DataFrame, controller) -> None:
    session = st.session_state.get("admin_session")
    if session:
        profile = session.get("profile", {})
        st.sidebar.markdown(f"👤 **{profile.get('full_name', 'Admin')}**")
        st.sidebar.caption(f"{profile.get('email', '')}")
        if st.sidebar.button("Sign Out", use_container_width=True):
            sign_out(session)
            st.session_state.pop("admin_session", None)
            if controller:
                controller.remove("complaintiq_admin_session")
            st.rerun()
        st.sidebar.divider()

    st.sidebar.header("RBI Compliance Report")
    stats = rbi_report.summary_stats(df)
    st.sidebar.metric("Total complaints", stats["total"])
    # Split the "Resolved" tile into the two underlying sources so the numbers
    # don't visually overlap with the standalone "Auto-resolved" tile.
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Resolved (manual)", stats["resolved"],
              delta="Closed by agent", delta_color="off")
    c2.metric("Auto-resolved", stats.get("auto_resolved", 0),
              delta="Duplicate + Standard", delta_color="off")
    c1.metric("Pending", stats["pending"],
              delta="Open + within SLA", delta_color="off")
    c2.metric("Breached", stats["breached"],
              delta="Open + past due", delta_color="off")
    st.sidebar.caption(
        f"Sum: {stats['resolved']} + {stats.get('auto_resolved', 0)} + "
        f"{stats['pending']} + {stats['breached']} = "
        f"{stats['resolved'] + stats.get('auto_resolved', 0) + stats['pending'] + stats['breached']} "
        f"complaints."
    )
    st.sidebar.markdown("**By category**")
    cat_df = pd.DataFrame(
        sorted(stats["by_category"].items(), key=lambda kv: -kv[1]),
        columns=["Type", "Count"],
    )
    st.sidebar.dataframe(cat_df, hide_index=True, use_container_width=True)

    csv_bytes = rbi_report.to_csv(df)
    st.sidebar.download_button(
        label="Download RBI Compliance Report (CSV)",
        data=csv_bytes,
        file_name=f"rbi_compliance_{date.today().isoformat()}.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )
    st.sidebar.caption("Format follows RBI Master Circular on Customer Service in Banks (2024).")


def render_model_performance() -> None:
    import joblib
    from pathlib import Path

    st.subheader("Model performance")
    ROOT = Path(__file__).resolve().parent.parent

    sla_path = ROOT / "models" / "sla_rf.joblib"
    cat_path = ROOT / "models" / "category_clf.joblib"
    pri_path = ROOT / "models" / "priority_gbm.joblib"

    # --- SLA model -----------------------------------------------------------
    st.markdown("### 1. SLA breach predictor (algorithm bake-off)")
    if not sla_path.exists():
        st.warning("SLA model not trained yet. Run `python -m models.train_sla_model`.")
    else:
        try:
            art = joblib.load(sla_path)
            # Backwards-compatible metric extraction: new artefact has cv_auc +
            # holdout_auc, the old one had a single `auc`.
            m = art.get("test_metrics", {})
            holdout_auc = m.get("holdout_auc", m.get("auc"))
            cv_auc = m.get("cv_auc")
            cv_auc_std = m.get("cv_auc_std")
            holdout_acc = m.get("holdout_accuracy", m.get("accuracy"))

            cols = st.columns(4)
            cols[0].metric("Winner", art.get("winner", "?"))
            cols[1].metric("Hold-out AUC",
                           f"{holdout_auc:.3f}" if holdout_auc is not None else "-")
            if cv_auc is not None:
                label = f"{cv_auc:.3f}" + (f" +/- {cv_auc_std:.3f}" if cv_auc_std else "")
                cols[2].metric("CV AUC (5-fold)", label)
            else:
                cols[2].metric("Accuracy",
                               f"{holdout_acc:.3f}" if holdout_acc is not None else "-")
            cols[3].metric("Training rows", art.get("training_rows", "?"))

            if art.get("xgb_best_params"):
                st.caption("XGBoost best params (GridSearchCV): "
                           + ", ".join(f"{k}={v}" for k, v in art["xgb_best_params"].items()))

            st.markdown("**Leaderboard:**")
            st.dataframe(pd.DataFrame(art.get("leaderboard", [])),
                         hide_index=True, use_container_width=True)

            if art.get("feature_importances"):
                st.markdown("**Top 15 features:**")
                fi = pd.DataFrame(list(art["feature_importances"].items())[:15],
                                  columns=["feature", "importance"])
                fig = px.bar(fi.sort_values("importance"), x="importance", y="feature",
                             orientation="h", height=420, title="Feature importance",
                             color_discrete_sequence=[PAL_BLUE])
                fig.update_layout(plot_bgcolor=PAL_BG, paper_bgcolor=PAL_BG,
                                  font_color=PAL_INK)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load SLA model diagnostics: {e}. Model detail views are disabled because training packages (like lightgbm) are not available.")

    st.divider()
    # --- Category classifier --------------------------------------------------
    st.markdown("### 2. Category classifier (TF-IDF + Logistic Regression)")
    if not cat_path.exists():
        st.warning("Category classifier not trained yet. Run "
                   "`python -m models.train_category_classifier`.")
    else:
        try:
            art = joblib.load(cat_path)
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{art['accuracy']:.3f}")
            c2.metric("Training rows", art["training_rows"])
            c3.metric("Label source", art.get("label_source", "?"))

            cm = pd.DataFrame(art["confusion_matrix"],
                              index=art["labels"], columns=art["labels"])
            st.markdown("**Confusion matrix:**")
            # Custom scale that stays in the brand palette: cream -> sand -> blue -> red.
            fig = px.imshow(cm, text_auto=True, aspect="auto",
                            labels=dict(x="Predicted", y="Actual"),
                            color_continuous_scale=[[0.0, PAL_BG], [0.25, PAL_SAND],
                                                    [0.6, PAL_BLUE], [1.0, PAL_RED]],
                            title="Category classifier confusion matrix")
            fig.update_layout(plot_bgcolor=PAL_BG, paper_bgcolor=PAL_BG, font_color=PAL_INK)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load category classifier diagnostics: {e}.")

    st.divider()
    # --- Priority scorer ------------------------------------------------------
    st.markdown("### 3. Priority scorer (Gradient Boosting)")
    if not pri_path.exists():
        st.warning("Priority model not trained yet. Run "
                   "`python -m models.train_priority_model`.")
    else:
        try:
            art = joblib.load(pri_path)
            c1, c2, c3 = st.columns(3)
            c1.metric("R^2 (hold-out)", f"{art['test_metrics']['r2']:.3f}")
            c2.metric("MAE", f"{art['test_metrics']['mae']:.2f}")
            c3.metric("Training rows", art["training_rows"])
            fi = pd.DataFrame(list(art["feature_importances"].items()),
                              columns=["feature", "importance"]).sort_values("importance")
            fig = px.bar(fi, x="importance", y="feature", orientation="h",
                         title="Feature importance", height=320,
                         color_discrete_sequence=[PAL_BLUE])
            fig.update_layout(plot_bgcolor=PAL_BG, paper_bgcolor=PAL_BG, font_color=PAL_INK)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load priority scorer diagnostics: {e}.")

    st.divider()
    # --- Sentiment model ------------------------------------------------------
    st.markdown("### 4. Sentiment model (Hugging Face Roberta)")
    st.markdown(
        "Model: `cardiffnlp/twitter-roberta-base-sentiment-latest` "
        "-- runs locally, no training needed. Compared per-complaint against "
        "the LLM sentiment label for agreement scoring."
    )

    st.divider()
    # --- LLM vs ML agreement (live, from DB) ---------------------------------
    st.markdown("### 5. Live LLM <-> ML agreement (from the database)")
    db.init_db()
    rows = db.list_complaints(where="ml_category IS NOT NULL OR ml_sentiment IS NOT NULL")
    if rows:
        ag = pd.DataFrame(rows)
        c1, c2 = st.columns(2)
        cat_counts = ag["category_confidence"].dropna().value_counts()
        sent_counts = ag["sentiment_confidence"].dropna().value_counts()
        cat_total = int(cat_counts.sum())
        sent_total = int(sent_counts.sum())
        cat_agree = int(cat_counts.get("High Confidence", 0))
        sent_agree = int(sent_counts.get("High Confidence", 0))
        c1.metric("Category agreement (LLM vs TF-IDF)",
                  f"{(cat_agree / cat_total * 100):.1f}%" if cat_total else "-",
                  delta=f"{cat_agree}/{cat_total}", delta_color="off")
        c2.metric("Sentiment agreement (LLM vs Roberta)",
                  f"{(sent_agree / sent_total * 100):.1f}%" if sent_total else "-",
                  delta=f"{sent_agree}/{sent_total}", delta_color="off")
    else:
        st.info("No ML-enriched rows yet. Run `python -m pipeline.ml_backfill` "
                "once the LLM pipeline has processed enough complaints.")

    st.divider()
    # --- Human feedback accuracy ---------------------------------------------
    st.markdown("### 6. Human feedback accuracy")
    fb_stats = db.feedback_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews submitted", fb_stats["total"])
    c2.metric("Marked correct", fb_stats["correct"])
    c3.metric("Corrections", fb_stats["corrections"])
    acc = fb_stats["accuracy_rate"]
    c4.metric("Accuracy rate", f"{acc:.0%}" if acc is not None else "-")
    if fb_stats["corrections_by_field"]:
        st.caption("Corrections by field: " + ", ".join(
            f"{k}={v}" for k, v in fb_stats["corrections_by_field"].items()))
    st.caption("Submit reviews from the Feedback tab to populate this section.")


def render_analytics(df: pd.DataFrame) -> None:
    st.subheader("Analytics")
    if df.empty:
        st.info("No data yet.")
        return

    layout = dict(plot_bgcolor=PAL_BG, paper_bgcolor=PAL_BG, font_color=PAL_INK)
    channel_color = SEVERITY_COLORS["Low"]   # neutral slate for non-categorical bars
    city_color    = CATEGORY_COLORS["General"]

    # --- 1. Complaints per day (spline area) --------------------------------
    by_day = (df.dropna(subset=["date"])
                .groupby(df["date"].dt.date).size().reset_index(name="count"))
    by_day.columns = ["date", "count"]
    fig1 = px.area(by_day, x="date", y="count",
                   title="Complaints per day trend",
                   color_discrete_sequence=[CATEGORY_COLORS["UPI"]])
    fig1.update_layout(**layout, xaxis_title="Date", yaxis_title=None)
    fig1.update_traces(line=dict(width=3, shape='spline'), fillcolor='rgba(59, 130, 246, 0.15)')

    # --- 2. Complaints by category (donut) ----------------------------------
    cat_counts = df["category"].fillna("Unknown").value_counts().reset_index()
    cat_counts.columns = ["category", "count"]
    total_cat = cat_counts["count"].sum()
    cat_text_labels = []
    for _, row in cat_counts.iterrows():
        pct = row["count"] / total_cat
        if pct >= 0.10:
            cat_text_labels.append(f"{row['category']}<br>{pct:.0%}")
        else:
            cat_text_labels.append("")  # hide from segment, only display in right legend

    fig2 = px.pie(cat_counts, names="category", values="count",
                  title="Complaints by category",
                  color="category", color_discrete_map=CATEGORY_COLORS, hole=0.55)
    fig2.update_traces(text=cat_text_labels, textinfo='text', textposition='inside')
    fig2.update_layout(**layout, showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02))

    # --- 3. Complaints by channel (horizontal bar) --------------------------
    chan_counts = df["channel"].fillna("Unknown").value_counts().reset_index()
    chan_counts.columns = ["channel", "count"]
    chan_counts = chan_counts.sort_values("count")
    fig3 = px.bar(chan_counts, x="count", y="channel", orientation="h",
                  title="Complaints by channel",
                  color_discrete_sequence=[channel_color])
    fig3.update_layout(**layout, xaxis_title=None, yaxis_title=None)

    # --- 4. Sentiment distribution (donut) ----------------------------------
    sent_counts = df["sentiment"].fillna("Unknown").value_counts().reset_index()
    sent_counts.columns = ["sentiment", "count"]
    total_sent = sent_counts["count"].sum()
    sent_text_labels = []
    for _, row in sent_counts.iterrows():
        pct = row["count"] / total_sent
        if pct >= 0.10:
            sent_text_labels.append(f"{row['sentiment']}<br>{pct:.0%}")
        else:
            sent_text_labels.append("")

    fig4 = px.pie(sent_counts, names="sentiment", values="count",
                  title="Sentiment distribution", hole=0.55,
                  category_orders={"sentiment": SENTIMENT_ORDER},
                  color="sentiment", color_discrete_map=SENTIMENT_COLORS)
    fig4.update_traces(text=sent_text_labels, textinfo='text', textposition='inside')
    fig4.update_layout(**layout, showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02))

    # --- 5. Severity distribution (bar) -------------------------------------
    sev_counts = df["severity"].fillna("Unknown").value_counts().reindex(
        SEVERITY_ORDER + [c for c in df["severity"].dropna().unique()
                          if c not in SEVERITY_ORDER]
    ).dropna().reset_index()
    sev_counts.columns = ["severity", "count"]
    fig5 = px.bar(sev_counts, x="severity", y="count",
                  title="Severity distribution",
                  category_orders={"severity": SEVERITY_ORDER},
                  color="severity", color_discrete_map=SEVERITY_COLORS)
    fig5.update_layout(**layout, showlegend=False,
                       xaxis_title=None, yaxis_title=None)

    # --- 6. Avg breach probability by category (bar) ------------------------
    breach_cat = (df.dropna(subset=["sla_breach_prob", "category"])
                    .groupby("category")["sla_breach_prob"].mean()
                    .reset_index().sort_values("sla_breach_prob", ascending=False))
    fig6 = px.bar(breach_cat, x="category", y="sla_breach_prob",
                  title="Average breach probability by category",
                  color="category", color_discrete_map=CATEGORY_COLORS)
    fig6.update_yaxes(tickformat=".0%", title=dict(text="Avg Breach Prob", standoff=15))
    fig6.update_layout(**layout, showlegend=False, xaxis_title=None)

    # --- 7. Top 10 cities (bar) ---------------------------------------------
    cities = (df["location"].fillna("Unknown")
                            .value_counts().head(10)
                            .reset_index())
    cities.columns = ["city", "count"]
    cities = cities.sort_values("count")
    fig7 = px.bar(cities, x="count", y="city", orientation="h",
                  title="Top 10 cities by complaint count",
                  color_discrete_sequence=[city_color])
    fig7.update_layout(**layout, xaxis_title=None, yaxis_title=None)

    # --- 8. Resolution status breakdown (donut) -----------------------------
    today = pd.Timestamp(date.today())
    def _bucket(row):
        s = (row.get("status") or "open")
        if s == "resolved":           return "Resolved"
        if s == "auto_resolved_dup":  return "Auto-Resolved (Dup)"
        if s == "auto_resolved_std":  return "Auto-Resolved (Standard)"
        # open
        due = row.get("sla_due_date")
        if pd.notna(due) and pd.Timestamp(due) < today:
            return "Breached"
        return "Pending"

    df_buckets = df.apply(_bucket, axis=1).value_counts().reset_index()
    df_buckets.columns = ["bucket", "count"]
    total_buckets = df_buckets["count"].sum()
    bucket_text_labels = []
    for _, row in df_buckets.iterrows():
        pct = row["count"] / total_buckets
        if pct >= 0.10:
            bucket_text_labels.append(f"{row['bucket']}<br>{pct:.0%}")
        else:
            bucket_text_labels.append("")

    fig8 = px.pie(df_buckets, names="bucket", values="count",
                  title="Resolution status breakdown",
                  color="bucket", color_discrete_map=RESOLUTION_COLORS, hole=0.55)
    fig8.update_traces(text=bucket_text_labels, textinfo='text', textposition='inside')
    fig8.update_layout(**layout, showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02))

    # --- render in a symmetric 2-col card grid ------------------------------
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        with st.container(border=True):
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
    with r1c2:
        with st.container(border=True):
            st.plotly_chart(fig8, use_container_width=True, config={"displayModeBar": False})
            
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        with st.container(border=True):
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    with r2c2:
        with st.container(border=True):
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
            
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        with st.container(border=True):
            st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})
    with r3c2:
        with st.container(border=True):
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
            
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        with st.container(border=True):
            st.plotly_chart(fig7, use_container_width=True, config={"displayModeBar": False})
    with r4c2:
        with st.container(border=True):
            st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})


def render_feedback(df: pd.DataFrame) -> None:
    st.subheader("Human-in-the-Loop AI Classification Feedback")
    stats = db.feedback_stats()
    
    cols = st.columns(4)
    
    def _render_fb_card(col, title, value, detail=None):
        detail_html = f"<div style='margin-top:6px; font-size:12px; color:#64748b; font-weight:400;'>{detail}</div>" if detail else "<div style='margin-top:6px; height:18px;'></div>"
        col.markdown(
            f"<div style='display:flex; flex-direction:column; justify-content:space-between; "
            f"height:120px; background-color:#ffffff; border:1px solid #e2e8f0; "
            f"border-radius:8px; padding:16px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05), "
            f"0 2px 4px -1px rgba(0,0,0,0.03); color:#0f172a;'>"
            f"<div style='font-size:11px; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;'>{title}</div>"
            f"<div style='font-size:28px; font-weight:700; margin-top:12px; line-height:1; color:#0f172a;'>{value}</div>"
            f"{detail_html}"
            f"</div>",
            unsafe_allow_html=True
        )
        
    _render_fb_card(cols[0], "Total Feedback", stats["total"])
    _render_fb_card(cols[1], "Marked Correct", stats["correct"])
    _render_fb_card(cols[2], "Corrections", stats["corrections"])
    acc = stats["accuracy_rate"]
    _render_fb_card(cols[3], "Accuracy Rate", f"{acc:.0%}" if acc is not None else "-", "ML classification rate")

    if stats["corrections_by_field"]:
        st.caption("Corrections by field: " + ", ".join(
            f"{k}={v}" for k, v in stats["corrections_by_field"].items()))

    pick = st.selectbox(
        "Pick a complaint to review",
        df.dropna(subset=["category"]).sort_values("date", ascending=False)["id"].tolist(),
        key="fb_pick",
    )
    if not pick:
        return

    r = db.get_complaint(pick)
    if not r:
        return

    with st.container(border=True):
        st.markdown(f"**{r['id']}** &bull; {r.get('customer_name', '?')} "
                    f"({r.get('channel', '?')}, {r.get('language', '?')})")
        st.write(r.get("complaint_text", ""))
        
        cols_fb = st.columns(3)
        _feedback_widget(cols_fb[0], r, "category", r.get("category"), FB_CATEGORIES)
        _feedback_widget(cols_fb[1], r, "severity", r.get("severity"), FB_SEVERITIES)
        _feedback_widget(cols_fb[2], r, "sentiment", r.get("sentiment"), FB_SENTIMENTS)

    prior = db.list_feedback(complaint_id=pick)
    if prior:
        st.markdown("**Prior feedback on this complaint:**")
        st.dataframe(
            pd.DataFrame(prior)[["created_at", "field", "is_correct",
                                 "original_value", "corrected_value"]],
            hide_index=True, use_container_width=True,
        )


def _feedback_widget(col, row, field: str, current: str | None, choices: list[str]) -> None:
    col.markdown(f"**{field.capitalize()}**: `{current or '?'}`")
    key_prefix = f"fb_{row['id']}_{field}"
    a, b = col.columns(2)
    if a.button("Correct", key=f"{key_prefix}_ok", use_container_width=True):
        db.record_feedback(row["id"], field, current, None, is_correct=True)
        load_complaints.clear()
        st.success("Recorded.")
        st.rerun()
    if b.button("Wrong", key=f"{key_prefix}_bad", use_container_width=True):
        st.session_state[f"{key_prefix}_correcting"] = True
    if st.session_state.get(f"{key_prefix}_correcting"):
        new = col.selectbox(
            "Correct value",
            [c for c in choices if c != current],
            key=f"{key_prefix}_new",
        )
        if col.button("Save correction", key=f"{key_prefix}_save",
                      type="primary", use_container_width=True):
            db.record_feedback(row["id"], field, current, new, is_correct=False)
            st.session_state[f"{key_prefix}_correcting"] = False
            load_complaints.clear()
            st.success(f"Updated {field} -> {new}.")
            st.rerun()


def render_login_screen() -> None:
    st.markdown(
        f"<h1 style='text-align: center; color:{PAL_INK}; margin-top: 10vh;'>"
        f"<span style='color:{PAL_BLUE}'>Complaint</span>"
        f"<span style='color:{PAL_RED}'>IQ</span></h1>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<p style='text-align: center; color:{PAL_INK};'>Bank Admin Portal</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="background-color:{PAL_SAND}55; padding: 20px; border-radius: 10px; border: 1px solid {PAL_BLUE}33; margin-top: 20px;">
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="admin@bank.com")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit:
                if not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    try:
                        session = sign_in(email, password)
                        update_last_login(session["user"]["id"])
                        st.session_state["admin_session"] = session
                        st.rerun()
                    except RuntimeError as e:
                        st.error(str(e))
        
        st.markdown("</div>", unsafe_allow_html=True)

def render_mfa_screen() -> None:
    st.markdown(
        f"<h1 style='text-align: center; color:{PAL_INK}; margin-top: 10vh;'>"
        f"<span style='color:{PAL_BLUE}'>Complaint</span>"
        f"<span style='color:{PAL_RED}'>IQ</span></h1>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<p style='text-align: center; color:{PAL_INK};'>Two-Factor Authentication</p>", unsafe_allow_html=True)

    session = st.session_state.get("admin_session")
    user_id = session["user"]["id"]
    email = session["user"]["email"]
    
    profile = session.get("profile", {})
    totp_secret = profile.get("totp_secret")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="background-color:{PAL_SAND}55; padding: 20px; border-radius: 10px; border: 1px solid {PAL_BLUE}33; margin-top: 20px;">
        """, unsafe_allow_html=True)
        
        if not totp_secret:
            st.info("Please set up Two-Factor Authentication.")
            if "setup_totp_secret" not in st.session_state:
                import pyotp
                st.session_state["setup_totp_secret"] = pyotp.random_base32()
            
            secret = st.session_state["setup_totp_secret"]
            import pyotp
            totp = pyotp.TOTP(secret)
            uri = totp.provisioning_uri(name=email, issuer_name="ComplaintIQ Admin")
            
            import qrcode
            import io
            img = qrcode.make(uri)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            
            st.image(buf.getvalue(), width=250, caption="Scan with Google Authenticator or Authy")
            st.write(f"Or enter code manually: `{secret}`")
            
            with st.form("mfa_setup_form"):
                code = st.text_input("Enter 6-digit code", max_chars=6)
                submit = st.form_submit_button("Verify & Save", use_container_width=True)
                if submit:
                    if totp.verify(code, valid_window=1):
                        from auth.supabase_auth import update_totp_secret
                        update_totp_secret(user_id, secret)
                        st.session_state["admin_session"]["profile"]["totp_secret"] = secret
                        st.session_state["mfa_verified"] = True
                        st.rerun()
                    else:
                        st.error("Invalid code. Try again.")
        else:
            with st.form("mfa_verify_form"):
                st.write("Enter the 6-digit code from your authenticator app.")
                code = st.text_input("Authenticator Code", max_chars=6)
                submit = st.form_submit_button("Verify", use_container_width=True)
                if submit:
                    import pyotp
                    totp = pyotp.TOTP(totp_secret)
                    if totp.verify(code, valid_window=1):
                        st.session_state["mfa_verified"] = True
                        st.rerun()
                    else:
                        st.error("Invalid code. Try again.")
        
        if st.button("Logout", key="mfa_logout"):
            st.session_state["logout_requested"] = True
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
    
    from streamlit_cookies_controller import CookieController
    controller = CookieController()

    if st.session_state.get("logout_requested"):
        controller.remove("complaintiq_admin_session")
        st.session_state.clear()
        render_login_screen()
        return
    try:
        from streamlit_cookies_controller import CookieController
        controller = CookieController()
    except ImportError:
        controller = None
    
    # 1. Fetch the cookie
    saved = controller.get("complaintiq_admin_session") if controller else None
    
    # 2. Sync cookie to session_state if we just loaded the page
    if "admin_session" not in st.session_state:
        if saved:
            st.session_state["admin_session"] = saved
        else:
            st.session_state["admin_session"] = None

    # 3. Sync session_state to cookie if we just logged in (saved is None)
    if st.session_state.get("admin_session") and not saved and controller:
        controller.set("complaintiq_admin_session", st.session_state["admin_session"], max_age=604800)

    # 4. If still not logged in, show login screen
    if not st.session_state.get("admin_session"):
        render_login_screen()
        return

    # 5. MFA Verification Phase
    if not st.session_state.get("mfa_verified"):
        render_mfa_screen()
        return
    st.markdown(
        f"<h1 style='color:{PAL_INK};margin-bottom:0'>"
        f"<span style='color:{PAL_BLUE}'>Complaint</span>"
        f"<span style='color:{PAL_RED}'>IQ</span></h1>",
        unsafe_allow_html=True,
    )
    st.caption("AI-powered unified complaint dashboard for Union Bank of India "
               "| iDEA 2.0 / PSBs Hackathon 2026 | Team AgentForge")

    df = load_complaints()
    alerts = load_alerts()
    if df.empty:
        st.warning("Database empty. Run `python -m database.db` to seed and "
                   "`python -m pipeline.orchestrator` to process.")
        return

    # 1. Dark slate sidebar navigation menu
    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center;padding:15px 0 25px 0;'>"
            f"<span style='font-size:24px;font-weight:800;color:#3b82f6;'>Complaint</span>"
            f"<span style='font-size:24px;font-weight:800;color:#dc2626;'>IQ</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        page = st.radio(
            "WORKSPACE",
            ["📊 Dashboard Overview", "📥 Operations Workspace", "👤 Customer Insights", "🗺️ Geospatial Intel", "⚙️ AI Diagnostics", "💬 Correction & Feedback"],
            key="navigation_page",
            label_visibility="collapsed"
        )

    # 2. Main Canvas Header (sophisticated title & metadata + export & profile top-right alignment)
    session = st.session_state.get("admin_session")
    profile = session.get("profile", {}) if session else {}
    user_name = profile.get("full_name", "Admin")
    email = profile.get("email", "admin@bank.com")
    
    col_title, col_export, col_user = st.columns([2.5, 0.8, 0.7])
    with col_title:
        st.markdown(
            f"<h2 style='color:{PAL_INK};margin:0;font-size:24px;font-weight:700;line-height:1.2;'>"
            f"<span style='color:#3b82f6'>Complaint</span>"
            f"<span style='color:#dc2626'>IQ</span></h2>"
            f"<p style='color:#64748b;margin:2px 0 0 0;font-size:12px;font-weight:400;'>AI-powered unified complaint dashboard &bull; Union Bank of India</p>",
            unsafe_allow_html=True,
        )
    with col_export:
        csv_bytes = rbi_report.to_csv(df)
        st.download_button(
            label="📥 Export RBI CSV",
            data=csv_bytes,
            file_name=f"rbi_compliance_{date.today().isoformat()}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_user:
        with st.popover(f"👤 {user_name}", use_container_width=True):
            st.markdown(f"**{user_name}**")
            st.caption(f"{email}")
            st.caption(f"Role: {profile.get('role', 'admin').upper()}")
            st.divider()
            if st.button("Sign Out", key="header_signout_btn", use_container_width=True):
                sign_out(session)
                st.session_state.pop("admin_session", None)
                if controller:
                    controller.remove("complaintiq_admin_session")
                st.rerun()

    st.divider()

    # 3. Workspace View Router
    if page == "📊 Dashboard Overview":
        # Global KPI indicators & Urgency alert banners rendered on Dashboard Overview only
        render_kpis(df)
        render_alert_banners(df, alerts)
        st.divider()
        
        # Columns layout for Alerts & SLA chart
        col_alerts, col_sla = st.columns([1, 1], gap="medium")
        with col_alerts:
            render_alerts(alerts)
        with col_sla:
            st.subheader("SLA Breach Risk Profile")
            render_sla_tracker_chart(df)
            
        st.divider()
        st.subheader("Operational Analytics & Trends")
        render_analytics(df)
    elif page == "📥 Operations Workspace":
        view, selected_row = render_live_feed(df)
        if selected_row is not None:
            render_selected_complaint_workspace(selected_row, df)
    elif page == "👤 Customer Insights":
        render_customer_view(df)
    elif page == "🗺️ Geospatial Intel":
        render_india_map(df)
    elif page == "⚙️ AI Diagnostics":
        render_model_performance()
    elif page == "💬 Correction & Feedback":
        tab_fb, tab_draft = st.tabs(["Feedback", "Drafts"])
        with tab_fb:
            render_feedback(df)
        with tab_draft:
            st.subheader("Drafted replies")
            drafted = df.dropna(subset=["draft_response"]).sort_values("date", ascending=False).head(20)
        if drafted.empty:
            st.info("No drafts yet -- pipeline still running, or all complaints were duplicates.")
        for _, r in drafted.iterrows():
            with st.expander(f"{r['id']} -- {r['customer_name']} -- {r['category']} ({r['severity']})"):
                st.caption(f"Channel: {r['channel']}  |  Language: {r['language']}  "
                           f"|  SLA due: {pd.to_datetime(r['sla_due_date']).date()}")
                st.markdown("**Original complaint:**")
                st.write(r["complaint_text"])
                st.markdown("**Drafted reply:**")
                st.write(r["draft_response"])


if __name__ == "__main__":
    main()
