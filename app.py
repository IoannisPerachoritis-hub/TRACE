import streamlit as st
from utils.pub_theme import apply_matplotlib_theme, build_plotly_template

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="TRACE",
    layout="wide",
    page_icon="T",
)
apply_matplotlib_theme()
build_plotly_template()

# --- Custom styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #2C5F2D;
    --primary-light: #4A8B4F;
    --accent: #3A7D44;
    --bg-card: #F5F5F5;
    --bg-dark: #1a1a1a;
    --border: #D5D5D5;
    --text-muted: #666666;
    --success: #2e7d32;
    --info: #1565c0;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

h1 { color: var(--primary); font-weight: 700; letter-spacing: -0.02em; }
h2 { color: #2C3E2C; font-weight: 600; }
h3 { color: #444444; font-weight: 600; }

.stButton>button {
    background-color: var(--primary-light);
    color: white;
    border-radius: 6px;
    border: none;
    font-weight: 500;
    font-family: 'IBM Plex Sans', sans-serif;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    background-color: var(--accent);
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(44, 95, 45, 0.3);
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f5f8f5 0%, #e8f0e8 100%);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
}

/* ── Scrollable tab bar (fixes long tab labels on narrow screens) ── */
div[data-testid="stTabs"] div[role="tablist"] {
    overflow-x: auto;
    overflow-y: hidden;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
    padding-bottom: 2px;
    gap: 0;
}
div[role="tab"] {
    flex-shrink: 0;
    white-space: nowrap;
}

/* ── Section headers in analysis pages ── */
.stMarkdown h3 {
    border-left: 3px solid var(--primary-light);
    padding-left: 10px;
    margin-top: 24px;
}
.stMarkdown h4 {
    color: #444444;
    margin-top: 16px;
}

/* ── Expander headers ── */
details summary {
    font-weight: 600;
    font-family: 'IBM Plex Sans', sans-serif;
    color: #2C3E2C;
}
details[open] summary {
    color: var(--primary);
}

/* ── Alert / notification boxes ── */
div[data-testid="stAlert"] {
    border-radius: 8px;
    border-left-width: 4px;
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #f5f7f5;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--primary);
}

/* ── Numeric / text inputs ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# Welcome
# ==========================================================
st.title("TRACE — Trait Resolution and Candidate Evaluation")

st.markdown("""
Integrated GWAS, LD analysis, haplotype mapping, and gene annotation
for **tomato, pepper, and other crop** breeding panels.

Use the **sidebar** to navigate to:
- **GWAS Analysis** — upload data and run the full pipeline
- **LD & Haplotype Analysis** — explore LD structure and haplotype effects
- **Help & Reference** — output format, column glossary, and methods reference
""")

st.info(
    "**Getting started?** Navigate to **GWAS Analysis** in the sidebar, "
    "upload your VCF and phenotype file, and click **One-Click Full Analysis**."
)

st.markdown("""
---
<div style="color: #666666; font-size: 0.85em;">
    © 2026 TRACE · Built for the NatGenCrop Project · v1.0
</div>
""", unsafe_allow_html=True)
