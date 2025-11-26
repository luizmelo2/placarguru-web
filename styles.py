"""Módulo para injeção de CSS customizado e temas de gráficos."""
import altair as alt
import streamlit as st
from string import Template


def inject_custom_css(dark_mode: bool = False):
    """Insere o CSS customizado na página, inspirado no redesign do Placar Guru."""

    theme = "dark" if dark_mode else "light"
    css_tpl = Template("""
<style>
  :root,
  [data-pg-theme="light"],
  .stApp[data-pg-theme="light"] {
    --bg: #f8fafc;
    --panel: #ffffff;
    --glass: rgba(255,255,255,0.65);
    --stroke: #e2e8f0;
    --text: #0f172a;
    --muted: #475569;
    --primary: #2563eb;
    --primary-2: #22d3ee;
    --neon: #bfff3b;
    --shadow: 0 20px 60px rgba(0,0,0,0.12);
  }

  [data-pg-theme="dark"],
  .stApp[data-pg-theme="dark"] {
    --bg: #0b1224;
    --panel: #0f172a;
    --glass: rgba(255,255,255,0.04);
    --stroke: #1f2937;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --primary: #60a5fa;
    --primary-2: #22d3ee;
    --neon: #bfff3b;
    --shadow: 0 20px 60px rgba(0,0,0,0.35);
  }

  html, body, .stApp {
    font-size: 16px;
    background: radial-gradient(circle at 20% 20%, rgba(96,165,250,0.08), transparent 25%),
                radial-gradient(circle at 80% 0%, rgba(34,211,238,0.06), transparent 30%),
                var(--bg);
    color: var(--text);
    transition: background 260ms ease, color 260ms ease;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
  }
  @media (max-width: 768px) {
    html, body, .stApp { font-size: 17px; }
    section[data-testid="stSidebar"] { display:none !important; }
  }

  .block-container { padding-top: 0.5rem !important; max-width: 1200px; }
  h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.01em; }
  h2 { font-size: 1.25rem; font-weight: 700; }
  h3 { font-size: 1.1rem; font-weight: 700; }

  .pg-hero {
    border: 1px solid var(--stroke);
    border-radius: 18px;
    padding: 18px;
    background: color-mix(in srgb, var(--panel) 92%, transparent);
    box-shadow: var(--shadow);
    display: grid;
    gap: 12px;
  }

  .pg-kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px,1fr)); gap: 12px; }
  .pg-kpi {
    border: 1px solid var(--stroke);
    border-radius: 14px;
    padding: 12px;
    background: color-mix(in srgb, var(--panel) 88%, transparent);
  }
  .pg-kpi .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
  .pg-kpi .value { font-size: 1.35rem; font-weight: 700; }
  .pg-kpi .delta { color: #10b981; font-size: 12px; font-weight: 600; }

  .tourn-box {
    border: 1px solid var(--stroke);
    border-radius: 14px;
    padding: 14px;
    background: color-mix(in srgb, var(--panel) 92%, transparent);
    box-shadow: var(--shadow);
    margin-bottom: 12px;
  }
  .tourn-title { font-weight: 800; font-size: 1.05rem; letter-spacing: -0.01em; }

  .badge { padding: 6px 10px; border-radius: 999px; font-weight: 700; font-size: 12px; border: 1px solid var(--stroke); }
  .badge-ok { background: #14532d; color: #d1fae5; }
  .badge-bad { background: #7f1d1d; color: #fee2e2; }
  .badge-wait { background: #1f2937; color: #e2e8f0; }
  .badge-finished { background: #1e3a8a; color: #dbeafe; }

  .accent-green { color:#22c55e; font-weight:700; }
  .text-odds { color: var(--muted); font-size: 0.9em; margin-left: 0.35rem; }
  .text-label { color: var(--muted); font-weight:600; }
  .text-muted { color: var(--muted); }

  .pg-card {
    position: relative;
    border: 1px solid var(--stroke);
    border-radius: 18px;
    padding: 14px 16px;
    background: var(--panel);
    box-shadow: var(--shadow);
    margin-bottom: 12px;
    transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
  }
  .pg-card:hover { transform: translateY(-2px); border-color: var(--primary); box-shadow: 0 20px 50px rgba(37,99,235,0.16); }
  .pg-card.neon {
    border: 1.5px solid var(--neon);
    box-shadow: 0 0 0 1px rgba(191,255,59,0.35), 0 10px 40px rgba(191,255,59,0.25);
  }
  .pg-card.neon::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    box-shadow: 0 0 24px rgba(191,255,59,0.18);
    opacity: 0;
    animation: pg-pulse 2.4s ease-in-out infinite;
    pointer-events: none;
  }
  @keyframes pg-pulse { 0% {opacity:0;} 50% {opacity:1;} 100% {opacity:0;} }

  .pg-meta { color: var(--muted); font-size: 13px; margin-top: -2px; margin-bottom: 8px; }
  .pg-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap:10px; }
  .pg-pill { border: 1px solid var(--stroke); border-radius: 12px; padding: 10px 12px; background: color-mix(in srgb, var(--panel) 88%, transparent); }
  .pg-pill .label { color: var(--muted); font-size: 12px; }
  .pg-pill .value { font-weight: 700; font-size: 1.05rem; }

  /* Tabelas com visual de card */
  div[data-testid="stTable"] table,
  div[data-testid="stDataFrameContainer"] table {
    border-collapse: collapse;
    width: 100%;
    background: color-mix(in srgb, var(--panel) 94%, transparent);
    color: var(--text);
    border-radius: 14px;
    overflow: hidden;
  }
  div[data-testid="stDataFrameContainer"] {
    border: 1px solid var(--stroke);
    border-radius: 14px;
    box-shadow: var(--shadow);
    background: color-mix(in srgb, var(--panel) 90%, transparent);
    padding: 4px;
  }
  div[data-testid="stTable"] thead th,
  div[data-testid="stDataFrameContainer"] thead th {
    background: color-mix(in srgb, var(--panel) 86%, transparent);
    color: var(--muted);
    font-weight: 800;
    border-bottom: 1px solid var(--stroke);
  }
  div[data-testid="stTable"] tbody td,
  div[data-testid="stDataFrameContainer"] tbody td {
    border-bottom: 1px solid color-mix(in srgb, var(--stroke) 70%, transparent);
    padding: 10px 12px;
  }
  div[data-testid="stTable"] tbody tr:hover,
  div[data-testid="stDataFrameContainer"] tbody tr:hover {
    background: color-mix(in srgb, var(--primary) 10%, transparent);
  }

  /* Moldura para gráficos Altair */
  .vega-embed {
    background: color-mix(in srgb, var(--panel) 90%, transparent) !important;
    border: 1px solid var(--stroke);
    border-radius: 16px;
    box-shadow: var(--shadow);
    padding: 8px;
  }

  .info-grid { display:grid; grid-template-columns: 1fr 1fr; gap: 0.1rem 0.8rem; margin-top: 0.25rem; }
  @media (max-width: 768px) { .info-grid { grid-template-columns: 1fr; gap: 0.25rem; } }

  button, .stButton>button {
    border-radius: 14px; padding: 10px 14px; font-weight: 700;
    background: linear-gradient(120deg, var(--primary), var(--primary-2));
    color: white; border: 1px solid color-mix(in srgb, var(--primary) 70%, var(--stroke));
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.2), 0 10px 30px rgba(37,99,235,0.25);
  }
  button:hover, .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 12px 40px rgba(37,99,235,0.35); }

  div[data-testid="stExpander"] summary { padding: 10px 12px; font-size: 1.02rem; font-weight: 700; }
  div[data-testid="stDataFrameContainer"] { border-radius: 12px; overflow: hidden; }
</style>
<script>
  (function() {
    const theme = '$theme';
    const targets = [document.documentElement, document.body, document.querySelector('.stApp')];
    targets.forEach(el => el && el.setAttribute('data-pg-theme', theme));

    // Atualiza color-scheme para melhorar inputs nativos em cada modo
    let meta = document.querySelector('meta[name="color-scheme"]');
    if (!meta) {
      meta = document.createElement('meta');
      meta.name = 'color-scheme';
      document.head.appendChild(meta);
    }
    meta.content = theme === 'dark' ? 'dark light' : 'light dark';
  })();
</script>
""")

    st.markdown(css_tpl.substitute(theme=theme), unsafe_allow_html=True)


def chart_tokens(dark_mode: bool):
    """Retorna tokens cromáticos para uso nos gráficos Altair."""

    if dark_mode:
        return {
            "background": "#0b1224",
            "panel": "#0f172a",
            "stroke": "#1f2937",
            "text": "#e2e8f0",
            "grid": "#1f2a3c",
            "palette": [
                "#60a5fa",
                "#22d3ee",
                "#bfff3b",
                "#a855f7",
                "#f97316",
                "#c7d2fe",
            ],
            "accent": "#60a5fa",
        }

    return {
        "background": "#f8fafc",
        "panel": "#ffffff",
        "stroke": "#e2e8f0",
        "text": "#0f172a",
        "grid": "#e2e8f0",
        "palette": [
            "#2563eb",
            "#22d3ee",
            "#bfff3b",
            "#a855f7",
            "#f97316",
            "#0ea5e9",
        ],
        "accent": "#2563eb",
    }


def apply_altair_theme(dark_mode: bool = False):
    """Configura tema do Altair com base na paleta do redesign."""

    tokens = chart_tokens(dark_mode)
    name = "pg-dark" if dark_mode else "pg-light"

    theme = {
        "config": {
            "background": tokens["background"],
            "view": {"stroke": "transparent", "cornerRadius": 14, "fill": tokens["panel"]},
            "axis": {
                "domainColor": tokens["stroke"],
                "gridColor": tokens["grid"],
                "labelColor": tokens["text"],
                "titleColor": tokens["text"],
            },
            "legend": {"labelColor": tokens["text"], "titleColor": tokens["text"]},
            "title": {"color": tokens["text"], "fontSize": 16, "fontWeight": 700},
            "range": {
                "category": tokens["palette"],
                "ordinal": tokens["palette"],
                "ramp": tokens["palette"],
            },
        }
    }

    alt.themes.register(name, lambda theme=theme: theme)
    alt.themes.enable(name)
