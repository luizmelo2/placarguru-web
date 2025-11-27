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
  .stApp[data-pg-theme="light"],
  .pg-theme-light {
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
  .stApp[data-pg-theme="dark"],
  .pg-theme-dark {
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
                var(--bg) !important;
    color: var(--text) !important;
    transition: background 260ms ease, color 260ms ease;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
  }
  @media (max-width: 768px) {
    html, body, .stApp { font-size: 17px; }
    section[data-testid="stSidebar"] { display:none !important; }
  }

  .block-container { padding-top: 0.5rem !important; max-width: 1200px; }
  .stMain, .block-container, .main { background: transparent !important; }
  h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.01em; }
  h2 { font-size: 1.25rem; font-weight: 700; }
  h3 { font-size: 1.1rem; font-weight: 700; }

  .pg-topbar {
    position: sticky;
    top: 0;
    z-index: 50;
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 14px;
    align-items: center;
    padding: 14px 18px;
    margin: 0 -1rem 12px -1rem;
    background: color-mix(in srgb, var(--bg) 90%, transparent);
    backdrop-filter: blur(18px);
    border-bottom: 1px solid var(--stroke);
    box-shadow: 0 14px 40px rgba(0,0,0,0.08);
    position: sticky;
  }
  .pg-topbar::before {
    content: '';
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: radial-gradient(circle at 16% 20%, color-mix(in srgb, var(--primary) 24%, transparent) 0, transparent 40%),
                radial-gradient(circle at 84% 10%, color-mix(in srgb, var(--primary-2) 22%, transparent) 0, transparent 42%);
    opacity: 0.8;
  }
  .pg-topbar > * { position: relative; z-index: 1; }
  .pg-topbar__brand { display: flex; align-items: center; gap: 12px; min-width: 220px; }
  .pg-logo {
    width: 44px; height: 44px;
    border-radius: 14px;
    background: linear-gradient(135deg, #38bdf8, #22d3ee, #bfff3b);
    box-shadow: 0 14px 36px rgba(34, 211, 238, 0.35);
  }
  .pg-eyebrow { margin: 0; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
  .pg-appname { font-size: 1.05rem; font-weight: 800; letter-spacing: -0.01em; }
  .pg-topbar__nav { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
  .pg-tab {
    display: inline-flex;
    align-items: center;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid color-mix(in srgb, var(--stroke) 70%, transparent);
    background: color-mix(in srgb, var(--panel) 92%, transparent);
    font-weight: 700;
    font-size: 13px;
    color: var(--muted);
    transition: border-color 160ms ease, background 160ms ease, color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
    cursor: pointer;
  }
  .pg-tab.active {
    color: var(--text);
    border-color: var(--primary);
    background: color-mix(in srgb, var(--primary) 12%, var(--panel));
    box-shadow: 0 10px 28px rgba(37,99,235,0.18), inset 0 1px 0 rgba(255,255,255,0.08);
  }
  .pg-tab:hover { border-color: var(--primary); color: var(--text); background: color-mix(in srgb, var(--panel) 96%, transparent); transform: translateY(-1px); box-shadow: 0 12px 32px rgba(37,99,235,0.18); }
  .pg-topbar__actions { display: flex; justify-content: flex-end; }
  .pg-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid var(--stroke);
    background: color-mix(in srgb, var(--panel) 88%, transparent);
    font-weight: 700;
    font-size: 12px;
    color: var(--text);
    transition: border-color 160ms ease, background 160ms ease, transform 160ms ease, box-shadow 160ms ease;
  }
  .pg-chip.ghost { background: color-mix(in srgb, var(--panel) 75%, transparent); color: var(--muted); }
  .pg-chip:hover { border-color: var(--primary); transform: translateY(-1px); box-shadow: 0 10px 30px rgba(37,99,235,0.18); }
  .pg-subhead { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 6px; }
  @media (max-width: 900px) {
    .pg-topbar { grid-template-columns: 1fr; padding: 12px 14px; }
    .pg-topbar__nav { justify-content: flex-start; }
    .pg-topbar__actions { width: 100%; }
  }

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
    transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease, background 160ms ease;
  }
  .pg-kpi:hover { transform: translateY(-1px); border-color: var(--primary); box-shadow: 0 14px 38px rgba(37,99,235,0.16); background: color-mix(in srgb, var(--panel) 95%, transparent); }
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
    transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
  }
  .tourn-box:hover { border-color: var(--primary); transform: translateY(-1px); box-shadow: 0 16px 44px rgba(37,99,235,0.16); }
  .tourn-title { font-weight: 800; font-size: 1.05rem; letter-spacing: -0.01em; }

  .badge { padding: 6px 10px; border-radius: 999px; font-weight: 700; font-size: 12px; border: 1px solid var(--stroke); transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease, background 140ms ease; }
  .badge:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(37,99,235,0.12); border-color: var(--primary); background: color-mix(in srgb, var(--panel) 94%, transparent); }
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
  .pg-pill {
    border: 1px solid var(--stroke);
    border-radius: 12px;
    padding: 10px 12px;
    background: color-mix(in srgb, var(--panel) 88%, transparent);
    transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease, background 160ms ease;
  }
  .pg-pill:hover { transform: translateY(-1px); border-color: var(--primary); box-shadow: 0 12px 32px rgba(37,99,235,0.12); background: color-mix(in srgb, var(--panel) 95%, transparent); }
  .pg-pill .label { color: var(--muted); font-size: 12px; }
  .pg-pill .value { font-weight: 700; font-size: 1.05rem; }

    /* Data cards (tabelas) com visual do protótipo Tailwind */
  .pg-table-card {
    border: 1px solid var(--stroke);
    border-radius: 18px;
    padding: 8px;
    background: linear-gradient(120deg, color-mix(in srgb, var(--panel) 90%, transparent), color-mix(in srgb, var(--panel) 96%, transparent));
    box-shadow: var(--shadow);
    margin-bottom: 12px;
  }
  .pg-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    overflow: hidden;
    border-radius: 14px;
  }
  .pg-table thead tr {
    background: linear-gradient(120deg, color-mix(in srgb, var(--panel) 82%, transparent), color-mix(in srgb, var(--panel) 95%, transparent));
  }
  .pg-table thead th {
    padding: 12px 14px;
    font-weight: 800;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    border-bottom: 1px solid color-mix(in srgb, var(--stroke) 70%, transparent);
  }
  .pg-table tbody tr {
    transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease;
  }
  .pg-table tbody tr:nth-child(odd) { background: color-mix(in srgb, var(--panel) 96%, transparent); }
  .pg-table tbody tr:nth-child(even) { background: color-mix(in srgb, var(--panel) 92%, transparent); }
  .pg-table tbody tr:hover {
    background: color-mix(in srgb, var(--primary) 12%, var(--panel));
    transform: translateY(-1px);
    box-shadow: 0 10px 32px rgba(37,99,235,0.14);
  }
  .pg-table td {
    padding: 12px 14px;
    border-bottom: 1px solid color-mix(in srgb, var(--stroke) 72%, transparent);
    border-right: 1px solid color-mix(in srgb, var(--stroke) 60%, transparent);
    color: var(--text);
  }
  .pg-table td:first-child { font-weight: 700; color: var(--text); }
  .pg-table td:last-child { border-right: none; }
  .pg-table tbody tr:last-child td { border-bottom: none; }
  .pg-table-caption { margin-top: 8px; color: var(--muted); font-size: 13px; }

  /* Moldura para gráficos Altair */
  .pg-chart-card {
    border: 1px solid var(--stroke);
    border-radius: 18px;
    padding: 10px 12px;
    background: linear-gradient(140deg, color-mix(in srgb, var(--panel) 88%, transparent), color-mix(in srgb, var(--panel) 96%, transparent));
    box-shadow: var(--shadow);
    margin-bottom: 14px;
  }
  .pg-chart-card .vega-embed {
    background: color-mix(in srgb, var(--panel) 92%, transparent) !important;
    border: 1px solid color-mix(in srgb, var(--stroke) 80%, transparent);
    border-radius: 14px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 10px 28px rgba(0,0,0,0.08);
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
</style>
  <script>
    (function() {
      const theme = '$theme';
      const applyTheme = (mode) => {
        const isDark = mode === 'dark';
        const targets = [
          document.documentElement,
          document.body,
          document.querySelector('.stApp'),
          document.querySelector('main'),
          document.querySelector('section.main'),
          document.querySelector('div.block-container'),
        ];
        targets.forEach(el => {
          if (!el) return;
          el.setAttribute('data-pg-theme', mode);
          el.classList.toggle('pg-theme-dark', isDark);
          el.classList.toggle('pg-theme-light', !isDark);
          el.style.background = 'var(--bg)';
          el.style.color = 'var(--text)';
        });

        let meta = document.querySelector('meta[name="color-scheme"]');
        if (!meta) {
          meta = document.createElement('meta');
          meta.name = 'color-scheme';
          document.head.appendChild(meta);
        }
        meta.content = isDark ? 'dark light' : 'light dark';
      };

      // aplica imediatamente e revalida quando o container do Streamlit é recriado
      applyTheme(theme);
      let ticks = 0;
      const interval = setInterval(() => {
        applyTheme(theme);
        ticks += 1;
        if (ticks > 12) clearInterval(interval);
      }, 200);
      const observer = new MutationObserver(() => {
        const app = document.querySelector('.stApp');
        if (app && app.getAttribute('data-pg-theme') !== theme) {
          applyTheme(theme);
        }
      });
      observer.observe(document.documentElement, { childList: true, subtree: true });
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
            "grid_soft": "#13233a",
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
        "grid_soft": "#f1f5f9",
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
            "view": {
                "stroke": "transparent",
                "cornerRadius": 14,
                "fill": tokens["panel"],
                "strokeWidth": 0
            },
            "axis": {
                "domainColor": tokens["stroke"],
                "gridColor": tokens["grid_soft"],
                "labelColor": tokens["text"],
                "titleColor": tokens["text"],
                "gridOpacity": 0.8,
                "tickColor": tokens["stroke"],
                "labelPadding": 8,
            },
            "legend": {
                "labelColor": tokens["text"],
                "titleColor": tokens["text"],
                "orient": "top",
                "labelFontWeight": 600,
                "padding": 10,
                "cornerRadius": 10,
                "fillColor": tokens["panel"],
                "strokeColor": tokens["stroke"],
            },
            "title": {
                "color": tokens["text"],
                "fontSize": 16,
                "fontWeight": 800,
                "font": "Inter"
            },
            "range": {
                "category": tokens["palette"],
                "ordinal": tokens["palette"],
                "ramp": tokens["palette"],
            },
            "bar": {
                "cornerRadiusTopLeft": 10,
                "cornerRadiusTopRight": 10,
                "cornerRadiusBottomLeft": 4,
                "cornerRadiusBottomRight": 4,
                "opacity": 0.92,
                "stroke": tokens["stroke"],
                "strokeWidth": 0.6,
            },
            "line": {"strokeWidth": 3, "point": True},
            "area": {"opacity": 0.25},
            "text": {"color": tokens["text"], "fontWeight": 700},
            "point": {
                "filled": True,
                "fill": tokens["accent"],
                "stroke": tokens["stroke"],
                "size": 72
            },
        }
    }

    alt.themes.register(name, lambda theme=theme: theme)
    alt.themes.enable(name)
