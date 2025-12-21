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
    --mobile-breakpoint: 1024px;
    --bg: #f8fafc;
    --panel: #ffffff;
    --glass: rgba(255,255,255,0.65);
    --glass-strong: rgba(255,255,255,0.82);
    --stroke: #e2e8f0;
    --text: #0f172a;
    --muted: #475569;
    --primary: #2563eb;
    --primary-2: #22d3ee;
    --neon: #a3e635;
    --positive: #10b981;
    --warning: #f59e0b;
    --info: #0ea5e9;
    --shadow: 0 16px 44px rgba(0,0,0,0.10);
    --shadow-strong: 0 20px 60px rgba(0,0,0,0.12);
    --font-xs: 12px;
    --font-sm: 13px;
    --font-md: 14px;
    --radius-sm: 10px;
    --radius-md: 14px;
    --radius-lg: 18px;
    --blur-bg: 14px;
    --shadow-card: 0 12px 32px rgba(37,99,235,0.12);
    --focus-ring: 0 0 0 3px color-mix(in srgb, var(--primary) 24%, transparent);
  }

  [data-pg-theme="dark"],
  .stApp[data-pg-theme="dark"],
  .pg-theme-dark {
    --mobile-breakpoint: 1024px;
    --bg: #0b1224;
    --panel: #0f172a;
    --glass: rgba(255,255,255,0.04);
    --glass-strong: rgba(255,255,255,0.08);
    --stroke: #1f2937;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --primary: #60a5fa;
    --primary-2: #22d3ee;
    --neon: #9bdd31;
    --positive: #34d399;
    --warning: #fbbf24;
    --info: #38bdf8;
    --shadow: 0 16px 44px rgba(0,0,0,0.32);
    --shadow-strong: 0 24px 70px rgba(0,0,0,0.36);
    --font-xs: 12px;
    --font-sm: 13px;
    --font-md: 14px;
    --radius-sm: 10px;
    --radius-md: 14px;
    --radius-lg: 18px;
    --blur-bg: 14px;
    --shadow-card: 0 16px 40px rgba(96,165,250,0.18);
    --focus-ring: 0 0 0 3px color-mix(in srgb, var(--primary) 36%, transparent);
  }

  html, body, .stApp {
    font-size: 16px;
    background: linear-gradient(120deg, color-mix(in srgb, var(--bg) 92%, transparent), color-mix(in srgb, var(--panel) 8%, transparent)),
                var(--bg) !important;
    color: var(--text) !important;
    transition: background 260ms ease, color 260ms ease;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
  }
  @media (max-width: 768px) {
    html, body, .stApp { font-size: clamp(15px, 2.8vw, 17px); }
  }

  .block-container { padding-top: 0.5rem !important; max-width: 1200px; }
  .stMain, .block-container, .main { background: transparent !important; }
  h1 { font-size: clamp(1.25rem, 2.9vw, 1.55rem); font-weight: 700; letter-spacing: -0.01em; }
  h2 { font-size: clamp(1.15rem, 2.5vw, 1.35rem); font-weight: 700; }
  h3 { font-size: clamp(1.05rem, 2.2vw, 1.2rem); font-weight: 700; }

  .pg-header {
    position: sticky;
    top: 0;
    z-index: 50;
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 14px;
    align-items: center;
    padding: 12px 16px;
    margin: 0 -1rem 10px -1rem;
    background: color-mix(in srgb, var(--panel) 94%, transparent);
    backdrop-filter: blur(var(--blur-bg));
    border: 1px solid var(--stroke);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-card);
  }
  .pg-header > * { position: relative; z-index: 1; }
  .pg-header__brand { display: flex; align-items: center; gap: 12px; min-width: 220px; }
  .pg-logo {
    width: 48px; height: 48px;
    border-radius: var(--radius-md);
    background: linear-gradient(135deg, color-mix(in srgb, var(--primary) 40%, #22d3ee), color-mix(in srgb, var(--neon) 48%, var(--primary-2)));
    box-shadow: 0 12px 30px rgba(34, 211, 238, 0.3);
    border: 1px solid color-mix(in srgb, var(--stroke) 70%, transparent);
    display: grid;
    place-items: center;
    position: relative;
    overflow: hidden;
  }
  .pg-logo::after {
    content: "";
    position: absolute;
    inset: -30% 30% auto auto;
    width: 70%; height: 70%;
    background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.28), transparent 55%);
    transform: rotate(-18deg);
    opacity: 0.65;
  }
  .pg-logo svg { width: 34px; height: 34px; position: relative; z-index: 1; }
  .pg-logo .pg-logo-shield { fill: rgba(12, 20, 38, 0.16); stroke: rgba(255, 255, 255, 0.65); stroke-width: 1.2; }
  .pg-logo .pg-logo-ball { fill: #f8fafc; stroke: rgba(15, 23, 42, 0.35); stroke-width: 1.1; }
  .pg-logo .pg-logo-chart { fill: rgba(12, 20, 38, 0.22); stroke: #0b1224; stroke-opacity: 0.22; }
  .pg-logo .pg-logo-glow { fill: rgba(163, 230, 53, 0.88); filter: drop-shadow(0 4px 10px rgba(163, 230, 53, 0.28)); }
  .pg-eyebrow { margin: 0; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
  .pg-appname { font-size: clamp(1.05rem, 2.6vw, 1.28rem); font-weight: 800; letter-spacing: -0.01em; }
  .pg-header__status { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
  .pg-header__actions { display: flex; gap: 8px; align-items: center; justify-content: flex-end; flex-wrap: wrap; }
  .pg-tab {
    display: inline-flex;
    align-items: center;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid color-mix(in srgb, var(--stroke) 70%, var(--primary));
    background: color-mix(in srgb, var(--panel) 92%, var(--glass-strong));
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
  .pg-tab:focus-visible { outline: none; box-shadow: var(--focus-ring); }
  .pg-topbar__actions { display: flex; justify-content: flex-end; }
  .pg-breadcrumbs { display: flex; gap: 6px; align-items: center; color: var(--muted); font-size: 12px; }
  .pg-breadcrumbs span:last-child { color: var(--text); font-weight: 700; }
  .pg-header__summary { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
  .pg-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid var(--stroke);
    background: color-mix(in srgb, var(--panel) 88%, transparent);
    font-weight: 700;
    font-size: clamp(12px, 1.8vw, 13px);
    color: var(--text);
    transition: border-color 160ms ease, background 160ms ease, transform 160ms ease, box-shadow 160ms ease;
  }
  .pg-chip.ghost { background: color-mix(in srgb, var(--panel) 82%, var(--glass-strong)); color: color-mix(in srgb, var(--text) 92%, var(--muted)); border-color: color-mix(in srgb, var(--stroke) 82%, var(--primary)); }
  .pg-chip:hover { border-color: var(--primary); transform: translateY(-1px); box-shadow: 0 10px 30px rgba(37,99,235,0.14); }
  .pg-chip:focus-visible { outline: none; box-shadow: var(--focus-ring); }
  .pg-header__meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
  .pg-header__secondary { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
  .pg-subhead { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 6px; }
  .pg-hero-breadcrumb { color: var(--muted); font-size: 13px; }
  .pg-sr { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
  @media (max-width: 1024px) {
    .pg-header { grid-template-columns: 1fr; padding: 12px 14px; }
    .pg-header__status { justify-content: flex-start; }
    .pg-header__actions { width: 100%; justify-content: flex-start; }
  }

  .pg-table-card [data-testid="stDataFrame"] *:focus-visible { box-shadow: var(--focus-ring) !important; outline: none !important; }
  .pg-table-card table th:first-child, .pg-table-card table td:first-child { position: sticky; left: 0; z-index: 3; background: color-mix(in srgb, var(--panel) 96%, transparent); }
  @media (max-width: 1024px) {
    .pg-table-card table th:first-child, .pg-table-card table td:first-child { position: static; }
  }

  /* Filtros principais */
  .pg-filter-shell {
    background: linear-gradient(135deg, color-mix(in srgb, var(--panel) 96%, transparent), color-mix(in srgb, var(--panel) 88%, var(--glass-weak)));
    border: 1px solid color-mix(in srgb, var(--stroke) 85%, var(--primary) 8%);
    border-radius: 16px;
    box-shadow: 0 12px 32px rgba(37,99,235,0.08), inset 0 1px 0 rgba(255,255,255,0.05);
    padding: 16px 18px 8px 18px;
    margin: 6px 0 14px 0;
    backdrop-filter: blur(14px);
  }
  .pg-filter-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 10px;
  }
  .pg-filter-title { margin: 0; }
  .pg-filter-sub { margin: 2px 0 0; color: var(--muted); font-size: 13px; }
  .pg-filter-actions { display: flex; gap: 6px; align-items: center; }
  .pg-filter-toggle-label { color: var(--muted); font-size: 12px; margin-bottom: 2px; text-align: right; }
  .pg-filter-shell .streamlit-expanderHeader,
  .pg-filter-shell label,
  .pg-filter-shell p { color: var(--text) !important; }
  .pg-filter-shell .stMultiSelect, .pg-filter-shell .stTextInput, .pg-filter-shell .stDateInput {
    margin-bottom: 8px;
  }
  .pg-filter-shell [data-baseweb="slider"] {
    margin-top: 6px;
  }
  .pg-filter-section { padding: 10px 12px 4px; border-radius: 12px; border: 1px dashed color-mix(in srgb, var(--stroke) 90%, var(--primary) 10%); margin-bottom: 8px; background: color-mix(in srgb, var(--panel) 92%, transparent); }
  .pg-filter-section__head { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
  .pg-filter-section__title { margin: 0; font-size: clamp(15px, 2.3vw, 16px); }
  .pg-filter-section__hint { margin: 2px 0 0; color: var(--muted); font-size: 13px; }
  .pg-filter-chip { font-weight: 700; font-size: 12px; padding: 6px 10px; border-radius: 10px; border: 1px solid color-mix(in srgb, var(--stroke) 78%, var(--primary) 22%); background: color-mix(in srgb, var(--panel) 86%, transparent); }
  .pg-filter-section--models { border-style: solid; border-color: color-mix(in srgb, var(--stroke) 72%, var(--primary) 18%); background: linear-gradient(135deg, color-mix(in srgb, var(--panel) 94%, transparent), color-mix(in srgb, var(--primary) 8%, var(--panel))); box-shadow: inset 0 1px 0 rgba(255,255,255,0.05); }
  .pg-filter-section--teams { border-style: dashed; border-color: color-mix(in srgb, var(--stroke) 82%, var(--accent) 22%); background: linear-gradient(135deg, color-mix(in srgb, var(--panel) 94%, transparent), color-mix(in srgb, var(--accent) 8%, var(--panel))); }
  .pg-filter-section--teams .pg-filter-chip { border-color: color-mix(in srgb, var(--stroke) 74%, var(--accent) 24%); }
  .pg-filter-section--period { border-style: solid; border-color: color-mix(in srgb, var(--stroke) 78%, var(--primary) 18%); background: linear-gradient(135deg, color-mix(in srgb, var(--panel) 96%, transparent), color-mix(in srgb, var(--primary) 8%, var(--panel))); box-shadow: inset 0 1px 0 rgba(255,255,255,0.04); }
  .pg-filter-section--period .pg-filter-chip { border-color: color-mix(in srgb, var(--stroke) 70%, var(--primary) 30%); }
  .pg-filter-section--suggestions { border-style: solid; border-color: color-mix(in srgb, var(--stroke) 76%, var(--highlight) 18%); background: linear-gradient(135deg, color-mix(in srgb, var(--panel) 94%, transparent), color-mix(in srgb, var(--highlight) 10%, var(--panel))); box-shadow: inset 0 1px 0 rgba(255,255,255,0.04); }
  .pg-filter-section--suggestions .pg-filter-chip { border-color: color-mix(in srgb, var(--stroke) 74%, var(--highlight) 22%); background: color-mix(in srgb, var(--panel) 84%, transparent); color: var(--text); }

  .pg-mobile-toolbar {
    border: 1px dashed var(--stroke);
    border-radius: var(--radius-md);
    padding: 8px 12px;
    background: color-mix(in srgb, var(--panel) 92%, transparent);
    margin: 10px 0 4px;
  }
  .pg-mobile-toolbar__title { font-weight: 700; font-size: var(--font-md); margin-bottom: 2px; }
  .pg-mobile-toolbar__hint { margin: 0; color: var(--muted); font-size: var(--font-sm); }

  /* Destaque para ocultar lista de jogos */
  .pg-hide-card {
    background: linear-gradient(135deg, color-mix(in srgb, var(--panel) 96%, transparent), color-mix(in srgb, var(--panel) 88%, transparent));
    border: 1px solid var(--stroke);
    border-radius: 16px;
    box-shadow: var(--shadow);
    padding: 12px 14px;
    margin: 6px 0 14px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }
  .pg-hide-copy { display: flex; flex-direction: column; gap: 6px; }
  .pg-hide-title { font-weight: 700; font-size: 1.05rem; color: var(--text); }
  .pg-hide-desc { color: var(--muted); margin: 0; }
  .pg-hide-chips { display: flex; gap: 8px; flex-wrap: wrap; }
  .pg-hide-card [data-testid="stToggle"] { justify-content: flex-end; }
  .pg-hide-card [data-testid="stWidgetLabel"] p { color: var(--text) !important; font-weight: 700; }
  @media (max-width: 980px) {
    .pg-hide-card { flex-direction: column; align-items: flex-start; }
    .pg-hide-card [data-testid="stToggle"] { width: 100%; }
  }

  /* Sessão de estatísticas premium */
  .pg-stats-stack { display: flex; flex-direction: column; gap: 12px; margin-top: 8px; }
  .pg-stats-section {
    border: 1px solid var(--stroke);
    border-radius: 18px;
    padding: 16px;
    background: linear-gradient(135deg, color-mix(in srgb, var(--panel) 94%, transparent), color-mix(in srgb, var(--panel) 88%, transparent));
    box-shadow: var(--shadow);
  }
  .pg-stats-header { display: flex; justify-content: space-between; gap: 12px; align-items: center; flex-wrap: wrap; }
  .pg-stats-desc { color: var(--muted); margin: 4px 0 0 0; }
  .pg-stats-tags { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .pg-stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-top: 10px; }
  .pg-stat-card {
    border: 1px solid color-mix(in srgb, var(--stroke) 82%, transparent);
    border-radius: 14px;
    padding: 12px;
    background: radial-gradient(circle at 20% 20%, color-mix(in srgb, var(--primary) 18%, transparent), transparent 40%),
                linear-gradient(160deg, color-mix(in srgb, var(--panel) 96%, transparent), color-mix(in srgb, var(--panel) 88%, transparent));
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 14px 30px rgba(0,0,0,0.08);
    transition: transform 160ms ease, box-shadow 180ms ease, border-color 160ms ease;
  }
  .pg-stat-card:hover { transform: translateY(-2px); border-color: var(--primary); box-shadow: 0 16px 36px rgba(37,99,235,0.22); }
  .pg-stat-label { color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-size: 12px; margin: 0 0 4px 0; }
  .pg-stat-value { font-weight: 800; font-size: 1.55rem; margin: 0; color: var(--text); }
  .pg-stat-foot { margin: 4px 0 0 0; color: var(--muted); font-size: 13px; }
  .pg-stats-panel {
    border: 1px solid var(--stroke);
    border-radius: 16px;
    padding: 14px;
    background: linear-gradient(135deg, color-mix(in srgb, var(--panel) 92%, transparent), color-mix(in srgb, var(--panel) 84%, transparent));
    box-shadow: var(--shadow);
  }

  .pg-hero {
    border: 1px solid var(--stroke);
    border-radius: 18px;
    padding: 14px;
    background: linear-gradient(135deg, color-mix(in srgb, var(--panel) 94%, transparent), color-mix(in srgb, var(--panel) 86%, transparent));
    box-shadow: var(--shadow);
    display: grid;
    gap: 8px;
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
  .pg-kpi .delta { color: var(--positive); font-size: 12px; font-weight: 600; }

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

  .badge {
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 12px;
    border: 1px solid var(--stroke);
    transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease, background 140ms ease;
    color: color-mix(in srgb, var(--text) 92%, #0b1224);
  }
  .badge:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(37,99,235,0.12); border-color: var(--primary); background: color-mix(in srgb, var(--panel) 94%, transparent); }
  .badge-ok { background: color-mix(in srgb, var(--positive) 22%, var(--panel)); color: color-mix(in srgb, #065f46 82%, var(--text)); }
  .badge-bad { background: color-mix(in srgb, #ef4444 22%, var(--panel)); color: color-mix(in srgb, #7f1d1d 82%, var(--text)); }
  .badge-wait { background: color-mix(in srgb, var(--stroke) 32%, var(--panel)); color: color-mix(in srgb, var(--text) 92%, #0b1224); }
  .badge-finished { background: color-mix(in srgb, var(--primary) 24%, var(--panel)); color: color-mix(in srgb, var(--text) 94%, #0b1224); }

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
  .pg-matchup { display:flex; align-items:center; gap:8px; flex-wrap:wrap; font-weight:800; font-size:1.05rem; }
  .pg-team { display:inline-flex; align-items:center; gap:8px; padding:6px 8px; border-radius:12px; border:1px solid color-mix(in srgb, var(--stroke) 80%, transparent); background: color-mix(in srgb, var(--panel) 90%, var(--glass-strong)); box-shadow: inset 0 1px 0 rgba(255,255,255,0.04); }
  .pg-team__logo { width:28px; height:28px; object-fit: contain; border-radius:10px; background: color-mix(in srgb, var(--panel) 80%, var(--glass-strong)); border: 1px solid color-mix(in srgb, var(--stroke) 80%, transparent); padding: 2px; }
  .pg-team__name { font-weight:800; letter-spacing:-0.01em; }
  .pg-vs { color: var(--muted); font-weight:700; }
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

  /* Detalhes dentro do card (substitui o expander solto) */
  .pg-details {
    margin-top: 12px;
    border: 1px solid var(--stroke);
    border-radius: 14px;
    background: linear-gradient(120deg, color-mix(in srgb, var(--panel) 92%, transparent), color-mix(in srgb, var(--panel) 98%, transparent));
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    overflow: hidden;
    transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
  }
  .pg-details[open] { box-shadow: 0 10px 36px rgba(37,99,235,0.14); transform: translateY(-1px); }
  .pg-details summary {
    cursor: pointer;
    list-style: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 12px 14px;
    font-weight: 700;
    color: var(--text-strong);
  }
  .pg-details summary::-webkit-details-marker { display: none; }
  .pg-details summary:after {
    content: "⇵";
    font-size: 12px;
    color: var(--muted);
    transition: transform 160ms ease;
  }
  .pg-details[open] summary:after { transform: rotate(180deg); }
  .pg-details-title { font-weight: 800; letter-spacing: -0.01em; }
  .pg-details-hint { color: var(--muted); font-weight: 600; font-size: 12px; }
  .pg-details-body { padding: 0 14px 14px; display: grid; gap: 12px; }
  .pg-details-block {
    border: 1px solid color-mix(in srgb, var(--stroke) 70%, transparent);
    border-radius: 12px;
    padding: 10px 12px;
    background: color-mix(in srgb, var(--panel) 90%, transparent);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
  }
  .pg-details-subtitle { font-weight: 800; color: var(--text-strong); margin-bottom: 6px; font-size: 0.95rem; }
  .pg-details-list { margin: 0; padding-left: 16px; color: var(--text); display: grid; gap: 4px; }
  .pg-details-two-cols { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }

  /* Accordions para os gráficos diários (similar ao card de jogo) */
  .pg-chart-accordion { margin-top: 8px; border: 1px solid color-mix(in srgb, var(--stroke) 72%, transparent); }
  .pg-chart-accordion summary { gap: 10px; }
  .pg-chart-accordion summary > div:first-child { display: grid; gap: 4px; }
  .pg-chart-accordion summary .pg-stats-tags { gap: 6px; }
  .pg-chart-accordion .pg-details-body { padding-top: 6px; }
  .pg-chart-accordion .pg-chart-grid { margin-top: 4px; }

    /* Data cards (tabelas) com visual do protótipo Tailwind */
  .pg-table-card {
    border: 1px solid var(--stroke);
    border-radius: 18px;
    padding: 8px;
    background: linear-gradient(120deg, color-mix(in srgb, var(--panel) 90%, transparent), color-mix(in srgb, var(--panel) 96%, transparent));
    box-shadow: var(--shadow);
    margin-bottom: 12px;
  }
  .pg-table-card--interactive [data-testid="stDataFrame"] {
    background: linear-gradient(120deg, color-mix(in srgb, var(--panel) 90%, transparent), color-mix(in srgb, var(--panel) 96%, transparent));
    border: 1px solid color-mix(in srgb, var(--stroke) 80%, transparent);
    border-radius: 14px;
    box-shadow: var(--shadow);
    overflow: hidden;
  }
  .pg-table-card--interactive [data-testid="stDataFrame"] > div {
    border-radius: 14px;
  }
  .pg-table-card--interactive table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.95rem;
  }
  .pg-table-card--interactive thead tr {
    background: linear-gradient(120deg, color-mix(in srgb, var(--panel) 82%, transparent), color-mix(in srgb, var(--panel) 95%, transparent));
  }
  .pg-table-card--interactive thead th {
    padding: 12px 14px;
    font-weight: 800;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    border-bottom: 1px solid color-mix(in srgb, var(--stroke) 70%, transparent);
    background: transparent;
  }
  .pg-table-card--interactive tbody tr:nth-child(odd) { background: color-mix(in srgb, var(--panel) 96%, transparent); }
  .pg-table-card--interactive tbody tr:nth-child(even) { background: color-mix(in srgb, var(--panel) 92%, transparent); }
  .pg-table-card--interactive tbody tr:hover {
    background: color-mix(in srgb, var(--primary) 12%, var(--panel));
    box-shadow: 0 10px 32px rgba(37,99,235,0.14);
  }
  @media (max-width: 1024px) {
    .pg-table-card--interactive tbody tr:nth-child(odd),
    .pg-table-card--interactive tbody tr:nth-child(even) { background: color-mix(in srgb, var(--panel) 94%, transparent); }
    .pg-table-card--interactive tbody tr:hover { box-shadow: none; }
  }
  .pg-table-card--interactive td {
    padding: 12px 14px;
    border-bottom: 1px solid color-mix(in srgb, var(--stroke) 72%, transparent);
    border-right: 1px solid color-mix(in srgb, var(--stroke) 60%, transparent);
    color: var(--text);
  }
  .pg-table-card--interactive td:first-child { font-weight: 700; color: var(--text); }
  .pg-table-card--interactive td:last-child { border-right: none; }
  .pg-table-card--interactive tbody tr:last-child td { border-bottom: none; }
  .pg-density-compact .pg-table-card--interactive table td,
  .pg-density-compact .pg-table-card--interactive table th { font-size: 12px !important; }
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
    position: relative;
    overflow: hidden;
    transition: transform 160ms ease, box-shadow 180ms ease;
  }
  .pg-chart-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 12% 16%, color-mix(in srgb, var(--primary) 18%, transparent), transparent 32%);
    pointer-events: none;
    opacity: 0.8;
  }
  .pg-chart-card:hover { box-shadow: 0 16px 38px rgba(37,99,235,0.22); transform: translateY(-2px); transition: all 160ms ease; }
  .pg-chart-card > * { position: relative; z-index: 1; }
  .pg-chart-card .vega-embed {
    background: color-mix(in srgb, var(--panel) 92%, transparent) !important;
    border: 1px solid color-mix(in srgb, var(--stroke) 80%, transparent);
    border-radius: 14px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 10px 28px rgba(0,0,0,0.08);
    padding: 8px;
  }
  .pg-chart-slot {
    width: 100%;
    height: 100%;
    min-height: 220px;
    border: 1px solid color-mix(in srgb, var(--stroke) 80%, transparent);
    border-radius: 14px;
    background: color-mix(in srgb, var(--panel) 92%, transparent);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 10px 28px rgba(0,0,0,0.08);
    overflow: hidden;
  }
  .pg-chart-slot .vega-embed {
    background: color-mix(in srgb, var(--panel) 95%, transparent) !important;
    border: none;
    box-shadow: none;
    height: 100%;
  }
  .pg-chart-slot .vega-actions { display: none; }
  .pg-chart-card.nested { margin-top: 6px; }
  .pg-chart-card__title { font-weight: 800; font-size: 15px; margin: 4px 0 10px; color: var(--text); }
  .pg-chart-grid { display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 8px; }
  .pg-chart-cluster { border: 1px solid color-mix(in srgb, var(--stroke) 75%, transparent); border-radius: 16px; padding: 12px; background: linear-gradient(140deg, color-mix(in srgb, var(--panel) 92%, transparent), color-mix(in srgb, var(--panel) 84%, transparent)); box-shadow: var(--shadow); }
  .pg-chart-cluster__head { display:flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 8px; }
  .pg-chart-cluster__head h4 { margin: 2px 0 0; }
  @media (min-width: 1000px) { .pg-chart-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }


  .info-grid { display:grid; grid-template-columns: 1fr 1fr; gap: 0.1rem 0.8rem; margin-top: 0.25rem; }
  @media (max-width: 768px) { .info-grid { grid-template-columns: 1fr; gap: 0.25rem; } }

  button, .stButton>button {
    border-radius: 10px;
    padding: 8px 12px;
    font-weight: 700;
    font-size: var(--font-sm);
    line-height: 1.3;
    background: linear-gradient(120deg, var(--primary), var(--primary-2));
    color: white;
    border: 1px solid color-mix(in srgb, var(--primary) 70%, var(--stroke));
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.14), 0 8px 22px rgba(37,99,235,0.22);
  }
  button:hover, .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 10px 28px rgba(37,99,235,0.28); }

  div[data-testid="stExpander"] summary { padding: 10px 12px; font-size: 1.02rem; font-weight: 700; }

  .pg-sofascore-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: color-mix(in srgb, var(--panel) 90%, var(--glass-strong));
    border: 1px solid var(--stroke);
    transition: all 160ms ease;
  }
  .pg-sofascore-link:hover {
    transform: translateY(-1px) scale(1.05);
    box-shadow: 0 8px 24px rgba(37,99,235,0.12);
    border-color: var(--primary);
  }
  .pg-sofascore-link svg {
    width: 18px;
    height: 18px;
    fill: var(--muted);
    transition: fill 160ms ease;
  }
  .pg-sofascore-link:hover svg {
    fill: var(--primary);
  }
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
