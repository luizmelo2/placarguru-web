
"""Módulo para injeção de CSS customizado e temas de gráficos."""
import altair as alt
import streamlit as st
from string import Template

def inject_custom_css(dark_mode: bool = False):
    """Insere o CSS customizado na página."""
    theme = "dark" if dark_mode else "light"
    css_template = Template("""
<style>
  :root {
    --bg: #${'0b1224' if dark_mode else 'f8fafc'};
    --panel: #${'0f172a' if dark_mode else 'ffffff'};
    --stroke: #${'1f2937' if dark_mode else 'e2e8f0'};
    --text: #${'e2e8f0' if dark_mode else '0f172a'};
    --muted: #${'94a3b8' if dark_mode else '475569'};
    --primary: #${'60a5fa' if dark_mode else '2563eb'};
    --neon: #${'9bdd31' if dark_mode else 'a3e635'};
  }
  html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
  }
</style>
  <script>
    document.documentElement.setAttribute('data-pg-theme', '$theme');
  </script>
""")
    st.markdown(css_template.substitute(theme=theme), unsafe_allow_html=True)

def chart_tokens(dark_mode: bool):
    """Retorna tokens cromáticos para os gráficos Altair."""
    if dark_mode:
        return {
            "background": "#0b1224", "panel": "#0f172a", "stroke": "#1f2937",
            "text": "#e2e8f0", "grid": "#1f2a3c", "accent": "#60a5fa",
            "palette": ["#60a5fa", "#22d3ee", "#bfff3b", "#a855f7", "#f97316"],
        }
    return {
        "background": "#f8fafc", "panel": "#ffffff", "stroke": "#e2e8f0",
        "text": "#0f172a", "grid": "#e2e8f0", "accent": "#2563eb",
        "palette": ["#2563eb", "#22d3ee", "#bfff3b", "#a855f7", "#f97316"],
    }

def apply_altair_theme(dark_mode: bool = False):
    """Configura tema do Altair."""
    tokens = chart_tokens(dark_mode)
    theme_name = "pg-dark" if dark_mode else "pg-light"

    theme_config = {
        "background": tokens["background"],
        "view": {"stroke": "transparent", "fill": tokens["panel"]},
        "axis": {
            "domainColor": tokens["stroke"], "gridColor": tokens["grid"],
            "labelColor": tokens["text"], "titleColor": tokens["text"],
        },
        "legend": {"labelColor": tokens["text"], "titleColor": tokens["text"]},
        "title": {"color": tokens["text"]},
        "range": {"category": tokens["palette"]},
    }

    alt.themes.register(theme_name, lambda: {"config": theme_config})
    alt.themes.enable(theme_name)
