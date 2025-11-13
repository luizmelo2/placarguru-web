"""Módulo para injeção de CSS customizado."""
import streamlit as st


def inject_custom_css():
    """Insere o CSS customizado na página."""
    st.markdown('''
<style>
html, body, .stApp { font-size: 16px; }
@media (max-width: 768px) {
  html, body, .stApp { font-size: 17px; }
  section[data-testid="stSidebar"] { display:none !important; }
}

/* Títulos menores */
h1 { font-size: 1.6rem; line-height: 1.2; margin-bottom: .25rem; }
h2 { font-size: 1.25rem; line-height: 1.25; margin: .5rem 0 .25rem 0; }
h3 { font-size: 1.10rem; line-height: 1.3; margin: .35rem 0 .15rem 0; }
@media (max-width: 768px) {
  h1 { font-size: 1.35rem; }
  h2 { font-size: 1.15rem; }
  h3 { font-size: 1.00rem; }
}

/* Containers e componentes */
.block-container { padding-top: 0.5rem !important; }
.card {
  border: 1px solid #1f2937; border-radius: 14px; padding: 12px;
  background: #0b0b0b; box-shadow: 0 1px 8px rgba(0,0,0,.2);
}
.badge { padding: 2px 8px; border-radius: 999px; font-weight: 600; }
.badge-ok { background:#14532d; color:#d1fae5; }
.badge-bad { background:#7f1d1d; color:#fee2e2; }
.badge-wait { background:#334155; color:#e2e8f0; }
.badge-finished { background: #1e3a8a; color: #dbeafe; } /* Royal Blue */
button, .stButton>button { border-radius: 12px; padding: 10px 14px; font-weight: 600; }
div[data-testid="stExpander"] summary { padding: 10px 12px; font-size: 1.05rem; font-weight: 700; }
.stDataFrame { overflow-x: auto; }

/* Tipografia/cores dentro dos cards */
.text-label { color:#9CA3AF; font-weight:600; }   /* rótulos (Prev., Placar, etc.) */
.text-muted { color:#9CA3AF; }
.text-strong { color:#E5E7EB; font-weight:700; }

/* ÚNICA cor destaque (verde) para previsão, placar, sugestões, probabilidades e odds */
.accent-green { color:#22C55E; font-weight:700; }
.text-odds { color: #9CA3AF; font-size: 0.9em; margin-left: 0.5rem; }

.value-soft { color:#E5E7EB; }
.info-line { margin-top:.25rem; }
.info-line > .sep { color:#6B7280; margin: 0 .35rem; }

/* Grid de detalhes */
.info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.1rem 0.8rem;
  margin-top: 0.25rem;
}
@media (max-width: 768px) {
  .info-grid {
    grid-template-columns: 1fr;
    gap: 0.2rem;
  }
}

/* Caixa do filtro (C2) */
.tourn-box {
  border: 1px solid #1f2937; border-radius: 12px; padding: 12px;
  background: #0c0c0c; margin-bottom: 8px;
}
.tourn-title { font-weight:800; font-size:1.05rem; }

/* Confiança ao lado da Previsão */
.conf-inline { margin-left: .4rem; color:#9CA3AF; font-weight:600; white-space:nowrap; }
</style>
''', unsafe_allow_html=True)
