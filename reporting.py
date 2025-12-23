"""Módulo para geração de relatórios em PDF."""
import pandas as pd
from fpdf import FPDF, XPos, YPos
from datetime import datetime

from utils import market_label, fmt_score_pred_text

def _sanitize_for_pdf(text: str) -> str:
    """Codifica o texto para latin-1, substituindo caracteres não suportados."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def generate_pdf_report(df: pd.DataFrame):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)

    title = f"Relatório de Jogos - {datetime.now().strftime('%d/%m/%Y')}"
    pdf.cell(0, 10, _sanitize_for_pdf(title), border=0, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("helvetica", "B", 8)

    # Cabeçalho da tabela
    headers = ["Data", "Casa", "Visitante", "Prev.", "Placar", "Sugestão", "Gols"]
    col_widths = [22, 35, 35, 15, 20, 30, 30]
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 8, _sanitize_for_pdf(h), border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln()

    # Linhas
    pdf.set_font("helvetica", "", 8)
    for _, row in df.iterrows():
        data_txt = row["date"].strftime("%d/%m %H:%M") if pd.notna(row.get("date")) else ""
        pdf.cell(col_widths[0], 8, _sanitize_for_pdf(str(data_txt)[:16]), border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[1], 8, _sanitize_for_pdf(str(row.get("home", ""))[:18]), border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[2], 8, _sanitize_for_pdf(str(row.get("away", ""))[:18]), border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[3], 8, _sanitize_for_pdf(str(market_label(row.get("result_predicted")))[:10]), border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[4], 8, _sanitize_for_pdf(str(fmt_score_pred_text(row.get("score_predicted")))[:10]), border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[5], 8, _sanitize_for_pdf(str(market_label(row.get("bet_suggestion")))[:18]), border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[6], 8, _sanitize_for_pdf(str(market_label(row.get("goal_bet_suggestion")))[:18]), border=1, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    out = pdf.output()
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    else:
        return out.encode("latin-1", "ignore")
