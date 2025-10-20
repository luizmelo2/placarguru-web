import pandas as pd
import pytest
from app import generate_pdf_report
from fpdf.errors import FPDFUnicodeEncodingException

def test_pdf_generation_does_not_crash_with_unsupported_chars():
    """
    Verifica se a geração de PDF não falha mais com um FPDFUnicodeEncodingException
    quando os dados contêm caracteres não suportados, graças à sanitização.
    """
    data = {
        'date': [pd.Timestamp('2023-10-27 15:00:00')],
        'home': ['São Paulo α'], # 'α' não está em cp1252
        'away': ['Vitória'],
        'result_predicted': ['H'],
        'score_predicted': ['2-1'],
        'bet_suggestion': ['H'],
        'goal_bet_suggestion': ['over_2_5']
    }
    df = pd.DataFrame(data)

    # A função não deve levantar uma exceção.
    try:
        generate_pdf_report(df)
    except FPDFUnicodeEncodingException:
        pytest.fail("generate_pdf_report levantou uma FPDFUnicodeEncodingException inesperadamente.")

if __name__ == "__main__":
    pytest.main()
