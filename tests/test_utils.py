
"""Testes para o módulo de utilitários."""
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import generate_sofascore_link

@pytest.mark.parametrize("home, away, expected", [
    ("Palmeiras", "Corinthians", "https://www.google.com/search?q=site%3Asofascore.com+%22Palmeiras%22+vs+%22Corinthians%22"),
    ("", "", ""),
    ("Palmeiras", "", ""),
    ("", "Corinthians", ""),
    ("São Paulo", "Atlético-MG", "https://www.google.com/search?q=site%3Asofascore.com+%22S%C3%A3o+Paulo%22+vs+%22Atl%C3%A9tico-MG%22"),
])
def test_generate_sofascore_link(home, away, expected):
    """Testa a geração de links do Sofascore."""
    assert generate_sofascore_link(home, away) == expected
