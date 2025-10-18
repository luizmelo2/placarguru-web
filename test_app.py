import pytest
from datetime import date, timedelta
from app import get_date_range

@pytest.mark.parametrize("button_clicked, expected_start_delta, expected_end_delta", [
    ("Hoje", 0, 0),
    ("Próx. 3 dias", 0, 3),
    ("Últimos 3 dias", -3, 0),
])
def test_get_date_range(button_clicked, expected_start_delta, expected_end_delta):
    """
    Tests the get_date_range function with various inputs.
    """
    today = date(2023, 10, 26)
    expected_start_date = today + timedelta(days=expected_start_delta)
    expected_end_date = today + timedelta(days=expected_end_delta)

    start_date, end_date = get_date_range(button_clicked, today)

    assert start_date == expected_start_date
    assert end_date == expected_end_date

def test_get_date_range_semana():
    """
    Tests the get_date_range function for the "Semana" case.
    """
    today = date(2023, 10, 26) # A Thursday
    expected_start_date = date(2023, 10, 23) # Monday
    expected_end_date = date(2023, 10, 29) # Sunday

    start_date, end_date = get_date_range("Semana", today)

    assert start_date == expected_start_date
    assert end_date == expected_end_date

def test_get_date_range_invalid():
    """
    Tests the get_date_range function with an invalid input.
    """
    today = date(2023, 10, 26)
    assert get_date_range("Invalid", today) is None