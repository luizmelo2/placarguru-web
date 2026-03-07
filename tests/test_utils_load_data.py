from io import BytesIO

import pandas as pd

from utils import load_data


def test_load_data_limits_to_last_two_months_by_default():
    now = pd.Timestamp.now().normalize()
    old_date = now - pd.DateOffset(months=4)
    recent_date = now - pd.DateOffset(days=10)

    src = pd.DataFrame(
        {
            "date": [old_date, recent_date, pd.NaT],
            "home": ["A", "B", "C"],
            "away": ["D", "E", "F"],
            "prob_H": [0.5, 0.6, 0.7],
            "prob_D": [0.2, 0.2, 0.1],
        }
    )

    bio = BytesIO()
    src.to_excel(bio, index=False)
    loaded = load_data(bio.getvalue())

    # Linha antiga deve sair, recente e NaT devem permanecer
    assert len(loaded) == 2
    assert (loaded["home"] == "B").any()
    assert (loaded["home"] == "C").any()


def test_load_data_can_disable_date_cutoff():
    now = pd.Timestamp.now().normalize()
    old_date = now - pd.DateOffset(months=4)
    recent_date = now - pd.DateOffset(days=10)

    src = pd.DataFrame(
        {
            "date": [old_date, recent_date],
            "home": ["A", "B"],
            "away": ["D", "E"],
            "prob_H": [0.5, 0.6],
            "prob_D": [0.2, 0.2],
        }
    )

    bio = BytesIO()
    src.to_excel(bio, index=False)
    loaded = load_data(bio.getvalue(), months_back=0)

    assert len(loaded) == 2
