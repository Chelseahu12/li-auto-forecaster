import numpy as np
import pytest
from src.model.bass import BassBaseline


def test_fit_produces_parameters():
    model = BassBaseline()
    early_sales = {1: 4000, 2: 7000, 3: 9500}
    model.fit(early_sales)
    assert hasattr(model, "p_")
    assert hasattr(model, "q_")
    assert hasattr(model, "M_")
    assert model.M_ > 0


def test_predict_returns_9_months():
    model = BassBaseline()
    early_sales = {1: 4000, 2: 7000, 3: 9500}
    model.fit(early_sales)
    preds = model.predict(months=list(range(4, 13)))
    assert len(preds) == 9


def test_predict_positive_values():
    model = BassBaseline()
    early_sales = {1: 4000, 2: 7000, 3: 9500}
    model.fit(early_sales)
    preds = model.predict(months=list(range(4, 13)))
    assert all(p >= 0 for p in preds)
