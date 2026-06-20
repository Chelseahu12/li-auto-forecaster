import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go
from src.viz.plots import fan_chart


def test_fan_chart_returns_figure():
    preds = np.array([[4000, 7000, 10000]] * 9)  # (9, 3)
    actuals = pd.DataFrame({"month_since_launch": range(4, 13),
                            "sales": [6000] * 9})
    fig = fan_chart(preds, actuals, model_name="L6", quantiles=[0.1, 0.5, 0.9])
    assert isinstance(fig, go.Figure)
