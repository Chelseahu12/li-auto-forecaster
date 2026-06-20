import numpy as np
import pandas as pd
import plotly.graph_objects as go


def fan_chart(
    predictions: np.ndarray,
    actuals: pd.DataFrame,
    model_name: str,
    quantiles: list[float],
) -> go.Figure:
    """Trajectory fan chart showing median + prediction interval.

    Args:
        predictions: shape (n_months, n_quantiles) for a single model
        actuals: DataFrame with columns [month_since_launch, sales]
        model_name: display name for the title
        quantiles: list of quantile values matching predictions axis-1
    """
    months = list(range(4, 4 + len(predictions)))
    q_low_idx = 0
    q_mid_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    q_high_idx = len(quantiles) - 1

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=months + months[::-1],
        y=predictions[:, q_high_idx].tolist() + predictions[:, q_low_idx].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(0, 100, 200, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"{int(min(quantiles)*100)}–{int(max(quantiles)*100)}% interval",
    ))

    fig.add_trace(go.Scatter(
        x=months,
        y=predictions[:, q_mid_idx].tolist(),
        mode="lines",
        line=dict(color="royalblue", width=2),
        name="Median forecast",
    ))

    if not actuals.empty:
        fig.add_trace(go.Scatter(
            x=actuals["month_since_launch"].tolist(),
            y=actuals["sales"].tolist(),
            mode="markers+lines",
            marker=dict(color="crimson", size=8),
            line=dict(color="crimson", dash="dot"),
            name="Actual sales",
        ))

    fig.update_layout(
        title=f"{model_name} — Sales Forecast (months 4–12)",
        xaxis_title="Month since launch",
        yaxis_title="Monthly sales (units)",
        template="plotly_white",
    )
    return fig
