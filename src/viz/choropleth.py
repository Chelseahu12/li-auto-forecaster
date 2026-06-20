import pandas as pd
import plotly.express as px


def build_choropleth_data(regional_df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Filter to model and return province × month sales."""
    return (
        regional_df[regional_df["model"] == model]
        [["province", "month_since_launch", "sales"]]
        .reset_index(drop=True)
    )


def regional_ranking_table(
    regional_df: pd.DataFrame,
    model: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Top N provinces by total sales for a given model."""
    df = regional_df[regional_df["model"] == model]
    return (
        df.groupby("province")["sales"]
        .sum()
        .reset_index()
        .rename(columns={"sales": "total_sales"})
        .sort_values("total_sales", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def plot_choropleth(regional_df: pd.DataFrame, model: str):
    """Animated province-level choropleth."""
    data = build_choropleth_data(regional_df, model)
    fig = px.choropleth(
        data,
        geojson="https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json",
        locations="province",
        featureidkey="properties.name",
        color="sales",
        animation_frame="month_since_launch",
        title=f"{model} Monthly Sales by Province",
        color_continuous_scale="Blues",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    return fig
