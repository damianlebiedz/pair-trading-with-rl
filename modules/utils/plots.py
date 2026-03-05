import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.performance.models import StrategyResult


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_results_dir(directory: str | None) -> Path:
    if directory and (Path(directory).is_absolute() or "results" in str(directory)):
        path = Path(directory)
    else:
        path = get_project_root() / "results"
        if directory:
            path = path / directory

    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_custom_tickvals(index: pd.Index, target_ticks: int = 8):
    n = len(index)
    if n <= target_ticks:
        return index

    step = n // (target_ticks - 1)

    indices = list(range(0, n, step))

    if indices[-1] != n - 1:
        if n - 1 - indices[-1] < (step * 0.3):
            indices[-1] = n - 1
        else:
            indices.append(n - 1)

    return index[indices]


def plot_zscore_pos(
    result,
    directory: str | None = None,
    save: bool = False,
) -> None:
    df = result.data.dropna(subset=["equity"]).copy()

    for col in ["entry_thr", "exit_thr", "sl_thr", "z_score", "position"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    results_dir = Path(directory) if directory else Path(".")

    custom_ticks = _get_custom_tickvals(df.index)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"Z-Score: {result.ticker_x}/{result.ticker_y}", "Positions"),
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["z_score"],
            mode="lines",
            name="Z-Score",
            line=dict(color="black", width=1.5),
            hovertemplate="Z-Score</b>: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["entry_thr"],
            mode="lines",
            name="Entry Threshold",
            line=dict(color="darkred", width=1.5),
            legendgroup="entry",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=-df["entry_thr"],
            mode="lines",
            name="-Entry Threshold",
            line=dict(color="darkred", width=1.5),
            legendgroup="entry",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["exit_thr"],
            mode="lines",
            name="Exit Threshold",
            line=dict(color="green", width=1.5),
            legendgroup="exit",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=-df["exit_thr"],
            mode="lines",
            name="-Exit Threshold",
            line=dict(color="green", width=1.5),
            legendgroup="exit",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    if "sl_thr" in df.columns and not df["sl_thr"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sl_thr"],
                mode="lines",
                name="Stop Loss Threshold",
                line=dict(color="red", width=1.5),
                connectgaps=False,
                legendgroup="sl",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=-df["sl_thr"],
                mode="lines",
                name="-Stop Loss Threshold",
                line=dict(color="red", width=1.5),
                connectgaps=False,
                legendgroup="sl",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["position"],
            mode="lines",
            name="Position",
            line=dict(color="grey", width=1.5, shape="hv"),
            fill="tozeroy",
            opacity=0.5,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=750,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=100),
    )

    fig.for_each_annotation(lambda a: a.update(font=dict(color="black")))

    fig.update_yaxes(
        title=dict(text="Z-Score", font=dict(color="black")),
        tickfont=dict(color="black"),
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title=dict(text="Position", font=dict(color="black")),
        tickfont=dict(color="black"),
        tickvals=[-1, 0, 1],
        range=[-1.2, 1.2],
        row=2,
        col=1,
    )

    fig.update_xaxes(
        hoverformat="%Y-%m-%d %H:%M",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        showline=False,
        fixedrange=True,
    )

    fig.update_xaxes(
        title=dict(text="Date", font=dict(color="black")),
        tickfont=dict(color="black"),
        tickvals=custom_ticks,
        tickformat="%Y-%m-%d",
        row=2,
        col=1,
    )

    fig.update_yaxes(fixedrange=True)

    filename = f"zscore_pos_{result.ticker_x}_{result.ticker_y}_{result.start}_{result.end}.html".replace(
        ":", "_"
    )

    if save:
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(results_dir / filename)


def plot_spread_pos(
    result,
    directory: str | None = None,
    save: bool = False,
) -> None:
    df = result.data.dropna(subset=["equity"]).copy()

    for col in [
        "entry_thr",
        "exit_thr",
        "sl_thr",
        "z_score",
        "position",
        "spread",
        "mean",
        "std",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    results_dir = Path(directory) if directory else Path(".")

    custom_ticks = _get_custom_tickvals(df.index)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"Spread: {result.ticker_x}/{result.ticker_y}", "Positions"),
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["spread"],
            mode="lines",
            name="Spread",
            line=dict(color="black", width=1.5),
            hovertemplate="Spread</b>: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["mean"],
            mode="lines",
            name="Mean",
            line=dict(color="grey", width=1.5),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["mean"] + (df["entry_thr"] * df["std"]),
            mode="lines",
            name="Entry Threshold",
            line=dict(color="darkred", width=1.5),
            legendgroup="entry",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["mean"] - (df["entry_thr"] * df["std"]),
            mode="lines",
            name="-Entry Threshold",
            line=dict(color="darkred", width=1.5),
            legendgroup="entry",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["mean"] + (df["exit_thr"] * df["std"]),
            mode="lines",
            name="Exit Threshold",
            line=dict(color="green", width=1.5),
            legendgroup="exit",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["mean"] - (df["exit_thr"] * df["std"]),
            mode="lines",
            name="-Exit Threshold",
            line=dict(color="green", width=1.5),
            legendgroup="exit",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    if "sl_thr" in df.columns and not df["sl_thr"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["mean"] + (df["sl_thr"] * df["std"]),
                mode="lines",
                name="Stop Loss Threshold",
                line=dict(color="red", width=1.5),
                connectgaps=False,
                legendgroup="sl",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["mean"] - (df["sl_thr"] * df["std"]),
                mode="lines",
                name="-Stop Loss Threshold",
                line=dict(color="red", width=1.5),
                connectgaps=False,
                legendgroup="sl",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["position"],
            mode="lines",
            name="Position",
            line=dict(color="grey", width=1.5, shape="hv"),
            fill="tozeroy",
            opacity=0.5,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=750,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=100),
    )

    fig.for_each_annotation(lambda a: a.update(font=dict(color="black")))

    fig.update_yaxes(
        title=dict(text="Spread", font=dict(color="black")),
        tickfont=dict(color="black"),
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title=dict(text="Position", font=dict(color="black")),
        tickfont=dict(color="black"),
        tickvals=[-1, 0, 1],
        range=[-1.2, 1.2],
        row=2,
        col=1,
    )

    fig.update_xaxes(
        hoverformat="%Y-%m-%d %H:%M",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        showline=False,
        fixedrange=True,
    )

    fig.update_xaxes(
        title=dict(text="Date", font=dict(color="black")),
        tickfont=dict(color="black"),
        tickvals=custom_ticks,
        tickformat="%Y-%m-%d",
        row=2,
        col=1,
    )

    fig.update_yaxes(fixedrange=True)

    filename = f"spread_pos_{result.ticker_x}_{result.ticker_y}_{result.start}_{result.end}.html".replace(
        ":", "_"
    )

    if save:
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(results_dir / filename)


def plot_returns(
    result: StrategyResult,
    btc_data: pd.DataFrame | None = None,
    ewp_data: pd.DataFrame | None = None,
    directory: str | None = None,
    save: bool = True,
    prefix: str | None = "",
) -> None:
    df = result.data.dropna(subset=["equity"]).copy()

    if result.ticker_x in df.columns:
        df[f"return_{result.ticker_x}"] = (
            df[result.ticker_x] / df[result.ticker_x].iloc[0]
        ) - 1
    if result.ticker_y in df.columns:
        df[f"return_{result.ticker_y}"] = (
            df[result.ticker_y] / df[result.ticker_y].iloc[0]
        ) - 1

    results_dir = Path(directory) if directory else Path(".")

    custom_ticks = _get_custom_tickvals(df.index)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["total_net_return"],
            mode="lines",
            name="Total Return (Net)",
            line=dict(color="black", width=1.5),
            hovertemplate="<b>Total Return (Net)</b>: %{y:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["total_return"],
            mode="lines",
            name="Total Return (Gross)",
            line=dict(color="grey", width=1.5),
            visible="legendonly",
            hovertemplate="<b>Total Return (Gross)</b>: %{y:.4f}<extra></extra>",
        )
    )

    if btc_data is not None:
        fig.add_trace(
            go.Scatter(
                x=btc_data.index,
                y=btc_data["BTC_return"],
                mode="lines",
                name="BTC Return",
                line=dict(color="orange", width=1, dash="dot"),
                visible="legendonly",
                hovertemplate="<b>BTC Return</b>: %{y:.4f}<extra></extra>",
            )
        )

    if ewp_data is not None:
        fig.add_trace(
            go.Scatter(
                x=ewp_data.index,
                y=ewp_data["ewp_return"],
                mode="lines",
                name="EWP Return",
                line=dict(color="red", width=1, dash="dot"),
                visible="legendonly",
                hovertemplate="<b>EWP Return</b>: %{y:.4f}<extra></extra>",
            )
        )

    if result.ticker_x in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[f"return_{result.ticker_x}"],
                name=f"{result.ticker_x} Return",
                line=dict(color="blue", width=1, dash="dot"),
                opacity=0.6,
                visible="legendonly",
                hovertemplate=f"<b>{result.ticker_x} Return</b>: %{{y:.4f}}<extra></extra>",
            )
        )
    if result.ticker_y in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[f"return_{result.ticker_y}"],
                name=f"{result.ticker_y} Return",
                line=dict(color="green", width=1, dash="dot"),
                opacity=0.6,
                visible="legendonly",
                hovertemplate=f"<b>{result.ticker_y} Return</b>: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Performance: {result.ticker_x} / {result.ticker_y}",
            x=0.5,
            y=0.9,
            xanchor="center",
            yanchor="top",
            font=dict(color="black"),
        ),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=100),
    )

    fig.update_yaxes(
        title=dict(text="Cumulative Return", font=dict(color="black")),
        tickfont=dict(color="black"),
        fixedrange=True,
    )

    fig.update_xaxes(
        title=dict(text="Date", font=dict(color="black")),
        tickfont=dict(color="black"),
        tickvals=custom_ticks,
        tickformat="%Y-%m-%d",
        hoverformat="%Y-%m-%d %H:%M",
        fixedrange=True,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
    )

    filename = f"{prefix}returns_{result.ticker_x}_{result.ticker_y}_{result.start}_{result.end}.html".replace(
        ":", "_"
    )

    if save:
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(results_dir / filename)
