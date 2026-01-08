import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as go

from modules.core.models import StrategyResult


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
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


def plot_zscore(
    result: StrategyResult,
    directory: str | None = None,
    save: bool = False,
    show: bool = False,
    sl_thr: bool = False,
) -> None:
    x, y = result.ticker_x, result.ticker_y
    start, end = result.start, result.end
    df = result.data
    results_dir = _resolve_results_dir(directory)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df.index, y=df["z_score"], color="grey")

    plt.plot(
        df.index, df["entry_thr"].astype(float), color="red", label="Entry Threshold"
    )
    plt.plot(df.index, -df["entry_thr"].astype(float), color="red")
    plt.plot(
        df.index, df["exit_thr"].astype(float), color="green", label="Exit Threshold"
    )
    plt.plot(df.index, -df["exit_thr"].astype(float), color="green")

    if sl_thr:
        plt.plot(
            df.index,
            df["sl_thr"].astype(float),
            color="red",
            linestyle="--",
            label="SL Threshold",
            zorder=10,
            marker="o",
            markersize=1,
        )
        plt.plot(
            df.index,
            -df["sl_thr"].astype(float),
            color="red",
            linestyle="--",
            zorder=10,
            marker="o",
            markersize=1,
        )

    plt.title(f"Z-Score: {x}/{y}")
    plt.ylabel("Z-Score")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.xlim(df.index.min(), df.index.max())
    plt.legend(loc="lower right", fontsize="small")

    if save:
        filename = f"z_score_{x}_{y}_{start}_{end}.png".replace(":", "-")
        save_path = results_dir / filename
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_positions(
    result: StrategyResult,
    directory: str | None = None,
    save: bool = False,
    show: bool = False,
) -> None:
    x, y, start, end = (
        result.ticker_x,
        result.ticker_y,
        result.start,
        result.end,
    )
    df = result.data
    results_dir = _resolve_results_dir(directory)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["position"], color="grey", linewidth=1.6)
    ax.set_ylabel("Position")
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis="y")
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("Date")
    ax.set_title(f"Position Over Time: {x}/{y}")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(df.index.min(), df.index.max())

    if save:
        filename = f"positions_{x}_{y}_{start}_{end}.png".replace(":", "-")
        save_path = results_dir / filename
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_returns(
        result: StrategyResult,
        btc_data: pd.DataFrame | None = None,
        directory: str | None = None,
        save: bool = False,
        show: bool = False,
        interactive: bool = True
) -> None:
    """
    Function to plot returns with assets and BTC cumulative returns.

    interactive=True -> Plotly
    interactive=False -> Matplotlib
    """
    df = result.data.copy()

    if result.ticker_x in df.columns:
        df[f"return_{result.ticker_x}"] = (df[result.ticker_x] / df[result.ticker_x].iloc[0]) - 1
    if result.ticker_y in df.columns:
        df[f"return_{result.ticker_y}"] = (df[result.ticker_y] / df[result.ticker_y].iloc[0]) - 1

    results_dir = Path(directory) if directory else Path(".")

    if interactive:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df.index, y=df["total_return_pct"],
            mode='lines', name='Total Return (Gross)',
            line=dict(color='red', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=df.index, y=df["net_return_pct"],
            mode='lines', name='Total Return (Net)',
            line=dict(color='darkred', width=2, dash='dash')
        ))

        if btc_data is not None:
            fig.add_trace(go.Scatter(
                x=btc_data.index, y=btc_data["BTC_c_return"],
                mode='lines', name='BTC Return',
                line=dict(color='grey', width=1),
                visible='legendonly'
            ))

        if result.ticker_x in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[f"return_{result.ticker_x}"],
                name=f'{result.ticker_x} Hold',
                line=dict(color='blue', width=1, dash='dot'),
                opacity=0.6, visible='legendonly'
            ))

        if result.ticker_y in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[f"return_{result.ticker_y}"],
                name=f'{result.ticker_y} Hold',
                line=dict(color='orange', width=1, dash='dot'),
                opacity=0.6, visible='legendonly'
            ))

        fig.update_layout(
            title=f"Performance: {result.ticker_x} / {result.ticker_y}",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_white",
            hovermode="x unified"
        )

        filename = f"returns_{result.ticker_x}_{result.ticker_y}_{result.start}_{result.end}.html".replace(
            ":", "_")

        if save:
            fig.write_html(results_dir / filename)

        if show:
            fig.show()

    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))

        if result.ticker_x in df.columns:
            ax1.plot(df.index, df[result.ticker_x], label=f"{result.ticker_x} Hold",
                     color="blue", alpha=0.3, linewidth=0.8, linestyle=":")
        if result.ticker_y in df.columns:
            ax1.plot(df.index, df[result.ticker_x], label=f"{result.ticker_y} Hold",
                     color="orange", alpha=0.3, linewidth=0.8, linestyle=":")

        if btc_data is not None:
            ax1.plot(btc_data.index, btc_data["BTC_c_return"], label="BTC Benchmark",
                     color="grey", alpha=0.5, linewidth=1, linestyle="--")

        ax1.plot(df.index, df["total_return_pct"], label="Total Return (Gross)",
                 color="red", linewidth=1.6)
        ax1.plot(df.index, df["net_return_pct"], label="Total Return (Net)",
                 color="darkred", linewidth=1.2, linestyle="--")

        ax1.set_title(f"Performance: {result.ticker_x}/{result.ticker_y}")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Total Return")
        ax1.legend(loc="upper left", fontsize="small")
        ax1.grid(True, alpha=0.3)
        plt.xlim(df.index.min(), df.index.max())

        filename = f"returns_{result.ticker_x}_{result.ticker_y}_{result.start}_{result.end}.png".replace(":", "-")

        if save:
            plt.savefig(results_dir / filename, dpi=150)

        if show:
            plt.show()

        plt.close()
