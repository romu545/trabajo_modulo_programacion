# pyright: reportGeneralTypeIssues=false
"""
viz.py
------
Visualization utilities for the Saber 11 results pipeline.

All public functions receive chart-ready DataFrames (output of ``process.py``)
and return a ``matplotlib.figure.Figure`` object so callers can either display
it (``plt.show()``) or save it (``fig.savefig(...)``).

Dependencies
------------
matplotlib, seaborn
"""

from __future__ import annotations

import textwrap
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns


# ── Defaults ──────────────────────────────────────────────────────────────────

_PALETTE_HIGHLIGHT = sns.color_palette("tab10")
_COLOR_GREY        = "#CCCCCC"
_COLOR_RIBBON      = "#4477AA"
_COLOR_MEAN_LINE   = "#111111"
_ALPHA_RIBBON      = 0.20
_ALPHA_GREY        = 0.50


def _build_auto_title(
    semester:       int | None,
    calendar:       str | list | None,
    location:       str | list | None,
    nature:         str | list | None,
    year_from:      int | None,
    year_to:        int | None,
) -> tuple[str, str]:
    """
    Build a default chart title and subtitle from the active filter values.

    Returns
    -------
    title : str
    subtitle : str
    """
    title = "EVOLUCIÓN DEL PUNTAJE GLOBAL EN LAS PRUEBAS SABER 11"

    parts: list[str] = []

    if nature:
        vals = [nature] if isinstance(nature, str) else nature
        parts.append(", ".join(vals).lower() + "es")

    if location:
        vals = [location] if isinstance(location, str) else location
        parts.append(", ".join(v.lower() for v in vals))

    if calendar:
        vals = [calendar] if isinstance(calendar, str) else calendar
        label = "calendario " + "/".join(vals)
        parts.append(label)

    if semester:
        parts.append(f"semestre {semester}")

    if year_from or year_to:
        y1 = str(year_from) if year_from else "…"
        y2 = str(year_to)   if year_to   else "…"
        parts.append(f"{y1}–{y2}")

    subtitle = "Instituciones " + " · ".join(parts) if parts else ""
    return title, subtitle


def _spread_labels_vertically(
    targets: list[float],
    min_gap: float,
    low:     float,
    high:    float,
) -> list[float]:
    """
    Separate a set of label y-positions so they never overlap.

    Each label wants to sit at its "true" y value (``targets``).  When two
    labels are closer than ``min_gap`` they are nudged apart, preserving their
    original vertical order and trying to stay inside the ``[low, high]`` band.
    This is the same idea behind R's ggrepel / directlabels for end-of-line
    labels.

    Parameters
    ----------
    targets : list of float
        Desired y-position of each label (e.g. the school's last score).
    min_gap : float
        Minimum vertical separation between two consecutive labels, in the
        same units as *targets*.
    low, high : float
        Lower and upper bounds of the plotting band (the y-axis limits).

    Returns
    -------
    list of float
        Adjusted y-positions, in the same order as *targets*.
    """
    n = len(targets)
    if n == 0:
        return []

    # Sort by desired position, remembering each label's original slot.
    order = sorted(range(n), key=lambda i: targets[i])
    ys    = [targets[i] for i in order]

    # Sweep up: push any label that sits too close to the one below it.
    for i in range(1, n):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap

    # If the stack ran past the top, pin the highest label and sweep back down.
    if ys[-1] > high:
        ys[-1] = high
        for i in range(n - 2, -1, -1):
            if ys[i + 1] - ys[i] < min_gap:
                ys[i] = ys[i + 1] - min_gap

    # Keep the lowest label inside the band as well.
    ys[0] = max(ys[0], low)

    # Restore the caller's original ordering.
    result = [0.0] * n
    for slot, original_idx in enumerate(order):
        result[original_idx] = ys[slot]
    return result


# ── Main chart ────────────────────────────────────────────────────────────────

def plot_school_trends(
    df_plot:      pd.DataFrame,
    period_stats: pd.DataFrame,
    labels:       pd.DataFrame,
    *,
    # --- appearance ---
    title:        str | None = None,
    subtitle:     str | None = None,
    y_min:        float = 150,
    y_max:        float = 350,
    y_breaks:     int   = 11,
    figsize:      tuple[float, float] = (14, 7),
    # --- filter metadata (used for auto-title only) ---
    semester:       int  | None = None,
    calendar:       str  | None = None,
    location:       str  | None = None,
    grade:          str  | None = None,
    nature:         str  | None = None,
    exclude_shifts: list | None = None,
    year_from:      int  | None = None,
    year_to:        int  | None = None,
    # --- ribbon ---
    ribbon_quantiles: tuple[float, float] = (0.25, 0.75),
) -> plt.Figure:
    """
    Replicate the R/ggplot2 school-trend chart in seaborn / matplotlib.

    The chart shows one line per school, with top-N and bottom-N schools
    highlighted and the rest drawn in grey.  A ribbon spans the
    inter-quantile range and a bold black line shows the weighted mean.

    Parameters
    ----------
    df_plot : pd.DataFrame
        School-period aggregated data with a ``grupo`` column
        (``"top"``, ``"bottom"``, ``"medio"``).
        Output of :func:`process.build_plot_data`.
    period_stats : pd.DataFrame
        Period-level summary with columns ``año``, ``promedio``,
        ``p25``, ``p75`` (or the columns matching *ribbon_quantiles*).
        Output of :func:`process.build_plot_data`.
    labels : pd.DataFrame
        Rows for direct labelling (top/bottom schools at the last period).
        Output of :func:`process.build_plot_data`.
    title : str, optional
        Chart title.  Auto-generated from filter params when ``None``.
    subtitle : str, optional
        Chart subtitle.  Auto-generated when ``None``.
    y_min : float, optional
        Lower bound of the y-axis.  Default ``150``.
    y_max : float, optional
        Upper bound of the y-axis.  Default ``350``.
    y_breaks : int, optional
        Number of y-axis tick marks.  Default ``11``.
    figsize : tuple, optional
        Figure size in inches ``(width, height)``.  Default ``(14, 7)``.
    semester : int, optional
        Passed to auto-title builder only.
    calendar : str, optional
        Passed to auto-title builder only.
    location : str, optional
        Passed to auto-title builder only.
    grade : str, optional
        Passed to auto-title builder only.
    nature : str, optional
        Passed to auto-title builder only.
    exclude_shifts : list, optional
        Passed to auto-title builder only.
    year_from : int, optional
        Passed to auto-title builder only.
    year_to : int, optional
        Passed to auto-title builder only.
    ribbon_quantiles : tuple of (float, float), optional
        Lower and upper quantile columns to use for the ribbon.
        Default ``(0.25, 0.75)`` → columns ``p25`` and ``p75``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> from process import build_plot_data
    >>> df_plot, period_stats, labels = build_plot_data(df)
    >>> fig = plot_school_trends(df_plot, period_stats, labels,
    ...                          year_from=2016, year_to=2025,
    ...                          nature="Oficial", semester=2)
    >>> fig.savefig("saber11_trends.png", dpi=150, bbox_inches="tight")
    """
    # ── resolve auto title ────────────────────────────────────────────────────
    if title is None or subtitle is None:
        auto_title, auto_subtitle = _build_auto_title(
            semester, calendar, location,
            nature, year_from, year_to,
        )
        title    = title    or auto_title
        subtitle = subtitle or auto_subtitle

    # ── year-level x-axis labels ──────────────────────────────────────────────
    df_plot = df_plot.copy()
    anio_order = sorted(df_plot["anio_prueba"].unique())
    anio_to_idx = {yr: i for i, yr in enumerate(anio_order)}
    df_plot["anio_order"]      = df_plot["anio_prueba"].map(anio_to_idx)
    period_stats = period_stats.copy()
    period_stats["anio_order"] = period_stats["anio_prueba"].map(anio_to_idx)

    # ── colour palette for highlighted schools ────────────────────────────────
    highlighted_codes = df_plot.loc[
        df_plot["grupo"].isin(["top", "bottom"]),
        "cole_cod_dane_establecimiento",
    ].unique()

    hl_palette = {
        code: _PALETTE_HIGHLIGHT[i % len(_PALETTE_HIGHLIGHT)]
        for i, code in enumerate(highlighted_codes)
    }

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("whitegrid")

    # Grey "background" lines (non-highlighted)
    for code, sub in df_plot[df_plot["grupo"] == "medio"].groupby(
        "cole_cod_dane_establecimiento", observed=True
    ):
        ax.plot(
            sub["anio_order"], sub["promedio"],
            color=_COLOR_GREY, linewidth=0.35, alpha=_ALPHA_GREY, zorder=1,
        )

    # Highlighted lines
    for code, sub in df_plot[df_plot["grupo"].isin(["top", "bottom"])].groupby(
        "cole_cod_dane_establecimiento", observed=True
    ):
        ax.plot(
            sub["anio_order"], sub["promedio"],
            color=hl_palette[code], linewidth=1.4, zorder=3,
        )

    # ── ribbon ────────────────────────────────────────────────────────────────
    q_lo_col = f"p{int(ribbon_quantiles[0]*100)}"
    q_hi_col = f"p{int(ribbon_quantiles[1]*100)}"

    ax.fill_between(
        period_stats["anio_order"],
        period_stats[q_lo_col],
        period_stats[q_hi_col],
        color=_COLOR_RIBBON,
        alpha=_ALPHA_RIBBON,
        zorder=2,
        label=f"IQR ({int(ribbon_quantiles[0]*100)}{int(ribbon_quantiles[1]*100)} pct.)",
    )

    # ── weighted mean line ────────────────────────────────────────────────────
    ax.plot(
        period_stats["anio_order"],
        period_stats["promedio"],
        color=_COLOR_MEAN_LINE,
        linewidth=1.6,
        zorder=4,
        label="Promedio ponderado",
    )

    # ── axes formatting ───────────────────────────────────────────────────────
    # El eje x usa el índice del año (0, 1, 2, …), no el año en sí.  Se parte de
    # un margen derecho amplio que más abajo se recorta a lo que realmente
    # ocupan los nombres de los colegios.
    ax.set_xlim(-0.5, len(anio_order) - 1 + 5)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(range(len(anio_order)))
    ax.set_xticklabels(anio_order, rotation=0)
    ax.yaxis.set_major_locator(mticker.LinearLocator(y_breaks))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    ax.set_xlabel("Año", fontsize=11)
    ax.set_ylabel("Promedio puntaje global", fontsize=11)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#AAAAAA")
    ax.grid(axis="x", visible=False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

    # ── direct labels (one per highlighted school, at the last period) ─────────
    # Se rotula cada colegio resaltado al final de su línea.  Cuando dos colegios
    # terminan con puntajes parecidos sus nombres se solaparían, así que se
    # separan verticalmente conservando el orden (idea de ggrepel / directlabels
    # de R) y se une cada etiqueta con su línea mediante un trazo delgado.
    last_x_idx = len(anio_order) - 1
    label_x    = last_x_idx + 0.20

    rows = list(labels.iterrows())
    text_objects: list[plt.Text] = [
        ax.text(
            label_x, row["promedio"],
            textwrap.shorten(row["cole_nombre_establecimiento"], width=40, placeholder="…"),
            fontsize=7.5,
            color=hl_palette.get(row["cole_cod_dane_establecimiento"], "black"),
            va="center", zorder=5, clip_on=False,
        )
        for _, row in rows
    ]

    if text_objects:
        # El hueco mínimo entre etiquetas se toma de la altura real del texto
        # (en unidades de datos) para que no dependa del tamaño de la figura.
        fig.canvas.draw()
        text_px  = text_objects[0].get_window_extent().height
        inv      = ax.transData.inverted()
        _, y0_px = inv.transform((0, 0))
        _, y1_px = inv.transform((0, text_px))
        min_gap  = abs(y1_px - y0_px) * 1.3

        true_ys   = [row["promedio"] for _, row in rows]
        spread_ys = _spread_labels_vertically(true_ys, min_gap, y_min, y_max)

        # Reubicar cada etiqueta y unirla con su línea mediante un trazo fino.
        for t, (_, row), new_y in zip(text_objects, rows, spread_ys):
            t.set_y(new_y)
            color = hl_palette.get(row["cole_cod_dane_establecimiento"], "black")
            ax.plot(
                [last_x_idx, label_x], [row["promedio"], new_y],
                color=color, linewidth=0.5, alpha=0.6, zorder=4, clip_on=False,
            )

        # Recortar el margen derecho para que quede solo el espacio que ocupan
        # los nombres y la gráfica aproveche el ancho.  Se itera porque al mover
        # el límite del eje también cambia el ancho (en datos) de cada texto.
        for _ in range(4):
            fig.canvas.draw()
            inv_x      = ax.transData.inverted()
            right_data = max(
                inv_x.transform((t.get_window_extent().x1, 0))[0]
                for t in text_objects
            )
            ax.set_xlim(-0.5, right_data + 0.15)

    # ── legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=_COLOR_MEAN_LINE, label="Promedio ponderado"),
        mpatches.Patch(color=_COLOR_RIBBON, alpha=0.5,
                       label=f"IQR ({int(ribbon_quantiles[0]*100)}-{int(ribbon_quantiles[1]*100)} pct.)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper left", framealpha=0.7)

    # ── titles ────────────────────────────────────────────────────────────────
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    ax.set_title(subtitle, fontsize=9.5, color="#444444", pad=4)

    # caption = (
    #     "Línea negra: promedio ponderado general  ·  "
    #     f"Franja azul: rango intercuartílico {int(ribbon_quantiles[0]*100)}-{int(ribbon_quantiles[1]*100)}"
    # )
    # fig.text(0.01, -0.01, caption, fontsize=8, color="#666666", ha="left")

    fig.tight_layout()
    return fig


# ── Descriptive plots ─────────────────────────────────────────────────────────


def plot_score_distribution(
    df: pd.DataFrame,
    score_col: str = "punt_global",
    group_col: str | None = None,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """
    Plot the distribution of a score column as a KDE + histogram.

    When *group_col* is provided the distributions are overlaid per group.

    Parameters
    ----------
    df : pd.DataFrame
        Clean Saber 11 DataFrame.
    score_col : str, optional
        Score column.  Default ``"punt_global"``.
    group_col : str, optional
        Categorical column for grouped distributions (e.g.
        ``"cole_naturaleza"``).  Default ``None``.
    title : str, optional
        Chart title.  Auto-generated when ``None``.
    figsize : tuple, optional
        Figure size.  Default ``(10, 5)``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_score_distribution(df, group_col="cole_naturaleza")
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    if group_col:
        groups = df[group_col].dropna().unique()
        palette = sns.color_palette("tab10", n_colors=len(groups))
        for color, grp in zip(palette, groups):
            sub = df.loc[df[group_col] == grp, score_col].dropna()
            sns.histplot(sub, ax=ax, kde=True, color=color,
                         alpha=0.35, label=str(grp), stat="density",
                         linewidth=0)
        ax.legend(title=group_col, fontsize=9)
    else:
        sns.histplot(df[score_col].dropna(), ax=ax, kde=True,
                     color=_COLOR_RIBBON, alpha=0.5, stat="density", linewidth=0)

    ax.set_xlabel(score_col, fontsize=11)
    ax.set_ylabel("Densidad", fontsize=11)
    ax.set_title(title or f"Distribución de {score_col}", fontsize=12, fontweight="bold")
    sns.despine(fig)
    fig.tight_layout()
    return fig


def plot_score_by_group(
    df: pd.DataFrame,
    group_col: str,
    x: str,
    score_col: str = "punt_global",
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """
    Box-plot of a score column broken down by a categorical variable.

    Groups are sorted by their median score (descending).

    Parameters
    ----------
    df : pd.DataFrame
        Clean Saber 11 DataFrame.
    group_col : str
        Column to group by (e.g. ``"cole_jornada"``, ``"cole_naturaleza"``).
    score_col : str, optional
        Score column.  Default ``"punt_global"``.
    title : str, optional
        Chart title.  Auto-generated when ``None``.
    figsize : tuple, optional
        Figure size.  Default ``(10, 5)``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_score_by_group(df, "cole_jornada")
    >>> plt.show()
    """
    if x is None:
        x = group_col
    order = (
        df.groupby(by=group_col, observed=True, as_index=False)[score_col]
        .median()
        .sort_values(by=x, ascending=True)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df, 
        x=x, 
        y=score_col, 
        order=order,
        palette="Blues_d", width=0.5, linewidth=0.8, ax=ax,
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
    )
    ax.set_xlabel(group_col, fontsize=11)
    ax.set_ylabel(score_col, fontsize=11)
    ax.set_title(title or f"{score_col} por {group_col}", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    sns.despine(fig)
    fig.tight_layout()
    return fig


def plot_trend(
    trend_df: pd.DataFrame,
    *,
    title: str = "Tendencia del puntaje global por año",
    ci: bool = True,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """
    Line chart of the year-level mean score with an optional 95 % CI ribbon.

    Parameters
    ----------
    trend_df : pd.DataFrame
        Output of :func:`process.score_trend_by_year`, with columns
        ``año``, ``mean``, ``ci95_lower``, ``ci95_upper``.
    title : str, optional
        Chart title.
    ci : bool, optional
        Draw the 95 % confidence-interval ribbon.  Default ``True``.
    figsize : tuple, optional
        Figure size.  Default ``(10, 5)``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> from process import score_trend_by_year
    >>> trend = score_trend_by_year(df, semester=2)
    >>> fig = plot_trend(trend)
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(trend_df["anio_prueba"], trend_df["mean"],
            color=_COLOR_MEAN_LINE, linewidth=1.8, marker="o", markersize=4)

    if ci:
        ax.fill_between(
            trend_df["anio_prueba"],
            trend_df["ci95_lower"],
            trend_df["ci95_upper"],
            color=_COLOR_RIBBON, alpha=_ALPHA_RIBBON,
            label="IC 95 %",
        )
    
    ax.legend(fontsize=9)
    ax.set_xbound(100, 400)
    ax.set_xlabel("Año", fontsize=11)
    ax.set_ylabel("Puntaje global promedio", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    sns.despine(fig)
    fig.tight_layout()
    plt.show()
    return fig
