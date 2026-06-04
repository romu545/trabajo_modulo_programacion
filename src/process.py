"""
process.py
----------
Data processing utilities for the Saber 11 results pipeline.

This module sits between the loading/cleaning layers and the visualization
layer.  All functions receive a *clean* DataFrame (output of the cleaning
module) and return either a filtered/aggregated DataFrame or a scalar summary,
ready to be passed to ``viz.py``.
"""

#from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd


# ── Type aliases ──────────────────────────────────────────────────────────────

Semester   = Literal[1, 2]
Calendar   = Literal["A", "B"]
Location   = Literal["Urbana", "Rural"]
Grade      = Literal["Grado 11", "Grado 10", "Ciclo 5", "Ciclo 6"]
Nature     = Literal["Oficial", "No oficial"]
Shift      = Literal["Mañana", "Tarde", "Noche", "Completa", "Sabatina", "Unica"]

_PERIOD_RE = r"^(\d{4})([12])$"


# ── Filtering ─────────────────────────────────────────────────────────────────

def filter_data(
    df: pd.DataFrame,
    *,
    semester: Semester | None = None,
    calendar: Calendar | list[Calendar] | None = None,
    location: Location | list[Location] | None = None,
    grade: Grade | list[Grade] | None = None,
    nature: Nature | list[Nature] | None = None,
    exclude_shifts: list[Shift] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> pd.DataFrame:
    """
    Apply one or more filters to the clean Saber 11 DataFrame.

    Every parameter is optional; when ``None`` the corresponding dimension is
    not filtered.  All string comparisons are case-sensitive and must match the
    values stored in the clean dataset exactly.

    Parameters
    ----------
    df : pd.DataFrame
        Clean Saber 11 DataFrame produced by the loading/cleaning pipeline.
    semester : {1, 2}, optional
        Keep only records from the given semester (last digit of *periodo*).
    calendar : str or list of str, optional
        School calendar(s) to keep.  Typical values: ``"A"``, ``"B"``.
    location : str or list of str, optional
        School area(s) to keep: ``"Urbana"`` or ``"Rural"``.
    grade : str or list of str, optional
        Student grade(s) to keep (e.g. ``"Grado 11"``).
    nature : str or list of str, optional
        School nature(s) to keep: ``"Oficial"`` or ``"No oficial"``.
    exclude_shifts : list of str, optional
        School shifts to *exclude* (e.g. ``["Sabatina", "Noche"]``).
    year_from : int, optional
        Keep only records with year >= *year_from* (inclusive).
    year_to : int, optional
        Keep only records with year <= *year_to* (inclusive).

    Returns
    -------
    pd.DataFrame
        Filtered copy of *df*.  Resets the index.

    Examples
    --------
    >>> filtered = filter_data(
    ...     df,
    ...     semester=2,
    ...     calendar="A",
    ...     location="Urbana",
    ...     grade="Grado 11",
    ...     nature="Oficial",
    ...     exclude_shifts=["Sabatina", "Noche"],
    ...     year_from=2015,
    ...     year_to=2025,
    ... )
    """
    mask = pd.Series(True, index=df.index)

    if semester is not None:
        mask &= df["semestre_prueba"] == semester

    if calendar is not None:
        vals = [calendar] if isinstance(calendar, str) else list(calendar)
        mask &= df["cole_calendario"].isin(vals)

    if location is not None:
        vals = [location] if isinstance(location, str) else list(location)
        mask &= df["cole_area_ubicacion"].isin(vals)

    if grade is not None:
        vals = [grade] if isinstance(grade, str) else list(grade)
        mask &= df["estu_grado"].isin(vals)

    if nature is not None:
        vals = [nature] if isinstance(nature, str) else list(nature)
        mask &= df["cole_naturaleza"].isin(vals)

    if exclude_shifts is not None:
        mask &= ~df["cole_jornada"].isin(exclude_shifts)

    if year_from is not None:
        mask &= df["anio_prueba"] >= year_from

    if year_to is not None:
        mask &= df["anio_prueba"] <= year_to

    result = df.loc[mask].copy().reset_index(drop=True)

    if result.empty:
        warnings.warn(
            "filter_data: the applied filters produced an empty DataFrame.",
            stacklevel=2
        )

    return result


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_by_school(
    df: pd.DataFrame,
    by: list,
    col_agg: str = "punt_global"
    ) -> pd.DataFrame:
    """
    Aggregate student-level records to school × period level.

    For each (period, school DANE code, school name) group the function
    computes:

    * ``recuento``  – number of students that sat the exam.
    * ``promedio``  – (unweighted) mean of ``punt_global``.

    It then appends a column ``nro_anios_presentados`` with the total number
    of distinct periods the school appears in across the whole dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Clean, optionally pre-filtered DataFrame with at least the columns
    by: Columns for gruop
        ``periodo``, ``cole_cod_dane_establecimiento``,
        ``cole_nombre_establecimiento``, and ``punt_global``.
    col_agg: column for aggregate

    Returns
    -------
    pd.DataFrame
        One row per (period, school) pair.

    Examples
    --------
    >>> agg = aggregate_by_school(filtered_df, ['anio_prueba'], [punt_global])
    >>> agg.head()
    """
    agg = (
        df.groupby(
            by,
            observed=True,
        )
        .agg(
            recuento=(col_agg, "count"),
            promedio=(col_agg, "mean"),
        )
        .reset_index()
    )

    years_per_school = (
        agg.groupby("cole_cod_dane_establecimiento", observed=True)["anio_prueba"]
        .nunique()
        .rename("nro_anios_presentados")
    )

    return agg.merge(years_per_school, on="cole_cod_dane_establecimiento", how="left")


def filter_schools_by_continuity(
    df: pd.DataFrame,
    min_years: int = 10,
    require_consecutive: bool = True,
    year_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """
    Keep only schools that meet a minimum data-continuity requirement.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated DataFrame produced by :func:`aggregate_by_school`.
    min_years : int, optional
        Minimum number of years a school must appear in.  Default is ``10``.
    require_consecutive : bool, optional
        If ``True`` (default) the school must have data in *every* year within
        *year_range*.  Requires *year_range* to be set.
    year_range : tuple of (int, int), optional
        ``(start_year, end_year)`` inclusive range used to check consecutive
        presence.  Ignored when ``require_consecutive=False``.

    Returns
    -------
    pd.DataFrame
        Filtered copy of *df* containing only schools that pass the continuity
        check.

    Raises
    ------
    ValueError
        If ``require_consecutive=True`` but ``year_range`` is not provided.

    Examples
    --------
    >>> continuous = filter_schools_by_continuity(
    ...     agg, min_years=10, require_consecutive=True, year_range=(2016, 2025)
    ... )
    """
    if require_consecutive and year_range is None:
        raise ValueError(
            "year_range must be provided when require_consecutive=True. "
            "Example: year_range=(2015, 2025)"
        )

    # Step 1: minimum years filter (already stored in nro_anios_presentados)
    df = df[df["nro_anios_presentados"] >= min_years].copy()

    if not require_consecutive or year_range is None:
        return df.reset_index(drop=True)

    # Step 2: all years in range must be present
    start, end = year_range
    required = set(range(start, end + 1))

    def _has_all_years(sub: pd.DataFrame) -> bool:
        years_present = set(sub["anio_prueba"])
        return required.issubset(years_present)

    valid_codes = (
        df.groupby("cole_cod_dane_establecimiento", observed=True)
        .filter(_has_all_years)["cole_cod_dane_establecimiento"]
        .unique()
    )

    return df[df["cole_cod_dane_establecimiento"].isin(valid_codes)].reset_index(drop=True)



def compute_period_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weighted descriptive statistics per period across all schools.

    For each *periodo* the function calculates:

    * ``promedio``  – weighted mean of school averages (weights = ``recuento``).
    * ``p10``, ``p25``, ``p75``, ``p90`` – quantiles of school averages.

    These statistics are used by the ribbon and reference line in the chart.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated DataFrame produced by :func:`aggregate_by_school`, which
        must contain columns ``periodo``, ``promedio``, and ``recuento``.

    Returns
    -------
    pd.DataFrame
        One row per period with columns ``periodo``, ``año`` (str),
        ``promedio``, ``p10``, ``p25``, ``p75``, ``p90``.

    Examples
    --------
    >>> stats = compute_period_stats(agg)
    """
    def _weighted_mean(sub: pd.DataFrame) -> float:
        return np.average(sub["promedio"], weights=sub["recuento"])

    rows = []
    for anio, sub in df.groupby("anio_prueba", observed=True):
        rows.append({
            "anio_prueba":  anio,
            "promedio": _weighted_mean(sub),
            "p10":      sub["promedio"].quantile(0.10),
            "p25":      sub["promedio"].quantile(0.25),
            "p75":      sub["promedio"].quantile(0.75),
            "p90":      sub["promedio"].quantile(0.90),
        })

    stats = pd.DataFrame(rows).sort_values("anio_prueba").reset_index(drop=True)
    return stats


def rank_schools(
    df: pd.DataFrame,
    n: int = 3,
) -> pd.DataFrame:
    """
    Assign a ``grupo`` label (``"top"``, ``"bottom"``, ``"medio"``) to each
    school based on its weighted-mean global score.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated DataFrame with columns ``cole_cod_dane_establecimiento``,
        ``promedio``, and ``recuento``.
    n : int, optional
        Number of schools to label as ``"top"`` and ``"bottom"``.  Default 3.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``cole_cod_dane_establecimiento``,
        ``promedio_global``, and ``grupo``.

    Examples
    --------
    >>> ranking = rank_schools(agg, n=3)
    """
    ranking = (
        df.groupby("cole_cod_dane_establecimiento", observed=True)
        .apply(lambda sub: pd.Series({
            "promedio_global": np.average(sub["promedio"], weights=sub["recuento"])
        }))
        .reset_index()
    )

    ranking = ranking.sort_values("promedio_global").reset_index(drop=True)
    n_schools = len(ranking)

    ranking["grupo"] = "medio"
    ranking.iloc[:n, ranking.columns.get_loc("grupo")] = "bottom"
    ranking.iloc[n_schools - n:, ranking.columns.get_loc("grupo")] = "top"

    return ranking


def build_plot_data(
    df: pd.DataFrame,
    *,
    semester: Semester | None = 2,
    calendar: Calendar | list[Calendar] | None = "A",
    location: Location | list[Location] | None = "Urbana",
    grade: Grade | list[Grade] | None = "Grado 11",
    nature: Nature | list[Nature] | None = "Oficial",
    exclude_shifts: list[Shift] | None = None,
    year_from: int | None = 2016,
    year_to: int | None = 2025,
    min_years: int = 10,
    require_consecutive: bool = True,
    n_highlight: int = 3,
    by: list,
    col_agg: str = "punt_global"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full processing pipeline from clean data to chart-ready tables.

    This is a convenience wrapper that chains :func:`filter_data`,
    :func:`aggregate_by_school`, :func:`filter_schools_by_continuity`,
    :func:`compute_period_stats`, and :func:`rank_schools` into a single call.

    Parameters
    ----------
    df : pd.DataFrame
        Clean Saber 11 DataFrame.
    semester : {1, 2}, optional
        Semester filter.  Default ``2``.
    calendar : str or list, optional
        Calendar filter.  Default ``"A"``.
    location : str or list, optional
        Location filter.  Default ``"Urbana"``.
    grade : str or list, optional
        Grade filter.  Default ``"Grado 11"``.
    nature : str or list, optional
        School nature filter.  Default ``"Oficial"``.
    exclude_shifts : list of str, optional
        Shifts to exclude.  Default ``None`` (keeps all).
    year_from : int, optional
        First year to include.  Default ``2016``.
    year_to : int, optional
        Last year to include.  Default ``2025``.
    min_years : int, optional
        Minimum number of years a school must appear in.  Default ``10``.
    require_consecutive : bool, optional
        Require the school to be present in every year of the range.
        Default ``True``.
    n_highlight : int, optional
        Number of top/bottom schools to highlight.  Default ``3``.

    Returns
    -------
    df_plot : pd.DataFrame
        School-period aggregated data with ``grupo`` column merged in.
    period_stats : pd.DataFrame
        Period-level summary statistics (for ribbon and reference line).
    labels : pd.DataFrame
        Subset of *df_plot* for the top/bottom schools in the last period
        (used for direct labelling).

    Examples
    --------
    >>> df_plot, period_stats, labels = build_plot_data(
    ...     df,
    ...     semester=2,
    ...     calendar="A",
    ...     location="Urbana",
    ...     grade="Grado 11",
    ...     nature="Oficial",
    ...     exclude_shifts=["Sabatina", "Noche"],
    ...     year_from=2016,
    ...     year_to=2025,
    ...     min_years=10,
    ...     n_highlight=3,
    ... )
    """
    if exclude_shifts is None:
        exclude_shifts = ["Sabatina", "Noche"]

    filtered      = filter_data(
        df,
        semester=semester,
        calendar=calendar,
        location=location,
        grade=grade,
        nature=nature,
        exclude_shifts=exclude_shifts,
        year_from=year_from,
        year_to=year_to,
    )
    agg           = aggregate_by_school(filtered, by, col_agg)
    year_range    = (year_from, year_to) if (require_consecutive and year_from and year_to) else None
    continuous    = filter_schools_by_continuity(
        agg,
        min_years=min_years,
        require_consecutive=require_consecutive,
        year_range=year_range
        )
    period_stats  = compute_period_stats(continuous)
    ranking       = rank_schools(continuous, n=n_highlight)

    df_plot = continuous.merge(ranking, on="cole_cod_dane_establecimiento", how="left")

    last_period = df_plot["anio_prueba"].max()
    labels = df_plot[
        df_plot["grupo"].isin(["top", "bottom"]) &
        (df_plot["anio_prueba"] == last_period)
    ].copy()

    return df_plot, period_stats, labels
