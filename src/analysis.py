import numpy as np
import pandas as pd
from typing import Literal

Semester = Literal[1, 2]


def describe_global_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estadísticas descriptivas de ``punt_global`` para el conjunto completo.

    Returns
    -------
    pd.DataFrame
        Una fila por estadístico: count, mean, std, min, p5, p10, p25,
        p50, p75, p90, p95, max, skewness, kurtosis.
    """
    s = df["punt_global"].dropna()
    q = s.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return pd.DataFrame({
        "punt_global": {
            "count":    s.count(),
            "mean":     s.mean(),
            "std":      s.std(),
            "min":      s.min(),
            "p5":       q[0.05],
            "p10":      q[0.10],
            "p25":      q[0.25],
            "p50":      q[0.50],
            "p75":      q[0.75],
            "p90":      q[0.90],
            "p95":      q[0.95],
            "max":      s.max(),
            "skewness": s.skew(),
            "kurtosis": s.kurt(),
        }
    })


def describe_by_group(
    df: pd.DataFrame,
    group_col: str,
    score_col: str = "punt_global",
) -> pd.DataFrame:
    """
    Estadísticas descriptivas agrupadas para una columna de puntaje.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame limpio de Saber 11.
    group_col : str
        Columna de agrupación (p. ej. ``"cole_naturaleza"``).
    score_col : str, optional
        Columna numérica a resumir.  Default ``"punt_global"``.

    Returns
    -------
    pd.DataFrame
        Una fila por grupo con: count, mean, std, median, p25, p75, iqr.
        Ordenado por mean descendente.
    """
    def _stats(s: pd.Series) -> pd.Series:
        q25, q50, q75 = s.quantile([0.25, 0.50, 0.75])
        return pd.Series({
            "count":  s.count(),
            "mean":   s.mean(),
            "std":    s.std(),
            "median": q50,
            "p25":    q25,
            "p75":    q75,
            "iqr":    q75 - q25,
        })

    return (
        df.groupby(group_col, observed=True)[score_col]
        .apply(_stats)
        .unstack()
        .sort_values("mean", ascending=False)
    )


def score_trend_by_year(
    df: pd.DataFrame,
    score_col: str = "punt_global",
    semester: Semester | None = None,
) -> pd.DataFrame:
    """
    Media anual y IC 95 % de un puntaje.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame limpio con columna ``periodo``.
    score_col : str, optional
        Columna de puntaje.  Default ``"punt_global"``.
    semester : {1, 2}, optional
        Filtrar por semestre.  Default ``None`` (ambos semestres).

    Returns
    -------
    pd.DataFrame
        Una fila por año con: anio_prueba, count, mean, std,
        ci95_lower, ci95_upper.
    """
    data = df.copy()
    if semester is not None:
        data = data[data["semestre_prueba"] == semester]

    def _agg(s: pd.Series) -> pd.Series:
        n    = s.count()
        mean = s.mean()
        std  = s.std()
        se   = std / np.sqrt(n) if n > 1 else np.nan
        return pd.Series({
            "count":      n,
            "mean":       mean,
            "std":        std,
            "ci95_lower": mean - 1.96 * se,
            "ci95_upper": mean + 1.96 * se,
        })

    return (
        data.groupby("anio_prueba")[score_col]
        .apply(_agg)
        .unstack()
        .reset_index()
        .sort_values("anio_prueba")
    )


def count_schools_per_period(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Cuenta instituciones distintas que presentaron la prueba por periodo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame limpio de Saber 11.
    group_cols : list of str, optional
        Columnas adicionales de agrupación.

    Returns
    -------
    pd.DataFrame
        Columnas: periodo, (extras), n_schools, n_students. Ordenado por periodo.
    """
    by = ["periodo"] + (group_cols or [])
    return (
        df.groupby(by, observed=True)
        .agg(
            n_schools=("cole_cod_dane_establecimiento", "nunique"),
            n_students=("punt_global", "count"),
        )
        .reset_index()
        .sort_values(by)
    )


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resumen de valores faltantes por columna.

    Returns
    -------
    pd.DataFrame
        Una fila por columna con n_missing, pct_missing, dtype.
        Ordenado por pct_missing descendente.
    """
    n = len(df)
    missing = df.isna().sum()
    return (
        pd.DataFrame({
            "n_missing":   missing,
            "pct_missing": (missing / n * 100).round(2),
            "dtype":       df.dtypes,
        })
        .sort_values("pct_missing", ascending=False)
    )
