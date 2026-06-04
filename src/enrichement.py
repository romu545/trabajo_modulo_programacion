# pyright: reportGeneralTypeIssues=false
import warnings
from typing import Sequence
import pandas as pd

from src.cleaning import (
    _require_columns,
    add_ordinal_encodings,
)


def _extract_year(period: pd.Series) -> pd.Series:
    return period.astype(str).str[:4].astype("Int64")


def _extract_semester(period: pd.Series) -> pd.Series:
    return period.astype(str).str[4].astype("Int64")


def add_period_columns(
    df: pd.DataFrame,
    *,
    col_periodo: str = "periodo",
) -> pd.DataFrame:
    """
    Descompone ``periodo`` (YYYYS) en año y semestre por separado.

    Agrega ``anio_prueba`` (Int64) y ``semestre_prueba`` (Int64).
    """
    _require_columns(df, [col_periodo], "add_period_columns")
    out = df.copy()
    out["anio_prueba"]     = _extract_year(out[col_periodo])
    out["semestre_prueba"] = _extract_semester(out[col_periodo])
    return out


def add_student_age(
    df: pd.DataFrame,
    *,
    col_birth:        str = "estu_fechanacimiento",
    col_period:       str = "periodo",
    valid_range:      tuple[int, int] = (13, 30),
    omit_age_invalid: bool = False,
) -> pd.DataFrame:
    """
    Agrega la edad del estudiante al momento del examen como ``estu_edad_prueba``.

    Edades fuera de *valid_range* se reemplazan por ``pd.NA`` cuando
    ``omit_age_invalid=True``, ya que indican errores en la fuente de datos.
    """
    _require_columns(df, [col_birth, col_period], "add_student_age")
    out        = df.copy()
    exam_year  = _extract_year(out[col_period])
    birth_year = out[col_birth].dt.year
    age        = (exam_year - birth_year).astype("Int64")

    lo, hi    = valid_range
    n_invalid = int(((age < lo) | (age > hi)).sum())
    if n_invalid:
        warnings.warn(
            f"add_student_age: {n_invalid} valor(es) de edad fuera de [{lo}, {hi}] "
            f"encontrados. Pase omit_age_invalid=True para reemplazarlos con pd.NA.",
            stacklevel=2,
        )

    if omit_age_invalid:
        out["estu_edad_prueba"] = age.where((age >= lo) & (age <= hi), other=pd.NA)
    else:
        out["estu_edad_prueba"] = age

    return out


def add_overage_flag(
    df: pd.DataFrame,
    *,
    col_age:      str = "estu_edad_prueba",
    expected_age: int = 17,
) -> pd.DataFrame:
    """
    Marca como ``True`` en ``estu_extraedad`` a estudiantes con edad superior
    a la esperada para grado 11 (por defecto 17 años).

    Requiere que ``estu_edad_prueba`` ya exista (ver :func:`add_student_age`).
    """
    _require_columns(df, [col_age], "add_overage_flag")
    out = df.copy()
    out["estu_extraedad"] = (out[col_age] > expected_age).where(
        out[col_age].notna(), other=pd.NA
    )
    return out


def enrich_all(
    df: pd.DataFrame,
    *,
    age_valid_range: tuple[int, int]      = (13, 30),
    expected_age:    int                  = 17,
) -> pd.DataFrame:
    """
    Aplica todos los pasos de enriquecimiento en el orden correcto.

    Pasos:
        1. add_period_columns    → anio_prueba, semestre_prueba
        2. add_student_age       → estu_edad_prueba
        3. add_overage_flag      → estu_extraedad
        4. add_ordinal_encodings → seis columnas *_num

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame limpio de Saber 11.
    age_valid_range : tuple, optional
        Rango válido de edades.  Default ``(13, 30)``.
    expected_age : int, optional
        Edad esperada para grado 11.  Default ``17``.

    Returns
    -------
    pd.DataFrame
        DataFrame con todas las columnas derivadas añadidas.
    """
    out = add_period_columns(df)
    out = add_student_age(out, valid_range=age_valid_range)
    out = add_overage_flag(out, expected_age=expected_age)
    out = add_ordinal_encodings(out)
    return out
