import pandas as pd
from typing import List, Sequence


_GRADE_MAPPING = {
    11: 'Grado 11',
    12: 'Grado 12',
    25: 'Ciclo 5',
    26: 'Ciclo 6'
}

_GENDER_MAPPING = {
    "F": "Femenino",
    "M": "Masculino"
}

_BOOL_MAPPING = {
    'S': True, 'SI': True,
    'N': False, 'NO': False
}

_ESTRATO_MAP: dict[str, int] = {
    "Sin Estrato": 0,
    "Estrato 1": 1, "Estrato 2": 2, "Estrato 3": 3,
    "Estrato 4": 4, "Estrato 5": 5, "Estrato 6": 6,
}

_EDUCATION_MAP: dict[str, int] = {
    "Ninguno":                                  0,
    "Primaria incompleta":                      1,
    "Primaria completa":                        2,
    "Secundaria (Bachillerato) incompleta":     3,
    "Secundaria (Bachillerato) completa":       4,
    "Técnica o tecnológica incompleta":         5,
    "Técnica o tecnológica completa":           6,
    "Educación profesional incompleta":         7,
    "Educación profesional completa":           8,
    "Postgrado":                                9,
    # Non-rankable values → pd.NA via map
    "No Aplica": None,
    "No sabe":   None,
}

_BOOKS_MAP: dict[str, int] = {
    "0 A 10 LIBROS":    1,
    "11 A 25 LIBROS":   2,
    "26 A 100 LIBROS":  3,
    "MÁS DE 100 LIBROS": 4,
}

_ROOMS_MAP: dict[str, int] = {
    "Uno":        1, "Dos":   2, "Tres":  3, "Cuatro": 4,
    "Cinco":      5, "Seis":  6, "Siete": 7, "Ocho":   8,
    "Nueve":      9, "Seis o mas": 6, "Diez o más": 10,
}

_PERSONS_MAP: dict[str, int] = {
    "Una":    1, "Dos":    2,  "Tres": 3,  "Cuatro": 4,
    "Cinco":  5, "Seis":   6,  "Siete": 7, "Ocho":   8,
    "Nueve":  9, "Diez":  10,  "Once": 11, "Doce o más": 12,
    "1 a 2":  1, "3 a 4":  3,  "5 a 6": 5, "7 a 8":  7, "9 o más": 9,
}

_COLUMNS_TO_DROP = [
    'cole_area_ubicacion',
    'cole_nombre_establecimiento',
    'cole_nombre_sede',
    'cole_caracter'
]


def _require_columns(df: pd.DataFrame, cols: Sequence[str], caller: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{caller}: columnas requeridas ausentes en el DataFrame: {missing}.\n"
            f"Columnas disponibles: {df.columns.tolist()}"
        )


def _apply_categorical_mappings(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    if 'estu_grado' in df_clean.columns:
        df_clean['estu_grado'] = df_clean['estu_grado'].map(_GRADE_MAPPING)
    if 'estu_genero' in df_clean.columns:
        df_clean['estu_genero'] = df_clean['estu_genero'].astype(str).str.upper().map(_GENDER_MAPPING)
    if 'cole_sede_principal' in df_clean.columns:
        df_clean['cole_sede_principal'] = df_clean['cole_sede_principal'].astype(str).str.upper().map(_BOOL_MAPPING)
    return df_clean


def _format_text_columns(
    df: pd.DataFrame,
    text_columns: list = ['cole_caracter', 'cole_jornada', 'cole_naturaleza', 'cole_genero']
) -> pd.DataFrame:
    df_clean = df.copy()
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.capitalize().str.strip()
    return df_clean


def _integrate_standardized_schools(df: pd.DataFrame, features_path: str) -> pd.DataFrame:
    df_clean = df.copy()
    cols_to_drop = [col for col in _COLUMNS_TO_DROP if col in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)
    try:
        features_school = pd.read_csv(features_path, dtype='str')
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo de estandarización no encontrado: {features_path}")
    return pd.merge(
        left=df_clean,
        right=features_school,
        on=['cole_cod_dane_establecimiento', 'cole_cod_dane_sede'],
        how='left'
    )


def _reorder_columns(df: pd.DataFrame, ordered_cols: List[str]) -> pd.DataFrame:
    if not ordered_cols:
        return df
    valid_cols = [col for col in ordered_cols if col in df.columns]
    extra_cols = [col for col in df.columns if col not in valid_cols]
    return df[valid_cols + extra_cols]


def add_ordinal_encodings(
    df: pd.DataFrame,
    *,
    col_estrato:    str = "fami_estratovivienda",
    col_educ_madre: str = "fami_educacionmadre",
    col_educ_padre: str = "fami_educacionpadre",
    col_libros:     str = "fami_numlibros",
    col_cuartos:    str = "fami_cuartoshogar",
    col_personas:   str = "fami_personashogar",
) -> pd.DataFrame:
    """
    Convierte variables categóricas del hogar a códigos enteros ordenados.

    Nuevas columnas (Int64, pd.NA donde el original es nulo o no mapeable):
        fami_estrato_num, fami_educ_madre_num, fami_educ_padre_num,
        fami_libros_num, fami_cuartos_num, fami_personas_num.

    Para educación, los valores 'No Aplica' y 'No sabe' se tratan como
    pd.NA, ya que no tienen posición clara en la escala ordinal.
    """
    required = [col_estrato, col_educ_madre, col_educ_padre,
                col_libros, col_cuartos, col_personas]
    _require_columns(df, required, "add_ordinal_encodings")

    out = df.copy()

    def _map(series: pd.Series, mapping: dict) -> pd.Series:
        return series.astype(object).map(mapping).astype("Int64")

    out["fami_estrato_num"]    = _map(out[col_estrato],    _ESTRATO_MAP)
    out["fami_educ_madre_num"] = _map(out[col_educ_madre], _EDUCATION_MAP)
    out["fami_educ_padre_num"] = _map(out[col_educ_padre], _EDUCATION_MAP)
    out["fami_libros_num"]     = _map(out[col_libros],     _BOOKS_MAP)
    out["fami_cuartos_num"]    = _map(out[col_cuartos],    _ROOMS_MAP)
    out["fami_personas_num"]   = _map(out[col_personas],   _PERSONS_MAP)

    return out


def cleaning_pipeline(
    df: pd.DataFrame,
    school_features_path: str,
    order_columns: list = None
) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de limpieza y estandarización de datos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame crudo cargado por el módulo loading.
    school_features_path : str
        Ruta al archivo CSV con los datos maestros de instituciones.
    order_columns : list, optional
        Lista con el orden deseado de columnas en el resultado.

    Returns
    -------
    pd.DataFrame
        DataFrame limpio y estandarizado.
    """
    df = _apply_categorical_mappings(df)
    df = _format_text_columns(df)
    df = _integrate_standardized_schools(df, school_features_path)
    df = _reorder_columns(df, order_columns)
    return df
