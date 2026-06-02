# pyright: reportGeneralTypeIssues=false
import re
import os
from pathlib import Path
import pandas as pd
import pyarrow


def _get_basename(path: str) -> str:
    """
    Extract the basename (filename) from a given path.

    Handles both absolute and relative paths, as well as
    different operating system path separators (Windows and Unix).

    Parameters
    ----------
    path : str
        File path, either absolute or relative.

    Returns
    -------
    str
        The filename extracted from the path.

    Examples
    --------
    >>> _get_basename('data/raw/Resultado_Saber_11_20151.parquet')
    'Resultado_Saber_11_20151.parquet'

    >>> _get_basename(r'C:\\Users\\romu5\\Documents\\Resultado_Saber_11_20151.parquet')
    'Resultado_Saber_11_20151.parquet'
    """
    return os.path.basename(path)


def _validate_names_files(path: str = '.', _ext: str = 'parquet') -> list:
    """
    List all files with a given extension in the specified directory.

    Parameters
    ----------
    path : str, optional
        Directory path to search. Defaults to current working directory.
    _ext : str, optional
        File extension to filter by, without the dot. Defaults to 'parquet'.

    Returns
    -------
    list
        List of full file paths matching the given extension.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist or is not a directory.

    Examples
    --------
    >>> _validate_names_files('data/raw/magdalena_parquet/')
    ['data/raw/magdalena_parquet/Resultado_Saber_11_20151.parquet', ...]

    >>> _validate_names_files('data/raw/magdalena_parquet/', _ext='csv')
    []
    """
    folder = Path(path)

    if not folder.is_dir():
        raise FileNotFoundError(f"Path not found or is not a directory: {path}")

    return [str(f) for f in folder.iterdir() if f.is_file() and f.suffix == f".{_ext}"]


def _validate_filename_structure(path: str) -> bool:
    """
    Validate that a filename follows the expected Saber 11 structure.

    The expected filename structure is:
        Resultado_Saber_11_YYYYS.parquet
    where YYYY is the year and S is the semester (1 or 2).

    Parameters
    ----------
    path : str
        File path, either absolute or relative.

    Returns
    -------
    bool
        True if the filename matches the expected structure, False otherwise.

    Examples
    --------
    >>> _validate_filename_structure('data/raw/Resultado_Saber_11_20151.parquet')
    True

    >>> _validate_filename_structure('data/raw/Resultado_Saber_11_20153.parquet')
    False

    >>> _validate_filename_structure(r'C:\\Users\\romu5\\data\\Resultado_Saber_11_20151.parquet')
    True
    """
    VALID_STRUCTURE = re.compile(r"^Resultado_Saber_11_(\d{4})([12])\.parquet$")
    basename = _get_basename(path)
    return bool(VALID_STRUCTURE.match(basename))


def select_files_by_period(list_files: list, _from: int | None = None, _to: int | None = None) -> list[str]:
    """
    Filter a list of Saber 11 parquet files by period.

    The period is encoded in the filename as a 5-digit integer (YYYYS),
    where YYYY is the year and S is the semester (1 or 2).
    Example: 'Resultado_Saber_11_20151.parquet' → period 20151 (2015, semester 1)

    Invalid filenames are skipped with a warning.

    Parameters
    ----------
    list_files : list
        List of parquet filenames or full paths to filter.
    _from : int, optional
        Start period (inclusive) in YYYYS format (e.g. 20151).
        If None, no lower bound is applied.
    _to : int, optional
        End period (inclusive) in YYYYS format (e.g. 20242).
        If None, no upper bound is applied.

    Returns
    -------
    list
        Filtered list of filenames within the specified period range.

    Examples
    --------
    >>> select_files_by_period(list_files)
    ['Resultado_Saber_11_20051.parquet', ..., 'Resultado_Saber_11_20252.parquet']

    >>> select_files_by_period(list_files, _from=20151, _to=20242)
    ['Resultado_Saber_11_20151.parquet', ..., 'Resultado_Saber_11_20242.parquet']

    >>> select_files_by_period(list_files, _from=20201)
    ['Resultado_Saber_11_20201.parquet', ..., 'Resultado_Saber_11_20252.parquet']
    """
    files = []
    for file in list_files:

        if not _validate_filename_structure(file):
            print(f"[WARNING] Skipping invalid filename: '{_get_basename(file)}'")
            continue

        period = int(re.search(r"(\d{5})\.parquet", file).group(1))

        from_ = _from if _from is not None else 0
        to_   = _to   if _to   is not None else 99999

        if from_ <= period <= to_:
            files.append(file)

    return files


# Maps the `tipo` tokens stored in columns_in_data.csv to the way each column
# should be converted. `fecha` is handled separately because the source format
# is day/month/year and needs an explicit parser.
_SIMPLE_DTYPES = {
    "entera": "Int64",        # nullable integer, supports nulls (e.g. punt_ingles)
    "string_id": "string",    # codes kept as text (DANE, consecutivo)
    "string_texto": "string", # free text names
    "categorica": "category",
}


def _selected_columns(file: str = "data/raw/columns_in_data.csv") -> pd.DataFrame:
    """Return the rows of the columns metadata that are flagged for selection."""
    col_sel = pd.read_csv(file)
    return col_sel[col_sel['seleccionada'] == 'si']


def _columns_to_select(file="data/raw/columns_in_data.csv") -> list[str]:
    return _selected_columns(file)['columna'].to_list()  # pyright: ignore


def _apply_dtypes(df: pd.DataFrame, file: str = "data/raw/columns_in_data.csv") -> pd.DataFrame:
    """
    Cast the dataframe columns to the types declared in the `tipo` column of
    the columns metadata CSV.

    Conversions applied per `tipo`:
    - ``entera``       -> nullable integer (``Int64``)
    - ``fecha``        -> datetime parsed as day/month/year
    - ``string_id`` / ``string_texto`` -> stripped ``string``
    - ``categorica``   -> stripped ``category``

    Columns without a declared ``tipo`` (or absent from the dataframe) are left
    untouched. Unparseable values become nulls instead of raising.
    """
    meta = _selected_columns(file)

    for column, tipo in zip(meta['columna'], meta['tipo']):
        if column not in df.columns or not isinstance(tipo, str) or not tipo.strip():
            continue
        tipo = tipo.strip()

        if tipo == "entera":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
        elif tipo == "fecha":
            df[column] = pd.to_datetime(df[column], dayfirst=True, errors="coerce")
        elif tipo == "categorica":
            df[column] = df[column].astype("string").str.strip().astype("category")
        elif tipo in _SIMPLE_DTYPES:
            df[column] = df[column].astype("string").str.strip()

    return df


def load_data(path='data/raw/magdalena_parquet/', starting_period=2015_1) -> pd.DataFrame:
    df = pd.read_parquet(
        select_files_by_period(
            _validate_names_files(path),
            _from=starting_period
        ),
        columns=_columns_to_select()
    )
    return _apply_dtypes(df)

# ['/home/camilo/PyCharmProjects/trabajo_modulo_programacion/data/raw/magdalena_parquet/Resultado_Saber_11_20051.parquet']
# print(select_files_by_period(
#     _validate_names_files('/home/camilo/PyCharmProjects/trabajo_modulo_programacion/data/raw/magdalena_parquet/'),
#     _from=2015_1
# ))
