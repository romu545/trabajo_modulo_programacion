# pyright: reportGeneralTypeIssues=false
import re
import os
from pathlib import Path
import pandas as pd


def _get_basename(path: str) -> str:
    """
    Extract the basename (filename) from a given path.
    Handles both absolute and relative paths, as well as
    different operating system path separators (Windows and Unix).
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
    "booleano": "bool"
}

def _selected_columns(
    metadata_file: str = "data/raw/columns_in_data.csv",
    attributes: list[str] | None = None,
    select_all: bool = False,
) -> pd.DataFrame:
    """
    Return metadata rows from the columns registry.

    Reads the columns metadata CSV.  By default filters to rows where
    ``seleccionada == 'si'``.  Optionally restricts the returned columns
    to a subset of fields in the registry.

    Parameters
    ----------
    file : str, optional
        Path to the columns metadata CSV.
        Default ``"data/raw/columns_in_data.csv"``.
    attributes : list of str, optional
        Column names from the registry to include in the result
        (e.g. ``["columna", "tipo"]``).  When ``None`` all columns
        of the registry are returned.
    select_all : bool, optional
        When ``True`` returns all rows regardless of the ``seleccionada``
        flag.  Default ``False``.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only the selected rows and, if requested,
        only the specified attributes.

    Raises
    ------
    ValueError
        If any name in *attributes* is not present in the registry.

    Examples
    --------
    >>> _selected_columns()                           # flagged rows, all attributes
    >>> _selected_columns(select_all=True)            # all rows, all attributes
    >>> _selected_columns(attributes=["columna"])     # flagged rows, one attribute
    >>> _selected_columns(attributes=["columna", "tipo"], select_all=True)
    """
    registry = pd.read_csv(metadata_file)
    attributes_default = ["columna", "tipo"]
    selected = registry[attributes_default] if select_all else registry.loc[registry["seleccionada"] == "si", attributes_default]

    if attributes is not None:
        missing = [a for a in attributes if a not in attributes_default]
        if missing:
            raise ValueError(
                f"_selected_columns: attributes {missing} not found in registry. "
                f"Available: {attributes_default}"
            )
        
        if len(attributes) > 1:
            raise ValueError(
                f"Must selected a attribute between {attributes_default}"
            )
    
        return selected[attributes[0]].to_list()

    return selected.reset_index(drop=True)


def _apply_dtypes(
    df: pd.DataFrame,
    metadata_file: str = "data/raw/columns_in_data.csv") -> pd.DataFrame:
    """
    Cast the dataframe columns to the types declared in the `tipo` column of
    the columns metadata CSV.

    Conversions applied per `tipo`:
    - ``entera``       -> nullable integer (``Int64``)
    - ``fecha``        -> datetime parsed as day/month/year
    - ``string_id`` / ``string_texto`` -> stripped ``string``
    - ``categorica``   -> stripped ``category``
    - ``boolenao``  -> ``bool``

    Columns without a declared ``tipo`` (or absent from the dataframe) are left
    untouched. Unparseable values become nulls instead of raising.
    """
    meta = _selected_columns(metadata_file)

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
        elif tipo == "booleano":
            df[column] = df[column].astype("string").str.strip().map({"Si": True, "No": False})
        elif tipo in _SIMPLE_DTYPES:
            df[column] = df[column].astype("string").str.strip()

    return df


def load_data(
    path='data/raw/magdalena_parquet/', 
    starting_period=2015_1,
    _columns: list = _selected_columns(attributes = ['columna'])
    ) -> pd.DataFrame:
    
    df = pd.read_parquet(
        select_files_by_period(
            _validate_names_files(path),
            _from=starting_period
        ),
        columns=_columns
    )
    return _apply_dtypes(df)

