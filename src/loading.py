import re
import os
from pathlib import Path
import pandas as pd


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


def select_files_by_period(list_files: list, _from: int = None, _to: int = None) -> list:
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