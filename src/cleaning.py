import pandas as pd
from typing import List


# CONSTANTS AND MAPPING DICTIONARIES

GRADE_MAPPING = {
    11: 'Grado 11',
    12: 'Grado 12',
    25: 'Ciclo 5 - grado 10 adultos',
    26: 'Ciclo 6 - grado 10 adultos'
}

GENDER_MAPPING = {
    "F": "Femenino",
    "M": "Masculino"
}

BOOL_MAPPING = {
    'S': True,
    'SI': True,
    'N': False,
    'NO': False
}

## Columns that will be replaced by the standardized schools master file
COLUMNS_TO_DROP = [
    'cole_area_ubicacion',
    'cole_nombre_establecimiento',
    'cole_nombre_sede',
    'cole_caracter'
]


# HELPER FUNCTIONS

def _apply_categorical_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps categorical variables to standardized, readable values.
    
    Args:
        df (pd.DataFrame): Original DataFrame.
    Returns:
        pd.DataFrame: DataFrame with mapped variables.
    """
    df_clean = df.copy()
    
    if 'estu_grado' in df_clean.columns:
        df_clean['estu_grado'] = df_clean['estu_grado'].map(GRADE_MAPPING)
        
    if 'estu_genero' in df_clean.columns:
        df_clean['estu_genero'] = df_clean['estu_genero'].astype(str).str.upper().map(GENDER_MAPPING)
        
    if 'cole_sede_principal' in df_clean.columns:
        df_clean['cole_sede_principal'] = df_clean['cole_sede_principal'].astype(str).str.upper().map(BOOL_MAPPING)
        
    return df_clean


def _format_text_columns(df: pd.DataFrame, text_columns: list = ['cole_caracter', 'cole_jornada', 'cole_naturaleza', 'cole_genero']) -> pd.DataFrame:
    """
    Applies capitalization and format cleaning to free-text columns.
    
    Args:
        df (pd.DataFrame): DataFrame currently being processed.
    Returns:
        pd.DataFrame: DataFrame with clean text formatting.
    """
    df_clean = df.copy()
    
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.capitalize().str.strip()
            
    return df_clean



def _integrate_standardized_schools(df: pd.DataFrame, features_path: str) -> pd.DataFrame:
    """
    Safely drops messy school columns and replaces them by performing a LEFT JOIN 
    with the standardized school features master file.
    
    Args:
        df (pd.DataFrame): Current DataFrame.
        features_path (str): Path to the standardized schools CSV file.
    Returns:
        pd.DataFrame: DataFrame containing official school information.
    """
    df_clean = df.copy()
    
    # 1. Safely drop messy columns (only if they exist in the dataframe)
    cols_to_drop_safe = [col for col in COLUMNS_TO_DROP if col in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop_safe)
    
    # 2. Load the standardization master file
    
    try:
        features_school = pd.read_csv(features_path, dtype='str')
    except FileNotFoundError:
        print(f"The file {features_path} was not found. Please check the path.")
        raise
        
    # 3. Perform the merge (LEFT JOIN)
    df_merged = pd.merge(
        left=df_clean,
        right=features_school,
        on=['cole_cod_dane_establecimiento', 'cole_cod_dane_sede'],
        how='left'
    )
    
    return df_merged


def _reorder_columns(df: pd.DataFrame, ordered_cols: List[str]) -> pd.DataFrame:
    """
    Reorders the columns in the DataFrame based on a given list.
    Any extra columns present in the DataFrame will be appended at the end.
    """

    if not ordered_cols:
        return df

    # Keep only the columns that actually exist in the dataframe
    valid_cols = [col for col in ordered_cols if col in df.columns]
    
    # Append any extra columns not present in the sorting list to avoid data loss
    extra_cols = [col for col in df.columns if col not in valid_cols]
    
    return df[valid_cols + extra_cols]

# 3. MAIN PIPELINE

def cleaning_pipeline(df: pd.DataFrame, school_features_path: str, order_columns: list = None) -> pd.DataFrame:
    """
    Executes the complete data cleaning and standardization pipeline.
    
    Args:
        df (pd.DataFrame): Raw loaded DataFrame.
        school_features_path (str): Path to the schools master CSV file.
    Returns:
        pd.DataFrame: Fully cleaned and standardized DataFrame.
    """
    
    df = _apply_categorical_mappings(df)
    df = _format_text_columns(df)
    df = _integrate_standardized_schools(df, school_features_path)
    
    # Define desired column order
    df = _reorder_columns(df, order_columns)
    
    return df
