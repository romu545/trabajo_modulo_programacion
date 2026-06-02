import pandas as pd
import numpy as np

DATA_FOLDER = "data"

FILENAMES = [
    "Examen_Saber_11_20151.txt", "Examen_Saber_11_20152.txt",
    "Examen_Saber_11_20161.txt", "Examen_Saber_11_20162.txt",
    "Examen_Saber_11_20171.txt", "Examen_Saber_11_20172.txt",
    "Examen_Saber_11_20181.txt", "Examen_Saber_11_20182.txt",
    "Examen_Saber_11_20191.txt", "Examen_Saber_11_20192.txt",
    "Examen_Saber_11_20201.txt", "Examen_Saber_11_20202.txt",
    "Examen_Saber_11_20211.txt", "Examen_Saber_11_20212.txt",
    "Examen_Saber_11_20221.txt", "Examen_Saber_11_20222.txt",
    "Examen_Saber_11_20231.txt", "Examen_Saber_11_20232.txt",
    "Examen_Saber_11_20241.txt", "Examen_Saber_11_20242.txt",
    "Examen_Saber_11_20251.txt", "Examen_Saber_11_20252.txt",
]

MISSING_VALUES = [
    "", "NA", "N/A", "nan", "NaN",
    "Unknown", "not recorded", "null", "NULL", "-"
]

TEXT_COLUMNS = {
    "estu_consecutivo":              "string",
    "estu_estudiante":               "string",
    "estu_tipodocumento":            "string",
    "cole_area_ubicacion":           "string",
    "cole_bilingue":                 "string",
    "cole_calendario":               "string",
    "cole_caracter":                 "string",
    "cole_cod_dane_establecimiento": "string",
    "cole_cod_dane_sede":            "string",
    "cole_depto_ubicacion":          "string",
    "cole_genero":                   "string",
    "cole_jornada":                  "string",
    "cole_mcpio_ubicacion":          "string",
    "cole_naturaleza":               "string",
    "cole_nombre_establecimiento":   "string",
    "cole_nombre_sede":              "string",
    "cole_sede_principal":           "string",
    "estu_depto_presentacion":       "string",
    "estu_depto_reside":             "string",
    "estu_discapacidad":             "string",
    "estu_etnia":                    "string",
    "estu_genero":                   "string",
    "estu_mcpio_presentacion":       "string",
    "estu_mcpio_reside":             "string",
    "estu_nacionalidad":             "string",
    "estu_pais_reside":              "string",
    "estu_privado_libertad":         "string",
    "estu_repite":                   "string",
    "estu_tieneetnia":               "string",
    "estu_tiporemuneracion":         "string",
    "estu_dedicacioninternet":       "string",
    "estu_dedicacionlecturadiaria":  "string",
    "estu_agregado":                 "string",
    "fami_comecarnepescadohuevo":    "string",
    "fami_comecerealfrutoslegumbre": "string",
    "fami_comelechederivados":       "string",
    "fami_educacionmadre":           "string",
    "fami_educacionpadre":           "string",
    "fami_estratovivienda":          "string",
    "fami_numlibros":                "string",
    "fami_situacioneconomica":       "string",
    "fami_tieneautomovil":           "string",
    "fami_tienecomputador":          "string",
    "fami_tieneconsolavideojuegos":  "string",
    "fami_tienehornomicroogas":      "string",
    "fami_tieneinternet":            "string",
    "fami_tienelavadora":            "string",
    "fami_tienemotocicleta":         "string",
    "fami_tieneserviciotv":          "string",
    "fami_trabajolabormadre":        "string",
    "fami_trabajolaborpadre":        "string",
    "desemp_c_naturales":            "string",
    "desemp_ingles":                 "string",
    "desemp_lectura_critica":        "string",
    "desemp_matematicas":            "string",
    "desemp_sociales_ciudadanas":    "string",
}

DATE_COLUMNS = ["estu_fechanacimiento"]


GENERO_MAPPING = {
    "F": "F", "FEMENINO": "F", "FEMALE": "F", "MUJER": "F",
    "M": "M", "MASCULINO": "M", "MALE": "M", "HOMBRE": "M",
}

AREA_UBICACION_MAPPING = {
    "URBANO": "URBANO", "URBAN": "URBANO", "U": "URBANO",
    "RURAL": "RURAL", "R": "RURAL",
}

NATURALEZA_MAPPING = {
    "OFICIAL": "OFICIAL",
    "NO OFICIAL": "NO OFICIAL", "NOOFICIAL": "NO OFICIAL", "PRIVADO": "NO OFICIAL",
}

CALENDARIO_MAPPING = {
    "A": "A", "CALENDARIO A": "A",
    "B": "B", "CALENDARIO B": "B",
}

BILINGUE_MAPPING = {
    "S": "S", "SI": "S", "YES": "S", "Y": "S",
    "N": "N", "NO": "N",
}

ESTRATO_MAPPING = {
    "ESTRATO 1": "1", "1": "1",
    "ESTRATO 2": "2", "2": "2",
    "ESTRATO 3": "3", "3": "3",
    "ESTRATO 4": "4", "4": "4",
    "ESTRATO 5": "5", "5": "5",
    "ESTRATO 6": "6", "6": "6",
}


# CARGA Y CONCATENACIÓN DE ARCHIVOS

frames = []

for filename in FILENAMES:
    filepath = DATA_FOLDER + "/" + filename

    saber = pd.read_csv(
        filepath,
        sep=";",
        dtype=TEXT_COLUMNS,
        na_values=MISSING_VALUES,
        parse_dates=DATE_COLUMNS,
    )

    # Extraer periodo, año y semestre desde el nombre del archivo
    periodo_str = filename.replace("Examen_Saber_11_", "").replace(".txt", "")
    saber["periodo"]   = periodo_str
    saber["anio"]      = int(periodo_str[:-1])
    saber["semestre"]  = int(periodo_str[-1])

    frames.append(saber)

saber11 = pd.concat(frames, ignore_index=True)
print(saber11.shape)


# ESTANDARIZACIÓN DE CATEGORÍAS por si acaso

saber11["estu_genero"] = saber11["estu_genero"].str.strip().str.upper().replace(GENERO_MAPPING)
saber11["estu_genero"].value_counts(dropna=False)

saber11["cole_area_ubicacion"] = saber11["cole_area_ubicacion"].str.strip().str.upper().replace(AREA_UBICACION_MAPPING)
saber11["cole_area_ubicacion"].value_counts(dropna=False)

saber11["cole_naturaleza"] = saber11["cole_naturaleza"].str.strip().str.upper().replace(NATURALEZA_MAPPING)
saber11["cole_naturaleza"].value_counts(dropna=False)

saber11["cole_calendario"] = saber11["cole_calendario"].str.strip().str.upper().replace(CALENDARIO_MAPPING)
saber11["cole_calendario"].value_counts(dropna=False)

saber11["cole_bilingue"] = saber11["cole_bilingue"].str.strip().str.upper().replace(BILINGUE_MAPPING)
saber11["cole_bilingue"].value_counts(dropna=False)

saber11["fami_estratovivienda"] = saber11["fami_estratovivienda"].str.strip().str.upper().replace(ESTRATO_MAPPING)
saber11["fami_estratovivienda"].value_counts(dropna=False)

print(saber11.shape)
