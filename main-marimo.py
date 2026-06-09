import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import src.loading_rafa as ld
    import numpy as np
    import statistics as st
    import matplotlib.pyplot as plt

    return ld, mo, np, plt, st


@app.cell
def _(ld):
    saber11 = ld.load_data()
    saber11
    return (saber11,)


@app.cell(hide_code=True)
def _(saber11):
    _integer_columns = {
        "estu_grado",
        "punt_c_naturales",
        "punt_global",
        "punt_ingles",
        "punt_lectura_critica",
        "punt_matematicas",
    }
    _date_columns = {"estu_fechanacimiento"}
    _string_identifier_columns = {
        "cole_cod_dane_establecimiento",
        "cole_cod_dane_sede",
        "estu_consecutivo",
    }
    _string_text_columns = {
        "cole_nombre_establecimiento",
        "cole_nombre_sede",
    }
    _categorical_columns = set(saber11.columns) - _integer_columns - _date_columns - _string_identifier_columns - _string_text_columns

    _schema_rows = []
    for _column in saber11.columns:
        _non_null_values = saber11[_column].dropna().astype(str)
        _categories = sorted(_non_null_values.unique().tolist()) if _column in _categorical_columns else None
        _schema_rows.append(
            {
                "columna": _column,
                "tipo_actual": str(saber11[_column].dtype),
                "tipo_recomendado": (
                    "fecha" if _column in _date_columns
                    else "entero" if _column in _integer_columns
                    else "string_identificador" if _column in _string_identifier_columns
                    else "string_texto" if _column in _string_text_columns
                    else "categorica"
                ),
                "nulos": int(saber11[_column].isna().sum()),
                "valores_unicos": int(saber11[_column].nunique(dropna=True)),
                "categorias": _categories,
            }
        )

    for _row in _schema_rows:
        _category_text = " | categorias: " + ", ".join(_row["categorias"]) if _row["categorias"] is not None else ""
        print(
            f"{_row['columna']}: {_row['tipo_recomendado']} "
            f"(nulos={_row['nulos']}, unicos={_row['valores_unicos']})"
            f"{_category_text}"
        )

    saber11_schema_sugerido = saber11.__class__(_schema_rows)
    saber11_schema_sugerido
    return


@app.cell(hide_code=True)
def _():
    _saber11_markdown_content = """# Análisis de tipos sugeridos para el dataframe `saber11`

    Dataset analizado: `saber11`  
    Dimensiones: 74.973 filas y 31 columnas.

    Actualmente todas las columnas están cargadas como `str`. A partir de los valores observados, se recomienda reclasificarlas en columnas enteras, fechas, strings identificadores, strings de texto y variables categóricas.

    ## Recomendación general

    ### Columnas enteras

    Estas columnas representan valores numéricos discretos y deberían convertirse a enteros, idealmente usando enteros nullable porque algunas tienen nulos.

    | Columna | Tipo recomendado | Comentario |
    |---|---|---|
    | `estu_grado` | entero | Grado escolar. Tiene 2 nulos. |
    | `punt_c_naturales` | entero | Puntaje de ciencias naturales. |
    | `punt_global` | entero | Puntaje global. |
    | `punt_ingles` | entero | Puntaje inglés. Tiene 302 nulos. |
    | `punt_lectura_critica` | entero | Puntaje lectura crítica. |
    | `punt_matematicas` | entero | Puntaje matemáticas. |

    ### Columnas de fecha

    | Columna | Tipo recomendado | Comentario |
    |---|---|---|
    | `estu_fechanacimiento` | fecha | Formato observado: día/mes/año, por ejemplo `05/10/1998`. |

    ### Columnas string / texto

    Estas columnas deben quedarse como texto porque son identificadores o nombres. Aunque algunas parecen números, no conviene convertirlas a entero porque no se usan para operaciones matemáticas y podrían tener ceros iniciales o códigos significativos.

    | Columna | Tipo recomendado | Comentario |
    |---|---|---|
    | `cole_cod_dane_establecimiento` | string identificador | Código DANE del establecimiento. |
    | `cole_cod_dane_sede` | string identificador | Código DANE de la sede. |
    | `estu_consecutivo` | string identificador | Identificador único del estudiante/examen. |
    | `cole_nombre_establecimiento` | string texto | Nombre del colegio. |
    | `cole_nombre_sede` | string texto | Nombre de la sede. |

    ## Columnas categóricas y sus categorías

    ### `cole_area_ubicacion`

    Categorías:

    ```text
    RURAL
    URBANA
    URBANO
    ```

    Nota: aquí hay una posible inconsistencia: `URBANA` y `URBANO` probablemente deberían homologarse.

    ### `cole_calendario`

    Categorías:

    ```text
    A
    B
    ```

    ### `cole_caracter`

    Categorías:

    ```text
    ACADÉMICO
    NO APLICA
    TÉCNICO
    TÉCNICO/ACADÉMICO
    ```

    Tiene 1.418 nulos.

    ### `cole_genero`

    Categorías:

    ```text
    FEMENINO
    MASCULINO
    MIXTO
    ```

    ### `cole_jornada`

    Categorías:

    ```text
    COMPLETA
    MAÑANA
    NOCHE
    SABATINA
    TARDE
    UNICA
    ```

    Tiene 4 nulos.

    ### `cole_naturaleza`

    Categorías:

    ```text
    NO OFICIAL
    OFICIAL
    ```

    ### `cole_sede_principal`

    Categorías:

    ```text
    N
    S
    S 
    ```

    Nota importante: aparece una categoría `S ` con espacio al final. Recomiendo limpiar espacios con `str.strip()` antes de convertirla a categórica.

    ### `estu_genero`

    Categorías:

    ```text
    F
    M
    ```

    Tiene 10 nulos.

    ### `fami_cuartoshogar`

    Categorías:

    ```text
    Cinco
    Cuatro
    Diez o más
    Dos
    Nueve
    Ocho
    Seis
    Seis o mas
    Siete
    Tres
    Uno
    ```

    Tiene 3.320 nulos.

    Nota: hay inconsistencia entre `Seis` y `Seis o mas`. También hay diferencia de acento entre `más` y `mas`.

    ### `fami_educacionmadre`

    Categorías:

    ```text
    Educación profesional completa
    Educación profesional incompleta
    Ninguno
    No Aplica
    No sabe
    Postgrado
    Primaria completa
    Primaria incompleta
    Secundaria (Bachillerato) completa
    Secundaria (Bachillerato) incompleta
    Técnica o tecnológica completa
    Técnica o tecnológica incompleta
    ```

    Tiene 4.792 nulos.

    ### `fami_educacionpadre`

    Categorías:

    ```text
    Educación profesional completa
    Educación profesional incompleta
    Ninguno
    No Aplica
    No sabe
    Postgrado
    Primaria completa
    Primaria incompleta
    Secundaria (Bachillerato) completa
    Secundaria (Bachillerato) incompleta
    Técnica o tecnológica completa
    Técnica o tecnológica incompleta
    ```

    Tiene 4.736 nulos.

    ### `fami_estratovivienda`

    Categorías:

    ```text
    Estrato 1
    Estrato 2
    Estrato 3
    Estrato 4
    Estrato 5
    Estrato 6
    Sin Estrato
    ```

    Tiene 4.363 nulos.

    Nota: esta podría convertirse también a una variable ordinal si se transforma `Estrato 1` a `1`, etc., dejando `Sin Estrato` como categoría especial.

    ### `fami_numlibros`

    Categorías:

    ```text
    0 A 10 LIBROS
    11 A 25 LIBROS
    26 A 100 LIBROS
    MÁS DE 100 LIBROS
    ```

    Tiene 6.077 nulos.

    Es categórica ordinal.

    ### `fami_personashogar`

    Categorías:

    ```text
    1 a 2
    3 a 4
    5 a 6
    7 a 8
    9 o más
    Cinco
    Cuatro
    Diez
    Doce o más
    Dos
    Nueve
    Ocho
    Once
    Seis
    Siete
    Tres
    Una
    ```

    Tiene 3.207 nulos.

    Nota importante: esta columna mezcla dos formatos: rangos como `1 a 2`, `3 a 4` y números escritos como `Cinco`, `Cuatro`, etc. Recomiendo limpiarla antes de modelar.

    ### `fami_tieneautomovil`

    Categorías:

    ```text
    No
    Si
    ```

    Tiene 3.487 nulos.

    ### `fami_tienecomputador`

    Categorías:

    ```text
    No
    Si
    ```

    Tiene 3.378 nulos.

    ### `fami_tieneinternet`

    Categorías:

    ```text
    No
    Si
    ```

    Tiene 4.816 nulos.

    ### `fami_tieneserviciotv`

    Categorías:

    ```text
    No
    Si
    ```

    Tiene 5.001 nulos.

    ### `periodo`

    Categorías:

    ```text
    20151
    20152
    20161
    20162
    20171
    20172
    20181
    20182
    20191
    20192
    20201
    20202
    20211
    20212
    20221
    20222
    20231
    20232
    20241
    20242
    20251
    20252
    ```

    Aunque parece numérica, recomiendo tratarla como categórica o como periodo, no como entero simple. Representa año y semestre/calendario, por ejemplo `20162`.

    ## Resumen por tipo

    ### Enteras

    ```text
    estu_grado
    punt_c_naturales
    punt_global
    punt_ingles
    punt_lectura_critica
    punt_matematicas
    ```

    ### Fecha

    ```text
    estu_fechanacimiento
    ```

    ### String identificador

    ```text
    cole_cod_dane_establecimiento
    cole_cod_dane_sede
    estu_consecutivo
    ```

    ### String texto

    ```text
    cole_nombre_establecimiento
    cole_nombre_sede
    ```

    ### Categóricas

    ```text
    cole_area_ubicacion
    cole_calendario
    cole_caracter
    cole_genero
    cole_jornada
    cole_naturaleza
    cole_sede_principal
    estu_genero
    fami_cuartoshogar
    fami_educacionmadre
    fami_educacionpadre
    fami_estratovivienda
    fami_numlibros
    fami_personashogar
    fami_tieneautomovil
    fami_tienecomputador
    fami_tieneinternet
    fami_tieneserviciotv
    periodo
    ```

    ## Observaciones de limpieza importantes

    Antes de convertir tipos, recomiendo:

    1. Aplicar `strip()` a todas las columnas string para quitar espacios al inicio/final.
    2. Homologar `URBANA` y `URBANO`.
    3. Homologar `S` y `S ` en `cole_sede_principal`.
    4. Revisar `fami_cuartoshogar`, porque mezcla `Seis` y `Seis o mas`.
    5. Revisar `fami_personashogar`, porque mezcla rangos y números escritos.
    6. Convertir `estu_fechanacimiento` con formato día/mes/año.
    7. Convertir puntajes a enteros nullable para soportar nulos en `punt_ingles`.
    8. Mantener códigos DANE como string, no como entero.
    """

    _saber11_markdown_path = "saber11_analisis_tipos.md"
    with open(_saber11_markdown_path, "w", encoding="utf-8") as _markdown_file:
        _markdown_file.write(_saber11_markdown_content)

    print(f"Archivo Markdown exportado: {_saber11_markdown_path}")
    return


@app.cell
def _(saber11, st):
    saber11_pre2022 = saber11[(saber11['periodo'] < 2022_0) & (saber11['periodo'] > 2019_0)]
    saber11_post2022 = saber11[saber11['periodo'] >= 2022_0]

    print(f'size: <2022={len(saber11_pre2022.punt_global)} | >=2022={len(saber11_post2022.punt_global)}')
    print(f'means: <2022={st.mean(saber11_pre2022.punt_global)} | >=2022={st.mean(saber11_post2022.punt_global)}')
    print(f'sd: <2022={st.stdev(saber11_pre2022.punt_global.to_list())} | >=2022={st.stdev(saber11_post2022.punt_global.to_list())}')
    return


@app.cell
def _(np, plt, saber11, st):
    X_per = saber11.periodo.unique().astype(int)
    X_per.sort()
    _X_periodo1 = X_per[X_per % 10 == 1]
    _X_periodo2 = X_per[X_per % 10 == 2]

    def Y_means(x):
        return st.mean(saber11[saber11.periodo == x]['punt_global'])

    def Y_std(x):
        return st.stdev(saber11[saber11.periodo == x]['punt_global'].to_list())


    mean_func = np.vectorize(Y_means)
    std_func = np.vectorize(Y_std)

    Y1 = mean_func(_X_periodo1)
    Y2 = mean_func(_X_periodo2)

    Ys1 = std_func(_X_periodo1) * .2
    Ys2 = std_func(_X_periodo2) * .2

    # fig, ax = plt.subplot(3, 3)
    plt.fill_between(_X_periodo1, Y1 - Ys1, Y1 + Ys1, color="#aaaaaa20")
    plt.fill_between(_X_periodo2, Y2 - Ys2, Y2 + Ys2, color="#aaaaaa20")
    plt.plot(_X_periodo2, Y2)
    plt.plot(_X_periodo1, Y1)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Separación de Datos
    Trabajamos solamente con uno de los periodos al tiempo porque la diferencia entre ambos grupos es bastante significativa, primero separamos. Vamos a separar los datos en periodos y vamos a evaluar el rendimiento de los icfes de las instituciones oficiales y no-oficiales.
    """)
    return


@app.cell
def _(plt, saber11):
    saber11['cohorte_periodo'] = saber11['periodo'] % 10
    vals = saber11.groupby(by=["cole_naturaleza", "periodo"]).agg({ 'punt_global': 'mean' })
    plt.plot(vals['periodo'], vals['punt_global'])
    plt.show()
    return


if __name__ == "__main__":
    app.run()
