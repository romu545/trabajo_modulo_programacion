# Análisis de Resultados Saber 11 – Magdalena (2005–2025)

Proyecto desarrollado para la especialización en Ciencia de Datos. Analiza la evolución del puntaje global en las Pruebas Saber 11 de las instituciones educativas del departamento de Magdalena, con foco en el periodo 2015–2025.

**Autores:** Rafael · Camilo · Sheimy

---

## Descripción

El proyecto responde tres preguntas:

1. ¿Qué tendencias se observan en los puntajes promedio de las instituciones educativas urbanas de calendario A entre 2015 y 2025?
2. ¿Cómo varían los puntajes según la naturaleza jurídica (oficial vs. no oficial)?
3. ¿Qué relación existe entre las condiciones socioeconómicas del hogar y el desempeño en la prueba?

Los datos provienen del portal de datos abiertos del ICFES (resultados por aplicación, nivel establecimiento e individuo).

---

## Estructura del proyecto

```
.
├── data/
│   ├── raw/
│   │   ├── magdalena_parquet/       # Archivos fuente por periodo (YYYYS.parquet)
│   │   ├── columns_in_data.csv      # Diccionario de variables seleccionadas
│   │   └── feature_standarizacion_school.csv  # Maestro de instituciones
│   └── processed/
│       └── result_saber_11_clean.parquet  # Dataset procesado (generado)
├── reports/
│   ├── Final Report.qmd             # Reporte principal (Quarto)
│   ├── Final Report.html            # Versión renderizada
│   └── Reference.bib                # Referencias bibliográficas
├── src/
│   ├── loading.py      # Carga y validación de archivos parquet
│   ├── cleaning.py     # Limpieza, estandarización y codificaciones ordinales
│   ├── enrichement.py  # Variables derivadas (edad, periodo, extraedad)
│   ├── process.py      # Filtrado, agregación y preparación para gráficas
│   ├── viz.py          # Visualizaciones (matplotlib/seaborn)
│   └── analysis.py     # Estadísticas descriptivas
├── constants.py        # Orden estándar de columnas
├── main.py             # Script de pipeline completo (deprecado, ver nota)
└── README.md
```

> **Nota:** `main.py` se conserva como referencia del pipeline de procesamiento, pero el flujo principal del proyecto corre desde el reporte Quarto (`reports/Final Report.qmd`).

---

## Requisitos

- Python ≥ 3.11
- Quarto ≥ 1.4
- Dependencias Python: `pandas`, `pyarrow`, `matplotlib`, `seaborn`, `numpy`

Instalación con `uv` (recomendado, el archivo `uv.lock` está incluido):

```bash
uv sync
```

O con pip:

```bash
pip install pandas pyarrow matplotlib seaborn numpy
```

---

## Cómo reproducir el reporte

Con el entorno activo, desde la raíz del proyecto:

```bash
quarto render "reports/Final Report.qmd"
```

El archivo `reports/Final Report.html` se actualizará con los resultados. Quarto ejecuta los bloques de código Python, lee el parquet procesado en `data/processed/` y genera todas las tablas y gráficas del informe.

---

## Datos

Los archivos en `data/raw/magdalena_parquet/` **no se incluyen en el repositorio** por su tamaño. Deben descargarse del portal del ICFES y ubicarse en esa carpeta respetando el nombre `Resultado_Saber_11_YYYYS.parquet` (p. ej. `Resultado_Saber_11_20152.parquet`).

El archivo `data/processed/result_saber_11_clean.parquet` es generado por el pipeline de limpieza y sí está incluido para permitir reproducir el reporte sin necesidad de reprocesar desde los datos crudos.
