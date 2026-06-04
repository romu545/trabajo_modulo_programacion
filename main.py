# DEPRECADO: este archivo no es parte del flujo principal del proyecto.
# El análisis completo se ejecuta desde el reporte Quarto en reports/Final Report.qmd.
# Se conserva únicamente como referencia del pipeline de procesamiento.

import src.loading as ld
import src.cleaning as cl
import src.enrichement as en
import src.process as pr
from constants import ORDER_COLUMN


def main():
    result_saber_11 = ld.load_data()
    result_saber_11_clean = cl.cleaning_pipeline(
        result_saber_11,
        'data/raw/feature_standarizacion_school.csv',
        ORDER_COLUMN
    )
    result_saber_11_enriched = en.add_period_columns(
        result_saber_11_clean,
        col_periodo="periodo"
    )
    data_to_plot = pr.build_plot_data(
        result_saber_11_enriched,
        semester=2,
        calendar="A",
        location="Urbana",
        grade="Grado 11",
        nature="Oficial",
        exclude_shifts=["Sabatina", "Noche"],
        year_from=2015,
        year_to=2025,
        min_years=10,
        require_consecutive=True,
        n_highlight=3,
        by=["anio_prueba", "cole_cod_dane_establecimiento", "cole_nombre_establecimiento"],
        col_agg="punt_global"
    )


if __name__ == "__main__":
    main()
