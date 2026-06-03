library(tidyverse)
library(duckdb)

con <- dbConnect(duckdb())

df <- dbGetQuery(
  con, "SELECT * FROM read_parquet('C:/Users/romu5/Documents/GitHub/trabajo_modulo_programacion/data/processed/result_saber_11_clean.parquet')")

datos_plot <- 
  df |> 
  subset(
    cole_calendario == "A" &
      cole_area_ubicacion == "Urbana" & 
      cole_naturaleza == "No oficial",
    c("periodo", "cole_cod_dane_sede", "cole_nombre_sede", "punt_global")
  ) |> 
  mutate(
    periodo = as.integer(str_sub(periodo, end = 4))
  ) |> 
  summarise(
    .by = c("periodo", "cole_cod_dane_sede", "cole_nombre_sede"),
    promedio = mean(punt_global)
  ) |> 
  filter(
    .by = "cole_cod_dane_sede",
    n() == 11
  ) 

datos_plot |>
  ggplot(
    aes(
      periodo,
      promedio
    )
  ) +
  geom_line(
    aes(colour = cole_cod_dane_sede),
    show.legend = FALSE
    ) +
  scale_x_continuous(
    n.breaks = 10
  )

datos_plot |> 
  count(
    cole_cod_dane_sede
  )
