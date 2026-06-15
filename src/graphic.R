library(tidyverse)
library(duckdb)
library(gghighlight)

# ── Datos ──────────────────────────────────────────────────────────────────────
con <- dbConnect(duckdb())

df <- dbGetQuery(
  con,
  "SELECT * FROM read_parquet('C:/Users/romu5/Documents/GitHub/trabajo_modulo_programacion/data/processed/result_saber_11_clean.parquet')"
)

dbDisconnect(con)

# ── Pipeline ───────────────────────────────────────────────────────────────────
df_plot <- df |>
  filter(
    str_sub(periodo, 5) == 2,
    cole_calendario        == "A",
    cole_area_ubicacion    == "Urbana",
    estu_grado             %in% c("Grado 11"),
    cole_naturaleza        == "Oficial",
    cole_jornada           %notin% c("Sabatina", "Noche")
  ) |>
  summarise(
    .by     = c("periodo", "cole_cod_dane_establecimiento", "cole_nombre_establecimiento"),
    recuento = n(),
    promedio = mean(punt_global)
  ) |>
  add_count(cole_nombre_establecimiento, name = "nro_anios_presentados") |>
  filter(nro_anios_presentados >= 10) |>
  mutate(
    .by  = "cole_cod_dane_establecimiento",
    anios = all(2016:2025 %in% as.integer(str_sub(periodo, end = 4)))
  ) |>
  filter(anios)

# ── Ranking top/bottom ─────────────────────────────────────────────────────────
n_resaltar <- 3

ranking <- df_plot |>
  summarise(
    .by           = cole_cod_dane_establecimiento,
    promedio_global = weighted.mean(promedio, recuento)
  ) |>
  mutate(
    grupo = case_when(
      rank(-promedio_global, ties.method = "first") <= n_resaltar ~ "top",
      rank( promedio_global, ties.method = "first") <= n_resaltar ~ "bottom",
      .default = "medio"
    )
  )

df_plot <- df_plot |>
  left_join(ranking, by = "cole_cod_dane_establecimiento")

# ── Promedio ponderado general (precalculado para evitar conflicto con gghighlight) ──
promedio_pond <- df_plot |>
  summarise(
    .by      = periodo,
    p25      = quantile(promedio, 0.25),
    p75      = quantile(promedio, 0.75),
    p10      = quantile(promedio, 0.10),
    p90      = quantile(promedio, 0.90),
    promedio = weighted.mean(promedio, recuento)
  ) |>
  mutate(año = factor(str_sub(periodo, end = 4)))


# ── Etiquetas finales ──────────────────────────────────────────────────────────
etiquetas <- df_plot |>
  filter(
    grupo.y %in% c("top", "bottom"),
    str_sub(periodo, end = 4) == "2025"
  )

# ── Gráfico ────────────────────────────────────────────────────────────────────
df_plot |>
  ggplot(
    aes(
      x      = factor(str_sub(periodo, end = 4)),
      y      = promedio,
      colour = cole_cod_dane_establecimiento,
      group  = cole_cod_dane_establecimiento
  )) +
  geom_line(linewidth = 0.4) +
  gghighlight(
    grupo.y %in% c("top", "bottom"),
    unhighlighted_params = list(colour = "grey80", linewidth = 0.3, alpha = 0.5),
    use_direct_label = FALSE
  ) +
  geom_ribbon(
    data        = promedio_pond,
    aes(x = año, ymin = p25, ymax = p75, group = 1),
    fill        = "blue",
    alpha       = 0.20,
    inherit.aes = FALSE
  ) +
  geom_line(
    data        = promedio_pond,
    aes(x = año, y = promedio, group = 1),
    colour      = "black",
    linewidth   = 1.1,
    inherit.aes = FALSE
  ) +
  ggrepel::geom_text_repel(
    data        = etiquetas,
    aes(label   = cole_nombre_establecimiento),
    nudge_x     = 0.1,
    direction   = "y",
    segment.size = 0,
    size        = 3,
    show.legend = FALSE,
    hjust        = "left",
    xlim    = c(11, NA)
  ) +
  scale_colour_manual(values = scales::hue_pal()(n_resaltar * 2)) +
  scale_x_discrete(
    limits = as.character(2015:2025),
    expand = expansion(add = c(0.2, 4))
    ) +
  scale_y_continuous(limits = c(150, 350), n.breaks = 11) +
  labs(
    title    = "EVOLUCIÓN DEL PUNTAJE GLOBAL EN LA PRUEBAS SABER 11",
    subtitle = "Datos de instituciones oficiales urbanas del calendario A con datos en 2015–2025",
    x        = "Año",
    y        = "Promedio puntaje global",
    caption  = "Línea negra: promedio ponderado general\nFranja azul: rango intercuartilicos"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position  = "none",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linetype = 3, linewidth = 0.3, color = "grey80"),
    panel.border = element_rect(colour = "black", linewidth = 0.2),
    plot.caption.position = "plot" 
  ) +
  coord_cartesian(clip = "off")
 