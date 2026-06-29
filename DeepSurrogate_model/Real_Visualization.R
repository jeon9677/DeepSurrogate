# ============================================================
# Spatial lower-bound map for FoS / coastal prediction results
# ============================================================

library(ggplot2)
library(viridis)
library(MBA)
library(dplyr)

# ------------------------------------------------------------
# 1. Input data
# ------------------------------------------------------------

coords_df <- data.frame(
  longitude = coords[, 1],
  latitude  = coords[, 2],
  lower     = lower
)

coords_df <- coords_df %>%
  filter(
    is.finite(longitude),
    is.finite(latitude),
    is.finite(lower)
  )

# ------------------------------------------------------------
# 2. Interpolate irregular spatial points onto a regular grid
# ------------------------------------------------------------
mba_fit <- mba.surf(
  data = data.frame(
    x = coords_df$longitude,
    y = coords_df$latitude,
    z = coords_df$lower
  ),
  no.X = 500,
  no.Y = 500,
  extend = FALSE
)

# MBA output -> plotting dataframe
grid_df <- expand.grid(
  longitude = mba_fit$xyz.est$x,
  latitude  = mba_fit$xyz.est$y
)

grid_df$lower <- as.vector(mba_fit$xyz.est$z)

# ------------------------------------------------------------
# 3. Plot
# ------------------------------------------------------------
p_lower <- ggplot(grid_df, aes(x = longitude, y = latitude)) +
  geom_raster(aes(fill = lower), interpolate = FALSE) +
  coord_fixed() +
  scale_fill_viridis_c(
    option = "plasma",
    na.value = "white",
    name = expression(hat(Y)[lower])
  ) +
  labs(
    x = "Longitude",
    y = "Latitude"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    panel.grid.major = element_line(
      color = "grey85",
      linewidth = 0.6
    ),
    panel.grid.minor = element_line(
      color = "grey92",
      linewidth = 0.4
    ),
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 14),
    legend.title = element_text(size = 17),
    legend.text = element_text(size = 14),
    legend.key.height = grid::unit(3.5, "cm"),
    legend.key.width = grid::unit(0.7, "cm"),
    plot.margin = margin(10, 20, 10, 10)
  )

print(p_lower)

# ------------------------------------------------------------
# 4. Save figure
# ------------------------------------------------------------
ggsave(
  filename = "FoS_lower_prediction_map.png",
  plot = p_lower,
  width = 7,
  height = 5.5,
  dpi = 400,
  bg = "white"
)