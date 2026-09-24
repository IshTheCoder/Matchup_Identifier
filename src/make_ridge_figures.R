#!/usr/bin/env Rscript
# Ridgeline ("caterpillar") posterior-density figures, one per (table, position) panel, for every
# paper table that reports a credible interval. Each figure shows the SAME players its table
# tabulates, so a reader sees the full posterior behind each interval rather than just its endpoints.
#
# Reads model_outputs/ridge_draws.parquet (written by scripts/export_ridge_draws.py, which owns the
# player selection and the snap thresholds) and writes figures/ridge_<table>[_<pos>].{png,pdf}.
# Run from the repo root:   Rscript src/make_ridge_figures.R

suppressPackageStartupMessages({
  library(ggplot2); library(ggridges); library(arrow); library(dplyr); library(grid)
})

dir.create("figures", showWarnings = FALSE)
d <- as.data.frame(read_parquet("model_outputs/ridge_draws.parquet"))

## ------------------------------------------------------------------ shared palette --
# One hue per position, used in every figure, and the hue family carries meaning: WARM for the
# defensive side (pass rushers), COOL for the offensive side (pass blockers), neutral grey for the
# quarterback, whose suppression score is a by-product rather than a player-vs-player rating.
# Kept in step with the rusher_2d palette in src/make_figures.R.
PAL <- c(Edge = "#B2182B", DT = "#D95F02", NT = "#E6AB02",   # crimson / burnt orange / gold
         T    = "#2166AC", G  = "#1B9E77", C  = "#7FBC41",   # blue / teal / green
         QB   = "#4D4D4D")

# x-axis label per table: the estimand's symbol (matching the table's column header) plus a gloss.
XLAB <- c(
  rusher_pm          = "R[j]~'   (adjusted peak STRAIN generated, centred on position mean)'",
  blocker_pm         = "B[b]~'   (peak STRAIN suppressed, centred on position mean)'",
  blocker_value      = "FN[b]~'   (front-normalized realized impedance)'",
  qb_suppression     = "-Q[q]~'   (STRAIN suppression)'",
  blocker_continuous = "B[b]^Delta~'   (per-frame STRAIN deceleration, centred on position mean)'",
  block_hold         = "-u[b]~'   (opponent-adjusted block-hold rating)'")

TITLE <- c(
  rusher_pm          = "Play-Level Plus-Minus: Pass Rushers",
  blocker_pm         = "Play-Level Plus-Minus: Pass Blockers",
  blocker_value      = "Front-Normalized Realized Impedance",
  qb_suppression     = "Quarterback STRAIN Suppression",
  blocker_continuous = "Continuous-Time Blocker Rating",
  block_hold         = "Opponent-Adjusted Block Hold")

# facet strip label: "<position> (<Top|Bottom>)", e.g. "G (Bottom)". Levels are ordered Top block
# first, then Bottom, with positions in their natural order within each block.
panel_factor <- function(pos, grp, pos_levels = sort(unique(pos))) {
  factor(sprintf("%s (%s)", pos, grp),
         levels = as.vector(t(outer(c("Top", "Bottom"), pos_levels,
                                    function(g, p) sprintf("%s (%s)", p, g)))))
}

## --------------------------------------------------------------------- one panel --
make_panel <- function(dd) {
  tbl <- dd$table[1]; pos <- dd$pos[1]
  unfaceted <- pos == "All"

  # order players by posterior mean; ggridges draws bottom-up, so the strongest ends up on top
  ord <- dd %>% group_by(name) %>% summarise(m = mean(value), .groups = "drop") %>% arrange(m)
  dd$name <- factor(dd$name, levels = ord$name)
  dd$grp <- factor(dd$grp, levels = c("Top", "Bottom"))
  dd$panel <- panel_factor(dd$pos, dd$grp, pos_levels = pos)
  n_players <- nlevels(dd$name)

  p <- ggplot(dd, aes(x = value, y = name, fill = player_pos)) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey60", linewidth = 0.4) +
    # quantile lines at 2.5/50/97.5: exactly the interval the companion table prints
    geom_density_ridges(
      quantile_lines = TRUE, quantiles = c(0.025, 0.5, 0.975),
      # scale = 1: the tallest density exactly touches the next row's baseline, so ridges
      # overlap slightly and the panel packs tighter without detaching a curve from its label.
      scale = 1, rel_min_height = 0.01, alpha = 0.80,
      colour = "grey25", linewidth = 0.3, vline_colour = "grey20", vline_linewidth = 0.35) +
    scale_fill_manual(values = PAL, name = NULL,
                      guide = if (unfaceted) "legend" else "none") +
    scale_y_discrete(expand = expansion(mult = c(0.01, 0.06))) +
    labs(x = parse(text = XLAB[[tbl]])[[1]], y = NULL,
         title = if (unfaceted) TITLE[[tbl]] else sprintf("%s — %s", TITLE[[tbl]], pos)) +
    theme_minimal(base_size = 12) +
    theme(panel.grid.major.y = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold"),
          legend.position = if (unfaceted) "bottom" else "none")

  # label the Top and Bottom blocks rather than letting them run together on one axis
  if (nlevels(droplevels(dd$grp)) > 1)
    p <- p + facet_grid(panel ~ ., scales = "free_y", space = "free_y")

  stem <- if (unfaceted) sprintf("figures/ridge_%s", tbl) else sprintf("figures/ridge_%s_%s", tbl, pos)
  h <- max(3.0, 0.32 * n_players + 1.7)
  ggsave(paste0(stem, ".png"), p, width = 7.2, height = h, dpi = 200)
  ggsave(paste0(stem, ".pdf"), p, width = 7.2, height = h)
  cat(sprintf("  wrote %s.{png,pdf}  (%d players)\n", stem, n_players))
}

panels <- split(d, list(d$table, d$pos), drop = TRUE)
cat(sprintf("%d panels from %s draws\n", length(panels), format(nrow(d), big.mark = ",")))
invisible(lapply(panels, make_panel))

## ------------------------------------------------- combined per-table figure (for the paper) --
# The paper needs ONE float per table, so for the position-faceted tables we also render all
# positions side by side (position across, Top/Bottom down). The standalone per-position files
# above stay for slides and for anyone wanting a single position at full size.
make_combined <- function(dd) {
  tbl <- dd$table[1]
  ord <- dd %>% group_by(pos, name) %>% summarise(m = mean(value), .groups = "drop") %>% arrange(m)
  dd$name <- factor(dd$name, levels = unique(ord$name))
  dd$grp <- factor(dd$grp, levels = c("Top", "Bottom"))
  dd$panel <- panel_factor(dd$pos, dd$grp)

  p <- ggplot(dd, aes(x = value, y = name, fill = player_pos)) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey60", linewidth = 0.4) +
    geom_density_ridges(
      quantile_lines = TRUE, quantiles = c(0.025, 0.5, 0.975),
      scale = 1, rel_min_height = 0.01, alpha = 0.80,
      colour = "grey25", linewidth = 0.25, vline_colour = "grey20", vline_linewidth = 0.3) +
    scale_fill_manual(values = PAL, guide = "none") +
    scale_y_discrete(expand = expansion(mult = c(0.01, 0.06))) +
    # facet_wrap, not facet_grid: facet_grid shares the y axis across a row, which would list
    # every position's players in every column. Rows are Top then Bottom, columns are positions.
    facet_wrap(~ panel, scales = "free_y", ncol = 3) +
    labs(x = parse(text = XLAB[[tbl]])[[1]], y = NULL, title = TITLE[[tbl]]) +
    theme_minimal(base_size = 11) +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold"),
          strip.text = element_text(face = "bold"),
          panel.spacing.x = unit(0.9, "lines"))

  n_rows <- dd %>% distinct(pos, grp, name) %>% count(pos, grp) %>% pull(n) %>% max()
  h <- max(3.2, 0.34 * n_rows * 2 + 1.7)
  ggsave(sprintf("figures/ridge_%s_all.png", tbl), p, width = 10, height = h, dpi = 200)
  ggsave(sprintf("figures/ridge_%s_all.pdf", tbl), p, width = 10, height = h)
  cat(sprintf("  wrote figures/ridge_%s_all.{png,pdf}\n", tbl))
}

faceted <- d %>% filter(pos != "All")
invisible(lapply(split(faceted, faceted$table, drop = TRUE), make_combined))
cat("done\n")
