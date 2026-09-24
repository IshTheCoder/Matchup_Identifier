#!/usr/bin/env Rscript
# Generates figures/tau_positions.{png,pdf} and figures/rusher_2d.{png,pdf} in R (ggplot2),
# replacing the matplotlib versions. The data prep that needs the Python model pickles stays in
# Python; this script only PLOTS, reading the CSVs that pipeline already writes:
#   - fitted_params_jax.csv  (HMM fitted params; `tau` = blocker position along the QB->rusher line)
#   - rusher_2d.csv          (per-rusher attention + plus-minus effect; written by rusher_2d.py)
# Run from the repo root:   Rscript src/make_figures.R
# Requires: ggplot2 (ggrepel optional, for nicer non-overlapping labels in the 2-D plot).

suppressPackageStartupMessages(library(ggplot2))
has_repel <- requireNamespace("ggrepel", quietly = TRUE)
dir.create("figures", showWarnings = FALSE)

## --------------------------------------------------------------- tau positions --
fp <- read.csv("fitted_params_jax.csv", stringsAsFactors = FALSE)

# `tau` is a bracketed, space-separated vector printed by numpy; the rusher component is the first
# element (mirrors np.fromstring(...)[0] in make_tables_figures.py).
first_num <- function(s) as.numeric(strsplit(trimws(gsub("\\[|\\]|\n", " ", s)), "\\s+")[[1]])[1]
fp$tau_r <- vapply(fp$tau, first_num, numeric(1))
fp <- fp[order(fp$tau_r), ]

labels <- c(T = "Tackle", G = "Guard", C = "Center", TE = "Tight End",
            RB = "Running Back", FB = "Fullback", WR = "Wide Receiver")
fp$label <- sprintf("%s (%.2f)",
                    ifelse(fp$position %in% names(labels), labels[fp$position], fp$position),
                    fp$tau_r)
# multi-level stagger + leader lines so the tight T/G/C/TE/FB cluster stays legible
levs <- c(0.32, -0.32, 0.52, -0.52, 0.72, -0.72)
fp$y <- levs[((seq_len(nrow(fp)) - 1L) %% length(levs)) + 1L]
fp$vjust <- ifelse(fp$y > 0, 0, 1)

p_tau <- ggplot(fp) +
  geom_hline(yintercept = 0, color = "grey60", linewidth = 1) +
  geom_segment(aes(x = tau_r, xend = tau_r, y = 0, yend = y), color = "grey55", linewidth = 0.4) +
  geom_point(aes(x = tau_r, y = 0), color = "#1f5fb4", size = 4) +
  geom_text(aes(x = tau_r, y = y, label = label, vjust = vjust), size = 3.3) +
  annotate("text", x = 0, y = -0.95, label = "Quarterback", fontface = "bold", size = 4, vjust = 1) +
  annotate("text", x = 1, y = -0.95, label = "Rusher", fontface = "bold", size = 4, vjust = 1) +
  scale_x_continuous(breaks = c(0, .25, .5, .75, 1)) +
  coord_cartesian(xlim = c(-0.08, 1.08), ylim = c(-1.05, 1.0)) +
  labs(x = expression(tau[r] * ": position along the QB" %->% "rusher line"), y = NULL,
       title = "Where each blocking group stands between the quarterback and its rusher") +
  theme_minimal(base_size = 12) +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(),
        panel.grid = element_blank(), axis.line.x = element_line(color = "grey50"),
        plot.title = element_text(size = 12))

ggsave("figures/tau_positions.png", p_tau, width = 11, height = 3.4, dpi = 200)
ggsave("figures/tau_positions.pdf", p_tau, width = 11, height = 3.4)
cat("wrote figures/tau_positions.{png,pdf}\n")

## --------------------------------------------------------------------- rusher 2-D --
g <- read.csv("rusher_2d.csv", stringsAsFactors = FALSE)
mx <- median(g$attention); my <- median(g$effect)               # quadrant crosshairs
# Warm hues for the defensive side, spread across crimson -> burnt orange -> gold so the three
# groups separate in the dense middle of the cloud (the old crimson/rose pair read as one colour).
# Kept in step with PAL in src/make_ridge_figures.R.
pal <- c(Edge = "#B2182B", DT = "#D95F02", NT = "#E6AB02")  # matches src/make_ridge_figures.R
g$grp <- factor(ifelse(g$pos %in% names(pal), g$pos, "other"), levels = c("Edge", "DT", "NT", "other"))

# annotate extremes: top 6 by effect, top 6 by attention, and top 6 high-on-both
hb <- which(g$attention > mx & g$effect > my)
lab_ids <- unique(c(head(order(-g$effect), 6), head(order(-g$attention), 6),
                    hb[head(order(-g$effect[hb]), 6)]))
g$lab <- NA_character_; g$lab[lab_ids] <- g$name[lab_ids]

p2 <- ggplot(g, aes(attention, effect)) +
  geom_vline(xintercept = mx, linetype = "dashed", color = "grey70") +
  geom_hline(yintercept = my, linetype = "dashed", color = "grey70") +
  geom_point(aes(color = grp, shape = grp, size = grp, alpha = grp)) +
  scale_color_manual(values = c(pal, other = "grey60"), name = NULL) +
  # shape as a second, redundant channel: in the crowded interior the marker outline separates the
  # groups even where two points overlap, and it survives greyscale printing.
  scale_shape_manual(values = c(Edge = 16, DT = 17, NT = 15, other = 16), name = NULL) +
  scale_size_manual(values = c(Edge = 2.2, DT = 2.1, NT = 2.0, other = 1.4), guide = "none") +
  # 0.8 rather than 0.6: at 0.6 on white the gold washed out to near-cream.
  scale_alpha_manual(values = c(Edge = .8, DT = .8, NT = .8, other = .3), guide = "none") +
  labs(x = "Attention commanded  (front-normalized: blockers drawn vs. average rusher on the play)",
       y = expression("Plus-minus effect " * R[j] * "  (strain generated, adjusted for blocking)"),
       title = "Two dimensions of pass rush: pressure generated vs. blocking attention drawn") +
  theme_minimal(base_size = 12) +
  theme(legend.position = c(0.92, 0.10), legend.background = element_blank())

if (has_repel) {
  p2 <- p2 + ggrepel::geom_text_repel(aes(label = lab), size = 2.6, na.rm = TRUE,
                                      max.overlaps = Inf, segment.color = "grey80", seed = 1)
} else {
  p2 <- p2 + geom_text(aes(label = lab), size = 2.6, na.rm = TRUE, hjust = -0.1, vjust = -0.4)
}

ggsave("figures/rusher_2d.png", p2, width = 9, height = 7, dpi = 160)
ggsave("figures/rusher_2d.pdf", p2, width = 9, height = 7)
cat(sprintf("wrote figures/rusher_2d.{png,pdf}  (rushers: %d, corr(att,effect): %.3f)\n",
            nrow(g), cor(g$attention, g$effect)))
