"""Regenerate play-level plus-minus rankings BY POSITION with a snap threshold,
reusing the saved posterior (no re-fit). Counts real snaps (mask for rushers,
assignment mass for blockers; padded slots use id 0 and are excluded).
Writes filtered CSVs and per-position top/bottom .tex tables."""
import pickle
import numpy as np
import pandas as pd

MIN_SNAPS = 50  # consistent with the attention / entropy tables

# Phase-2.5 (asymmetric null + block-failure hazard) is the canonical model for the headline
# plus-minus tables; the *norm (weight-normalized) variant is a robustness refit reported separately.
data, enc = pickle.load(open("play_design_phase25.pkl", "rb"))
samples = pickle.load(open("play_model_samples_phase25.pkl", "rb"))
players = pd.read_csv("data/players.csv").set_index("nflId")

mask = np.asarray(data["mask"])
rid = np.asarray(data["rusher_ids"])
bid = np.asarray(data["blocker_ids"])
asg = np.asarray(data["assignment"])

# --- real snap counts (exclude padded id-0 slots) ---
rusher_snaps = np.bincount(rid[mask > 0].ravel(), minlength=int(data["N_rushers"]))
bpres = asg.sum(1) > 0  # (N, blk_slot): blocker present if it has attention mass
blocker_snaps = np.bincount(bid[bpres].ravel(), minlength=int(data["N_blockers"]))


def rankings(cov_key, w_key, z_key, sig_key, enc_key, snaps):
    # effect = covariates @ weight + sigma * z  (per posterior draw)
    draws = samples[w_key] @ data[cov_key].T + samples[sig_key][:, None] * samples[z_key]
    inv = {v: k for k, v in enc[enc_key].items()}
    ids = [inv[i] for i in range(draws.shape[1])]
    df = pd.DataFrame({
        "nflId": ids,
        "name": [players["displayName"].get(i, "?") for i in ids],
        "pos": [players["officialPosition"].get(i, "?") for i in ids],
        "effect": draws.mean(0),
        "hdi_lo": np.percentile(draws, 2.5, axis=0),
        "hdi_hi": np.percentile(draws, 97.5, axis=0),
        "snaps": snaps,
    })
    df = df[df["snaps"] >= MIN_SNAPS]
    return df.sort_values(["pos", "effect"], ascending=[True, False]).reset_index(drop=True)


bl = rankings("blocker_covariates", "blocker_weight", "z_blocker", "sigma_blocker", "blocker", blocker_snaps)
ru = rankings("rusher_covariates", "rusher_weight", "z_rusher", "sigma_rusher", "rusher", rusher_snaps)
ru["pos"] = ru["pos"].replace({"DE": "Edge", "OLB": "Edge"})   # 4-3 DE + 3-4 OLB = one edge-rush group
bl.to_csv("blocker_rankings.csv", index=False)
ru.to_csv("rusher_rankings.csv", index=False)


# ------------------------------------------------ .tex tables by position --
# Effect = posterior mean; brackets = 95% credible interval (2.5--97.5 pct of the draws).
def _rows(df):
    return " \\\\\n".join(
        f"{r['name']} & {r['effect']:.2f} & $[{r['hdi_lo']:.2f},\\,{r['hdi_hi']:.2f}]$"
        for _, r in df.iterrows()) + " \\\\"


def _subtable(df, cap, sym, width="0.48"):
    # `sym` is the parameter's symbol in the play-level model of the paper (R_j for rushers,
    # B_b for blockers), so the column header names the estimand rather than "Effect".
    return ("\\begin{subtable}{" + width + "\\textwidth}\n\\centering\n\\footnotesize\n"
            "\\begin{tabular}{lcc}\n\\toprule\nName & " + sym + " & 95\\% CI \\\\\n\\midrule\n"
            + _rows(df) + "\n\\bottomrule\n\\end{tabular}\n\\caption{" + cap + "}\n\\end{subtable}")


def _table(groups, df, top, label, caption, sym, width="0.48"):
    # two subtables per row (the CI column needs the width); break the line after each pair
    subs = [_subtable(df[df["pos"] == g].sort_values("effect", ascending=not top).head(5), g, sym, width)
            for g in groups]
    body = ""
    for i in range(0, len(subs), 2):
        body += "\n\\hfill\n".join(subs[i:i + 2])
        body += "\n\n\\bigskip\n" if i + 2 < len(subs) else "\n"
    return ("\\begin{table}[h!]\n\\centering\n" + body
            + "\\caption{" + caption + "}\n\\label{" + label + "}\n\\end{table}\n")


RUSH = ["Edge", "DT", "NT"]
BLK = ["T", "G", "C"]  # TE/RB/FB rarely reach the pass-block snap threshold
NOTE = f"(min {MIN_SNAPS} pass snaps)"

# Top tables go in Results; the bottom-five-by-position tables are emitted as SEPARATE files so the
# paper can place them in the appendix.
with open("tables/rusher_plusminus.tex", "w") as f:
    f.write("% Play-level plus-minus rusher effect (higher = more peak pressure generated),\n"
            "% attribute-centered, per position, filtered by snaps. TOP only; bottom in _bot file.\n")
    f.write(_table(RUSH, ru, True, "tab:rusher_pm_top",
                   f"Top 5 pass rushers by plus-minus effect, by position {NOTE}. "
                   "Brackets are 95\\% posterior credible intervals.", "$R_j$"))
with open("tables/rusher_plusminus_bot.tex", "w") as f:
    f.write("% Bottom-five pass rushers by plus-minus effect (appendix).\n")
    f.write(_table(RUSH, ru, False, "tab:rusher_pm_bot",
                   f"Bottom 5 pass rushers by plus-minus effect, by position {NOTE}. "
                   "Brackets are 95\\% posterior credible intervals.", "$R_j$"))

with open("tables/blocker_plusminus.tex", "w") as f:
    f.write("% Play-level plus-minus blocker effect (higher = better pressure suppression,\n"
            "% since the blocker term is subtracted), per position. TOP only; bottom in _bot file.\n")
    f.write(_table(BLK, bl, True, "tab:blocker_pm_top",
                   f"Top 5 pass blockers by plus-minus effect, by position {NOTE}. "
                   "Brackets are 95\\% posterior credible intervals.", "$B_b$"))
with open("tables/blocker_plusminus_bot.tex", "w") as f:
    f.write("% Bottom-five pass blockers by plus-minus effect (appendix).\n")
    f.write(_table(BLK, bl, False, "tab:blocker_pm_bot",
                   f"Bottom 5 pass blockers by plus-minus effect, by position {NOTE}. "
                   "Brackets are 95\\% posterior credible intervals.", "$B_b$"))

# ------------------------------------------------ QB strain suppression --
# Q_q enters the peak-strain predictor additively, so a LOWER QB effect means
# less pressure on that QB's dropbacks. We report suppression = -Q_q (higher =
# better at avoiding/limiting pressure: quick release, mobility, pocket presence).
qb_ids = np.asarray(data["quarterback_ids"])
qb_snaps = np.bincount(qb_ids, minlength=int(data["N_quarterbacks"]))
qb_draws = (samples["quarterback_weight"] @ data["quarterback_covariates"].T
            + samples["sigma_quarterback"][:, None] * samples["z_quarterback"])
qb_inv = {v: k for k, v in enc["qb"].items()}
qb_idlist = [qb_inv[i] for i in range(qb_draws.shape[1])]
qb = pd.DataFrame({
    "nflId": qb_idlist,
    "name": [players["displayName"].get(i, "?") for i in qb_idlist],
    "suppression": -qb_draws.mean(0),
    "hdi_lo": -np.percentile(qb_draws, 97, axis=0),
    "hdi_hi": -np.percentile(qb_draws, 3, axis=0),
    "snaps": qb_snaps,
})
qb = qb[qb["snaps"] >= MIN_SNAPS].sort_values("suppression", ascending=False).reset_index(drop=True)
qb.to_csv("qb_suppression.csv", index=False)


def _qb_block(df):
    return " \\\\\n".join(
        f"{r['name']} & {r['suppression']:.3f} & $[{r['hdi_lo']:.3f},\\,{r['hdi_hi']:.3f}]$"
        for _, r in df.iterrows()) + " \\\\"


with open("tables/qb_suppression.tex", "w") as f:
    f.write("% QB strain suppression = -Q_q (higher = less peak pressure allowed on this\n"
            "% QB's dropbacks), from the play-level plus-minus model, min "
            + str(MIN_SNAPS) + " dropbacks.\n")
    f.write("\\begin{table}[h!]\n\\centering\n")
    for d, cap in [(qb.head(10), "Most strain-suppressing"),
                   (qb.tail(10).iloc[::-1], "Least strain-suppressing")]:
        f.write("\\begin{subtable}{0.48\\textwidth}\n\\centering\n\\footnotesize\n"
                "\\begin{tabular}{lcc}\n\\toprule\nName & Strain Suppr. ($-Q_q$) & 95\\% CI \\\\\n"
                "\\midrule\n"
                + _qb_block(d) + "\n\\bottomrule\n\\end{tabular}\n\\caption{" + cap
                + "}\n\\end{subtable}\n\\hfill\n")
    f.write("\\caption{Quarterbacks ranked by strain suppression (min " + str(MIN_SNAPS)
            + " dropbacks). Brackets are 95\\% posterior credible intervals.}"
            "\n\\label{tab:qb_suppression}\n\\end{table}\n")

# ------------------------------------------------------------- console --
print(f"snap threshold = {MIN_SNAPS}")
print(f"rushers kept: {len(ru)} / {int(data['N_rushers'])} | blockers kept: {len(bl)} / {int(data['N_blockers'])}")
for g in RUSH:
    d = ru[ru["pos"] == g].head(3)
    print(f"  rusher {g:>3}: " + ", ".join(f"{r['name']} {r['effect']:.2f}({int(r['snaps'])})" for _, r in d.iterrows()))
for g in BLK:
    d = bl[bl["pos"] == g].head(3)
    print(f"  blocker {g:>2}: " + ", ".join(f"{r['name']} {r['effect']:.2f}({int(r['snaps'])})" for _, r in d.iterrows()))
print(f"  QBs kept: {len(qb)} | most suppr: " + ", ".join(
    f"{r['name']} {r['suppression']:.3f}" for _, r in qb.head(3).iterrows()))
print("wrote rusher_rankings.csv, blocker_rankings.csv, qb_suppression.csv, "
      "tables/{rusher_plusminus,blocker_plusminus,qb_suppression}.tex and "
      "tables/{rusher,blocker}_plusminus_bot.tex (appendix bottom-five)")
