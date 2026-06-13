"""Regenerate play-level plus-minus rankings BY POSITION with a snap threshold,
reusing the saved posterior (no re-fit). Counts real snaps (mask for rushers,
assignment mass for blockers; padded slots use id 0 and are excluded).
Writes filtered CSVs and per-position top/bottom .tex tables."""
import pickle
import numpy as np
import pandas as pd

MIN_SNAPS = 50  # consistent with the attention / entropy tables

data, enc = pickle.load(open("play_design_8wk.pkl", "rb"))
samples = pickle.load(open("play_model_samples.pkl", "rb"))
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
        "hdi_lo": np.percentile(draws, 3, axis=0),
        "hdi_hi": np.percentile(draws, 97, axis=0),
        "snaps": snaps,
    })
    df = df[df["snaps"] >= MIN_SNAPS]
    return df.sort_values(["pos", "effect"], ascending=[True, False]).reset_index(drop=True)


bl = rankings("blocker_covariates", "blocker_weight", "z_blocker", "sigma_blocker", "blocker", blocker_snaps)
ru = rankings("rusher_covariates", "rusher_weight", "z_rusher", "sigma_rusher", "rusher", rusher_snaps)
bl.to_csv("blocker_rankings.csv", index=False)
ru.to_csv("rusher_rankings.csv", index=False)


# ------------------------------------------------ .tex tables by position --
def _rows(df):
    return " \\\\\n".join(f"{r['name']} & {r['effect']:.2f}" for _, r in df.iterrows()) + " \\\\"


def _subtable(df, cap, width="0.24"):
    return ("\\begin{subtable}{" + width + "\\textwidth}\n\\centering\n\\footnotesize\n"
            "\\begin{tabular}{lc}\n\\toprule\nName & Effect \\\\\n\\midrule\n"
            + _rows(df) + "\n\\bottomrule\n\\end{tabular}\n\\caption{" + cap + "}\n\\end{subtable}")


def _table(groups, df, top, label, caption, width="0.24"):
    subs = []
    for g in groups:
        d = df[df["pos"] == g].sort_values("effect", ascending=not top).head(5)
        subs.append(_subtable(d, g, width))
    return ("\\begin{table}[h!]\n\\centering\n" + "\n\\hfill\n".join(subs)
            + "\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\end{table}\n")


RUSH = ["DE", "DT", "NT", "OLB"]
BLK = ["T", "G", "C"]  # TE/RB/FB rarely reach the pass-block snap threshold
NOTE = f"(min {MIN_SNAPS} pass snaps)"

with open("tables/rusher_plusminus.tex", "w") as f:
    f.write("% Play-level plus-minus rusher effect (higher = more peak pressure generated),\n"
            "% attribute-centered, per position, filtered by snaps.\n")
    f.write(_table(RUSH, ru, True, "tab:rusher_pm_top",
                   f"Top 5 pass rushers by plus-minus effect, by position {NOTE}."))
    f.write("\n")
    f.write(_table(RUSH, ru, False, "tab:rusher_pm_bot",
                   f"Bottom 5 pass rushers by plus-minus effect, by position {NOTE}."))

with open("tables/blocker_plusminus.tex", "w") as f:
    f.write("% Play-level plus-minus blocker effect (higher = better pressure suppression,\n"
            "% since the blocker term is subtracted), attribute-centered, per position.\n")
    f.write(_table(BLK, bl, True, "tab:blocker_pm_top",
                   f"Top 5 pass blockers by plus-minus effect, by position {NOTE}.", width="0.32"))
    f.write("\n")
    f.write(_table(BLK, bl, False, "tab:blocker_pm_bot",
                   f"Bottom 5 pass blockers by plus-minus effect, by position {NOTE}.", width="0.32"))

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
    return " \\\\\n".join(f"{r['name']} & {r['suppression']:.3f}" for _, r in df.iterrows()) + " \\\\"


with open("tables/qb_suppression.tex", "w") as f:
    f.write("% QB strain suppression = -Q_q (higher = less peak pressure allowed on this\n"
            "% QB's dropbacks), from the play-level plus-minus model, min "
            + str(MIN_SNAPS) + " dropbacks.\n")
    f.write("\\begin{table}[h!]\n\\centering\n")
    for d, cap in [(qb.head(10), "Most strain-suppressing"),
                   (qb.tail(10).iloc[::-1], "Least strain-suppressing")]:
        f.write("\\begin{subtable}{0.48\\textwidth}\n\\centering\n\\footnotesize\n"
                "\\begin{tabular}{lc}\n\\toprule\nName & Strain Suppr. \\\\\n\\midrule\n"
                + _qb_block(d) + "\n\\bottomrule\n\\end{tabular}\n\\caption{" + cap
                + "}\n\\end{subtable}\n\\hfill\n")
    f.write("\\caption{Quarterbacks ranked by strain suppression "
            "(min " + str(MIN_SNAPS) + " dropbacks).}\n\\label{tab:qb_suppression}\n\\end{table}\n")

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
      "tables/{rusher_plusminus,blocker_plusminus,qb_suppression}.tex")
