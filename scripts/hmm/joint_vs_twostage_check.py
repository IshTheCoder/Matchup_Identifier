"""Does the two-stage fit sit at a stationary point of the JOINT (n_i+1)-state likelihood?  (appendix, Inference)

Usage (from repo root, in the football container): python3 scripts/hmm/joint_vs_twostage_check.py T,G,C 10
Output of that exact command on 2026-09-15 is in joint_vs_twostage_check.week0.txt alongside this file.
From the Phase-2.5 solution (emission/prior frozen at Phase 1, transition fit with the null
state), run K further EM iterations on week 0 in two modes:
  (a) transition-only  -- the Stage-2 coordinate ascent continued (control)
  (b) joint            -- tau, sigma, beta ALSO re-estimated from the (n_i+1)-state posteriors
and report full-model log-likelihood, parameter drift, and assignment drift vs. iteration 0.
"""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu"); sys.path.insert(0, "src")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np, pandas as pd
import baum_welch_jax as bwj

POSITIONS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["T", "G", "C"]
N_ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 8
C_B, NU, CONC, ORIENT = -5.0, 0.05, 1.0, "vonmises"
WEEKS = ["data/sample_data_week_0.csv"]

fitted = pd.read_csv("fitted_params_jax_phase25.csv").set_index("position")
rdf = pd.read_csv("rho_by_player_phase25.csv")
voxels = bwj.load_position_voxels(WEEKS, tuple(POSITIONS), with_features=True)


def estep(tau, sigma, beta, rho_pl, pf_pl, rho_null, groups):
    """full-model E-step over all K-groups -> accumulators for every M-step + LL + posteriors"""
    XtWX = jnp.zeros((2, 2)); XtWy = jnp.zeros(2); yty = 0.0; sden = 0.0; ll = 0.0
    n = rho_pl.shape[0]
    A = jnp.zeros(n); Bc = jnp.zeros(n); F = jnp.zeros(n); Nstay = Nrec = 0.0
    beta_data, gammas = [], {}
    for K, G in groups.items():
        idx = jnp.clip(G.blocker_id, 0, None); valid = G.blocker_id >= 0
        le = bwj.emission_logprob(tau, sigma, G.O, G.B, G.D, ORIENT, CONC, null_state=True, c_b=C_B)
        lp = bwj._log_pi_null_from_features(G.X, beta, NU)
        lT = bwj.build_log_transition_null_batched(rho_pl[idx], pf_pl[idx], rho_null, K)
        gamma, xi, ll_j = bwj.forward_backward_group_perT(le, lp, lT, G.t_mask)
        ll += jnp.sum(ll_j * G.j_mask)
        gammas[K] = np.asarray(gamma * G.j_mask[:, None, :, None])  # drop padded blocker slots
        # --- emission accumulators on ENGAGED responsibilities only (null column dropped) ---
        g = gamma[..., :K] * G.j_mask[:, None, :, None]
        O2, B2, D2 = G.O[..., :2], G.B[..., :2], G.D[..., :2]
        S_OO = jnp.einsum("ptjk,ptkc,ptkc->", g, O2, O2); S_OB = jnp.einsum("ptjk,ptkc,ptc->", g, O2, B2)
        S_BB = jnp.einsum("ptjk,ptc,ptc->", g, B2, B2)
        y_O = jnp.einsum("ptjk,ptkc,ptjc->", g, O2, D2); y_B = jnp.einsum("ptjk,ptc,ptjc->", g, B2, D2)
        XtWX += jnp.array([[S_OO, S_OB], [S_OB, S_BB]]); XtWy += jnp.array([y_O, y_B])
        yty += jnp.einsum("ptjk,ptjc,ptjc->", g, D2, D2); sden += 2.0 * g.sum()
        # --- matchup-prior soft targets: engaged snap responsibilities, renormalized ---
        g0 = gamma[:, 0, :, :K]; g0 = g0 / jnp.clip(g0.sum(-1, keepdims=True), 1e-12)
        beta_data.append((G.X, g0, G.j_mask))
        # --- transition directional counts (as in fit_null_transition_by_position) ---
        xi = xi * G.j_mask[:, :, None, None, None]
        A_pj = jnp.einsum("pjtii->pj", xi[:, :, :, :K, :K]); leave = jnp.einsum("pjtab->pj", xi[:, :, :, :, :K])
        F_pj = jnp.einsum("pjtb->pj", xi[:, :, :, K, :K]); B_pj = leave - A_pj - F_pj
        Nstay += jnp.sum(xi[:, :, :, K, K]); Nrec += jnp.sum(xi[:, :, :, :K, K])
        A = A.at[idx].add(jnp.where(valid, A_pj, 0.)); Bc = Bc.at[idx].add(jnp.where(valid, B_pj, 0.))
        F = F.at[idx].add(jnp.where(valid, F_pj, 0.))
    return dict(XtWX=XtWX, XtWy=XtWy, yty=yty, sden=sden, ll=float(ll), beta_data=beta_data,
                counts=jnp.clip(jnp.stack([A, Bc, F], 1), 0, None), Nstay=Nstay, Nrec=Nrec, gammas=gammas)


def drift(g_ref, g_new):
    """mean |delta gamma| per valid blocker-frame (engaged states), null mass %, attention corr.

    Padded (frame, blocker) slots carry gamma == 0 across all states, so the total mass
    sum(gamma) counts exactly the valid blocker-frames."""
    d, mass, null_new, att_ref, att_new = 0.0, 0.0, 0.0, [], []
    for K in g_ref:
        a, b = g_ref[K], g_new[K]
        d += np.abs(a[..., :K] - b[..., :K]).sum(); mass += b.sum(); null_new += b[..., K].sum()
        att_ref.append(a[..., :K].sum(2).ravel()); att_new.append(b[..., :K].sum(2).ravel())
    att_ref, att_new = np.concatenate(att_ref), np.concatenate(att_new)
    m = att_ref > 0
    return d / mass, None, null_new / mass, np.corrcoef(att_ref[m], att_new[m])[0, 1]


for pos in POSITIONS:
    B_list, O_list, D_list, X_list, BID_list = voxels[pos]
    row = fitted.loc[pos]
    tau0 = jnp.asarray(bwj._parse_vec(row["tau"])); sigma0 = float(row["sigma"])
    beta0 = jnp.asarray(bwj._parse_vec(row["beta"])); fmean, fstd = bwj._parse_vec(row["feat_mean"]), bwj._parse_vec(row["feat_std"])
    rho_null0 = float(row["rho_null"]); alpha0 = jnp.asarray(bwj._parse_vec(row["dir_alpha"]))
    X_std = [(np.asarray(x) - fmean) / fstd for x in X_list]
    pids = np.unique(np.concatenate([np.asarray(b) for b in BID_list])); id2 = {int(p): i for i, p in enumerate(pids)}
    bid = [np.array([id2[int(p)] for p in b], np.int32) for b in BID_list]
    groups = bwj.group_and_pad(B_list, O_list, D_list, X_std, bid)
    r = rdf[rdf.position == pos].set_index("nflId")
    rho_pl0 = jnp.asarray(np.clip([r.rho.get(int(p), float(row["rho"])) for p in pids], 1e-3, 1 - 1e-3))
    pf_pl0 = jnp.asarray(np.clip([r.p_fail.get(int(p), 0.01) for p in pids], 1e-4, 0.5))
    nposs = sum(1 for O in O_list if O.shape[1] > 1)
    print(f"\n================ {pos}: {nposs} week-0 possessions, {len(pids)} blockers ================", flush=True)
    print(f"start (two-stage): tau={np.asarray(tau0).round(4)} sigma={sigma0:.4f} rho_null={rho_null0:.4f} "
          f"mean rho_b={float(rho_pl0.mean()):.4f} mean p_fail={float(pf_pl0.mean()):.4f}", flush=True)

    for mode in ("transition-only", "joint"):
        tau, sigma, beta = tau0, sigma0, beta0
        rho_pl, pf_pl, rho_null, alpha = rho_pl0, pf_pl0, rho_null0, alpha0
        t0 = time.time(); ref = None; ll0 = None
        print(f"\n--- {mode} ---", flush=True)
        print(f"{'it':>2} {'loglik':>14} {'dLL':>10} {'tau_r':>7} {'sigma':>7} {'|dbeta|':>8} {'rho_null':>8} "
              f"{'m_rho':>6} {'m_pfail':>7} {'null%':>6} {'mean|dg|':>8} {'att_corr':>8}", flush=True)
        for it in range(N_ITER + 1):
            E = estep(tau, sigma, beta, rho_pl, pf_pl, rho_null, groups)
            if ref is None:
                ref, ll0 = E["gammas"], E["ll"]
            dg, nm0, nm, ac = drift(ref, E["gammas"])
            print(f"{it:>2} {E['ll']:>14.1f} {E['ll']-ll0:>10.1f} {float(tau[0]):>7.4f} {float(sigma):>7.4f} "
                  f"{float(jnp.abs(beta-beta0).max()):>8.4f} {rho_null:>8.4f} {float(rho_pl.mean()):>6.4f} "
                  f"{float(pf_pl.mean()):>7.4f} {100*nm:>6.2f} {dg:>8.5f} {ac:>8.5f}", flush=True)
            if it == N_ITER:
                break
            # ---- M-steps ----
            alpha = bwj.dirichlet_eb(E["counts"], alpha0=alpha, n_steps=50)
            pm = bwj.dirichlet_posterior_mean(E["counts"], alpha)
            rho_pl, pf_pl = pm[:, 0], pm[:, 2]
            rho_null = float((E["Nstay"] + 1.0) / (E["Nstay"] + E["Nrec"] + 2.0))
            if mode == "joint":
                tau = bwj.simplex_project(E["XtWX"], E["XtWy"])
                sigma = float((E["yty"] - 2.0 * (tau @ E["XtWy"]) + tau @ (E["XtWX"] @ tau)) / E["sden"])
                beta = bwj.fit_beta(E["beta_data"], beta, n_newton=10, ridge=1e-2)
        print(f"({mode}: {time.time()-t0:.0f}s)  final tau={np.asarray(tau).round(4)} sigma={float(sigma):.4f} "
              f"alpha={np.asarray(alpha).round(2)}", flush=True)
    jax.clear_caches()
