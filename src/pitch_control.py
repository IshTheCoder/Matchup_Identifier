"""Fernandez-Bornn-style pitch/space control adapted to the football pocket, with a speed/
acceleration-SKEWED influence (NOT a symmetric Gaussian): each player's influence is forward-skewed
along his motion -- isotropic small Gaussian when stationary, morphing to forward-concentrated,
backward-suppressed, laterally-tight density as speed/accel grow (a fast player can't access the
space behind him). Offense=blockers(+QB), Defense=rushers; ball=QB (radius scales with dist-to-QB).
Pure NumPy; per-frame field over a QB-centered grid. See build_pocket_design in feature_engineering."""
import numpy as np

DEFAULTS = dict(VMAX=9.5, AMAX=10.0, R_MIN=1.0, R_MAX=6.0, SIG_MID=6.0, SIG_SCALE=1.5,
                lead=0.5, alpha=1.0, alpha_a=0.5, beta=0.8, gamma=0.5, vsoft=0.5, floor=0.3)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def influence(G, x, v, a, dqb, p=DEFAULTS, radius=None):
    """Skewed influence of ONE player over grid G (P,2). Peak 1 at the player's lead point.
    Isotropic (small radius) when |v|<vsoft; else split-normal along v: longer forward reach
    (+ along-motion accel), suppressed backward, tight perpendicular -- all scaling with speed.
    `radius` override: if given, used directly (full-field/downfield use); else the pocket's
    distance-to-QB sigmoid in `dqb`."""
    if radius is None:
        radius = p["R_MIN"] + (p["R_MAX"] - p["R_MIN"]) * _sigmoid((dqb - p["SIG_MID"]) / p["SIG_SCALE"])
    speed = float(np.hypot(v[0], v[1]))
    if speed < p["vsoft"]:
        r = G - x
        return np.exp(-0.5 * (r[:, 0] ** 2 + r[:, 1] ** 2) / radius ** 2)
    vhat = v / speed
    r = G - (x + p["lead"] * v)                       # relative to the lead point mu
    rpar = r @ vhat
    rperp2 = np.maximum((r[:, 0] ** 2 + r[:, 1] ** 2) - rpar ** 2, 0.0)
    sr = min(speed / p["VMAX"], 0.95)
    apar = float(np.dot(a, vhat)) / p["AMAX"]
    s_fwd = radius * (1.0 + p["alpha"] * sr + p["alpha_a"] * max(apar, 0.0))   # reach ahead
    s_back = max(radius * (1.0 - p["beta"] * sr), p["floor"])                  # suppressed behind
    s_perp = max(radius * (1.0 - p["gamma"] * sr), p["floor"])                 # tight laterally
    spar = np.where(rpar >= 0, s_fwd, s_back)
    return np.exp(-0.5 * ((rpar / spar) ** 2 + rperp2 / s_perp ** 2))


def team_influences(G, X, V, A, D, mask, p=DEFAULTS, radius=None):
    """Per-player influences for a team on grid G -> (K_present, P) and the present slot indices.
    `radius` override (constant per player) for full-field/downfield use; else pocket dist-to-QB."""
    idx = np.where(mask > 0)[0]
    if len(idx) == 0:
        return np.zeros((0, len(G))), idx
    return np.stack([influence(G, X[k], V[k], A[k], 0.0 if D is None else D[k], p, radius=radius)
                     for k in idx]), idx


def frame_fields(G, data, n, p=DEFAULTS):
    """Rusher influences (Kr,P)+idx, blocker influences (Kb,P)+idx, QB influence (P,) for frame n."""
    Ir, ri = team_influences(G, data["x_rush"][n], data["v_rush"][n], data["a_rush"][n],
                             data["d_rush"][n], data["rmask"][n], p)
    Ib, bi = team_influences(G, data["x_blk"][n], data["v_blk"][n], data["a_blk"][n],
                             data["d_blk"][n], data["bmask"][n], p)
    Iqb = influence(G, data["x_qb"][n], data["v_qb"][n], data["a_qb"][n], 0.0, p)
    return Ir, ri, Ib, bi, Iqb


def pitch_control(Ir, Ib, Iqb=None, include_qb=False):
    """PC(p) = sigmoid(offense influence - defense influence) = P(offense controls p)."""
    off = Ib.sum(0) + (Iqb if (include_qb and Iqb is not None) else 0.0)
    return _sigmoid(off - Ir.sum(0))


def control_shares(Ir, Ib, Iqb, include_qb=True):
    """Per-player share of total control at each grid point: w_k = I_k / (sum_all I)."""
    tot = Ir.sum(0) + Ib.sum(0) + (Iqb if include_qb else 0.0) + 1e-9
    wr = Ir / tot if len(Ir) else Ir
    wb = Ib / tot if len(Ib) else Ib
    wqb = Iqb / tot if include_qb else None
    return wr, wb, wqb


def qb_grid(x_qb, half=8.0, n=64):
    """QB-centered square grid. Returns flat G (P,2), gx, gy (n,n), and cell area (yd^2)."""
    gx, gy = np.meshgrid(np.linspace(x_qb[0] - half, x_qb[0] + half, n),
                         np.linspace(x_qb[1] - half, x_qb[1] + half, n))
    G = np.stack([gx.ravel(), gy.ravel()], 1)
    cell = (2.0 * half / (n - 1)) ** 2
    return G, gx, gy, cell


FIELD_W = 53.3


def field_grid(los_x, forward=40.0, back=2.0, nx=96, ny=64):
    """Downfield grid in NORMALIZED coords (offense attacks +x): x in [los_x-back, los_x+forward],
    y across the full field width. Returns flat G (P,2), gx, gy (ny,nx), cell area (yd^2)."""
    gx, gy = np.meshgrid(np.linspace(los_x - back, los_x + forward, nx),
                         np.linspace(0.0, FIELD_W, ny))
    G = np.stack([gx.ravel(), gy.ravel()], 1)
    cell = ((forward + back) / (nx - 1)) * (FIELD_W / (ny - 1))
    return G, gx, gy, cell
