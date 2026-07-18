"""Fernandez-Bornn Space Occupation Gain (SOG), pocket analog. A RUSHER occupies pocket space by
winning control of valuable (near-QB) ground; he GAINS space when that owned-space quality rises.
 - Q_j(t) = value-weighted controlled space = sum over a QB disk of [control share w_j(p) * V(p)],
   with value V(p)=exp(-|p-QB|/lambda) (the rush analog of Bornn's distance-to-goal value: space AT
   the passer is worth most). This is the per-player Q_i = PC_i * V of Wide Open Spaces (Eq. 6).
 - per-frame gain g_j(t) = Q_j(t) - Q_j(t-1); SOG = positive gains >= eps, SOL = losses (Eq. 7-9).
 - active (rusher running > 1.5 m/s) vs passive occupation, as in the paper.
Also Space GENERATION Gain (SGG, Eq. 11, compute_pocket_generation): a rusher is credited with space
his TEAMMATES win because he pulled a blocker's HMM ASSIGNMENT off them. Where Bornn would soft-weight
the drag-off-a-teammate event by a radial kernel of distance, we weight it directly by the assignment
probability theta -- the fitted soft kernel -- showing the assignment framework plugging into
space-control analytics the way it plugs into plus-minus. Bornn's complementary Space RECEIVED (SR)
credits the same freed space to the teammate who benefits from it (every yard generated is received)."""
import numpy as np, pandas as pd
from collections import defaultdict
import pitch_control as pc

VYD = 1.64   # 1.5 m/s running-pace threshold, in yd/s


def _frame_Q(data, n, rad, lam, grid_n):
    """Value-weighted controlled pocket space per present rusher AND per present blocker at frame n.
    Q = sum over a QB disk of [control share * V(p)], V(p)=exp(-|p-QB|/lambda) (value highest at the
    passer). Returns (Qr, rusher slots, Qb, blocker slots) -- both from the SAME space-control density,
    so the lineman's responsibility is encoded by his own control, not an external HMM assignment."""
    G, gx, gy, cell = pc.qb_grid(data["x_qb"][n], half=6.0, n=grid_n)
    dq = np.hypot(G[:, 0] - data["x_qb"][n, 0], G[:, 1] - data["x_qb"][n, 1])
    inreg = dq <= rad
    Ir, ri, Ib, bi, Iqb = pc.frame_fields(G, data, n)
    wr, wb, _ = pc.control_shares(Ir, Ib, Iqb, include_qb=True)
    V = np.exp(-dq / lam)[inreg]
    Qr = cell * (wr[:, inreg] * V[None, :]).sum(1) if len(ri) else np.zeros(0)
    Qb = cell * (wb[:, inreg] * V[None, :]).sum(1) if len(bi) else np.zeros(0)
    return Qr, ri, Qb, bi


def _player_gains(prev, key, k, Q, slots, slot_ids, vel, n, rows):
    """Append consecutive-frame gains for one team's present players; update the prev-Q cache."""
    for j, slot in enumerate(slots):
        pid = int(slot_ids[n, slot]); spd = float(np.hypot(*vel[n, slot])); kk = (pid, key)
        if kk in prev and prev[kk][0] == k - 1:
            rows.append((pid, key, k, Q[j], Q[j] - prev[kk][1], spd))
        prev[kk] = (k, Q[j])


def compute_pocket_gains(data, rad=5.0, lam=3.0, grid_n=40, eps=0.0):
    """Single pass over a pocket design, computing Bornn space gain/loss from the SAME space-control
    density for both sides (no HMM assignment needed -- the control density already encodes who is
    responsible). Returns:
     rusher_df: per-(rusher,play,frame) Q/gain/sog/sol/active -- the rusher's Space Occupation GAIN
       (he wins valuable near-QB ground; SOG = positive gains).
     blocker_df: per-(blocker,play,frame) Q/gain/sog/sol/active -- the lineman's Space Occupation LOSS
       (SOL = he LOSES his value-weighted control of the protected pocket when his man beats him)."""
    N = len(data["x_qb"]); ps = data["play_start"]; pr = data["play_row"]
    prevR = {}; prevB = {}; rrows = []; brows = []
    for n in range(N):
        Qr, ri, Qb, bi = _frame_Q(data, n, rad, lam, grid_n)
        p = int(pr[n]); k = n - ps[p]
        _player_gains(prevR, p, k, Qr, ri, data["rush_slot_id"], data["v_rush"], n, rrows)
        _player_gains(prevB, p, k, Qb, bi, data["blk_slot_id"], data["v_blk"], n, brows)

    def _frame(rows):
        df = pd.DataFrame(rows, columns=["pid", "play", "k", "Q", "gain", "speed"])
        df["sog"] = np.where(df.gain >= eps, df.gain, 0.0)
        df["sol"] = np.where(df.gain <= -eps, -df.gain, 0.0)
        df["active"] = (df.speed > VYD).astype(int)
        return df
    return _frame(rrows).rename(columns={"pid": "rid"}), _frame(brows).rename(columns={"pid": "bid"})


def compute_pocket_sog(data, rad=5.0, lam=3.0, grid_n=40, eps=0.0):
    """Backward-compatible rusher-only entry point."""
    return compute_pocket_gains(data, rad, lam, grid_n, eps)[0]


def _aggregate(df, idcol, metric):
    """Bornn Table-1 analog: per player total/mean of the metric ('sog' or 'sol'), # event frames,
    active share, and the per-frame rate. `metric` is the focal one; the other is also summed."""
    other = "sol" if metric == "sog" else "sog"
    out = df.groupby(idcol).agg(frames=("gain", "size"), tot=(metric, "sum"),
                                tot_other=(other, "sum")).reset_index()
    ev = df[df[metric] > 0].groupby(idcol).agg(n_ev=(metric, "size"), active=("active", "mean")).reset_index()
    out = out.merge(ev, on=idcol, how="left")
    out["mu"] = out.tot / out.n_ev.replace(0, np.nan)
    out["rate"] = out.tot / out.frames                          # per-frame intensity (volume-free)
    return out


def aggregate_by_rusher(df):
    """Per-rusher Space Occupation GAIN summary. sumSOG/muSOG/active/rate (+ sumSOL)."""
    a = _aggregate(df, "rid", "sog")
    return a.rename(columns={"tot": "sumSOG", "mu": "muSOG", "tot_other": "sumSOL", "n_ev": "nSOG"})


def aggregate_by_blocker(df):
    """Per-lineman Space Occupation LOSS summary. sumSOL/muSOL/active/rate (+ sumSOG recovered)."""
    a = _aggregate(df, "bid", "sol")
    return a.rename(columns={"tot": "sumSOL", "mu": "muSOL", "tot_other": "sumSOG", "n_ev": "nSOL"})


def _frame_theta_Q(data, n, rad, lam, grid_n):
    """For frame n: {rid: owned-space Q} and {bid: {rid: HMM attention theta(b,rid)}} over present
    players (theta is the fitted blocker->rusher assignment, data['theta'][n] = (B, R) slot-indexed)."""
    Qr, ri, _, _ = _frame_Q(data, n, rad, lam, grid_n)
    rids = data["rush_slot_id"][n][ri]
    rQ = {int(rids[j]): float(Qr[j]) for j in range(len(ri))}
    th = data["theta"][n]; bi = np.where(data["bmask"][n] > 0)[0]
    btheta = {int(data["blk_slot_id"][n][bs]): {int(rids[j]): float(th[bs, ri[j]])
              for j in range(len(ri))} for bs in bi}
    return rQ, btheta


def compute_pocket_generation(data, w=5, gain_eps=0.0, pairs=False, rad=5.0, lam=3.0, grid_n=40):
    """Fernandez-Bornn Space Generation Gain (SGG), pocket analog (Wide Open Spaces, Eq. 10-11).

    A GENERATOR rusher i pulls a blocker b off a teammate (RECEIVER) rusher i', freeing i' to win
    near-QB space; the generator is credited with the receiver's space gain. This shows the HMM
    ASSIGNMENT framework plugging into the space-control analytics the way it plugs into plus-minus.

    Bornn's Eq. 10 decides ``b is on this player'' with a HARD distance cutoff 1[d<=delta] -- but he
    could equally have used a soft radial kernel exp(-d/ell) of distance. Our analog of that soft kernel
    is the HMM assignment PROBABILITY theta itself: a fitted, in-[0,1] membership of ``b is blocking
    this rusher'' that already grades engagement (no distance, no threshold). So theta replaces Bornn's
    distance kernel directly, and the ``b switched from i' onto i'' event over [t, t+w] is weighted by
    the probabilities,
        W = theta(b,i';t) * (1 - theta(b,i';t+w)) * theta(b,i;t+w) * (1 - theta(b,i;t))
            [   on i' @t  ] [     left i' @t+w     ] [   on i @t+w  ] [  not on i @t   ]
    and the generator is credited (Eq. 11) with that fraction of the receiver's owned-space gain,
        SGG_i += W * (Q_{i'}(t+w) - Q_{i'}(t))     when the gain > gain_eps
    (Q is the value-weighted near-QB control of _frame_Q). So a rusher scores -- in proportion to how
    much a blocker's assignment left a teammate for him -- for the space that teammate then wins: the
    space-control realization of the assignment-based ``drawing attention frees teammates'' spillover.
    Requires a pocket design built with HMM attention (data['theta']).

    Bornn's complementary view, SPACE RECEIVED (SR), credits the SAME W*gain to the freed receiver i'
    instead of the generator i -- who benefits most from space others create. Every yard generated is a
    yard received, so the league totals of sumSGG and sumSR match: two sides of one transaction.

    Returns a per-rusher DataFrame: rid, sumSGG/nSGG (space generated for teammates + effective switch
    weight) and sumSR/nSR (space received from teammates' generation). With pairs=True also returns a
    per-(generator,receiver) DataFrame of the directed flow -- who frees whom within a front.
    """
    if "theta" not in data:
        raise ValueError("pocket design must carry HMM attention; build with assignment=...")
    ps = data["play_start"]
    sgg = defaultdict(float); cnt = defaultdict(float)
    srv = defaultdict(float); rcnt = defaultdict(float)         # space RECEIVED by the freed teammate
    pair = defaultdict(float)                                   # (generator rid, receiver rid) -> space
    for p in range(len(ps) - 1):
        a0, b0 = int(ps[p]), int(ps[p + 1]); L = b0 - a0
        if L <= w:
            continue
        rQ = [None] * L; bth = [None] * L
        for off in range(L):                                    # pass 1: owned space + attention
            rQ[off], bth[off] = _frame_theta_Q(data, a0 + off, rad, lam, grid_n)
        for t in range(L - w):                                  # pass 2: windowed assignment switch
            tw = t + w
            for bid, th_w in bth[tw].items():
                th_t = bth[t].get(bid)
                if th_t is None:
                    continue
                for i, qi_w in th_w.items():                    # generator: b's assignment onto i @t+w
                    onset = qi_w * (1.0 - th_t.get(i, 0.0))     # ... that was not on i @t
                    if onset <= 0.0 or i not in rQ[t] or i not in rQ[tw]:
                        continue
                    for ip, qip_t in th_t.items():              # receiver: b was on i' @t ...
                        leave = qip_t * (1.0 - th_w.get(ip, 0.0))   # ... and left i' by @t+w
                        if ip == i or leave <= 0.0 or ip not in rQ[t] or ip not in rQ[tw]:
                            continue
                        gain = rQ[tw][ip] - rQ[t][ip]           # receiver's owned-space gain (Eq. 11)
                        if gain > gain_eps:
                            wt = onset * leave                  # soft switch weight W (theta kernel)
                            sgg[i] += wt * gain; cnt[i] += wt   # generator i made the space ...
                            srv[ip] += wt * gain; rcnt[ip] += wt   # ... and receiver i' got it
                            pair[(i, ip)] += wt * gain          # ... in the directed flow i -> i'
    df = pd.DataFrame([(rid, sgg[rid], cnt[rid], srv[rid], rcnt[rid]) for rid in set(sgg) | set(srv)],
                      columns=["rid", "sumSGG", "nSGG", "sumSR", "nSR"])
    if not pairs:
        return df
    pdf = pd.DataFrame([(g, r, v) for (g, r), v in pair.items()],
                       columns=["gen_rid", "rec_rid", "space"])
    return df, pdf


def _frame_theta_Q_byrusher(data, n, rad, lam, grid_n):
    """For frame n: {rid: owned-space Q} and {rid: {bid: theta(b,rid)}} -- each rusher's distribution
    over the blockers responsible for him (the transpose of _frame_theta_Q, used for the blocker
    concession dual)."""
    Qr, ri, _, _ = _frame_Q(data, n, rad, lam, grid_n)
    rids = data["rush_slot_id"][n][ri]; bi = np.where(data["bmask"][n] > 0)[0]
    bids = data["blk_slot_id"][n][bi]; th = data["theta"][n]
    rQ = {int(rids[j]): float(Qr[j]) for j in range(len(ri))}
    rtheta = {int(rids[j]): {int(bids[k]): float(th[bi[k], ri[j]]) for k in range(len(bi))}
              for j in range(len(ri))}
    return rQ, rtheta


def compute_pocket_concession(data, w=5, gain_eps=0.0, pairs=False, rad=5.0, lam=3.0, grid_n=40):
    """Space CONCEDED -- the blocker-side dual of compute_pocket_generation (Wide Open Spaces, Eq. 10-11).

    Where the rusher metric tracks a blocker's assignment shifting *among rushers* (drawing attention),
    this tracks a RUSHER's assignment shifting *among blockers* -- a pass-off. Over [t, t+w] rusher j's
    responsibility moves from a RELEASING blocker i' to a RECEIVING blocker i; if j wins valuable near-QB
    space in that window, the blockers CONCEDED it. The handoff is weighted by the same theta soft kernel,
        W = theta(i',j;t)*(1-theta(i',j;t+w)) * theta(i,j;t+w)*(1-theta(i,j;t)),
    and that fraction of the rusher's space gain Q_j(t+w)-Q_j(t) is charged to BOTH blockers:
      - the releaser i' as Space Conceded GENERATED (SCG)  -- handed his man off into space, and
      - the receiver i  as Space Conceded RECEIVED  (SCR)  -- failed to pick up the stunt.
    High concession = poor pass-off / stunt pickup, so unlike the rusher metric it should correlate
    POSITIVELY with PFF pressures allowed. By construction the league totals of SCG and SCR coincide
    (every yard conceded is both released and received). Requires data['theta'].

    Returns a per-blocker DataFrame: bid, sumSCG/nSCG (conceded by releasing), sumSCR/nSCR (conceded by
    receiving). With pairs=True also returns the per-(releaser,receiver) directed flow -- who hands off
    to whom into space.
    """
    if "theta" not in data:
        raise ValueError("pocket design must carry HMM attention; build with assignment=...")
    ps = data["play_start"]
    scg = defaultdict(float); gcnt = defaultdict(float)        # conceded by RELEASING the rusher
    scr = defaultdict(float); rcnt = defaultdict(float)        # conceded by RECEIVING the rusher
    pair = defaultdict(float)                                  # (releaser bid, receiver bid) -> conceded
    for p in range(len(ps) - 1):
        a0, b0 = int(ps[p]), int(ps[p + 1]); L = b0 - a0
        if L <= w:
            continue
        rQ = [None] * L; rth = [None] * L
        for off in range(L):                                  # pass 1: owned space + per-rusher blocker dist
            rQ[off], rth[off] = _frame_theta_Q_byrusher(data, a0 + off, rad, lam, grid_n)
        for t in range(L - w):                                # pass 2: windowed pass-off detection
            tw = t + w
            for jid, bd_w in rth[tw].items():                 # rusher j, his blocker distribution at t+w
                bd_t = rth[t].get(jid)
                if bd_t is None or jid not in rQ[t] or jid not in rQ[tw]:
                    continue
                gain = rQ[tw][jid] - rQ[t][jid]               # rusher's near-QB space gain = space conceded
                if gain <= gain_eps:
                    continue
                for i, qi_w in bd_w.items():                  # receiver: blocker i took on j @t+w ...
                    onset = qi_w * (1.0 - bd_t.get(i, 0.0))   # ... not on j @t
                    if onset <= 0.0:
                        continue
                    for ip, qip_t in bd_t.items():            # releaser: blocker i' was on j @t ...
                        leave = qip_t * (1.0 - bd_w.get(ip, 0.0))   # ... and let j go by @t+w
                        if ip == i or leave <= 0.0:
                            continue
                        wt = onset * leave                    # soft handoff weight W (theta kernel)
                        scg[ip] += wt * gain; gcnt[ip] += wt  # releaser i' conceded by handing off
                        scr[i] += wt * gain; rcnt[i] += wt    # receiver i  conceded by failing to pick up
                        pair[(ip, i)] += wt * gain            # directed flow i' -> i
    df = pd.DataFrame([(bid, scg[bid], gcnt[bid], scr[bid], rcnt[bid]) for bid in set(scg) | set(scr)],
                      columns=["bid", "sumSCG", "nSCG", "sumSCR", "nSCR"])
    if not pairs:
        return df
    pdf = pd.DataFrame([(g, r, v) for (g, r), v in pair.items()],
                       columns=["rel_bid", "rec_bid", "space"])
    return df, pdf
