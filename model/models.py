import jax
import jax.numpy as jnp
import numpyro
from abc import ABC, abstractmethod
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive, autoguide


class BaseModel(ABC):
    def __init__(self):
        self.samples = None

    @staticmethod
    def _vec(name, n):
        """non-centered standard-normal vector of per-entity effects"""
        return numpyro.sample(name, dist.Normal(0.0, 1.0).expand([n]).to_event(1))

    @abstractmethod
    def model_fn(self, data):
        ...

    def run_mcmc_inference(self, data, num_warmup=1000, num_samples=500, num_chains=4, seed=0):
        kernel = NUTS(self.model_fn, target_accept_prob=0.9)
        self.mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                         num_chains=num_chains, progress_bar=True)
        self.mcmc.run(jax.random.PRNGKey(seed), data)
        self.samples = self.mcmc.get_samples()
        return self.samples

    def run_svi_inference(self, data, num_steps=20000, lr=1e-2, seed=0, n_draws=1000):
        self.guide = autoguide.AutoNormal(self.model_fn)
        svi = SVI(self.model_fn, self.guide, numpyro.optim.Adam(lr), Trace_ELBO())
        self.svi_result = svi.run(jax.random.PRNGKey(seed), num_steps, data, progress_bar=True)
        self.samples = self.guide.sample_posterior(
            jax.random.PRNGKey(seed + 1), self.svi_result.params, sample_shape=(n_draws,))
        return self.samples

    def get_posterior_samples(self):
        return self.samples

    def predict(self, data, seed=2):
        return Predictive(self.model_fn, posterior_samples=self.samples)(
            jax.random.PRNGKey(seed), data)


class RusherPlusMinusModel(BaseModel):
    """Play-level max-strain plus-minus model.

    For each (play n, rusher r) we model that rusher's peak strain as its (attribute-
    centered) effect, COUNTERED by the attention-weighted sum of the blockers assigned
    to it, plus game-situation effects. The play's max strain over rushers is the
    reported metric. See build_play_design in src/feature_engineering.py for the
    expected `data` dict.
    """

    def model_fn(self, data: dict):
        assignment = data["assignment"]              # (N, R, B) time-avg attention
        outcome, mask = data["outcome"], data["mask"].astype(bool)  # (N, R)
        rusher_ids, blocker_ids = data["rusher_ids"], data["blocker_ids"]  # (N,R), (N,B)
        qb_ids, off_ids, def_ids = data["quarterback_ids"], data["offense_ids"], data["defense_ids"]
        down_ids, quarter_ids = data["down_ids"], data["quarter_ids"]
        oform_ids, dform_ids = data["offensive_formation_ids"], data["defensive_formation_ids"]
        Rc, Bc, Qc = data["rusher_covariates"], data["blocker_covariates"], data["quarterback_covariates"]
        score_diff = data["score_diff_covariates"]
        time_remaining = data["time_remaining_covariates"]
        yard_line = data["yard_line_covariates"]

        intercept = numpyro.sample("intercept", dist.Normal(0.0, 1.0))
        # attribute-coefficient vectors (position/height/weight archetypes)
        a_R = self._vec("rusher_weight", Rc.shape[1])
        a_B = self._vec("blocker_weight", Bc.shape[1])
        a_Q = self._vec("quarterback_weight", Qc.shape[1])

        ig = lambda nm: numpyro.sample(nm, dist.InverseGamma(2.0, 1.0))
        s_R, s_B, s_Q = ig("sigma_rusher"), ig("sigma_blocker"), ig("sigma_quarterback")
        s_off, s_def = ig("sigma_offense"), ig("sigma_defense")
        s_down, s_qtr = ig("sigma_down"), ig("sigma_quarter")
        s_of, s_df = ig("sigma_offensive_formation"), ig("sigma_defensive_formation")
        sigma = ig("sigma")

        # non-centered, attribute-centered random effects
        rusher_effect = Rc @ a_R + s_R * self._vec("z_rusher", data["N_rushers"])
        blocker_effect = Bc @ a_B + s_B * self._vec("z_blocker", data["N_blockers"])
        qb_effect = Qc @ a_Q + s_Q * self._vec("z_quarterback", data["N_quarterbacks"])
        # zero-centered random intercepts
        off_effect = s_off * self._vec("z_offense", data["N_offense"])
        def_effect = s_def * self._vec("z_defense", data["N_defense"])
        down_effect = s_down * self._vec("z_down", data["N_down"])
        qtr_effect = s_qtr * self._vec("z_quarter", data["N_quarter"])
        of_effect = s_of * self._vec("z_offensive_formation", data["N_offensive_formation"])
        df_effect = s_df * self._vec("z_defensive_formation", data["N_defensive_formation"])
        # numeric situation coefficients
        g_score = numpyro.sample("score_diff_theta", dist.Normal(0.0, 1.0))
        g_time = numpyro.sample("time_remaining_theta", dist.Normal(0.0, 1.0))
        g_yard = numpyro.sample("yard_line_theta", dist.Normal(0.0, 1.0))

        # attention-weighted blocker sum, per (play, rusher), SUBTRACTED (countering)
        blocker_sum = jnp.einsum("nrb,nb->nr", assignment, blocker_effect[blocker_ids])
        play = (intercept + qb_effect[qb_ids] + off_effect[off_ids] + def_effect[def_ids]
                + down_effect[down_ids] + qtr_effect[quarter_ids]
                + of_effect[oform_ids] + df_effect[dform_ids]
                + g_score * score_diff + g_time * time_remaining + g_yard * yard_line)
        strain = rusher_effect[rusher_ids] - blocker_sum + play[:, None]  # (N, R)

        with numpyro.handlers.mask(mask=mask):
            numpyro.sample("y", dist.Normal(strain, sigma), obs=outcome)


class ContinuousRusherPlusMinusModel(BaseModel):
    """Continuous-time (AR(1)) per-rusher plus-minus model.

    Observation = (play, rusher, frame). Forecasts that rusher's OWN next-frame strain
    (STRAIN_{i,j,t+1}) from its OWN current-frame strain via a single shared AR(1)
    coefficient rho in (-1,1), plus a per-rusher (attribute-centered) intercept countered
    by the frame-level attention-weighted blocker sum, plus game-situation effects. Each
    rusher-frame is therefore a distinct observation (unlike a shared play-frame max), and
    the play's peak pressure is a reported max over rushers, not the regression target.
    See build_continuous_design in src/feature_engineering.py for the (long) `data` dict.
    """

    def model_fn(self, data: dict):
        assignment = data["assignment"]              # (N_obs, B) frame-t attention
        outcome = data["outcome"]                    # (N_obs,) this rusher's OWN STRAIN at t+1
        strain_current = data["strain_current"]      # (N_obs,) this rusher's own STRAIN_t (AR predictor)
        rusher_ids, blocker_ids = data["rusher_ids"], data["blocker_ids"]
        qb_ids, off_ids, def_ids = data["quarterback_ids"], data["offense_ids"], data["defense_ids"]
        down_ids, quarter_ids = data["down_ids"], data["quarter_ids"]
        oform_ids, dform_ids = data["offensive_formation_ids"], data["defensive_formation_ids"]
        Rc, Bc, Qc = data["rusher_covariates"], data["blocker_covariates"], data["quarterback_covariates"]
        score_diff = data["score_diff_covariates"]
        time_remaining = data["time_remaining_covariates"]
        yard_line = data["yard_line_covariates"]

        intercept = numpyro.sample("intercept", dist.Normal(0.0, 1.0))
        rho = numpyro.sample("rho_ar", dist.Uniform(-1.0, 1.0))  # shared stationary AR(1)
        a_R = self._vec("rusher_weight", Rc.shape[1])
        a_B = self._vec("blocker_weight", Bc.shape[1])
        a_Q = self._vec("quarterback_weight", Qc.shape[1])

        ig = lambda nm: numpyro.sample(nm, dist.InverseGamma(2.0, 1.0))
        s_R, s_B, s_Q = ig("sigma_rusher"), ig("sigma_blocker"), ig("sigma_quarterback")
        s_off, s_def = ig("sigma_offense"), ig("sigma_defense")
        s_down, s_qtr = ig("sigma_down"), ig("sigma_quarter")
        s_of, s_df = ig("sigma_offensive_formation"), ig("sigma_defensive_formation")
        sigma = ig("sigma")

        rusher_effect = Rc @ a_R + s_R * self._vec("z_rusher", data["N_rushers"])
        blocker_effect = Bc @ a_B + s_B * self._vec("z_blocker", data["N_blockers"])
        qb_effect = Qc @ a_Q + s_Q * self._vec("z_quarterback", data["N_quarterbacks"])
        off_effect = s_off * self._vec("z_offense", data["N_offense"])
        def_effect = s_def * self._vec("z_defense", data["N_defense"])
        down_effect = s_down * self._vec("z_down", data["N_down"])
        qtr_effect = s_qtr * self._vec("z_quarter", data["N_quarter"])
        of_effect = s_of * self._vec("z_offensive_formation", data["N_offensive_formation"])
        df_effect = s_df * self._vec("z_defensive_formation", data["N_defensive_formation"])
        g_score = numpyro.sample("score_diff_theta", dist.Normal(0.0, 1.0))
        g_time = numpyro.sample("time_remaining_theta", dist.Normal(0.0, 1.0))
        g_yard = numpyro.sample("yard_line_theta", dist.Normal(0.0, 1.0))

        # frame-level attention-weighted blocker sum, per observation, SUBTRACTED
        blocker_sum = jnp.einsum("ob,ob->o", assignment, blocker_effect[blocker_ids])
        strain = (intercept + rho * strain_current + rusher_effect[rusher_ids] - blocker_sum
                  + qb_effect[qb_ids] + off_effect[off_ids] + def_effect[def_ids]
                  + down_effect[down_ids] + qtr_effect[quarter_ids]
                  + of_effect[oform_ids] + df_effect[dform_ids]
                  + g_score * score_diff + g_time * time_remaining + g_yard * yard_line)
        numpyro.sample("y", dist.Normal(strain, sigma), obs=outcome)


class ContinuousBlockerDeltaModel(BaseModel):
    """First-difference blocker model -- isolates the blocker effect from the rusher confound.

    Outcome = DELTA STRAIN = STRAIN_{t+1} - STRAIN_t. A strict first-difference of the STRAIN level
    model STRAIN_t = alpha_j (rusher FE) + gamma_play (play context) - sum_b theta(b,j,t) B_b + eps
    cancels EVERYTHING time-invariant within a play -- the rusher ability alpha_j AND all play-level
    controls (QB, offense/defense, down/quarter/formation, situation are all play-constant). Only the
    time-varying blocking survives:

        DELTA STRAIN = intercept - sum_b dtheta(b,j,t) B_b + eps,   dtheta = theta(t+1) - theta(t)

    so the per-blocker effect B_b is identified purely from how a rusher's strain responds when blocker
    b's attention on him SHIFTS -- free of which rushers/alignment he faces. B_b is attribute-centered
    (Bc @ a_B + s_B z_b) as in the plus-minus model; SUBTRACTED, so B_b>0 = a lineman whose increased
    engagement decelerates the rush (good) and validates NEGATIVELY vs PFF pressures allowed. The
    intercept is a single global per-frame strain drift (the only play-invariant survivor). Expects
    data["dtheta"] (N_obs, B) = assignment_next - assignment. See build_continuous_design.
    """

    def model_fn(self, data: dict):
        dtheta = data["dtheta"]                       # (N_obs, B) theta(t+1) - theta(t)
        blocker_ids = data["blocker_ids"]
        outcome = data["outcome"]                     # (N_obs,) DELTA STRAIN
        Bc = data["blocker_covariates"]

        intercept = numpyro.sample("intercept", dist.Normal(0.0, 1.0))  # global strain drift
        a_B = self._vec("blocker_weight", Bc.shape[1])
        s_B = numpyro.sample("sigma_blocker", dist.InverseGamma(2.0, 1.0))
        sigma = numpyro.sample("sigma", dist.InverseGamma(2.0, 1.0))
        blocker_effect = Bc @ a_B + s_B * self._vec("z_blocker", data["N_blockers"])

        blocker_sum = jnp.einsum("ob,ob->o", dtheta, blocker_effect[blocker_ids])
        mean = intercept - blocker_sum
        numpyro.sample("y", dist.Normal(mean, sigma), obs=outcome)


class ContinuousBlockerDoseModel(BaseModel):
    """Dose-response blocker model -- treatment is the assignment LEVEL (dose) at time t, not its change.

        DELTA STRAIN_{j,t} = STRAIN_{t+1} - STRAIN_t = intercept - sum_b theta(b,j,t) B_b [+ rho STRAIN_t] + eps

    The outcome is still the one-frame strain change (so the rusher's strain LEVEL differences out of the
    response), but the regressor is the dose theta(b,j,t) a rusher is receiving AT t -- a genuine
    dose->response: how much blocker b's engagement now decelerates the rush over the next frame. Unlike
    the first-difference (dtheta) model, this credits SUSTAINED engagement (a blocker locked on at theta
    approx constant has dtheta approx 0 and is invisible there, but a full dose here). B_b is
    attribute-centered and SUBTRACTED, so B_b>0 = a lineman whose engagement decelerates the rush (good),
    validating NEGATIVELY vs PFF pressures allowed.

    Regressing a change on a level predictor is exposed to regression-to-the-mean (theta(t) correlates
    with STRAIN(t), which predicts DELTA STRAIN by mean reversion). Setting data['baseline']=STRAIN_t adds
    a rho*STRAIN_t control to absorb that baseline dependence (at the cost of conditioning on a mediator);
    if B_b is stable with vs without it, the dose effect is not merely mean reversion. Expects
    data['dose'] (N_obs,B)=theta(.,j,t); optional data['baseline'] (N_obs,)=STRAIN_t.
    """

    def model_fn(self, data: dict):
        dose = data["dose"]                           # (N_obs, B) theta(b,j,t) -- the dose at t
        blocker_ids = data["blocker_ids"]
        outcome = data["outcome"]                     # (N_obs,) DELTA STRAIN
        Bc = data["blocker_covariates"]

        intercept = numpyro.sample("intercept", dist.Normal(0.0, 1.0))
        a_B = self._vec("blocker_weight", Bc.shape[1])
        s_B = numpyro.sample("sigma_blocker", dist.HalfNormal(0.05))   # data-scaled: IG(2,1) put ~0
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))          # density at the true tiny scales
        # CENTERED parameterization: sample B_b directly ~ Normal(attribute mean, s_B). With ~2000
        # obs/blocker the likelihood pins each B_b, so centered avoids the non-centered funnel that
        # collapsed the NUTS step size (tiny s_B -> microscopic steps, max-tree-depth trajectories).
        blocker_effect = numpyro.sample("blocker_effect", dist.Normal(Bc @ a_B, s_B).to_event(1))

        mean = intercept - jnp.einsum("ob,ob->o", dose, blocker_effect[blocker_ids])
        if "baseline" in data:                        # optional baseline-strain control (mean-reversion test)
            rho = numpyro.sample("rho", dist.Normal(0.0, 1.0))
            mean = mean + rho * data["baseline"]
        numpyro.sample("y", dist.Normal(mean, sigma), obs=outcome)


class BlockFailureHazardModel(BaseModel):
    """Matchup survival model of block failure, built on the null-state 'beaten' events.

    Each engagement spell pairs a blocker $b$ with the rusher $j$ he primarily engages; we observe a
    time-to-beat $t_i$ (exposure, in seconds) and an event indicator $e_i$ (1 if the block was beaten,
    0 if censored at the throw). An exponential frailty hazard decomposes the per-spell failure rate
    into crossed blocker (hold) and rusher (shed) random effects:

        h_i = exp(alpha + u_{b(i)} + v_{j(i)}),   u_b ~ N(0, s_b^2),  v_j ~ N(0, s_r^2)
        log L_i = e_i * log h_i - h_i * t_i        (exponential survival likelihood)

    u_b < 0 is a lineman who holds longer (lower failure hazard) NET of the rushers he faced; v_j > 0
    is a rusher who beats blocks faster NET of the blockers he faced. The crossed effects net each
    side out of the other, fixing the elite-tackles-face-elite-rushers confound. The expected hold
    time of a matchup is 1/h_{bj}. Constant-hazard (exponential) per spell, matching the Phase-2.5
    p_fail; a Weibull shape is a natural extension.
    """

    def model_fn(self, data: dict):
        t = jnp.asarray(data["time"]); e = jnp.asarray(data["event"])
        bid = jnp.asarray(data["blocker_id"]); rid = jnp.asarray(data["rusher_id"])
        Nb, Nr = int(data["N_blockers"]), int(data["N_rushers"])

        alpha = numpyro.sample("alpha", dist.Normal(jnp.log(float(data["base_hazard"])), 1.0))
        # tight half-normal priors regularize the crossed frailties: with heavy censoring (~92%) and
        # sparse beats, loose scales overfit (extreme per-player hazards, absurd exponential-tail
        # hold times). 0.4 keeps per-player log-hazard shifts to O(1) -- a few-fold hazard range.
        s_b = numpyro.sample("sigma_b", dist.HalfNormal(0.4))   # blocker (hold) frailty scale
        s_r = numpyro.sample("sigma_r", dist.HalfNormal(0.4))   # rusher (shed) frailty scale
        u = s_b * self._vec("z_blocker", Nb)                    # log-hazard shift per blocker
        v = s_r * self._vec("z_rusher", Nr)                     # log-hazard shift per rusher
        logh = alpha + u[bid] + v[rid]
        numpyro.factor("surv", jnp.sum(e * logh - jnp.exp(logh) * t))  # exponential survival


class BlockHoldDiscreteModel(BaseModel):
    """Discrete-time (per-frame) logistic hazard of block failure -- the bounded counterpart of the
    exponential frailty, avoiding its separation blow-up. For each frame a blocker remains engaged,

        logit h_{i,t} = alpha + g1 * tstd + g2 * tstd^2 + beta_opp * R_opp(j) + u_{b}

    where tstd is standardized frames-since-engagement (a dwell/duration effect), R_opp(j) is the
    engaged rusher's STRAIN plus-minus as a FIXED opponent-quality covariate (not a frailty -- the
    rusher frailty was invalid), and u_b is a tightly-regularized blocker hold frailty. The sigmoid is
    bounded so never-beaten blockers shrink rather than diverge. Hold rating = -u_b, opponent-adjusted
    (beta_opp nets out the quality of rushers faced) -- the test is whether this survives WITHIN
    position, where every marginal blocker metric fails."""

    def model_fn(self, data: dict):
        tstd = jnp.asarray(data["t"]); t2 = jnp.asarray(data["t2"])
        ropp = jnp.asarray(data["r_opp"]); bid = jnp.asarray(data["blocker_id"])
        y = jnp.asarray(data["beaten"]); Nb = int(data["N_blockers"])

        alpha = numpyro.sample("alpha", dist.Normal(-3.0, 2.0))     # rare per-frame beat
        g1 = numpyro.sample("g1", dist.Normal(0.0, 1.0))            # dwell dependence
        g2 = numpyro.sample("g2", dist.Normal(0.0, 1.0))
        beta_opp = numpyro.sample("beta_opp", dist.Normal(0.0, 1.0))  # opponent quality -> hazard (expect >0)
        s_u = numpyro.sample("sigma_b", dist.HalfNormal(0.5))       # blocker hold frailty scale (regularized)
        u = s_u * self._vec("z_blocker", Nb)
        logit = alpha + g1 * tstd + g2 * t2 + beta_opp * ropp + u[bid]
        numpyro.sample("y", dist.Bernoulli(logits=logit), obs=y)


class QBForceFieldModel(BaseModel):
    """Structural QB force-field model (milestone).

    Regresses the OBSERVED 2D QB acceleration a_QB,t = (d2x_smooth_qb, d2y_smooth_qb) on a sum
    of per-rusher screened-Coulomb repulsion forces, each (i) SHIELDED by the rusher's blocking
    engagement (a blocked rusher's repulsive charge is screened by the blocker) and (ii) scaled
    by the rusher's CLOSING kinematics (a standing rusher exerts no force):

        a_QB,t ~ Normal( sum_k F_k,t + drift , sigma^2 I_2 )
        F_k,t  = g(e_k) * Phi(d_k) * Psi(s_k, a_k) * uhat_{k->QB}
        g(e)   = sigmoid(eta0 - eta1 * e)                # blocker shielding; eta1>0 = shields
        Phi(d) = exp(-d/ell) / (d + d0)^2                # screened Coulomb (Yukawa), soft-core d0
        Psi    = beta_v * relu(s) + beta_a * relu(a)     # closing speed/accel toward QB; 0 if standing

    Fitting needs NO ODE solver: acceleration is observed, so this is an instantaneous force-
    balance regression. The overall scale kappa is dropped (confounded with beta_v/beta_a, which
    carry the force scale). eta1/beta_v/beta_a use NORMAL priors so their SIGN is a testable
    hypothesis (free rushers push harder; rushers bearing down push harder). drift_mode='const'
    is a single learned 2D bias for the milestone. See build_qb_force_design in feature_engineering.
    """

    def __init__(self, drift_mode="const", kernel="exp", dmin=0.5, d0=0.5, eta0=0.0,
                 use_archetype=False):
        super().__init__()
        self.drift_mode, self.kernel, self.dmin, self.d0 = drift_mode, kernel, dmin, d0
        # eta0 (free-rusher gate level) is FIXED, not sampled: with kappa dropped it is
        # confounded with the beta force-scales (gate level x scale), so we pin it and let
        # only the gate SHAPE eta1 carry the shielding test and beta carry the scale.
        self.eta0 = eta0
        # use_archetype: add an interior-vs-edge charge term to test whether the (mis-signed)
        # engagement gate eta1 is just proxying rusher archetype (interior bull-rush vs edge arc).
        self.use_archetype = use_archetype

    def model_fn(self, data: dict):
        a_qb = jnp.asarray(data["a_qb"])         # (N, 2) observed QB acceleration
        uhat = jnp.asarray(data["uhat"])         # (N, R, 2) rusher->QB unit vectors (repulsion dir)
        dr = jnp.asarray(data["dist"])           # (N, R) rusher-QB distance
        sclose = jnp.asarray(data["sclose"])     # (N, R) rusher closing speed toward QB
        aclose = jnp.asarray(data["aclose"])     # (N, R) rusher closing acceleration toward QB
        dose = jnp.asarray(data["dose"])         # (N, R) engagement dose (sum_b theta)
        rmask = jnp.asarray(data["rmask"])       # (N, R) 1 where a real rusher occupies the slot

        eta1 = numpyro.sample("eta1", dist.Normal(0.0, 1.0))    # shielding shape (sign-test)
        beta_v = numpyro.sample("beta_v", dist.Normal(0.0, 5.0))  # closing-speed force scale (sign-test)
        beta_a = numpyro.sample("beta_a", dist.Normal(0.0, 5.0))  # closing-accel force scale
        ell = numpyro.sample("ell", dist.LogNormal(jnp.log(3.0), 0.5))  # decay/screening length (yd)
        sigma = numpyro.sample("sigma", dist.InverseGamma(2.0, 1.0))
        drift = numpyro.sample("drift", dist.Normal(0.0, 1.0).expand([2]).to_event(1))

        gate = jax.nn.sigmoid(self.eta0 - eta1 * dose)          # (N, R) blocker shielding (eta1>0 shields)
        dclip = jnp.maximum(dr, self.dmin)
        if self.kernel == "screened":
            Phi = jnp.exp(-dclip / ell) / (dclip + self.d0) ** 2  # screened Coulomb (ell weakly identified)
        elif self.kernel == "coulomb":
            Phi = 1.0 / (dclip + self.d0) ** 2                    # pure soft-core Coulomb (ell unused)
        else:
            Phi = jnp.exp(-dclip / ell)                           # plain exponential (ell identified)
        Psi = beta_v * jnp.maximum(sclose, 0.0) + beta_a * jnp.maximum(aclose, 0.0)  # 0 if standing
        fmag = gate * Phi * Psi * rmask                         # (N, R) scalar force, padded slots -> 0
        if self.use_archetype:                                 # interior-vs-edge charge control
            gamma_int = numpyro.sample("gamma_int", dist.Normal(0.0, 1.0))
            fmag = fmag * jnp.exp(gamma_int * jnp.asarray(data["is_interior"]))
        mean = jnp.einsum("nr,nrd->nd", fmag, uhat) + drift     # (N, 2) superposed repulsion + drift
        numpyro.sample("a", dist.Normal(mean, sigma).to_event(1), obs=a_qb)


class QBSpatialFieldModel(BaseModel):
    """Spatial-field model of QB motion: the push on the QB is the negative gradient of a SUM of
    per-rusher ANISOTROPIC Gaussian pressure potentials, evaluated at the QB. Each rusher deposits a
    pressure bump centered on himself, elongated ALONG his direction of motion -- a rusher bearing
    down threatens the space ahead of him; one running the arc threatens only tangentially:

        a_QB,t ~ Normal( sum_k q_k * psi_k * (Sigma_k^-1 d_k) + drift , sigma^2 I_2 )
        d_k        = x_QB - x_k                                  # rusher -> QB displacement
        Sigma_k^-1 = (1/l_perp^2) I + (1/l_par^2 - 1/l_perp^2) vhat_k vhat_k^T   # motion-aligned precision
        psi_k      = exp(-1/2 d_k^T Sigma_k^-1 d_k)             # anisotropic Gaussian bump
        q_k        = softplus(c0 + c_a relu(aclose_k))          # rushers DRIVING in carry more charge

    The force direction is NOT purely radial: anisotropy bends it, so an edge rusher whose velocity is
    tangential to the QB produces little QB-ward force WITHOUT any engagement gate -- geometry replaces
    the mis-signed shielding gate of QBForceFieldModel. l_par >= l_perp keeps the field reaching
    further forward. The same superposition, evaluated on a grid, renders a pressure field. No ODE
    solver needed (acceleration observed). See build_qb_force_design for the data dict.
    """

    def __init__(self, vsoft=0.5, profile="exp"):
        super().__init__()
        self.vsoft = vsoft       # soft velocity-normalization (slow rushers -> ~isotropic field)
        self.profile = profile   # radial profile: "exp" (long tails) or "gauss"

    def model_fn(self, data: dict):
        uhat = jnp.asarray(data["uhat"]); dr = jnp.asarray(data["dist"])
        rvel = jnp.asarray(data["rvel"]); aclose = jnp.asarray(data["aclose"])
        is_int = jnp.asarray(data["is_interior"]); rmask = jnp.asarray(data["rmask"])
        a_qb = jnp.asarray(data["a_qb"])
        d = dr[..., None] * uhat                                 # (N,R,2) displacement rusher->QB

        l_perp = numpyro.sample("l_perp", dist.LogNormal(jnp.log(3.0), 0.5))
        dl = numpyro.sample("dl", dist.HalfNormal(3.0))         # l_par = l_perp + dl  (>= l_perp)
        l_par = numpyro.deterministic("l_par", l_perp + dl)
        sigma = numpyro.sample("sigma", dist.InverseGamma(2.0, 1.0))
        drift = numpyro.sample("drift", dist.Normal(0.0, 1.0).expand([2]).to_event(1))
        c0 = numpyro.sample("c0", dist.Normal(0.0, 2.0))
        c_a = numpyro.sample("c_a", dist.Normal(0.0, 2.0))      # closing-accel -> charge
        c_int = numpyro.sample("c_int", dist.Normal(0.0, 1.0))  # interior-DL archetype -> charge (bull rush)

        vmag = jnp.linalg.norm(rvel, axis=-1, keepdims=True)    # (N,R,1)
        vhat = rvel / (vmag + self.vsoft)                       # slow rushers -> small anisotropy
        ip2, ia2 = 1.0 / l_perp ** 2, 1.0 / l_par ** 2
        proj = jnp.sum(d * vhat, axis=-1)                       # (N,R) d . vhat
        Sinv_d = ip2 * d + (ia2 - ip2) * proj[..., None] * vhat  # (N,R,2) Sigma^-1 d
        quad = ip2 * jnp.sum(d * d, axis=-1) + (ia2 - ip2) * proj ** 2  # (N,R) d^T Sigma^-1 d
        q = jax.nn.softplus(c0 + c_a * jnp.maximum(aclose, 0.0) + c_int * is_int)  # (N,R) charge
        if self.profile == "exp":                              # exp(-r): long tails; F = q psi Sinv_d / r
            r = jnp.sqrt(quad + 1e-6)
            psi = jnp.exp(-r)
            grad = Sinv_d / r[..., None]
        else:                                                  # Gaussian bump; F = q psi Sinv_d
            psi = jnp.exp(-0.5 * quad)
            grad = Sinv_d
        F = (q * psi * rmask)[..., None] * grad                # (N,R,2) per-rusher force
        mean = jnp.sum(F, axis=1) + drift                      # (N,2) superposed field gradient
        numpyro.sample("a", dist.Normal(mean, sigma).to_event(1), obs=a_qb)


class QBConeFieldModel(BaseModel):
    """Forward-coned, PER-PLAYER spatial field. Each rusher exerts force only in a CONE ahead of his
    motion -- almost no field behind him (you cannot push backward) -- with a hierarchical per-player
    charge that serves as a shrinkage-regularized rusher rating:

        a_QB,t ~ Normal( sum_k F_k + drift , sigma^2 I_2 )
        F_k    = q_k * rho(r_k) * A(cos_k) * uhat_{k->QB}
        A(cos) = exp(kappa (cos - 1))                  # forward von-Mises cone; cos = uhat . vhat (=1 ahead, ->0 behind)
        rho(r) = exp(-1/2 (r/l)^2) [gauss] | exp(-r/l) [exp]   # radial reach
        q_k    = softplus(c0 + c_a relu(aclose) + c_int is_int) * exp(u_player)   # u_j ~ N(0, tau)

    The angular cone makes the field one-sided (forward only); the per-player log-charge u_j is
    partially pooled so low-snap blitzers shrink toward the population mean. exp(u_j) is the per-player
    field-strength rating. No ODE solver (acceleration observed). See build_qb_force_design.
    """

    def __init__(self, profile="gauss", vsoft=0.5, v0=1.0, use_player=True, direction="radial",
                 use_situation=False):
        super().__init__()
        self.profile, self.vsoft, self.v0, self.use_player = profile, vsoft, v0, use_player
        # direction: "velocity" -> force points along the rusher's MOMENTUM (v_k) plus a small radial
        # contact term; "radial" -> force points rusher->QB (the best-fitting directional variant).
        self.direction = direction
        # use_situation: add a per-QB and per-game-situation 2-D drift, so the rusher field isolates
        # forced motion rather than a mobile QB's volitional/scheme movement.
        self.use_situation = use_situation

    def model_fn(self, data: dict):
        uhat = jnp.asarray(data["uhat"]); dr = jnp.asarray(data["dist"]); rvel = jnp.asarray(data["rvel"])
        aclose = jnp.asarray(data["aclose"]); is_int = jnp.asarray(data["is_interior"])
        rmask = jnp.asarray(data["rmask"]); sid = jnp.asarray(data["rusher_slot_id"])
        a_qb = jnp.asarray(data["a_qb"]); nR = int(data["N_rushers"])

        ell0 = numpyro.sample("ell0", dist.LogNormal(jnp.log(2.0), 0.4))   # base reach (yd)
        ell_v = numpyro.sample("ell_v", dist.HalfNormal(1.0))              # reach growth per unit speed (yd per yd/s)
        kappa = numpyro.sample("kappa", dist.LogNormal(jnp.log(4.0), 0.5))  # forward-gate sharpness
        c_rad = numpyro.sample("c_rad", dist.HalfNormal(2.0))              # radial contact strength (yd/s)
        sigma = numpyro.sample("sigma", dist.InverseGamma(2.0, 1.0))
        drift = numpyro.sample("drift", dist.Normal(0.0, 1.0).expand([2]).to_event(1))
        c0 = numpyro.sample("c0", dist.Normal(0.0, 2.0))
        c_a = numpyro.sample("c_a", dist.Normal(0.0, 2.0))     # closing-accel -> charge

        vmag = jnp.linalg.norm(rvel, axis=-1)                  # (N,R) speed
        vhat = rvel / (vmag[..., None] + self.vsoft)
        cos = jnp.sum(uhat * vhat, axis=-1)                    # (N,R) motion vs QB-direction
        sw = vmag / (vmag + self.v0)                           # speed weight: ~0 slow, ~1 fast
        A = 1.0 - sw * (1.0 - jax.nn.sigmoid(kappa * cos))     # forward gate: kills only fast OVERRUN
        ell = ell0 + ell_v * vmag                             # (N,R) REACH grows with rusher speed
        rho = jnp.exp(-0.5 * (dr / ell) ** 2) if self.profile == "gauss" else jnp.exp(-dr / ell)
        arg = c0 + c_a * jnp.maximum(aclose, 0.0)
        if self.use_player:
            # ATTRIBUTE-CENTERED per-player charge (as in the plus-minus model): u_j = alpha_R^T z_j
            # + tau eps_j over z_j = [position, height, weight], shrinking toward the positional archetype.
            Rc = jnp.asarray(data["rusher_covariates"])        # (N_rushers, F)
            a_R = self._vec("rusher_weight", Rc.shape[1])
            tau = numpyro.sample("tau", dist.HalfNormal(0.5))
            u = Rc @ a_R + tau * self._vec("z_rusher", nR)     # (N_rushers,) attribute-centered log-charge
            arg = arg + u[sid]
        q = jax.nn.softplus(arg)                               # (N,R) charge (bounded; no exp blow-up)
        fmag = q * rho * A * rmask                             # (N,R) scalar magnitude
        vec = rvel + c_rad * uhat if self.direction == "velocity" else uhat  # force direction
        b2 = drift
        if self.use_situation and "quarterback_ids" in data:
            def reff2(nm, n):                                  # 2-D random intercept per level
                s = numpyro.sample("s_" + nm, dist.InverseGamma(2.0, 1.0))
                return s * numpyro.sample("z_" + nm, dist.Normal(0., 1.).expand([int(n), 2]).to_event(2))
            b2 = (drift
                  + reff2("qb", data["N_qb"])[jnp.asarray(data["quarterback_ids"])]
                  + reff2("down", data["N_down"])[jnp.asarray(data["down_ids"])]
                  + reff2("qtr", data["N_quarter"])[jnp.asarray(data["quarter_ids"])]
                  + reff2("cov", data["N_cov"])[jnp.asarray(data["coverage_ids"])]
                  + reff2("form", data["N_form"])[jnp.asarray(data["form_ids"])]
                  + numpyro.sample("g_score", dist.Normal(0., 1.).expand([2]).to_event(1)) * jnp.asarray(data["score_diff_cov"])[:, None]
                  + numpyro.sample("g_time", dist.Normal(0., 1.).expand([2]).to_event(1)) * jnp.asarray(data["time_cov"])[:, None]
                  + numpyro.sample("g_yard", dist.Normal(0., 1.).expand([2]).to_event(1)) * jnp.asarray(data["yard_cov"])[:, None])
        mean = jnp.einsum("nr,nrd->nd", fmag, vec) + b2       # (N,2) field + QB/situation 2-D drift
        numpyro.sample("a", dist.Normal(mean, sigma).to_event(1), obs=a_qb)


class QBPressureModel(BaseModel):
    """Direction-agnostic pressure model -- credit a rusher for FORCING the QB to move (any direction).

    The repulsion models mis-attribute because a step-up is the QB moving TOWARD an in-front rusher,
    which reads as anti-repulsion and wrongly debits him. Here the response is the MAGNITUDE of the
    QB's forced movement |a_QB|, regressed on a sum of per-rusher SCALAR pressures (charge x proximity
    x closing). Any rusher compressing the QB's space -- edge or interior -- is credited, regardless of
    which way the QB escapes:

        |a_QB,t| ~ Normal( b0 + sum_k q_k * exp(-r_k/ell) * relu(closing_k) , sigma )
        q_k = softplus(c0 + u_j),   u_j = alpha_R^T z_j + tau eps_j   (attribute-centered)

    exp(per-player u_j) is the rusher rating: how much he compresses the QB's space per snap.
    """

    def __init__(self, use_player=True):
        super().__init__()
        self.use_player = use_player

    def model_fn(self, data: dict):
        dr = jnp.asarray(data["dist"]); sclose = jnp.asarray(data["sclose"])
        rmask = jnp.asarray(data["rmask"]); sid = jnp.asarray(data["rusher_slot_id"])
        a_qb = jnp.asarray(data["a_qb"]); nR = int(data["N_rushers"])
        m = jnp.linalg.norm(a_qb, axis=-1)                     # (N,) forced-movement magnitude

        ell = numpyro.sample("ell", dist.LogNormal(jnp.log(3.0), 0.4))  # pressure reach (yd)
        b0 = numpyro.sample("b0", dist.HalfNormal(2.0))        # global baseline movement
        c0 = numpyro.sample("c0", dist.Normal(0.0, 2.0))
        sigma = numpyro.sample("sigma", dist.InverseGamma(2.0, 1.0))

        # QB identity + game-situation baseline: a mobile QB / certain situations move the QB
        # for volitional/scheme reasons; absorbing them lets the pressure term isolate FORCED movement.
        baseline = b0
        if "quarterback_ids" in data:
            def reff(nm, n):
                s = numpyro.sample("s_" + nm, dist.InverseGamma(2.0, 1.0))
                return s * self._vec("z_" + nm, int(n))
            baseline = (b0
                        + reff("qb", data["N_qb"])[jnp.asarray(data["quarterback_ids"])]
                        + reff("down", data["N_down"])[jnp.asarray(data["down_ids"])]
                        + reff("qtr", data["N_quarter"])[jnp.asarray(data["quarter_ids"])]
                        + reff("cov", data["N_cov"])[jnp.asarray(data["coverage_ids"])]
                        + reff("form", data["N_form"])[jnp.asarray(data["form_ids"])]
                        + numpyro.sample("g_score", dist.Normal(0., 1.)) * jnp.asarray(data["score_diff_cov"])
                        + numpyro.sample("g_time", dist.Normal(0., 1.)) * jnp.asarray(data["time_cov"])
                        + numpyro.sample("g_yard", dist.Normal(0., 1.)) * jnp.asarray(data["yard_cov"]))

        arg = c0 * jnp.ones(())                                # charge logit
        if self.use_player:
            Rc = jnp.asarray(data["rusher_covariates"])
            a_R = self._vec("rusher_weight", Rc.shape[1])
            tau = numpyro.sample("tau", dist.HalfNormal(0.5))
            u = Rc @ a_R + tau * self._vec("z_rusher", nR)     # attribute-centered per-player charge
            arg = arg + u[sid]
        q = jax.nn.softplus(arg)                               # (N,R) charge
        pressure = q * jnp.exp(-dr / ell) * jnp.maximum(sclose, 0.0) * rmask  # (N,R) scalar compression
        mean = baseline + jnp.sum(pressure, axis=1)            # baseline (QB+situation) + forced movement
        numpyro.sample("m", dist.Normal(mean, sigma), obs=m)


class QBEscapeModel(BaseModel):
    """Thermal-activation / Kramers escape model of QB release. The QB sits in the pocket well;
    rusher pressure lowers the escape barrier, so the per-frame RELEASE hazard rises with total
    pressure. Discrete-time logistic survival on QB-frames (event = ball out / sack at the play's
    last frame). Each rusher's attribute-centered charge u_j = how much he accelerates the forced
    release -- direction-AGNOSTIC and edge-inclusive (a fast-closing edge raises the hazard), and
    grounded in a real, high-signal outcome (the snap-to-throw clock) instead of volitional motion:

        logit P(release at t) = b0 + dur(t) + sum_k u_k phi_k(t) + QB + situation
        phi_k = exp(-r_k/ell) relu(closing_k);   u_j = alpha_R^T z_j + tau eps_j  (attribute-centered)
    """

    def __init__(self, use_situation=True, pressure_kind="proximity", use_blocking=False,
                 use_concept=True, use_openness=False):
        super().__init__()
        self.use_situation = use_situation
        self.use_concept = use_concept   # play-concept controls (dropback/coverage/box/play-action/drop-depth)
        # use_openness: condition the release hazard on downfield receiver openness (space-control). A QB
        # releases when a receiver is open, so g_open>0 expected; controlling for it lets the rusher charge
        # u isolate FORCED (pressure-driven) release from volitional throw-to-open-man release.
        self.use_openness = use_openness
        # pressure_kind: "proximity" -> phi = exp(-r/ell) (a close rusher lowers the barrier regardless
        # of his velocity direction; credits arc/speed edges); "closing" -> times relu(closing speed).
        self.pressure_kind = pressure_kind
        # use_blocking: gate the escape-pressure by HMM engagement (dose = sum_b theta(b,j)). A blocked
        # rusher is contained and should lower the barrier less than a free/shed one. This is where the
        # assignment model feeds the pressure; eta>0 means blocking attenuates pressure (the expected sign).
        self.use_blocking = use_blocking

    def model_fn(self, data: dict):
        dr = jnp.asarray(data["dist"]); sclose = jnp.asarray(data["sclose"])
        rmask = jnp.asarray(data["rmask"]); sid = jnp.asarray(data["rusher_slot_id"])
        Rc = jnp.asarray(data["rusher_covariates"]); nR = int(data["N_rushers"])
        y = jnp.asarray(data["release"]); ts = jnp.asarray(data["t_since_std"])

        ell = numpyro.sample("ell", dist.LogNormal(jnp.log(3.0), 0.4))  # pressure reach (yd)
        b0 = numpyro.sample("b0", dist.Normal(-3.0, 2.0))     # baseline per-frame release log-odds (rare)
        gt1 = numpyro.sample("gt1", dist.Normal(0., 1.))      # duration dependence (must throw eventually)
        gt2 = numpyro.sample("gt2", dist.Normal(0., 1.))
        a_R = self._vec("rusher_weight", Rc.shape[1])
        tau = numpyro.sample("tau", dist.HalfNormal(0.5))
        u = Rc @ a_R + tau * self._vec("z_rusher", nR)        # attribute-centered per-player pressure charge

        phi = jnp.exp(-dr / ell) * rmask                      # (N,R) proximity pressure exposure
        if self.pressure_kind == "closing":
            phi = phi * jnp.maximum(sclose, 0.0)
        if self.use_blocking:                                 # attenuate by HMM engagement (blocked => less)
            eta = numpyro.sample("eta_block", dist.Normal(0., 1.0))
            phi = phi * jnp.exp(-eta * jnp.asarray(data["dose"]))
        logit = b0 + gt1 * ts + gt2 * ts ** 2 + jnp.einsum("nr,nr->n", u[sid], phi)  # + sum_k u_k phi_k

        def reff(nm, n):
            s = numpyro.sample("s_" + nm, dist.InverseGamma(2.0, 1.0))
            return s * self._vec("z_" + nm, int(n))
        if self.use_situation and "quarterback_ids" in data:
            logit = (logit
                     + reff("qb", data["N_qb"])[jnp.asarray(data["quarterback_ids"])]
                     + reff("down", data["N_down"])[jnp.asarray(data["down_ids"])]
                     + reff("qtr", data["N_quarter"])[jnp.asarray(data["quarter_ids"])]
                     + reff("cov", data["N_cov"])[jnp.asarray(data["coverage_ids"])]
                     + reff("form", data["N_form"])[jnp.asarray(data["form_ids"])]
                     + numpyro.sample("g_score", dist.Normal(0., 1.)) * jnp.asarray(data["score_diff_cov"])
                     + numpyro.sample("g_time", dist.Normal(0., 1.)) * jnp.asarray(data["time_cov"])
                     + numpyro.sample("g_yard", dist.Normal(0., 1.)) * jnp.asarray(data["yard_cov"]))
        if self.use_concept and "dropback_ids" in data:    # play-concept de-confounding (quick game etc.)
            logit = (logit
                     + reff("dropback", data["N_dropback"])[jnp.asarray(data["dropback_ids"])]
                     + reff("cov2", data["N_cov2"])[jnp.asarray(data["cov2_ids"])]
                     + numpyro.sample("g_box", dist.Normal(0., 1.)) * jnp.asarray(data["box_cov"])
                     + numpyro.sample("g_pa", dist.Normal(0., 1.)) * jnp.asarray(data["pa_cov"])
                     + numpyro.sample("g_depth", dist.Normal(0., 1.)) * jnp.asarray(data["depth_cov"]))
        if self.use_openness and "open_cov" in data:       # downfield openness raises the release hazard
            logit = logit + numpyro.sample("g_open", dist.Normal(0., 1.)) * jnp.asarray(data["open_cov"])
        numpyro.sample("y", dist.Bernoulli(logits=logit), obs=y)


class QBZoneEscapeModel(BaseModel):
    """QB-specific danger-zone escape model. Each QB has a Gaussian danger zone around him with a
    QB-SPECIFIC radius sigma_q (pocket presence varies); each rusher is a velocity-anisotropic density
    bump (the cone field). The overlap T_k = how much of rusher k's density falls inside the QB's zone
    drives the per-frame RELEASE hazard, and u_k * T_k is the per-rusher attribution:

        T_k = exp(-1/2 d_k^T (Sigma_q + Sigma_k)^{-1} d_k)        # closed-form Gaussian overlap
        Sigma_q = sigma_q^2 I  (sigma_q ~ per-QB);  Sigma_k = lperp^2 I + (lpar^2-lperp^2) vhat vhat^T
        logit P(release at t) = b0 + dur(t) + sum_k u_k T_k + QB-hazard + situation
        u_j = alpha_R^T z_j + tau eps_j   (attribute-centered)

    The QB enters twice -- a baseline release-rate intercept AND his danger-zone radius. Predicts the
    snap-to-throw clock; edge-inclusive (overlap is direction-agnostic in r, anisotropy only adds reach).
    """

    def __init__(self, use_situation=True, aniso=True):
        super().__init__()
        self.use_situation, self.aniso = use_situation, aniso

    def model_fn(self, data: dict):
        uhat = jnp.asarray(data["uhat"]); dr = jnp.asarray(data["dist"]); rvel = jnp.asarray(data["rvel"])
        rmask = jnp.asarray(data["rmask"]); sid = jnp.asarray(data["rusher_slot_id"])
        Rc = jnp.asarray(data["rusher_covariates"]); nR = int(data["N_rushers"])
        y = jnp.asarray(data["release"]); ts = jnp.asarray(data["t_since_std"])
        qb = jnp.asarray(data["quarterback_ids"]); Nqb = int(data["N_qb"])

        b0 = numpyro.sample("b0", dist.Normal(-3.0, 2.0))
        gt1 = numpyro.sample("gt1", dist.Normal(0., 1.)); gt2 = numpyro.sample("gt2", dist.Normal(0., 1.))
        log_sig0 = numpyro.sample("log_sig0", dist.Normal(jnp.log(2.5), 0.3))  # baseline zone radius (yd)
        s_sig = numpyro.sample("s_sig", dist.HalfNormal(0.3))                  # per-QB zone-radius spread
        sigma_q = jnp.exp(log_sig0 + s_sig * self._vec("z_sig_qb", Nqb))[qb]   # (N,) QB-specific radius
        lperp = numpyro.sample("lperp", dist.LogNormal(jnp.log(1.5), 0.4))     # rusher field perp scale
        a_R = self._vec("rusher_weight", Rc.shape[1])
        tau = numpyro.sample("tau", dist.HalfNormal(0.5))
        u = Rc @ a_R + tau * self._vec("z_rusher", nR)                         # attribute-centered charge

        a = sigma_q[:, None] ** 2 + lperp ** 2                                 # (N,1) combined perp variance
        if self.aniso:
            dl = numpyro.sample("dl", dist.HalfNormal(1.5)); b = (lperp + dl) ** 2 - lperp ** 2
            vmag = jnp.linalg.norm(rvel, axis=-1); vhat = rvel / (vmag[..., None] + 0.5)
            proj = jnp.sum((dr[..., None] * uhat) * vhat, axis=-1)             # (N,R) d . vhat
            quad = (dr ** 2 - (b / (a + b)) * proj ** 2) / a                   # d^T (Sigma_q+Sigma_k)^-1 d
        else:
            quad = dr ** 2 / a
        T = jnp.exp(-0.5 * quad) * rmask                                       # (N,R) zone<->rusher overlap
        logit = b0 + gt1 * ts + gt2 * ts ** 2 + jnp.einsum("nr,nr->n", u[sid], T)
        if self.use_situation:
            def reff(nm, n):
                s = numpyro.sample("s_" + nm, dist.InverseGamma(2.0, 1.0))
                return s * self._vec("z_" + nm, int(n))
            logit = (logit
                     + reff("qb", Nqb)[qb]
                     + reff("down", data["N_down"])[jnp.asarray(data["down_ids"])]
                     + reff("qtr", data["N_quarter"])[jnp.asarray(data["quarter_ids"])]
                     + reff("cov", data["N_cov"])[jnp.asarray(data["coverage_ids"])]
                     + reff("form", data["N_form"])[jnp.asarray(data["form_ids"])]
                     + numpyro.sample("g_score", dist.Normal(0., 1.)) * jnp.asarray(data["score_diff_cov"])
                     + numpyro.sample("g_time", dist.Normal(0., 1.)) * jnp.asarray(data["time_cov"])
                     + numpyro.sample("g_yard", dist.Normal(0., 1.)) * jnp.asarray(data["yard_cov"]))
        numpyro.sample("y", dist.Bernoulli(logits=logit), obs=y)
