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
                         num_chains=num_chains, progress_bar=False)
        self.mcmc.run(jax.random.PRNGKey(seed), data)
        self.samples = self.mcmc.get_samples()
        return self.samples

    def run_svi_inference(self, data, num_steps=20000, lr=1e-2, seed=0, n_draws=1000):
        self.guide = autoguide.AutoNormal(self.model_fn)
        svi = SVI(self.model_fn, self.guide, numpyro.optim.Adam(lr), Trace_ELBO())
        self.svi_result = svi.run(jax.random.PRNGKey(seed), num_steps, data, progress_bar=False)
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
    """Continuous-time (AR(1)) max-strain plus-minus model.

    Observation = (play, rusher, frame). Forecasts the next frame's MAX strain across
    all rushers (the play's peak pressure at t+1) from each rusher's OWN current-frame
    strain via a single shared AR(1) coefficient rho in (-1,1), plus a per-rusher
    (attribute-centered) intercept countered by the frame-level attention-weighted
    blocker sum, plus game-situation effects. Rows at the same (play, frame) share the
    response but carry rusher-specific predictors. See build_continuous_design in
    src/feature_engineering.py for the expected (long) `data` dict.
    """

    def model_fn(self, data: dict):
        assignment = data["assignment"]              # (N_obs, B) frame-t attention
        outcome = data["outcome"]                    # (N_obs,) max-over-rushers STRAIN at t+1
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
