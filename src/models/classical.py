"""M1 — the classical statistical baseline (plan Part 5): per-channel ARIMA.

M0 (seasonal-naive) only knows "tomorrow looks like the last season". M1 is the
first model that actually *fits parameters*: an **ARIMA** — AutoRegressive Integrated
Moving Average — learns the short-memory linear structure (how today depends on
recent days and recent shocks) on top of a differenced random walk. On the Exchange
series, whose levels are near-unit-root (lag-1 autocorrelation ≈ 0.999, see
``docs/EDA_EXCHANGE.md``), the integration order ``d = 1`` is the natural choice and
the auto-selector usually lands on a very small ``(p, q)``.

Why statsmodels (not statsforecast)?
------------------------------------
The plan names *statsforecast* (Nixtla's ``AutoARIMA``) for this rung. That pulls in
``numba``/``llvmlite`` and is reserved for the group's primary-dataset repo. This
sandbox stays light and offline, so we use **statsmodels**, which is already a
dependency (ADF test, ACF/PACF in the EDA). The forecasts are the same family of
linear-Gaussian model; the group repo can later swap in ``AutoARIMA`` behind the same
``fit``/``predict`` interface without touching the evaluation code.

Leakage-free conditioning (the key design choice)
-------------------------------------------------
We **fit the ARIMA parameters once, on the training split**, per channel. To forecast
a test window we do *not* refit; we call ``results.apply(context, refit=False)``,
which keeps the trained coefficients and merely re-runs the Kalman filter over that
window's own ``H`` context to set the model state, then forecasts ``tau`` steps. So
every window is conditioned only on its own past, with parameters that never saw the
test data — fair by construction, and ~100× cheaper than refitting per window.

Making it probabilistic
------------------------
ARIMA is a linear-Gaussian model, so its ``h``-step predictive distribution is
*exactly* Gaussian: ``get_forecast(tau)`` returns the per-step mean and standard
error (the latter grows with ``h`` as forecast uncertainty accumulates). We draw the
``S`` predictive trajectories from those Gaussians. Sampling each horizon step
independently is sufficient here because every metric we report (CRPS, interval
coverage, width) is a **marginal-per-position** functional — invariant to the
cross-horizon correlation of the samples.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

# Default AIC search grid: a tiny set spanning random-walk to a richer ARMA on the
# first difference. Kept small so per-channel selection on Exchange is a few seconds.
DEFAULT_CANDIDATE_ORDERS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 0),  # pure random walk (the integrated naive) — M0's point forecast
    (0, 1, 1),  # + MA(1): one-step shock smoothing
    (1, 1, 0),  # + AR(1) on the difference: momentum/mean-reversion of changes
    (1, 1, 1),  # the workhorse ARIMA(1,1,1)
    (2, 1, 2),  # a little extra short memory, if the data earns the AIC
)
_FALLBACK_ORDER = (0, 1, 0)


@dataclass
class ClassicalForecaster:
    """Per-channel ARIMA with leakage-free per-window conditioning (plan Part 5, M1).

    Parameters
    ----------
    horizon : forecast length ``tau``.
    method : only ``"arima"`` for now (ETS is a documented future branch; on a
        driftless random walk like Exchange it offers no edge over ARIMA).
    order : fixed ARIMA ``(p, d, q)`` for *every* channel, or ``None`` to auto-select
        each channel's order by AIC over :attr:`candidate_orders`.
    candidate_orders : the AIC search grid used when ``order is None``.
    n_samples : number ``S`` of Gaussian predictive trajectories drawn per window.
    seed : RNG seed for the predictive draw (logged for reproducibility).

    Attributes (set by :meth:`fit`)
    -------------------------------
    results_ : list of length ``D`` of fitted statsmodels ARIMA results (one/channel).
    orders_  : list of length ``D`` of the ``(p, d, q)`` actually used per channel.
    """

    horizon: int
    method: str = "arima"
    order: tuple[int, int, int] | None = None
    candidate_orders: tuple[tuple[int, int, int], ...] = DEFAULT_CANDIDATE_ORDERS
    n_samples: int = 100
    seed: int = 0
    results_: list = field(default=None, repr=False)
    orders_: list = field(default=None, repr=False)

    # -- fit ------------------------------------------------------------------
    def fit(self, train_series: np.ndarray) -> "ClassicalForecaster":
        """Fit one ARIMA per channel on the **training split** (original scale).

        ``train_series`` : ``(L_train, D)`` time-major training values. Each channel
        is fit independently (Exchange's channels are separate currencies); the
        selected order and the fitted results object are cached for :meth:`predict`.
        """
        if self.method != "arima":
            raise NotImplementedError(
                f"ClassicalForecaster only implements method='arima', got {self.method!r}."
            )
        x = np.asarray(train_series, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"Expected (L_train, D) train series, got shape {x.shape}.")
        D = x.shape[1]
        self.results_ = []
        self.orders_ = []
        for d in range(D):
            order, res = self._fit_channel(x[:, d])
            self.orders_.append(order)
            self.results_.append(res)
        return self

    def _fit_channel(self, series: np.ndarray):
        """Select an order (fixed or by AIC) and fit ARIMA on one channel."""
        from statsmodels.tsa.arima.model import ARIMA

        if self.order is not None:
            return self.order, self._safe_fit(series, self.order)

        best_order, best_res, best_aic = None, None, np.inf
        for cand in self.candidate_orders:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ARIMA(series, order=cand).fit()
                aic = float(res.aic)
            except Exception:  # noqa: BLE001 — a bad order just loses the AIC race
                continue
            if np.isfinite(aic) and aic < best_aic:
                best_order, best_res, best_aic = cand, res, aic
        if best_res is None:  # every candidate failed → integrated naive
            return _FALLBACK_ORDER, self._safe_fit(series, _FALLBACK_ORDER)
        return best_order, best_res

    @staticmethod
    def _safe_fit(series: np.ndarray, order: tuple[int, int, int]):
        """Fit a single ARIMA order, falling back to the random walk if it fails."""
        from statsmodels.tsa.arima.model import ARIMA

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ARIMA(series, order=order).fit()
        except Exception:  # noqa: BLE001
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ARIMA(series, order=_FALLBACK_ORDER).fit()

    # -- predict --------------------------------------------------------------
    def predict(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(point, samples)`` for a batch of context windows.

        ``contexts`` : ``(N, H, D)`` on the **original** scale. For each window/channel
        we re-apply the trained ARIMA to that window's context (``refit=False``) and
        read the Gaussian ``tau``-step forecast.

        Returns
        -------
        point   : ``(N, tau, D)`` — the ARIMA conditional-mean forecast.
        samples : ``(N, S, tau, D)`` — Gaussian draws ``mean + se · ε`` per position.
        """
        if self.results_ is None:
            raise RuntimeError("predict() called before fit().")
        contexts = np.asarray(contexts, dtype=np.float64)
        if contexts.ndim != 3:
            raise ValueError(f"Expected (N, H, D) contexts, got shape {contexts.shape}.")
        N, _, D = contexts.shape
        if D != len(self.results_):
            raise ValueError(
                f"contexts has D={D} channels but the model was fit on {len(self.results_)}."
            )
        tau = self.horizon
        means = np.empty((N, tau, D), dtype=np.float64)
        ses = np.empty((N, tau, D), dtype=np.float64)
        for d in range(D):
            res_d = self.results_[d]
            for i in range(N):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fc = res_d.apply(contexts[i, :, d], refit=False).get_forecast(tau)
                means[i, :, d] = np.asarray(fc.predicted_mean, dtype=np.float64)
                ses[i, :, d] = np.asarray(fc.se_mean, dtype=np.float64)
        # Guard degenerate (zero) standard errors so the Gaussian draw is well-defined.
        ses[ses <= 0.0] = np.finfo(np.float64).eps
        rng = np.random.default_rng(self.seed)
        noise = rng.standard_normal((N, self.n_samples, tau, D))
        samples = means[:, None, :, :] + noise * ses[:, None, :, :]
        return means, samples
