"""M2 — DeepAR, the *fair* deep probabilistic baseline (plan Part 5).

Where M0/M1 are non-neural (a repeated value; a linear-Gaussian ARIMA), DeepAR
(Salinas et al., 2020) is the first **neural and probabilistic** rung. Its core is an
**autoregressive RNN** (LSTM): at each step it reads the recent value plus calendar /
lag covariates and emits not a number but the **parameters of a distribution** for the
next step (here a Student-t). It trains by maximising the log-likelihood of the
observed series under those per-step distributions.

This is the model the diffusion centerpiece (M3) must actually beat: a serious deep
probabilistic forecaster, not a toy. If diffusion buys richer / better-calibrated
uncertainty, it has to earn it *against DeepAR*.

How it produces a predictive distribution
------------------------------------------
At inference DeepAR is **autoregressive by sampling**: it draws a value from the
predicted distribution at step *t*, feeds it back in to predict *t+1*, and repeats to
the horizon ``tau``. Running that ``S`` times yields ``S`` trajectories — the same
``(N, S, tau, D)`` tensor the shared metrics module already scores, so M0/M1/M2 are
compared cell-for-cell.

Library & environment
---------------------
Built on **GluonTS** (``gluonts.torch`` DeepAR, a PyTorch + Lightning implementation).
This is the first rung that leaves the light/local env — it needs ``gluonts`` and
``lightning`` (see ``requirements.txt`` "heavy group"). All GluonTS imports are
deferred into the methods so the light env can still import this module (the guarded
tests skip when GluonTS is absent), mirroring ``ForecastDataset.to_gluonts``.

Per-window protocol (matches M1 for a leakage-free, like-for-like comparison)
-----------------------------------------------------------------------------
We **train one global DeepAR** on the full training split (one univariate series per
channel — GluonTS scales each internally). To score the test set we do *not* refit:
for every test window we hand DeepAR that window's own ``H``-step context as the
series history and ask it to sample ``tau`` steps ahead. Trained weights, fresh state
per window, no leakage — exactly the ``ARIMA.apply(context)`` discipline from M1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _import_gluonts():
    """Deferred import of the GluonTS pieces (heavy group only).

    Returns ``(DeepAREstimator, ListDataset)``. Raises a clear ImportError in the
    light env so the failure points at the missing heavy dependency, not a typo.
    """
    try:
        from gluonts.dataset.common import ListDataset
        from gluonts.torch.model.deepar import DeepAREstimator
    except ImportError as exc:  # pragma: no cover - heavy/Colab env only
        raise ImportError(
            "M2 DeepAR needs the heavy group (gluonts + lightning), which is not in "
            "the light local env. Install them on the Colab/GPU box (see "
            "requirements.txt) before running run_deepar.py."
        ) from exc
    return DeepAREstimator, ListDataset


@dataclass
class DeepARForecaster:
    """A thin, contract-aware wrapper around GluonTS ``gluonts.torch`` DeepAR (M2).

    Parameters
    ----------
    horizon : forecast length ``tau``.
    context_length : ``H`` steps of history DeepAR conditions on (matches the windows).
    freq : pandas offset alias of the series cadence (e.g. ``"D"`` for Exchange).
    start : nominal series start (calendar features only); taken from the dataset meta.
    num_layers, hidden_size : the LSTM depth / width.
    max_epochs, num_batches_per_epoch, batch_size, lr : the Lightning training budget.
    n_samples : number ``S`` of sampled trajectories drawn per window at predict time.
    disable_time_features : if ``True``, train DeepAR with ``time_features=[]`` — i.e.
        with **no** calendar covariates. On Exchange the ``start`` date is nominal
        (a placeholder 1990-01-01, the series is not truly calendar-aligned), so the
        default daily time features are spurious noise the model can over-fit. The
        M2-tuned run sets this to strip them; the M2-baseline run leaves them on.
    seed : seed for ``lightning.seed_everything`` + the sampler (best-effort determinism).

    Attributes (set by :meth:`fit`)
    -------------------------------
    predictor_ : the trained GluonTS predictor (``None`` until :meth:`fit`).
    """

    horizon: int
    context_length: int
    freq: str
    start: str | None = None
    num_layers: int = 2
    hidden_size: int = 40
    max_epochs: int = 20
    num_batches_per_epoch: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    n_samples: int = 100
    disable_time_features: bool = False
    accelerator: str = "auto"  # "auto" uses GPU on Colab, CPU on a laptop
    seed: int = 0
    predictor_: object = field(default=None, repr=False)

    # -- fit ------------------------------------------------------------------
    def fit(self, train_dataset) -> "DeepARForecaster":
        """Train one global DeepAR on a GluonTS dataset (one entry per channel).

        ``train_dataset`` is a GluonTS ``ListDataset`` — build it from the data
        contract with ``ds.to_gluonts("train")`` so the raw (un-scaled) series are
        used and GluonTS applies its own per-series scaling.
        """
        DeepAREstimator, _ = _import_gluonts()
        try:
            from lightning.pytorch import seed_everything
        except ImportError:  # pragma: no cover - older Lightning packaging
            from pytorch_lightning import seed_everything
        seed_everything(self.seed, workers=True)

        # Strip calendar covariates when asked: ``time_features=[]`` tells GluonTS to
        # build no time features instead of the default daily set. We only pass the
        # kwarg in that case so the baseline run keeps GluonTS's own defaults untouched.
        extra_kwargs = {"time_features": []} if self.disable_time_features else {}

        estimator = DeepAREstimator(
            freq=self.freq,
            prediction_length=self.horizon,
            context_length=self.context_length,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            lr=self.lr,
            batch_size=self.batch_size,
            num_batches_per_epoch=self.num_batches_per_epoch,
            trainer_kwargs={
                "max_epochs": self.max_epochs,
                "accelerator": self.accelerator,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            },
            **extra_kwargs,
        )
        self.predictor_ = estimator.train(train_dataset)
        return self

    # -- predict --------------------------------------------------------------
    def predict(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(point, samples)`` for a batch of context windows.

        ``contexts`` : ``(N, H, D)`` on the **original** scale. We build one GluonTS
        entry per (window, channel) whose target is that window's context, then let the
        trained predictor sample ``tau`` steps beyond each. GluonTS batches the whole
        list internally, so this is a single vectorised ``predict`` call.

        Returns
        -------
        point   : ``(N, tau, D)`` — the per-position sample **mean** (DeepAR's natural
            minimum-MSE point forecast).
        samples : ``(N, S, tau, D)`` — the sampled predictive trajectories.
        """
        if self.predictor_ is None:
            raise RuntimeError("predict() called before fit().")
        _, ListDataset = _import_gluonts()
        contexts = np.asarray(contexts, dtype=np.float64)
        if contexts.ndim != 3:
            raise ValueError(f"Expected (N, H, D) contexts, got shape {contexts.shape}.")
        N, _, D = contexts.shape

        # One entry per (window, channel), in window-major / channel-minor order.
        entries = [
            {"start": self.start, "target": contexts[i, :, d].astype("float32")}
            for i in range(N)
            for d in range(D)
        ]
        pred_ds = ListDataset(entries, freq=self.freq)
        forecasts = list(self.predictor_.predict(pred_ds, num_samples=self.n_samples))
        if len(forecasts) != N * D:
            raise RuntimeError(
                f"Expected {N * D} forecasts, got {len(forecasts)} from the predictor."
            )

        tau, S = self.horizon, self.n_samples
        samples = np.empty((N, S, tau, D), dtype=np.float64)
        point = np.empty((N, tau, D), dtype=np.float64)
        for k, fc in enumerate(forecasts):
            i, d = divmod(k, D)
            s = np.asarray(fc.samples, dtype=np.float64)  # (S, tau)
            if s.shape != (S, tau):
                raise RuntimeError(
                    f"Forecast {k} has samples shape {s.shape}, expected {(S, tau)}."
                )
            samples[i, :, :, d] = s
            point[i, :, d] = s.mean(axis=0)
        return point, samples
