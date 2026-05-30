"""M3 — TimeGrad, the conditional-diffusion centerpiece (plan Part 5).

This is the model the whole ladder builds toward. M0/M1/M2 are a repeated value, a
linear-Gaussian ARIMA, and an RNN that emits a parametric (Student-t) distribution.
**TimeGrad** (Rasul et al., 2021) instead emits a *learned, non-parametric* predictive
distribution by running a **conditional denoising diffusion model** at every forecast
step — the exact generative family the PML course derives in its final chapter, here
made **conditional on the past**.

How it works (one sentence per moving part)
-------------------------------------------
* An **autoregressive RNN** reads the history and produces a hidden state ``h_t`` that
  summarises "everything seen so far".
* Conditioned on ``h_t``, a **DDPM** denoises a sample of the *next* multivariate
  observation ``x_t`` (all ``D`` channels jointly) from pure Gaussian noise — the same
  ``epsilon``-prediction objective the course uses, but conditioned on ``h_t``.
* It is **autoregressive by sampling**: draw ``x_t``, feed it back to advance the RNN,
  repeat to the horizon ``tau``. Running that ``S`` times yields ``S`` trajectories —
  the ``(N, S, tau, D)`` tensor the shared metrics module already scores, so M0..M3 are
  compared cell-for-cell.

Multivariate by construction
----------------------------
Unlike M2 (which we ran as ``D`` independent univariate DeepARs), TimeGrad denoises the
**joint** ``D``-vector, so it can capture cross-channel structure. It therefore consumes
the multivariate bridge ``ForecastDataset.to_gluonts_multivariate`` (one entry whose
``target`` has shape ``(D, L)``), not the per-channel ``to_gluonts``.

Library & environment
---------------------
Built on **PyTorchTS** (``pts``), a PyTorch forecasting library that reuses GluonTS's
data/transform machinery. PyTorchTS's pinning against a *specific* GluonTS version is
fragile (see ``requirements.txt`` and ``notebooks/colab_m3_timegrad.ipynb``); the exact
known-good combo is pinned in the Colab notebook, not the light env. All ``pts`` /
GluonTS imports are deferred into the methods so the light env still imports this module
(the guarded tests skip when ``pts`` is absent), mirroring M2.

Per-window protocol (identical discipline to M1/M2 — leakage-free, like-for-like)
---------------------------------------------------------------------------------
We **train one global TimeGrad** on the full training split (one multivariate series).
To score the test set we do *not* refit: for every test window we hand TimeGrad that
window's own ``H``-step multivariate context as the series history and ask it to sample
``tau`` steps ahead. Trained weights, fresh state per window, no leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _import_pts():
    """Deferred import of the PyTorchTS / GluonTS pieces (heavy group only).

    Returns ``(TimeGradEstimator, Trainer, ListDataset)``. Raises a clear ImportError
    in the light env so the failure points at the missing heavy dependency, not a typo.
    """
    try:
        from gluonts.dataset.common import ListDataset
        from pts import Trainer
        from pts.model.time_grad import TimeGradEstimator
    except ImportError as exc:  # pragma: no cover - heavy/Colab env only
        raise ImportError(
            "M3 TimeGrad needs the heavy group (pytorchts + a matching gluonts), which "
            "is not in the light local env. Install the pinned combo on the Colab/GPU "
            "box (see requirements.txt and notebooks/colab_m3_timegrad.ipynb) before "
            "running run_timegrad.py."
        ) from exc
    return TimeGradEstimator, Trainer, ListDataset


def _patch_predictor_freq_kwarg():
    """Make GluonTS's ``PyTorchPredictor`` tolerate the legacy ``freq`` kwarg.

    The last PyTorchTS commit compatible with gluonts 0.13 (pinned at ``81be06bcc``)
    builds its predictor with ``PyTorchPredictor(..., freq=self.freq, ...)``. gluonts
    0.13 removed ``freq`` from that constructor in its freq-handling refactor, so the
    call dies with ``TypeError: __init__() got an unexpected keyword argument 'freq'``
    *after a full training run* — the worst place to fail on a paid GPU. This wraps the
    constructor to drop a stray ``freq`` (gluonts 0.13 never used it). No-op and
    idempotent: it does nothing if gluonts still accepts ``freq`` or if already patched.
    """
    import inspect

    from gluonts.torch.model.predictor import PyTorchPredictor

    init = PyTorchPredictor.__init__
    if getattr(init, "_freq_tolerant", False):
        return  # already patched this session
    if "freq" in inspect.signature(init).parameters:
        return  # this gluonts still takes freq — nothing to strip

    def __init__(self, *args, **kwargs):
        kwargs.pop("freq", None)
        init(self, *args, **kwargs)

    __init__._freq_tolerant = True
    PyTorchPredictor.__init__ = __init__


@dataclass
class TimeGradForecaster:
    """A thin, contract-aware wrapper around PyTorchTS ``TimeGradEstimator`` (M3).

    Parameters
    ----------
    target_dim : number ``D`` of channels modelled **jointly** by the diffusion step.
    horizon : forecast length ``tau``.
    context_length : ``H`` steps of history the RNN conditions on (matches the windows).
    freq : pandas offset alias of the series cadence (e.g. ``"D"`` for Exchange).
    start : nominal series start (calendar features only); taken from the dataset meta.
    num_cells, num_layers, cell_type : the conditioning RNN's width / depth / kind.
    diff_steps : number of diffusion steps ``T`` in the per-position DDPM.
    beta_end, beta_schedule : the noise-schedule end value and shape ("linear"/"cosine").
    loss_type : DDPM regression loss ("l2" = the course's epsilon-MSE).
    scaling : let TimeGrad mean-scale each series internally (as DeepAR does).
    max_epochs, num_batches_per_epoch, batch_size, lr : the training budget.
    n_samples : number ``S`` of sampled trajectories drawn per window at predict time.
    input_size : the RNN input width. This is TimeGrad's notorious must-match argument;
        left ``None`` it is computed from the GluonTS lags for ``freq`` plus the feature
        count. If ``pts`` raises a dimension-mismatch the message states the expected
        value — pass it explicitly (runner flag ``--input-size``) to override.
    device : torch device string ("cuda" on Colab GPU, "cpu" on a laptop).
    seed : best-effort determinism seed.
    train_series_copies : how many times the single multivariate training series is
        repeated in the GluonTS dataset before training. TimeGrad trains on **one**
        series; GluonTS ties its ``max_idle_transforms`` guard to ``len(dataset)`` and
        the instance sampler can draw zero windows on a pass, so a 1-entry dataset
        aborts with "Reached maximum number of idle transformation calls". Repeating the
        identical series lifts the guard and feeds the sampler many sources per cycle —
        same data, just offered repeatedly (see :meth:`_replicate_for_sampler`).

    Attributes (set by :meth:`fit`)
    -------------------------------
    predictor_ : the trained PyTorchTS predictor (``None`` until :meth:`fit`).
    input_size_ : the input width actually used (handy when it was auto-computed).
    """

    target_dim: int
    horizon: int
    context_length: int
    freq: str
    start: str | None = None
    num_cells: int = 64
    num_layers: int = 2
    cell_type: str = "GRU"
    diff_steps: int = 100
    beta_end: float = 0.1
    beta_schedule: str = "linear"
    loss_type: str = "l2"
    scaling: bool = True
    max_epochs: int = 20
    num_batches_per_epoch: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    n_samples: int = 100
    input_size: int | None = None
    device: str = "cuda"
    seed: int = 0
    train_series_copies: int = 64
    predictor_: object = field(default=None, repr=False)
    input_size_: int | None = field(default=None, repr=False)

    # -- input-size helper ----------------------------------------------------
    def _guess_input_size(self) -> int:
        """A first guess for ``input_size`` from the GluonTS lags + time features.

        TimeGrad concatenates, per step: the lagged sub-sequences of all channels
        (``target_dim * n_lags``), the time features, and a bookkeeping scalar. This
        guess is only a *starting* value — :meth:`fit` auto-corrects it from the RNN's
        own shape check if it is wrong (see ``_INPUT_SIZE_RE``), so a mismatch costs a
        few seconds, not a Colab round-trip. ``input_size`` (or ``--input-size``)
        overrides it entirely.
        """
        from gluonts.time_feature import (
            get_lags_for_frequency,
            time_features_from_frequency_str,
        )

        n_lags = len(get_lags_for_frequency(self.freq))
        n_time_feat = len(time_features_from_frequency_str(self.freq))
        return self.target_dim * n_lags + n_time_feat + 1

    def _build_estimator(self, input_size: int):
        """Construct a ``TimeGradEstimator`` for a given ``input_size`` (classic API)."""
        import torch

        TimeGradEstimator, Trainer, _ = _import_pts()
        return TimeGradEstimator(
            target_dim=self.target_dim,
            prediction_length=self.horizon,
            context_length=self.context_length,
            cell_type=self.cell_type,
            num_cells=self.num_cells,
            num_layers=self.num_layers,
            input_size=input_size,
            freq=self.freq,
            loss_type=self.loss_type,
            scaling=self.scaling,
            diff_steps=self.diff_steps,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            trainer=Trainer(
                device=torch.device(self.device),
                epochs=self.max_epochs,
                learning_rate=self.lr,
                num_batches_per_epoch=self.num_batches_per_epoch,
                batch_size=self.batch_size,
            ),
        )

    def _train(self, estimator, train_dataset):
        """Run ``estimator.train`` with a modern-torch-safe DataLoader config.

        PyTorchTS (pinned at the gluonts-0.13-compatible commit, mid-2024) builds its
        training ``DataLoader`` with ``num_workers=0`` **and** ``prefetch_factor=2``.
        torch >= 2.0 rejects that pair ("prefetch_factor option could only be specified
        in multiprocessing"): with ``num_workers=0`` the only legal value is
        ``prefetch_factor=None``. ``PyTorchEstimator.train`` forwards both kwargs
        straight through to the DataLoader, so we pass the safe pair here — single
        process, deterministic, no multiprocessing surprises on Colab.
        """
        return estimator.train(train_dataset, num_workers=0, prefetch_factor=None)

    def _replicate_for_sampler(self, train_dataset):
        """Repeat the single multivariate series so GluonTS won't abort training.

        TimeGrad trains on **one** multivariate series. GluonTS ties its
        ``max_idle_transforms`` guard to ``len(dataset)`` (here 1), while the
        ``ExpectedNumInstanceSampler`` draws a *random* number of windows per pass and
        can legitimately return zero; with the guard at 1, two such empty passes abort
        training with "Reached maximum number of idle transformation calls". Presenting
        the identical series ``train_series_copies`` times raises the guard to that many
        and gives the sampler many sources per cycle, so an empty-pass run becomes
        statistically impossible — **without changing the data the model sees** (the same
        single series, only offered repeatedly for random-window sampling). As a bonus it
        makes the ``input_size`` auto-correction reliable: the retry trains instead of
        tripping this same abort.
        """
        k = int(self.train_series_copies)
        if k <= 1:
            return train_dataset
        # GluonTS 0.11–0.13's ``ListDataset`` returns a plain *list* of processed
        # entries (not an object with ``.list_data``), so we materialise and repeat it:
        # ``k`` identical series lift the ``max_idle_transforms`` guard (which pts ties
        # to ``len(dataset)``) and give the sampler ``k`` sources per cycle. GluonTS
        # copies each entry before transforming, so repeating references is safe.
        try:
            entries = list(train_dataset)
        except TypeError:
            return train_dataset
        return entries * k if entries else train_dataset

    # -- fit ------------------------------------------------------------------
    def fit(self, train_dataset) -> "TimeGradForecaster":
        """Train one global TimeGrad on a multivariate GluonTS dataset.

        ``train_dataset`` is a GluonTS multivariate ``ListDataset`` — build it with
        ``ds.to_gluonts_multivariate("train")`` (one entry, ``target`` shape ``(D, L)``)
        so the raw series are used and TimeGrad applies its own per-series scaling.

        ``input_size`` is TimeGrad's notorious must-match argument. If it was not given
        explicitly we start from :meth:`_guess_input_size` and, on the RNN's
        "Expected X, got Y" shape error, parse the **true** width ``Y`` and retrain once
        — so the right value is found automatically instead of by trial-and-error.
        """
        import re

        _import_pts()  # surface the clean "heavy group" error before anything else
        import torch

        # pts@81be06bcc builds its predictor with PyTorchPredictor(..., freq=...), a kwarg
        # gluonts 0.13 dropped; patch it away *before* training so the run that just cost a
        # GPU hour doesn't die at predictor-construction time. Idempotent + version-safe.
        _patch_predictor_freq_kwarg()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Repeat the single multivariate series so GluonTS's idle-transform guard
        # (tied to len(dataset)=1) cannot abort training on an empty sampler pass.
        train_dataset = self._replicate_for_sampler(train_dataset)

        explicit = self.input_size is not None
        size = int(self.input_size) if explicit else self._guess_input_size()

        try:
            estimator = self._build_estimator(size)
            self.predictor_ = self._train(estimator, train_dataset)
        except RuntimeError as exc:
            # torch RNN check: "input.size(-1) must be equal to input_size. Expected X,
            # got Y" — Y is the real feature width. Auto-correct once (unless the caller
            # pinned input_size, in which case respect it and surface the error).
            m = re.search(r"got (\d+)", str(exc))
            if explicit or m is None:
                raise
            size = int(m.group(1))
            estimator = self._build_estimator(size)
            self.predictor_ = self._train(estimator, train_dataset)

        self.input_size_ = size
        return self

    # -- predict --------------------------------------------------------------
    def predict(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(point, samples)`` for a batch of context windows.

        ``contexts`` : ``(N, H, D)`` on the **original** scale. We build one multivariate
        GluonTS entry per window (``target`` shape ``(D, H)``) and let the trained
        predictor sample ``tau`` steps beyond each. A multivariate ``SampleForecast`` has
        ``samples`` of shape ``(S, tau, D)``, which drops straight into our tensor.

        Returns
        -------
        point   : ``(N, tau, D)`` — the per-position sample **mean**.
        samples : ``(N, S, tau, D)`` — the sampled predictive trajectories.
        """
        if self.predictor_ is None:
            raise RuntimeError("predict() called before fit().")
        contexts = np.asarray(contexts, dtype=np.float64)
        if contexts.ndim != 3:
            raise ValueError(f"Expected (N, H, D) contexts, got shape {contexts.shape}.")
        N, _, D = contexts.shape
        if D != self.target_dim:
            raise ValueError(
                f"contexts have D={D} channels but target_dim={self.target_dim}."
            )
        _, _, ListDataset = _import_pts()

        # One multivariate entry per window: target is channels-first (D, H).
        entries = [
            {"start": self.start, "target": contexts[i].T.astype("float32")}
            for i in range(N)
        ]
        pred_ds = ListDataset(entries, freq=self.freq, one_dim_target=False)
        forecasts = list(self.predictor_.predict(pred_ds, num_samples=self.n_samples))
        if len(forecasts) != N:
            raise RuntimeError(
                f"Expected {N} forecasts, got {len(forecasts)} from the predictor."
            )

        tau, S = self.horizon, self.n_samples
        samples = np.empty((N, S, tau, D), dtype=np.float64)
        point = np.empty((N, tau, D), dtype=np.float64)
        for i, fc in enumerate(forecasts):
            s = np.asarray(fc.samples, dtype=np.float64)  # (S, tau, D)
            if s.shape != (S, tau, D):
                raise RuntimeError(
                    f"Forecast {i} has samples shape {s.shape}, expected {(S, tau, D)}."
                )
            samples[i] = s
            point[i] = s.mean(axis=0)
        return point, samples
