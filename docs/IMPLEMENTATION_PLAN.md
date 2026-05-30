# Implementation Plan — Probabilistic Time Series Forecasting with Diffusion Models
### PML Exam Project · University of Trieste · Prof. Luca Bortolussi · Team of 3

> **Status:** planning document only. No code yet. This is the blueprint the team will execute in three steps.
> **PDF page note:** the notes PDF is offset +10 from the printed page numbers (printed p.X = PDF p.X+10). Citations below use **section numbers** (unambiguous) with printed page in parentheses.

---

## Part 0 — How to use this document

**What this is.** A complete, execution-ready plan: the research question, the exact connection to the PML course, the models, datasets, metrics, experiments, software architecture, the 3-step team workflow, risks, and the final deliverables (slides + oral prep). It is written so that any team member can open it and know what to do next.

**What this is not.** It is not the report, not the code, not the slides. Those come later, *from* this plan.

**How to read it.**
- Read Part 1–3 together as a team first. Align on the *story*.
- Each person then owns the parts mapped to their step in Part 9.
- Treat Part 7 (experiments) as the source of truth for "what counts as done."
- Appendix A is a ready-to-send proposal for the professor.

**The golden rule of this project:** *we are not presenting a survey, and we are not building a black box.* We take one well-defined probabilistic question, answer it with real experiments, and we can open every model we use and explain it in the professor's own notation.

---

## Part 1 — The project in one page

### 1.1 Research question (one sentence)
> Can a **diffusion model** produce **probabilistic forecasts** of a future time series — a *distribution* of plausible futures rather than a single number — that are more **accurate**, better **calibrated**, or more **informative** than classical and deep-probabilistic baselines, and at what **computational cost**?

### 1.2 Abstract (the 6-sentence version)
Real forecasting is not "tomorrow's electricity demand will be 100"; it is "here is the distribution of plausible tomorrows." We frame multi-step forecasting as learning the **conditional generative distribution** `p(future | past)` and we instantiate it with a **conditional denoising diffusion model**, trained by the same ELBO / noise-prediction objective the course derives in §11.2. We compare it head-to-head against a naive baseline, a classical statistical model (ARIMA/ETS), and a deep autoregressive probabilistic model (DeepAR), on a real multivariate dataset. We evaluate **three** things the exam rewards: **point accuracy** (MAE/RMSE), **probabilistic quality** (CRPS, coverage, calibration), and **cost** (training/inference time, and the quality-vs-denoising-steps trade-off). Our thesis is deliberately non-triumphalist: diffusion buys *richer, better-calibrated uncertainty*, but its advantage depends on horizon, dataset, and a real sampling-cost penalty. The deliverable is an 8-minute talk and a clean repo that demonstrates we understood the course *and* extended its final chapter into a working application.

### 1.3 The thesis we will defend (the "story")
**"For forecasting, the right question is not *what* will happen but *what could* happen and *how sure are we*. Diffusion models answer that second question natively — they generate samples of the future — but uncertainty quality is the prize, not a lower MAE, and it comes at a sampling cost we can measure and tune."**

This story is strong because it is *falsifiable* (maybe diffusion loses — that is still a result), *probabilistic* (it lives in the heart of PML), and *practical* (the cost trade-off is a real deployment concern).

### 1.4 Success criteria, mapped to the professor's grading rubric
| Grading criterion (from the exam description) | How this project scores on it |
|---|---|
| Clarity & completeness of exposition | 8 tight slides, one message each (Part 11); a clean repo; honest results. |
| **Originality w.r.t. course content** | We extend §11.2 (diffusion) from *unconditional generation* to *conditional forecasting* — exactly the survey's framing. Plus an aleatoric-vs-epistemic analysis (§1.4 here) the course sets up but does not apply to diffusion. |
| Understanding of theory & practice | The whole pipeline is mapped to course chapters (Part 2); we can derive the diffusion loss in the professor's notation; we built (at least a toy of) it ourselves. |
| Clarity & precision in the oral | Per-person oral prep with likely questions mapped to course pages (Part 11.3). |

### 1.5 Scope decisions (locked, from the planning Q&A)
- **Timeline:** flexible → the plan is **modular** with stop-after tiers (Part 9.4). You can stop after Tier 0 and still have a complete project.
- **Compute:** Colab Pro / paid GPU → TimeGrad-class models are feasible; we budget for it (Part 8.2).
- **Scope:** **balanced** → one frontier diffusion model + full baselines + probabilistic evaluation + **one robustness experiment** (regime shift). A second dataset / second diffusion model is a *stretch*, not a requirement.
- **Team skills:** comfortable Python, new to deep-learning frameworks → **library-first** (GluonTS / PyTorchTS / Darts / statsforecast), with a small **from-scratch toy DDPM** as the "understanding artifact" (Part 5.4). We use libraries to get results and read their internals to avoid black-box usage.

---

## Part 2 — Connection to the PML course (the theoretical spine)

This is the highest-value differentiator for the grade. The course *ends* on diffusion (Ch 11), so we are extending the final chapter, not importing something alien. Every component of the project maps to a chapter.

### 2.1 The mapping table (project component → course → what to say in the oral)
| Project component | Course location (printed page) | One-line oral answer |
|---|---|---|
| Forecasting as `p(future\|past)` | §1.2 Generative Modelling (p.3–4) | "It's conditional generative modelling: we learn a distribution, then sample futures." |
| Why a *distribution*, not a point | §1.1.1 Uncertainty, aleatoric vs epistemic (p.2–3) | "Decisions need uncertainty; a point estimate hides risk." |
| Diffusion forward process | §11.2.1 (p.123–124) | "A fixed Gaussian Markov chain that corrupts data to white noise: `p(x_t\|x_{t-1})=N(√(1−β_t)x_{t-1},β_tI)`." |
| Closed-form noising | §11.2.1 (p.124) | "`x_t=√ᾱ_t x_0+√(1−ᾱ_t)ε` — sample any step in one shot, via the reparameterization trick." |
| Reverse / generative process | §11.2.2 (p.124–125) | "Learn Gaussian reverse kernels `q_θ(x_{t-1}\|x_t)`; valid because slow noising ⇒ ~Gaussian reverse." |
| Training objective | §11.2.2–11.2.3 (p.125–126) | "Minimize `KL(forward‖backward)` = maximum likelihood; in practice the noise-prediction MSE." |
| The actual loss we optimize | §11.2.3 (p.126), Alg. 1 | "`E‖ε − ε_θ(√ᾱ_t x_0+√(1−ᾱ_t)ε, t)‖²`." |
| Sampling a forecast | §11.2.3 Alg. 2 (p.126); ancestral sampling §3.3.1 (p.26) | "Start from noise, run the reverse Markov chain; each run = one future trajectory." |
| ELBO machinery (shared with VAE) | §9.2 ELBO (p.96); §11.1 VAE (p.119–122) | "Same evidence lower bound as the VAE: reconstruction − KL-to-prior." |
| Reparameterization trick | §10.4.1 (p.112–113); used at §11.1 (p.122) | "Move randomness to `ε~N(0,I)` so gradients flow through sampling." |
| Score-based view / Langevin | §11.2.5 (p.127–128); sampling Ch 8 (p.79–91) | "Diffusion ≈ estimating `∇_x log p(x)` and doing Langevin-type sampling — links to MCMC." |
| DeepAR baseline (likelihood + sampling) | §1.2 (p.3–4); predictive dist. §6.5 (p.64) | "Autoregressive likelihood model; Monte-Carlo roll-outs give the predictive distribution." |
| Classical ARIMA/ETS baseline | state-space ≈ HMM Ch 5 (p.49); Bayesian lin. reg. Ch 6 | "Linear-Gaussian dynamics; predictive intervals from the Gaussian noise model." |
| CRPS / proper scoring | KL & scoring §2.3 (p.20) | "CRPS generalizes MAE to distributions; a proper scoring rule." |
| (Optional) GP baseline | Ch 12 Kernels & GPs (p.129+, starred) | "A fully-Bayesian forecaster with epistemic uncertainty — the contrast case." |

### 2.2 The one derivation we must own (in the professor's notation)
The diffusion training loss is a re-parameterized ELBO. The chain the team must be able to reproduce on a whiteboard:

1. **Forward (fixed):** `p(x_t|x_{t-1}) = N(x_t; √(1−β_t)·x_{t-1}, β_t·I)`, with `α_t = 1−β_t`, `ᾱ_t = ∏_{i≤t} α_i`.
2. **Closed form (reparam. trick, §10.4.1):** `x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε`, `ε ~ N(0,I)` ⟹ `p(x_t|x_0)=N(√ᾱ_t x_0, 1−ᾱ_t)`; as `t→∞`, `→ N(0,I)`.
3. **Reverse (learned):** `q_θ(x_{t-1}|x_t) = N(μ_θ(x_t,t), Σ_θ(x_t,t))`.
4. **Objective:** minimize `KL(p(x_{0:T}) ‖ q_θ(x_{0:T}))` `= −E[Σ_t log q_θ(x_{t-1}|x_t)] + const` (= maximum likelihood / negative ELBO).
5. **Noise-prediction reparam. (§11.2.3):** predicting `ε` instead of `x_0` gives the simple loss
   `L(θ) = E_{t,x_0,ε} ‖ ε − ε_θ(√ᾱ_t·x_0 + √(1−ᾱ_t)·ε, t) ‖²` (Algorithm 1).
6. **Sampling (Algorithm 2):** `x_{t-1} = (1/√α_t)(x_t − (1−α_t)/√(1−ᾱ_t)·ε_θ(x_t,t)) + σ_t·z`, `z~N(0,I)`.

**Our novelty in one line:** make the denoiser **conditional on the past**, `ε_θ(x_t, t, c)`, where `c = Encoder(x_{past})`. The survey calls this *conditioning source = historical series* and *condition integration = how `c` enters the network*. Everything else is the course's §11.2.

### 2.3 The "original + sophisticated" point (uncertainty decomposition)
The course distinguishes **aleatoric** (irreducible, data-noise) vs **epistemic** (model/parameter) uncertainty (§1.1.1) and shows how Bayesian models (Ch 6), BNNs (§10.6), and GPs (Ch 12) capture *epistemic* uncertainty by putting distributions on parameters. A diffusion forecaster with point-estimated weights `θ` captures **aleatoric** uncertainty (the spread of plausible futures) but **not epistemic** uncertainty (it is sure about its own parameters). This gives us a genuinely original discussion slide — *"what uncertainty are we actually quantifying, and what are we missing?"* — and a natural "future work" hook (ensemble / Bayesian diffusion). Few exam projects make this distinction explicitly; it directly targets the "depth of understanding" criterion.

---

## Part 3 — Problem formalization

### 3.1 Notation
- A multivariate series `x_{1:L} ∈ R^{D×L}` (`D` channels, length `L`).
- **Context / history window** `H`: the past we condition on, `c = x_{t-H+1 : t}`.
- **Prediction horizon** `τ`: the future we forecast, `y = x_{t+1 : t+τ}`.
- A **forecast** is the conditional distribution `p(y | c)`. Different model classes represent it differently (Part 5).

### 3.2 The task, precisely
Estimate and sample from `p(x_{t+1:t+τ} | x_{t-H+1:t})`. For diffusion and DeepAR we draw `S` sample trajectories `{y^{(s)}}_{s=1..S}`; for ARIMA/ETS we get parametric predictive intervals; for the naive baseline we get a point (and an empirical band via residual bootstrap).

### 3.3 Deterministic vs probabilistic forecasting (the framing we keep hammering)
- **Deterministic:** output `ŷ` (a number). Evaluated by MAE/RMSE.
- **Probabilistic:** output `p(y|c)` (a distribution). Evaluated by CRPS, coverage, calibration — *and* you can still extract a point (the mean/median) for MAE/RMSE, so the comparison is fair.

### 3.4 Forecast representation per model (so comparison is apples-to-apples)
| Model | Native output | Point forecast = | Predictive interval = |
|---|---|---|---|
| Seasonal-naive | point | the value itself | residual-bootstrap band |
| ARIMA/ETS | Gaussian predictive | mean | ±z·σ analytic interval |
| DeepAR | sampled trajectories | sample mean/median | sample quantiles |
| Diffusion (TimeGrad) | sampled trajectories | sample mean/median | sample quantiles |

---

## Part 4 — Data

We keep the *plan* dataset-agnostic (as requested) but constrain the *choice* so the project stays rigorous and the datasets stay interesting and reusable, not toy/didactic.

### 4.1 Dataset selection rubric (pick a dataset that satisfies all of these)
1. **Multivariate** (`D ≥ ~8`) so diffusion's joint modelling is meaningful.
2. **Has published TimeGrad/CSDI numbers** → enables E0 (reproduce-a-known-result gate). This is the single most important constraint for credibility.
3. **Regular sampling, real-world, industrially relevant** → reusable skill, good story.
4. **Fits Colab memory/time** at a sensible subset (a few hundred series max, or a channel subset).
5. **Has a plausible "regime shift"** we can exploit for the robustness experiment E4 (seasonality, a COVID-era break, a volatility cluster, etc.).

### 4.2 Shortlist (interesting + reusable + benchmarked)
| Dataset | D / cadence | Why interesting / reusable | In TimeGrad bench? |
|---|---|---|---|
| **Solar** (PV production, 137 stations) | 137 / hourly | renewable energy, intermittency, strong daily cycle | ✅ |
| **Electricity** (370 clients) | 370 / hourly | industrial load, classic, easy to narrate | ✅ |
| **Traffic** (San-Francisco, 963 sensors) | 963 / hourly | spatio-temporal, road occupancy | ✅ |
| **Exchange** (8 currencies) | 8 / daily | small & fast → great for iteration / fallback | ✅ |
| **Wind power** (SCADA / generation) | varies | renewable, weather-driven, "Industry 4.0" feel | ⚠️ varies |
| **ETT** (electricity transformer temp.) | 7 / 15-min·hourly | used in all modern long-horizon papers | ⚠️ (forecasting, not always CRPS) |

### 4.3 Recommendation
- **Primary:** **Solar** *or* **Electricity** (multivariate, energy, in the TimeGrad benchmark → E0 possible).
- **Fast fallback / iteration:** **Exchange** (tiny; run the whole pipeline end-to-end here first, *then* scale up).
- **"Interesting twist" stretch (E5):** **Wind power** or **Traffic**.
- **Decision point for the team:** confirm one primary + Exchange-for-iteration. *Do not* start on the big dataset; get the pipeline green on Exchange first.

### 4.4 Preprocessing spec (write this once in `src/data/`, reuse everywhere)
- **Temporal split, no leakage:** contiguous `train | val | test` in time order (e.g., 70/10/20). Never shuffle across time. Test is the *most recent* slice.
- **Scaling:** fit a scaler (per-series standardization or mean-scaling à la GluonTS) **on train only**; apply to val/test. Scaling leakage is the #1 silent bug — guard it.
- **Windowing:** sliding `(context H, horizon τ)` windows *within* each split; never let a window straddle the split boundary.
- **Missing values / zeros:** document the policy (forward-fill short gaps, mask long ones); note physical meaning (a zero meter may mean "offline," not "0 demand").
- **Frequency & calendar features:** keep it minimal first (hour-of-day, day-of-week) — these matter for DeepAR/TimeGrad covariates.
- **Determinism:** one function builds the datasets from a fixed seed; output a `manifest.json` recording split dates, scaler stats, `H`, `τ`.

### 4.5 The data contract
A single loader returns `(train, val, test)` GluonTS-style datasets + a metadata dict (`D, freq, H, τ, scaler`). Every model consumes the *same* object. This guarantees the comparison is fair by construction.

---

## Part 5 — Models (the comparison ladder)

Five rungs, increasing sophistication. M0–M3 are the **required** ladder; M4 and the GP are **stretch**. For each model: its role, the library, what to configure, what to understand for the oral, and the "open-the-box" task that keeps it from being a black box.

### 5.1 M0 — Naive / Seasonal-naive (the honesty anchor)
- **Role:** the "stupid but essential" baseline. If diffusion can't beat *this*, that's already a headline result.
- **Library:** Darts (`NaiveSeasonal`) or 5 lines of NumPy.
- **Probabilistic version:** residual bootstrap → empirical predictive band (so it has a CRPS/coverage too).
- **Oral point:** "a strong seasonal-naive is a famously hard baseline in forecasting; beating it must be earned."

### 5.2 M1 — Classical statistical (ARIMA/SARIMA or ETS)
- **Role:** the statistical-tradition rung; shows we know pre-deep forecasting.
- **Library:** `statsforecast` (Nixtla) `AutoARIMA`/`AutoETS` — fast, batched, beginner-friendly; or Darts.
- **Configure:** seasonal period, auto-order search; produce analytic predictive intervals.
- **Oral point:** ARIMA = linear-Gaussian state space ⇒ connects to HMM (Ch 5) and the Gaussian predictive distribution (Ch 6).
- **Open-the-box:** inspect residual autocorrelation; show where the Gaussian/linear assumption breaks.

### 5.3 M2 — Deep probabilistic (DeepAR)
- **Role:** the *fair* deep baseline — it already outputs a distribution, so diffusion isn't compared against a deterministic net.
- **Library:** **GluonTS** (`DeepAREstimator`, PyTorch backend). Well-documented, handles loading/training/eval.
- **Configure:** likelihood (Student-t/Gaussian), context length, RNN size, epochs; `num_parallel_samples` for predictive sampling.
- **Oral point:** autoregressive likelihood factorization + Monte-Carlo roll-out = predictive distribution; same generative spirit as diffusion, different mechanism.
- **Open-the-box:** plot sampled trajectories; show how sample spread = predictive uncertainty.

### 5.4 M3 — Frontier: conditional diffusion (TimeGrad) ← the centerpiece
- **Role:** the model the whole project is about.
- **Library:** **PyTorchTS** (`TimeGradEstimator`, by the TimeGrad author, built on GluonTS). This is the canonical, reproducible implementation.
- **What it is:** autoregressive diffusion — an RNN encodes the history into a hidden state `h_t` (the conditioning `c`), and at each forecast step a **conditional DDPM** denoises a sample of the next multivariate vector given `h_t`. Exactly §11.2 made conditional.
- **Configure:** diffusion steps `T` (e.g., 100), `β` schedule, RNN/denoiser sizes, epochs, `num_parallel_samples`.
- **Oral point:** be able to point at the denoising loss and say "this is §11.2.3, conditioned on `h_t`."
- **Open-the-box (mandatory — this is how we avoid 'black box'):**
  - vary `T` (the denoising steps) → E3;
  - swap the `β` schedule (linear vs cosine) → ablation;
  - plot intermediate denoising states of a forecast (noise → trajectory);
  - read `TimeGradEstimator`'s forward/loss in the source and annotate it against the notes.

### 5.5 The "understanding artifact": a from-scratch toy DDPM (small, high value)
Because the team is new to DL frameworks and the oral rewards real understanding, build **one** minimal conditional DDPM *from scratch* on a *univariate* or 1-channel version, following the notes' **Algorithm 1 & 2** literally:
- denoiser = small MLP or 1D-CNN `ε_θ(x_t, t, c)`;
- `T ≈ 100`, linear `β`;
- train with the exact loss in §11.2.3; sample with Alg. 2.
- ~150 lines, runs on CPU/Colab in minutes.
**Why:** it is the cleanest possible evidence of understanding, it powers the E3 steps-vs-quality story without library friction, and it is "original w.r.t. course content" because the course never builds the *conditional* version. Mark it optional only if time is dire.

### 5.6 (Optional) M4 — CSDI, or a GP contrast
- **CSDI** (conditional score-based diffusion, NeurIPS'21): forecasting-as-imputation with self-attention. Heavier; the *second diffusion model* if we go to Tier 2.
- **GP regression** (Ch 12): a fully-Bayesian forecaster with *epistemic* uncertainty — the perfect foil for the §2.3 uncertainty-decomposition discussion. Cheap on Exchange-size data via `GPyTorch`/`scikit-learn`.

---

## Part 6 — Evaluation protocol

Evaluation *is* the project's rigor. We measure **three** families (point · probabilistic · cost) + enforce statistical hygiene.

### 6.1 Point accuracy
- **MAE**, **RMSE** (on the predictive **mean/median**), and **MASE** (scale-free, lets us compare across series/datasets honestly). Optionally sMAPE.

### 6.2 Probabilistic quality (the PML core)
- **CRPS** — headline metric; generalizes MAE to full distributions; proper scoring rule (cite Gneiting & Raftery 2007). For multivariate, report **CRPS-sum** (sum channels, then CRPS) as TimeGrad does, for comparability.
- **Coverage @ {50, 80, 90}%** — do the real values fall inside the predicted intervals at the claimed rate?
- **Interval width** — sharpness; narrow *and* calibrated is the goal.
- **Calibration:** PIT histogram / reliability diagram — is the forecast distribution statistically consistent with reality?
- **Library:** GluonTS `Evaluator` gives MASE, sMAPE, **weighted-quantile-loss (a CRPS proxy)**, and coverage out of the box. Cross-check CRPS on a toy against a hand-rolled implementation (E0 hygiene).

### 6.3 Cost / efficiency (the practical edge)
- **Training** wall-clock (per dataset, fixed hardware) and **#parameters**.
- **Inference latency**: time to produce `S` sample trajectories for one window.
- **GPU-hours and a rough € estimate** (Colab/cloud) — this is a real, gradeable practical insight.
- **The diffusion-specific curve:** quality (CRPS) vs **number of denoising steps `T`** (E3). This is where diffusion's famous slow-sampling limitation becomes a *measured* result, tying directly to the survey's stated limitation.

### 6.5 Statistical hygiene (do not skip — it's cheap originality)
- **≥3 random seeds** per learned model; report **mean ± std**, not single runs.
- **Identical** splits, scaling, `H`, `τ`, and test windows across all models.
- **E0 reproduce-gate:** before trusting any number, reproduce a published CRPS-sum for one (model, dataset) within a sane tolerance. If we can't, the pipeline is wrong — fix it before proceeding.
- **Fairness rules:** point metrics computed from the predictive mean for *all* models; never compare a tuned diffusion against an untuned baseline; never pick the best horizon post-hoc.

### 6.6 Evaluation pitfalls to actively guard against
Look-ahead leakage · scaling fit on test · straddling-window leakage · comparing distributional vs point models unfairly · cherry-picking the favorable horizon/seed · reporting CRPS without saying CRPS-sum-vs-mean. Put these in the code-review checklist (Part 8.6).

---

## Part 7 — Experiments (numbered, falsifiable)

Each experiment states a **hypothesis**, **setup**, **what we vary/measure**, the **result that would support or refute the thesis**, and the **artifact** (table/plot) it produces. This is the definition of "done." *(Appendix E explains this phase in depth: first the methodology, then the operational runbook.)*

### E0 — Pipeline validation (gate)
- **Hypothesis:** our pipeline reproduces a known result.
- **Setup:** one model (DeepAR or TimeGrad) on one benchmarked dataset (Electricity/Solar/Exchange).
- **Measure:** CRPS-sum vs the published value.
- **Pass condition:** within tolerance (say same order, ≲20%). **Until E0 passes, no other numbers are trusted.**
- **Artifact:** a single "reproduction" row + a note on remaining gaps.

### E1 — Main comparison (the core result)
- **Hypothesis:** diffusion is competitive on point accuracy and *better on probabilistic quality* than baselines.
- **Setup:** M0,M1,M2,M3 on the primary dataset, fixed `H`, `τ`, ≥3 seeds.
- **Measure:** MAE/RMSE/MASE · CRPS-sum · coverage{50,80,90} · interval width · train/infer time · #params.
- **Supports thesis if:** M3 ≤ baselines on CRPS & calibration, even if MAE is only comparable.
- **Refutes/complicates if:** a seasonal-naive or DeepAR matches M3 on CRPS — *also a publishable-grade finding* ("diffusion didn't pay off here, and here's why").
- **Artifact:** the master results table + a forecast-with-intervals plot per model.

### E2 — Horizon sweep
- **Hypothesis:** diffusion's advantage **grows with horizon** (more future ⇒ more uncertainty ⇒ generative models help more).
- **Setup:** `τ ∈ {short, medium, long}` (e.g., 24/48/96), all models.
- **Measure:** CRPS & coverage vs `τ`.
- **Artifact:** CRPS-vs-horizon line plot. A clean monotone story is gold for a slide.

### E3 — Denoising steps vs quality & cost (the diffusion-specific experiment)
- **Hypothesis:** there's a **knee** — quality saturates while cost grows linearly in `T`.
- **Setup:** sweep `T ∈ {5,10,25,50,100,(250)}` on M3 (and/or the toy DDPM).
- **Measure:** CRPS vs `T` and inference-time vs `T` on the same axes.
- **Why it matters:** it operationalizes the survey's stated limitation (slow iterative sampling) and demonstrates we *understand* the mechanism, not just the API.
- **Artifact:** the quality/cost trade-off plot — likely the most memorable slide.

### E4 — Robustness / regime shift (the "balanced-scope" extra)
- **Hypothesis:** under distribution shift, calibration degrades; we test whether diffusion degrades **gracefully** vs baselines.
- **Setup:** train on a "normal" period, test on a shifted one (different season / a structural break / a volatility cluster). The survey explicitly motivates this (non-stationarity, regime shifts).
- **Measure:** CRPS & coverage drop, train→shift; which model keeps intervals honest.
- **Artifact:** a before/after calibration plot + a degradation table.

### E5 — Generality (stretch, Tier 2)
- **Either** a second dataset (does the E1 story hold?) **or** a second diffusion model (CSDI) **or** the GP epistemic-uncertainty contrast.
- **Artifact:** a "does it generalize?" table or the uncertainty-decomposition figure.

---

## Part 8 — Software architecture & engineering best practices

The repo must be reproducible, config-driven, and reviewable by three people who code at different levels.

### 8.1 Repository layout
```
pml-diffusion-tsf/
├── README.md                 # what/how to run, results summary
├── pyproject.toml / requirements.txt   # pinned deps
├── configs/                  # one YAML per experiment (model+data+seed)
│   ├── data_exchange.yaml
│   ├── model_deepar.yaml
│   └── exp_E1_main.yaml
├── src/
│   ├── data/                 # loaders, splitting, scaling, windowing, manifest
│   ├── models/               # thin wrappers: naive, arima, deepar, timegrad, toy_ddpm
│   ├── eval/                 # metrics (CRPS, coverage, calibration), runners
│   ├── viz/                  # consistent plotting (forecasts, intervals, curves)
│   └── utils/                # seeds, logging, timing, config loading
├── notebooks/                # exploration + the toy-DDPM teaching notebook
├── experiments/              # entry scripts: run_E0.py ... run_E4.py
├── results/                  # CSVs (the results registry) — committed
├── figures/                  # generated plots for slides — committed
└── runs/                     # checkpoints, logs — gitignored
```

### 8.2 Environment & compute
- **Two environments:** a light local one (data, classical baselines, eval, plots) and a Colab/GPU one (DeepAR, TimeGrad). Pin versions — **PyTorchTS↔GluonTS compatibility is fragile**; record the known-good combo in `requirements.txt` and the README.
- **Colab discipline (team is new to it):** mount Google Drive for data/checkpoints; put the known-good `pip install` block at the top of every training notebook; checkpoint to Drive so a disconnect doesn't lose a run; keep a "Colab setup" cell documented once.

### 8.3 Config & reproducibility
- Every run is fully described by a **YAML config** (model, data, `H`, `τ`, `T`, seed). No magic numbers in code.
- **Global seed** set for numpy/torch; log it; enable deterministic flags where feasible.
- Each run writes a row to `results/registry.csv` with config hash + all metrics + timing → the results table is *generated*, never hand-typed.

### 8.4 Experiment tracking
- Minimum: structured `results/registry.csv` + saved configs.
- Nice-to-have: Weights & Biases (free) for loss curves — helps the new-to-DL members *see* training.

### 8.5 Using Claude Code well (this matters, given the skill mix)
Give it **small, scoped, testable tasks**, not "build the project." Good prompts:
- "Write `src/data/load_exchange.py`: download, temporal 70/10/20 split, train-only mean-scaling, return GluonTS datasets + manifest. Include a `__main__` that prints shapes."
- "Implement `crps_sample(samples, target)` and a unit test comparing to a 3-point hand-computed example."
- "Given a GluonTS forecast object, plot median + 50/90% bands vs ground truth; save to `figures/`."
- "Write `run_E3.py` that sweeps `T` and appends CRPS + inference-time rows to the registry."
Always ask it to add a tiny test or a `__main__` smoke check. Review every diff against Part 6.6 pitfalls.

### 8.6 Code-review checklist (run before each merge)
☐ no scaling/leakage across splits ☐ same `H,τ`,test-window across models ☐ seed logged ☐ metric matches definition (CRPS-sum vs mean stated) ☐ figure regenerable from a script ☐ config committed.

---

## Part 9 — Team plan: 3 isolated steps × 3 people

The work is split into **three fundamental, isolated steps** (the team's time-units). Each step has a goal, a *definition of done*, deliverables, and per-person tasks. Roles are assigned to the "comfortable Python, new to DL" profile: spread the DL learning, give everyone one thing they fully own for the oral.

### Roles (each person owns a vertical slice end-to-end so each can answer in the oral)
- **Person A — Baselines & Statistics owner:** M0, M1, the classical/PML-statistics narrative, point metrics.
- **Person B — Diffusion & Infra owner:** M2, M3, the toy DDPM, Colab/training, configs.
- **Person C — Evaluation, Viz & Story owner:** CRPS/coverage/calibration, all figures, the slide narrative, oral-question bank.
Everyone reads Part 2 (the spine) — the oral grades individuals.

### Step 1 — Understand, restrict, formalize *(close when you can say the project in 30 seconds)*
- **Goal:** turn the survey + notes into a precise, frozen experimental question + a sent proposal.
- **Definition of done:** dataset chosen (primary + Exchange-for-iteration); `H`, `τ`, baselines, metrics fixed; 1-page paper/notes summary written; proposal emailed (Appendix A); repo skeleton + Exchange loader exist and run.
- **Per person:**
  - A: study ARIMA/ETS + the classical-forecasting framing; draft the metrics definitions (MAE/RMSE/MASE/CRPS/coverage).
  - B: study §11.1–11.2 + the TimeGrad paper; stand up the repo skeleton, environments, and the Exchange loader (E-prep).
  - C: study the survey taxonomy + §1.1.1/§9–10; draft the proposal and the 1-page summary; set up the figures/results scaffolding.
- **Output artifacts:** `PROPOSAL` sent · `summary.md` · repo skeleton · green Exchange loader.

### Step 2 — Build pipeline & run experiments *(close when E0+E1 are green on the primary dataset)*
- **Goal:** produce solid quantitative results.
- **Definition of done:** E0 passes; E1 table complete with ≥3 seeds; E3 trade-off curve done; E4 attempted; all figures generated from scripts.
- **Per person:**
  - A: M0 + M1 running and evaluated; point-metric tables; residual diagnostics figure.
  - B: M2 (DeepAR) + M3 (TimeGrad) training on Colab; the toy DDPM; E3 sweep; checkpoints saved.
  - C: the evaluation module (CRPS/coverage/calibration) + the master results runner; all comparison/interval/calibration plots; keep the registry clean.
- **Output artifacts:** `results/registry.csv` populated · master table · forecast/interval plots · E3 curve · (E4 degradation).

### Step 3 — Interpret, write the story, prepare the oral *(close when the 8-min deck + oral bank are done)*
- **Goal:** turn tables into an 8-minute argument and individual oral readiness.
- **Definition of done:** 8–9 slides; speaker notes; per-person oral Q&A bank mapped to course pages; clean repo + README; an explicit "what we deliberately did NOT show" list.
- **Per person:**
  - A: the baselines/statistics slides + the "is the baseline actually beaten?" honesty point; ARIMA↔state-space oral answers.
  - B: the diffusion-mechanism slide (forward/reverse/loss in the notes' notation) + the E3 cost story; diffusion oral answers (derive the loss).
  - C: the story arc, the results & calibration slides, the uncertainty-decomposition (§2.3) slide; rehearsal + timing; assemble the oral bank.
- **Output artifacts:** `slides.pdf` · `speaker_notes.md` · `oral_qa.md` · polished repo.

### 9.4 Modular "stop-after" tiers (because the timeline is flexible)
| Tier | Contains | Coherent project on its own? |
|---|---|---|
| **Tier 0 — MVP** | Steps 1–2 partial: E0 + E1 (M0–M3) on one dataset; basic slides | **Yes** — a complete comparison study. |
| **Tier 1 — Target (your "balanced")** | + E3 (steps vs cost) + E4 (regime shift) + calibration | Yes, and clearly exam-strong. |
| **Tier 2 — Stretch** | + E2 full sweep, E5 (2nd dataset / CSDI / GP), toy-DDPM deep-dive, uncertainty-decomposition figure | Yes, competitive for *laude*. |

Drive to Tier 0 first; *then* climb. Never leave Tier 0 half-done to start Tier 2.

---

## Part 10 — Risk register & mitigations
| Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|
| PyTorchTS↔GluonTS version hell | High | High | pin the known-good combo; isolate in its own Colab env; fallback to a from-scratch conditional DDPM (toy scaled up) | B |
| TimeGrad won't converge / OOM | Med | High | subset channels; fewer epochs/steps; smaller batch; Exchange first | B |
| CRPS implemented wrong | Med | High | use GluonTS Evaluator + cross-check on a toy; E0 gate | C |
| Data leakage (scaling/window) | Med | High | central loader; review checklist (8.6); manifest of split dates | A/C |
| Compute / € overrun | Med | Med | iterate on Exchange; cache; cap seeds at 3; cheap models first | B |
| Scope creep (chasing Tier 2 early) | High | Med | enforce tier gates (9.4); MVP before stretch | all |
| Oral exposes shallow understanding | Med | High | the spine (Part 2) + toy DDPM + per-person Q&A bank | all |
| One person becomes a bottleneck | Med | Med | vertical slices (each owns a runnable path); weekly sync | all |

---

## Part 11 — Deliverables

### 11.1 The 8-minute presentation (≈8–9 slides, one message each)
1. **Problem** — "forecasting = plausible futures, not one number." (the hook)
2. **PML framing** — `p(y_future | x_past)`; generative + probabilistic; aleatoric vs epistemic.
3. **Diffusion idea** — true future → add noise → learn to denoise *conditioned on the past* → sample trajectories. (one clean schematic)
4. **Where it sits in the course / survey** — §11.2 made conditional; survey taxonomy (conditioning source + integration) in one line.
5. **Experimental setup** — dataset, `H`, `τ`, temporal split, the model ladder.
6. **Main result (E1)** — the table: MAE/RMSE · CRPS · coverage · cost. Plus one forecast-with-intervals plot.
7. **The diffusion-specific insight (E3, +E4)** — quality-vs-denoising-steps trade-off; (regime-shift calibration).
8. **Discussion** — when diffusion wins / loses; what uncertainty we capture and miss; cost caveat.
9. **Conclusion** — "richer, better-calibrated uncertainty, at a tunable sampling cost; useful when uncertainty matters." + future work (Bayesian/ensemble diffusion).
- **Deliberately NOT shown:** every ablation, every seed, library plumbing, failed runs. Keep only what supports the thesis.

### 11.2 Supporting artifacts
- `speaker_notes.md` (1 page), the repo + README + `results/registry.csv` + `figures/`.

### 11.3 Oral-exam preparation (per person, mapped to course pages)
Build `oral_qa.md` with answers each member can give. Seed questions:
- *Derive the DDPM loss; why predict ε not x₀?* → §11.2.3 (p.126). (B leads)
- *Why is the reverse process ~Gaussian? when does that fail?* → §11.2.2 (p.125). (B)
- *Where is the ELBO / reparameterization trick here?* → §9.2 (p.96), §10.4.1 (p.112). (B/C)
- *What is CRPS, why is it "proper," how vs MAE?* → §2.3 (p.20). (C)
- *Aleatoric vs epistemic — which does your diffusion capture?* → §1.1.1 (p.2). (C)
- *How does DeepAR produce a distribution? vs diffusion?* → §1.2 (p.3). (A/B)
- *ARIMA as a probabilistic model — assumptions?* → Ch 5/6. (A)
- *Score-based view & link to Langevin/MCMC?* → §11.2.5 (p.127), Ch 8. (B)
- *Why is diffusion sampling slow; how did you measure/mitigate it?* → E3. (B/C)

---

## Part 12 — Study plan (what to read, when, who)
Recall the team has studied through Ch 5. The project needs Ch 9–11 most.
| Read | Who | When (step) | Why |
|---|---|---|---|
| §9.2 ELBO, §10.4.1 reparam., §10.6 BNN | all | Step 1 | the machinery diffusion reuses |
| §11.1 VAE, §11.2 Diffusion (whole) | all (B deepest) | Step 1 | the core; B must derive it |
| Survey *Diffusion Models for TSF* (2507.14507) | C (+all skim) | Step 1 | taxonomy, framing, related work |
| TimeGrad paper (Rasul et al., ICML'21) | B | Step 1–2 | the model we run |
| DeepAR paper (Salinas et al.) | A/B | Step 2 | the deep baseline |
| CRPS / proper scoring (Gneiting & Raftery '07) | C | Step 1–2 | metric correctness |
| §1.1.1 uncertainty; Ch 12 GP (skim) | C | Step 2–3 | the discussion slide |
| (opt.) CSDI paper | B | Step 3 (Tier 2) | second diffusion model |

---

## Appendix A — Ready-to-send proposal (copy-paste / translate for the professor)

> **Subject:** Project proposal — Probabilistic Time Series Forecasting with Diffusion Models
>
> Dear Professor Bortolussi,
>
> our group (3 students) proposes a project on **probabilistic time-series forecasting with diffusion models**, extending the diffusion material of Chapter 11 from unconditional generation to *conditional* forecasting.
>
> **Question.** Can a conditional diffusion model estimate `p(future | past)` — a distribution of plausible future trajectories — with better-calibrated uncertainty than classical and deep-probabilistic baselines, and at what sampling cost?
>
> **Method.** On a real multivariate dataset (e.g., Solar/Electricity), we compare a seasonal-naive baseline, a classical model (ARIMA/ETS), a deep autoregressive probabilistic model (DeepAR), and a diffusion model (TimeGrad), plus a small from-scratch conditional DDPM built from the Algorithms in §11.2 to demonstrate understanding. We evaluate point accuracy (MAE/RMSE/MASE), **probabilistic quality (CRPS, coverage, calibration)**, and **computational cost** (including a quality-vs-denoising-steps trade-off and a regime-shift robustness test).
>
> **Connection to the course.** The training objective is the §11.2 noise-prediction loss (a reparameterized ELBO, §9.2/§10.4.1); sampling is reverse-chain ancestral sampling (§3.3.1); we also discuss the score-based/Langevin view (§11.2.5) and what *kind* of uncertainty — aleatoric vs epistemic (§1.1.1) — the model captures.
>
> Would this scope be appropriate for the project? We are happy to adjust the dataset or the set of baselines.
>
> Kind regards, [names]

## Appendix B — Glossary (term → plain English → course page)
- **Aleatoric / epistemic uncertainty** — irreducible data noise / reducible model ignorance — §1.1.1 (p.2).
- **ELBO** — the lower bound we maximize instead of the intractable likelihood — §9.2 (p.96).
- **Reparameterization trick** — write `z=μ+σ·ε` so gradients flow through sampling — §10.4.1 (p.112).
- **Forward/Reverse diffusion** — fixed noising chain / learned denoising chain — §11.2.1–2 (p.123).
- **Noise-prediction loss** — train `ε_θ` to predict the added noise — §11.2.3 (p.126).
- **CRPS** — distributional generalization of MAE; a proper scoring rule — §2.3 (p.20).
- **Coverage / calibration** — do x% intervals contain the truth x% of the time — eval (Part 6).
- **CRPS-sum** — multivariate CRPS used by TimeGrad (sum channels, then score).
- **Pinball (quantile) loss** — the asymmetric loss whose minimizer is a given quantile; CRPS = its average over all quantile levels — §2.3 (p.20).

## Appendix C — References
- *Diffusion Models for Time Series Forecasting: A Survey*, arXiv:2507.14507 (2025).
- Rasul et al., *Autoregressive Denoising Diffusion Models for Multivariate Probabilistic TSF* (TimeGrad), ICML 2021.
- Tashiro et al., *CSDI*, NeurIPS 2021.
- Salinas et al., *DeepAR*, Int. J. Forecasting 2020.
- Ho et al., *Denoising Diffusion Probabilistic Models* (DDPM), NeurIPS 2020.
- Gneiting & Raftery, *Strictly Proper Scoring Rules*, JASA 2007.
- Gneiting & Katzfuss, *Probabilistic Forecasting*, Annual Review of Statistics 2014 — calibration and proper scoring of probabilistic forecasts.
- Bortolussi, *PML Lecture Notes* — Ch 1, 2, 3, 5, 8, 9, 10, 11, 12.
- Yang et al., *Diffusion Models: A Comprehensive Survey* (2022) — already in the PML folder.

## Appendix D — Dataset quick-facts (fill during Step 1)
| | primary = ? | iteration = Exchange | stretch = ? |
|---|---|---|---|
| D (channels) | | 8 | |
| frequency | | daily | |
| length | | ~6k | |
| chosen H / τ | | / | |
| regime-shift idea (E4) | | | |
| published CRPS-sum (E0 target) | | | |

---

## Appendix E — The experimental phase, explained in depth (methodology + operational runbook)

This appendix exists because the **experimental phase is the gradeable heart of the project**. Part 7 lists *which* experiments we run; here we explain **first the "why" (the methodology)** and **then the step-by-step "how" (the runbook)**. Read the first half to understand, the second half to execute.

### E.1 — Methodology: what makes an experiment valid

#### E.1.1 An experiment is not "running the code": it is a test that can fail
In our project an experiment has five mandatory parts (the same as Part 7):
1. **Hypothesis** — a precise, *falsifiable* prediction ("M3 has lower CRPS than DeepAR on the primary dataset").
2. **Setup** — the controlled environment: dataset, `H`, `τ`, seed, models involved.
3. **What varies / what is measured** — the *independent variable* we move (e.g. `T`) and the *dependent variable* we observe (e.g. CRPS, time).
4. **Outcome that supports vs refutes** — decided *before* looking at the numbers. If you can't say in advance which result would refute you, it's not an experiment, it's a demo.
5. **Artifact** — the table or plot that remains as evidence (and that ends up on a slide).

The mental rule: **a good experiment can disappoint our thesis, and that is still a result.** If diffusion loses on a dataset, we report it and explain why — this *raises* the grade (honesty, understanding), it does not lower it.

#### E.1.2 The fair-comparison principle (it's what makes the numbers credible)
The entire value of the E1 comparison depends on every model playing *by the same rules*. Concretely:
- **Same "data contract"** (Part 4.5): identical splits, scaling, `H`, `τ`, test windows for all models. One loader serves them all.
- **Same way of extracting the point and the interval** (Part 3.4): the point metric is always computed from the predictive mean/median, even for probabilistic models.
- **No asymmetric tuning:** don't compare an optimized TimeGrad against a DeepAR left at defaults. Either tune both, or neither, and state it.
- **No post-hoc choices:** the horizon, the seed and the threshold are fixed *beforehand*. Choosing the most favorable afterward is cherry-picking (Part 6.6).

#### E.1.3 The reproduction gate (E0): why we start there
Before trusting *any* number we produce, we reproduce an **already-published** number (a TimeGrad or DeepAR CRPS-sum on a known dataset) within a sane tolerance (same order of magnitude, ≲20%). The logic:
- If we reproduce a known result, the pipeline (loading, scaling, metric) is probably correct.
- If we **cannot**, there's an upstream bug (leakage, scaling, wrong CRPS) and *all* the later numbers would be garbage.
- It is also where you learn to use the tools on a target whose answer you already know — the least frustrating way to make mistakes.

**Until E0 is green, no other experiment counts.** It is literally a gate.

#### E.1.4 Statistical hygiene: the noise of randomness
Learned models depend on the random seed (weight initialization, batch order). A single number can be lucky or unlucky. Therefore:
- Run every learned model with **≥3 seeds** and report **mean ± standard deviation**.
- If two models are within one standard deviation of each other, do *not* declare a winner: say "indistinguishable on this data".
- Keep the seeds fixed and logged, so anyone can re-run and get the same numbers.

#### E.1.5 How to read the output (what "looks good")
- **CRPS / CRPS-sum:** lower is better. Alone it says nothing: it only makes sense *relative* (vs a baseline) or *vs the published number* (E0).
- **Coverage @ 90%:** you want ~90%. Well below = the model is overconfident (narrow, lying intervals); well above = too cautious (wide, useless intervals).
- **Interval width:** at equal coverage, narrower is better (sharper).
- **PIT histogram:** you want it **flat**. U-shaped = overconfident; bell-shaped = too cautious. It is the visual honesty check of the forecast.
- **CRPS-vs-`T` curve (E3):** you're looking for a **knee** — a point beyond which adding denoising steps no longer improves quality but keeps costing time. That knee is the "memorable slide".

#### E.1.6 Threats to validity (the list to fear)
These are the ways an experiment can *look* successful and be false. Print it and use it as a checklist:
- **Look-ahead leakage:** the model sees future information (e.g. a covariate computed over the whole series).
- **Scaling leakage:** the scaler was fit including val/test.
- **Straddling window:** a `(H, τ)` window crosses the train/test boundary.
- **Unfair comparison:** distributional vs point model without bringing both to the same metric.
- **Cherry-picking:** post-hoc choice of favorable seed/horizon/threshold.
- **CRPS ambiguity:** reporting "CRPS" without saying whether it's per-channel mean or CRPS-sum (they're not comparable).

### E.2 — Runbook: how you actually run it, step by step

> Assumes the repo skeleton of Part 8.1 and the two environments of Part 8.2. Golden order: **first get everything running on Exchange** (minutes, even without a GPU), *then* move to the primary dataset on Colab.

#### E.2.0 Environment setup (once)
1. **Local (light):** create a virtual environment; install the "light" group (numpy, pandas, Darts/statsforecast, the metrics, matplotlib). Data, classical baselines, evaluation and plots run here.
2. **Colab (GPU):** in a notebook, first cell = the known-good `pip install` block (pinned, compatible versions of PyTorchTS + GluonTS + torch — compatibility is fragile, Part 8.2). Second cell = mount Google Drive. Third cell = clone/update the repo. Save checkpoints to Drive, not to Colab's ephemeral disk.
3. **Smoke check:** run the Exchange loader's `__main__`; it must print train/val/test shapes with no errors. If this doesn't run, do not proceed.

#### E.2.1 Run E0 (the gate) — do it first
1. Pick the (model, dataset) pair with a **published** CRPS-sum (e.g. TimeGrad on Exchange or Electricity). Note the target number in Appendix D.
2. Write/use `experiments/run_E0.py`: load data with the central loader, train the model with the config (YAML), sample `S` trajectories on the test set, compute CRPS-sum with the evaluation module.
3. Compare with the published number.
   - **Within tolerance?** → E0 green. Write the "reproduction" row in the registry and a note on the gaps. Proceed.
   - **Out of tolerance?** → *do not proceed*. Check, in this order: CRPS definition (sum vs mean?), scaling leakage, `H/τ` alignment, number of samples `S`. Iterate until it falls in range.
4. Cross-check the CRPS: compute it by hand on a 3-point example and compare with the module, once, to trust the code.

#### E.2.2 Run E1 (the main comparison)
1. Fix, *beforehand*, `H`, `τ`, the primary dataset, the set of seeds (≥3).
2. For each model `M0, M1, M2, M3` and each seed: run via its YAML config; every run **appends a row** to `results/registry.csv` with config-hash + all metrics + timings. No hand-typed numbers.
3. Generate the master table *from* `registry.csv` (a script, not copy-paste): for each model, mean ± standard deviation over seeds.
4. Generate a forecast-with-intervals plot per model (median + 50/90% bands vs ground truth).
5. Read the results with the mindset of §E.1.5; write a conclusion sentence *before* polishing the slides.

#### E.2.3 Run E3 (denoising steps vs quality/cost)
1. Keep everything fixed except `T`; define the grid `T ∈ {5,10,25,50,100,(250)}`.
2. `experiments/run_E3.py` loops over `T`: for each value, sample from the *already-trained* model (do not retrain!), record CRPS *and* inference time per window.
3. Plot two curves on the same axes (CRPS vs `T`; time vs `T`). Look for the knee (§E.1.5).
4. If you also use the toy DDPM, repeat: this is where it pays off, because you control it completely.

#### E.2.4 Run E4 (regime-shift robustness)
1. Define two periods: "normal" (train) and "shifted" (test) — a different season, a structural break, a volatility cluster. Document the criterion.
2. Train on the normal period; evaluate on both the normal and the shifted period.
3. Measure the *drop* in CRPS and coverage from normal→shifted, for each model.
4. Artifact: a before/after calibration plot + a degradation table. The story is "who keeps the intervals honest under shift?".

#### E.2.5 Cross-cutting discipline (applies to every experiment)
- **One config = one run:** no magic numbers in code; everything in YAML (model, data, `H`, `τ`, `T`, seed).
- **Checkpoints saved to Drive** after every training run, so a Colab disconnect doesn't cost a run.
- **The registry is the truth:** the slide table is *regenerated* from `results/registry.csv`. If a number isn't in the registry, it doesn't exist.
- **Figures are regenerated from scripts** (Part 8.6): never a "hand-made" figure you can't reconstruct.
- **Pre-merge checklist (Part 8.6)** at every step: no leakage, same `H/τ`/window, seed logged, metric stated, figure regenerable, config committed.

#### E.2.6 "Definition of done" of the experimental phase
The experimental phase is closed (Tier 1, the "balanced" one) when:
- ☐ E0 is green and documented (reproduction row + gap note);
- ☐ the E1 table is complete, with mean ± standard deviation over ≥3 seeds, script-generated;
- ☐ the E3 trade-off curve exists with a readable knee;
- ☐ E4 has been attempted, with a before/after calibration plot;
- ☐ every number in every figure is traceable back to a row of `results/registry.csv` and a committed YAML config.
