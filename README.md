# Probabilistic Time-Series Forecasting with Diffusion Models

> **⚠️ Personal sandbox / test fork.** This is a personal experimentation copy, pinned to the **Exchange** dataset, forked from the group base [`Icaica14/pml-diffusion-tsf`](https://github.com/Icaica14/pml-diffusion-tsf). The canonical, group-developed project (targeting the **Electricity** dataset) lives there — this fork is for trying things out and may be broken or half-finished at any time.

> **Can a diffusion model produce *probabilistic* forecasts of a future time series — a distribution of plausible futures rather than a single number — that are more accurate, better calibrated, or more informative than classical and deep-learning baselines, and at what computational cost?**

![status](https://img.shields.io/badge/status-planning-yellow)
![license](https://img.shields.io/badge/license-MIT-blue)
![course](https://img.shields.io/badge/course-PML%20%C2%B7%20UniTS-8A2BE2)
![python](https://img.shields.io/badge/python-3.10%2B-3776AB)

**Exam project for the *Probabilistic Machine Learning* (PML) course — University of Trieste, Prof. Luca Bortolussi.**

🇬🇧 English (below) · 🇮🇹 [Versione italiana](#-in-italiano)

---

## Overview

Most forecasts give you a single number — *"tomorrow's electricity demand will be 100."* But the future is uncertain, and one number hides that uncertainty entirely. A more useful forecast says: *"here are the plausible tomorrows, and how likely each one is"* — a whole **distribution** of futures instead of one guess.

That is what this project builds. We treat forecasting as one question: *given everything seen so far (the past), what does the distribution of possible futures look like?* — written `p(future | past)`. To answer it we use a **diffusion model**: the same family of generative models behind modern image generators, retrained here to *generate plausible future trajectories of a time series* instead of pictures. It learns from the exact objective the PML course derives in its final chapter — our twist is to make it **conditional on the past**.

We compare it head-to-head against a naive baseline, a classical statistical model (ARIMA/ETS), and a deep autoregressive probabilistic model (DeepAR), and we measure **four** things the exam rewards:

- **Point accuracy** — MAE, RMSE, MASE
- **Probabilistic quality** — CRPS, interval coverage, calibration
- **Cost** — training/inference time, and the quality-vs-denoising-steps trade-off
- **Economic value** — the *money saved* when a real decision (scheduling a battery against a time-of-use price) is driven by each model's forecast, because a better-calibrated distribution makes cheaper, more robust decisions (see [Economic value](#economic-value--turning-forecasts-into-money) below)

Our thesis is deliberately non-triumphalist: diffusion buys *richer, better-calibrated uncertainty*, but its advantage depends on horizon, dataset, and a real sampling-cost penalty we can measure and tune.

## Status

🟡 **Planning phase — no code yet.** The full blueprint is written and lives in [`docs/`](docs/). Everything below will be built *from* that plan.

| Document | Description |
|---|---|
| [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | The complete, execution-ready implementation plan (English), including a deep appendix (E) on the experimental phase. |
| [`docs/IMPLEMENTATION_PLAN_IT.md`](docs/IMPLEMENTATION_PLAN_IT.md) | Full Italian translation, with every acronym expanded on first use and the same deep appendix on the experimental phase. |
| [`docs/EDA_EXCHANGE.md`](docs/EDA_EXCHANGE.md) | Figure-backed exploratory analysis of the Exchange iteration dataset (random-walk levels, volatility clustering, heavy tails) and what each finding implies for the models. |

## Approach at a glance

**The model ladder** (each rung a stronger opponent):

| Model | Role | Library |
|---|---|---|
| **M0 — Seasonal-naive** | the honesty anchor — must be beaten | NumPy / Darts |
| **M1 — ARIMA / ETS** | the classical statistical baseline | statsforecast |
| **M2 — DeepAR** | the *fair* deep probabilistic baseline | GluonTS |
| **M3 — TimeGrad** | the frontier: conditional diffusion *(centerpiece)* | PyTorchTS |
| *+ toy DDPM* | a ~150-line from-scratch conditional DDPM — the "understanding artifact" | PyTorch |

**The experiments** (each numbered, falsifiable, producing one artifact):

`E0` reproduce-a-published-result gate · `E1` main comparison · `E2` horizon sweep · `E3` denoising-steps vs quality & cost · `E4` regime-shift robustness · `E6` **economic value (battery dispatch)** · `E5` generality (stretch).

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) Parts 5–7 for the full specification.

## Economic value — turning forecasts into money

A forecast only matters if it changes a **decision**. Our fourth evaluation pillar makes that concrete: we use each model's forecast to **schedule a battery** (charge when power is cheap, discharge when it's expensive) against a time-of-use electricity price, then price the result.

- A **point** forecast plans against a single guessed future; a **distribution** (diffusion / DeepAR samples) plans against the whole spread of plausible futures, hedging its bets.
- The schedule is a small **linear program**; with a distribution we minimize the *expected* bill over the model's sampled trajectories (sample-average approximation).
- We apply every schedule to the **true** future and read off the realized bill, then report **money saved** vs a naive baseline (ceiling) and a perfect-foresight **oracle** (lower bound) — so euros are always shown as a fraction of what was actually achievable.
- The punchline ties the pillars together: optimal storage decisions use a **quantile** of the predictive distribution (a *newsvendor* structure), and **CRPS is the average decision regret over all cost ratios** — so a better-calibrated forecast should literally save more money. `E6` tests whether it does.

This is a **bolt-on module** (`src/eval/economic.py`) that runs *over the forecasts E1 already produces* — **no extra training** — so it adds a headline result without enlarging the core project. Full protocol in [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) §3.5, §6.4, and experiment E6.

## Planned repository structure

```
pml-diffusion-tsf/
├── README.md
├── docs/                     # the implementation plan (EN + IT)
├── configs/                  # one YAML per experiment (model + data + seed)
├── src/
│   ├── data/                 # loaders, splitting, scaling, windowing, manifest
│   ├── models/               # wrappers: naive, arima, deepar, timegrad, toy_ddpm
│   ├── eval/                 # metrics (CRPS, coverage, calibration), economic.py (battery dispatch), runners
│   ├── viz/                  # consistent plotting (forecasts, intervals, curves)
│   └── utils/                # seeds, logging, timing, config loading
├── experiments/              # entry scripts: run_E0.py ... run_E4.py
├── notebooks/                # exploration + the toy-DDPM teaching notebook
├── results/                  # CSV results registry (committed)
└── figures/                  # generated plots for the slides (committed)
```

## Getting started

> Code is not here yet (planning phase). When it lands, this section will hold the exact setup. For now:

1. **Read the plan.** Start with [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) Parts 1–3 (the story), then your owned parts.
2. **Two environments are planned:** a light local one (data, classical baselines, evaluation, plots) and a Colab/GPU one (DeepAR, TimeGrad). The PyTorchTS ↔ GluonTS version combo is fragile and will be pinned in `requirements.txt`.
3. **Golden rule:** get the whole pipeline green on the small **Exchange** dataset first; *only then* scale up to the primary dataset.

## Team & roles

Each member owns one vertical slice end-to-end (so each can defend it in the individual oral exam):

| Role | Owns | Member |
|---|---|---|
| **A — Baselines & Statistics** | M0, M1, classical/PML-statistics narrative, point metrics | _add name · GitHub handle_ |
| **B — Diffusion & Infrastructure** | M2, M3, the toy DDPM, Colab/training, configs | _add name · GitHub handle_ |
| **C — Evaluation, Viz & Story** | CRPS/coverage/calibration, all figures, slide narrative, oral Q&A bank | _add name · GitHub handle_ |

## Course context

The PML course *ends* on diffusion models (unconditional generation). This project's originality is to **extend that final chapter from unconditional generation to *conditional* forecasting**, `p(future | past)`, and to discuss explicitly *which* uncertainty the model captures — **aleatoric** (the spread of plausible futures) vs **epistemic** (model/parameter uncertainty, which a point-estimated diffusion model does *not* capture). The implementation plan maps every component back to a specific chapter/page of the course notes.

## Contributing

Colleagues: see [`CONTRIBUTING.md`](CONTRIBUTING.md) for branch conventions, how to pick up a role, and the definition of "done" for each experiment.

## A note on course materials

This repository **intentionally does not include** the professor's lecture notes or any course textbooks. Those are copyrighted, and the notes carry an explicit request not to be redistributed. The plan cites them by section and page number only. `*.pdf` files are gitignored as a safeguard.

## License

Released under the [MIT License](LICENSE). If your university's coursework policy requires otherwise, change this before making the work widely public.

---

## 🇮🇹 In italiano

**Progetto d'esame per il corso di *Probabilistic Machine Learning* (PML, apprendimento automatico probabilistico) — Università di Trieste, Prof. Luca Bortolussi. Gruppo di 3.**

**Domanda di ricerca.** Un *diffusion model* può produrre **forecast probabilistici** di una serie temporale — una *distribuzione* di futuri plausibili invece di un singolo numero — meglio calibrati delle baseline classiche e di deep learning, e a quale costo computazionale?

**Idea.** Inquadriamo il forecasting come l'apprendimento della distribuzione generativa condizionata `p(futuro | passato)` e la realizziamo con un diffusion model condizionato, addestrato con lo stesso obiettivo (ELBO / predizione del rumore) che il corso ricava nell'ultimo capitolo. Lo confrontiamo con una baseline ingenua, un modello classico (ARIMA/ETS) e un modello probabilistico di deep learning (DeepAR), misurando **quattro** cose: **accuratezza puntuale** (MAE/RMSE/MASE), **qualità probabilistica** (CRPS, copertura, calibrazione), **costo** e **valore economico** — il denaro risparmiato quando il forecast di ciascun modello programma una batteria contro un prezzo a fasce orarie (esperimento E6, un modulo *sopra i forecast già prodotti*, senza addestramento aggiuntivo).

**Stato:** 🟡 fase di pianificazione, nessun codice ancora. Il piano completo è in [`docs/IMPLEMENTATION_PLAN_IT.md`](docs/IMPLEMENTATION_PLAN_IT.md) — include un'appendice che spiega a fondo la fase sperimentale (prima la metodologia, poi le istruzioni operative passo-passo).

**Per i colleghi:** leggete prima il piano (Parti 1–3), poi la parte del vostro ruolo (A / B / C, vedi tabella sopra). La regola d'oro: far girare tutta la pipeline sul piccolo dataset **Exchange** prima di salire di scala.
