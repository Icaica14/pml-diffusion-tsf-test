# Probabilistic Time-Series Forecasting with Diffusion Models

> **⚠️ Personal sandbox / test fork.** This is a personal experimentation copy, pinned to the **Exchange** dataset, forked from the group base [`Icaica14/pml-diffusion-tsf`](https://github.com/Icaica14/pml-diffusion-tsf). The canonical, group-developed project (targeting the **Electricity** dataset) lives there — this fork is for trying things out and may be broken or half-finished at any time.

> **Can a diffusion model produce *probabilistic* forecasts of a future time series — a distribution of plausible futures rather than a single number — that are more accurate, better calibrated, or more informative than classical and deep-learning baselines, and at what computational cost?**

![status](https://img.shields.io/badge/status-WIP-orange)
![license](https://img.shields.io/badge/license-MIT-blue)
![course](https://img.shields.io/badge/course-PML%20%C2%B7%20UniTS-8A2BE2)
![python](https://img.shields.io/badge/python-3.10%2B-3776AB)

**Exam project for the *Probabilistic Machine Learning* (PML) course — University of Trieste, Prof. Luca Bortolussi.**

🇬🇧 English (below) · 🇮🇹 [Versione italiana](#-in-italiano)

---

## Overview

Most forecasts give you a single number — *"tomorrow's electricity demand will be 100."* But the future is uncertain, and one number hides that uncertainty entirely. A more useful forecast says: *"here are the plausible tomorrows, and how likely each one is"* — a whole **distribution** of futures instead of one guess.

That is what this project builds. We treat forecasting as one question: *given everything seen so far (the past), what does the distribution of possible futures look like?* — written `p(future | past)`. To answer it we use a **diffusion model**: the same family of generative models behind modern image generators, retrained here to *generate plausible future trajectories of a time series* instead of pictures. It learns from the exact objective the PML course derives in its final chapter — our twist is to make it **conditional on the past**.

We compare it head-to-head against a naive baseline, a classical statistical model (ARIMA/ETS), and a deep autoregressive probabilistic model (DeepAR), and we measure **three** things the exam rewards:

- **Point accuracy** — MAE, RMSE, MASE
- **Probabilistic quality** — CRPS, interval coverage, calibration
- **Cost** — training/inference time, and the quality-vs-denoising-steps trade-off

Our thesis is deliberately non-triumphalist: diffusion buys *richer, better-calibrated uncertainty*, but its advantage depends on horizon, dataset, and a real sampling-cost penalty we can measure and tune.

## Status

🟠 **Sandbox — building the model ladder on the Exchange dataset.** The blueprint lives in [`docs/`](docs/); the first rungs are now code. Already in place: the **data contract** (`src/data/` — contiguous temporal split, train-only scaling, per-split windowing) with an offline test suite; a figure-backed **EDA of Exchange**; the **evaluation module** (point + probabilistic metrics — MAE/RMSE/MASE, CRPS, interval coverage); the **M0 seasonal-naive baseline**; and now the **M1 ARIMA baseline** (per-channel auto-order, leakage-free per-window conditioning, Gaussian predictive bands) — both rows are in [`results/registry.csv`](results/registry.csv). The honest early finding: on the near-random-walk Exchange series M1 *ties* M0 on point accuracy (MASE ≈ 4.52 — an optimal linear forecast collapses to persistence on a unit-root series) but delivers **better-calibrated** uncertainty (90% coverage ≈ 0.93 vs 0.86). The **M2 DeepAR** code and a ready-to-run Colab notebook ([`notebooks/colab_m2_deepar.ipynb`](notebooks/colab_m2_deepar.ipynb)) are now in place — DeepAR is the first rung that leaves the light env, so its registry row is produced on Colab/GPU. Next after that: **M3 (TimeGrad)**, the conditional-diffusion centerpiece.

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
| **M1 — ARIMA / ETS** | the classical statistical baseline | statsmodels *(sandbox; statsforecast in the group repo)* |
| **M2 — DeepAR** | the *fair* deep probabilistic baseline | GluonTS |
| **M3 — TimeGrad** | the frontier: conditional diffusion *(centerpiece)* | PyTorchTS |
| *+ toy DDPM* | a ~150-line from-scratch conditional DDPM — the "understanding artifact" | PyTorch |

**The experiments** (each numbered, falsifiable, producing one artifact):

`E0` reproduce-a-published-result gate · `E1` main comparison · `E2` horizon sweep · `E3` denoising-steps vs quality & cost · `E4` regime-shift robustness · `E5` generality (stretch).

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) Parts 5–7 for the full specification.

## Planned repository structure

```
pml-diffusion-tsf/
├── README.md
├── docs/                     # the implementation plan (EN + IT)
├── configs/                  # one YAML per experiment (model + data + seed)
├── src/
│   ├── data/                 # loaders, splitting, scaling, windowing, manifest
│   ├── models/               # wrappers: naive, arima, deepar, timegrad, toy_ddpm
│   ├── eval/                 # metrics (CRPS, coverage, calibration), runners
│   ├── viz/                  # consistent plotting (forecasts, intervals, curves)
│   └── utils/                # seeds, logging, timing, config loading
├── experiments/              # entry scripts: run_E0.py ... run_E4.py
├── notebooks/                # exploration + the toy-DDPM teaching notebook
├── results/                  # CSV results registry (committed)
└── figures/                  # generated plots for the slides (committed)
```

## Getting started

The light local environment (data contract, classical baselines, evaluation, plots) is enough for everything built so far:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest tests/                 # offline data-contract invariants
python3 tests/test_data_contract.py      # plain-python fallback (no pytest needed)
python3 -m src.data.exchange             # smoke-test the Exchange ForecastDataset
```

Then:

1. **Read the plan.** Start with [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) Parts 1–3 (the story).
2. **Two environments are planned:** a light local one (data, classical baselines, evaluation, plots) and a Colab/GPU one (DeepAR, TimeGrad). The PyTorchTS ↔ GluonTS version combo is fragile and is pinned in `requirements.txt`.
3. **Golden rule:** this sandbox stays on the small **Exchange** dataset — it is the "get-everything-green-first" playground.

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

**Idea.** Inquadriamo il forecasting come l'apprendimento della distribuzione generativa condizionata `p(futuro | passato)` e la realizziamo con un diffusion model condizionato, addestrato con lo stesso obiettivo (ELBO / predizione del rumore) che il corso ricava nell'ultimo capitolo. Lo confrontiamo con una baseline ingenua, un modello classico (ARIMA/ETS) e un modello probabilistico di deep learning (DeepAR), misurando **tre** cose: **accuratezza puntuale** (MAE/RMSE/MASE), **qualità probabilistica** (CRPS, copertura, calibrazione) e **costo** (tempi di addestramento/inferenza e il compromesso qualità-vs-passi di denoising).

**Stato:** 🟠 sandbox personale, sto costruendo la scala dei modelli sul dataset **Exchange**. Già pronti: il **contratto dei dati** (`src/data/` — split temporale contiguo, scaling solo su train, windowing per-split) con la sua suite di test offline; una **EDA di Exchange** corredata di figure; il **modulo di valutazione** (metriche puntuali + probabilistiche — MAE/RMSE/MASE, CRPS, copertura degli intervalli); la **baseline M0 seasonal-naive**; e ora la **baseline M1 ARIMA** (ordine auto per canale, condizionamento per-finestra senza leakage, bande predittive gaussiane) — entrambe le righe sono in [`results/registry.csv`](results/registry.csv). Primo risultato onesto: sulla serie Exchange quasi-random-walk M1 *pareggia* M0 sull'accuratezza puntuale (MASE ≈ 4.52 — su una serie con radice unitaria il forecast lineare ottimo collassa sulla persistenza) ma offre un'incertezza **meglio calibrata** (copertura al 90% ≈ 0.93 contro 0.86). Il codice di **M2 DeepAR** e un notebook Colab pronto all'uso ([`notebooks/colab_m2_deepar.ipynb`](notebooks/colab_m2_deepar.ipynb)) sono ora pronti — DeepAR è il primo gradino che esce dall'ambiente leggero, quindi la sua riga del registry si genera su Colab/GPU. Subito dopo: **M3 (TimeGrad)**, il cuore a diffusione condizionata. Il piano completo è in [`docs/IMPLEMENTATION_PLAN_IT.md`](docs/IMPLEMENTATION_PLAN_IT.md) — include un'appendice che spiega a fondo la fase sperimentale (prima la metodologia, poi le istruzioni operative passo-passo).

**Per i colleghi:** leggete prima il piano (Parti 1–3), poi la parte del vostro ruolo (A / B / C, vedi tabella sopra). La regola d'oro: far girare tutta la pipeline sul piccolo dataset **Exchange** prima di salire di scala.
