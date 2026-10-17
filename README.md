# GreenDC Intelligence

**AI-driven sustainable data-centre operations and energy optimisation**

B.Tech CSE (AI & ML) capstone project, IILM University, Greater Noida (2026-27)

Team: Md Farhan Ansari, Aman Kumar Prasad, Syed Maaz | Guide: Dr. Pranav Shrivastava

GreenDC Intelligence analyses 10 years of real metered data from a high-performance computing data centre to **forecast energy use, detect anomalies, track efficiency (PUE), and recommend actions**, all in one operator dashboard. It supports human decisions; it does not control equipment.

---

## What it does

| Module | Question it answers | Method |
|---|---|---|
| **Forecasting** | How much power will we need in the next 1 h / 24 h? | Linear Regression, Random Forest, XGBoost, LightGBM vs persistence baselines |
| **Anomaly detection** | Is something wrong right now? | Model residuals, robust z-scores, Isolation Forest, meter-fault rules |
| **PUE tracking** | Are we getting more or less efficient? | Energy-weighted PUE, component breakdown, avoidable-overhead metric |
| **Recommendations** | What should the operator do? | YAML-configured rule engine with evidence and estimated savings |
| **Dashboard** | One place to see it all | Next.js + Recharts over a FastAPI backend |

**Novelty:** the techniques are established; the contribution is integrating them. The expected-overhead model's residuals drive anomaly detection, anomaly events drive recommendations, and every result reaches one operator interface.

---

## Key results

**Forecasting** (held-out 2025 test period; skill = MAE reduction vs persistence)

| Horizon | Best model | MAE | MAPE | Skill vs persistence |
|---|---|---|---|---|
| Day-ahead (24 h) | LightGBM / XGBoost | 110-112 kW | 4.9 % | **+7.6 to +8.5 %** |
| Hour-ahead (1 h) | XGBoost | 24.5 kW | 0.93 % | **+4.1 %** |

Predicting the *ratio change vs persistence* instead of raw kW made models robust to IT regime shifts (1.2 to 3.7 to 2.6 MW). It turned XGBoost from -162 % to +8 % skill.

**Anomaly detection**: synthetic faults injected into 2025 data, 30 trials per type

| Fault | Recall | 95 % CI |
|---|---|---|
| Meter dropout | 100 % | 88.6 - 100 % |
| IT outage | 100 % | 88.6 - 100 % |
| Cooling-overhead excess | 96.7 % | 83.3 - 99.4 % |
| Stuck meter | 100 % | 88.6 - 100 % |

False alarms: 0.06 % of test hours.

**Findings on the real data**
- From 2020 to 2024, IT energy grew **2.6x** but overhead energy grew **5.9x**; PUE rose from about 1.03 to 1.07.
- The rise traces to an **April 2024 HVAC step-change** (about 10 kW to 190 kW), which the system flags automatically.
- Low PUE in 2020-21 was partly a **metering artefact**: excluding detected meter-fault hours corrects 2021 PUE from 1.0265 to 1.0317.

---

## Architecture

```
data/raw (parquet, 1-min)  ->  src/greendc  (cleaning, models, anomalies, PUE, recommendations)
                                    |
                     docs/results + data/processed (pipeline outputs)
                                    |
                            api/main.py  (FastAPI, JSON)
                                    |
                      frontend/  (Next.js dashboard, Recharts)
```

---

## Quick start (Windows)

**Requirements:** Python 3.10+, Node.js 20+, Git.

```cmd
git clone https://github.com/farhanxansari/greendc-intelligence.git
cd greendc-intelligence
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
cd frontend && npm install && cd ..
```

**Data:** download the two Parquet files (Power Metrics and Outside Weather Station) from the [NLR data catalog](https://data.nlr.gov/submissions/300) and save them as `data\raw\esif\power.parquet` and `data\raw\esif\weather.parquet`.

**Run the analytics pipeline once:**

```cmd
python -m greendc.pipeline --with-eval
```

**Start the API and dashboard:**

```cmd
dev.bat
```

- Dashboard: http://localhost:3000
- API docs: http://localhost:8000/docs
- Tests: `pytest -q`

---

## Project structure

```
src/greendc/
  config.py                       shared paths, date splits, horizon, seed
  data/esif.py                    download, glitch removal, hourly resampling, PUE
  features/forecast_features.py   leakage-free features, ratio target
  models/forecast.py              baselines + LR, RF, XGBoost, LightGBM
  models/overhead.py              expected-overhead model (anomaly baseline)
  anomaly/detect.py               4 detectors -> events with confidence + category
  anomaly/evaluate.py             synthetic-fault injection, recall with 95% CI
  pue/metrics.py                  energy-weighted PUE, components, avoidable overhead
  recommend/engine.py             rule engine with historical replay (--as-of)
  pipeline.py                     runs every step in dependency order
configs/recommendation_rules.yaml recommendation thresholds and wording
api/main.py                       FastAPI endpoints
frontend/                         Next.js dashboard (5 pages)
notebooks/01_eda.py               exploratory analysis and report figures
tests/                            pytest suite
```

---

## Data

NLR (formerly NREL) ESIF HPC data centre PUE dataset: Clark, S. and Strelka, J. (2025), *NLR HPC Facility Power Usage Effectiveness (PUE) Data*, NLR Data Catalog, DOI: [10.7799/3015212](https://doi.org/10.7799/3015212).
Minute-level IT, cooling, HVAC, pump and lighting power plus outdoor weather, November 2015 to August 2025 (about 4.5 million rows).

## Method notes

- Time-based split only: train Jan 2019 - Jun 2024, validate Jul - Dec 2024, test Jan - Aug 2025.
- Every forecasting feature uses information at least `horizon` hours old, so there is no future leakage.
- PUE over any period is energy-weighted (total facility energy / IT energy), as recommended by The Green Grid.
- Anomaly thresholds were validated on both injected-fault recall and the real-data alert rate across all years.

## Roadmap

- LSTM forecaster under the same evaluation protocol (completes Objective 2).
- What-if simulation of the facility under Indian city climates (Objective 7).

## Limitations

- Batch analysis on historical data; no live sensor streaming.
- One facility with a highly efficient warm-water cooling design (PUE about 1.03-1.07); results may differ for air-cooled sites.
- Recommendation savings for hot-weather actions rely on a stated recoverable-fraction assumption.
- Decision support only; the system does not control equipment.