# Basketball Prediction Agent

A supervised machine learning agent that predicts the winner of NBA games before they are played, achieving ~69.4% accuracy & an AUC-ROC of 0.733 on a held out backtest of 1230 games.

## Overview

This project builds an NBA game outcome prediction system by :
- Scraping live game logs and injury reports from official sources
- Engineering a rich six-category feature set per team
- Training a calibrated Logistic Regression classifier on five simulated NBA seasons

## Features
 
The model constructs a **15-dimensional matchup feature vector** from the following per-team statistics:
 
| Feature | Description |
|---|---|
| Avg. points scored (last 10 games) | Short-term offensive form |
| Avg. points allowed (last 10 games) | Short-term defensive form |
| Win % (last 10 games) | Recent momentum |
| Avg. days of rest | Scheduling fatigue / back-to-back impact |
| Home / Away win % | Performance splits by venue |
| ELO rating | Continuously updated team strength rating |
| Active injury count | Proxy for current roster availability |
 
Home-minus-away differentials are computed for each feature, then combined with the ELO derived win probability and raw injury counts for each side.
 
---

## Results

On a backtest of **1,230 games**:

| Metric | Score | Notes |
|---|---|---|
| Accuracy | **69.4%** | Random baseline = 50% |
| Log Loss | **0.613** | Random baseline = 0.693 |
| AUC-ROC | **0.733** | Random baseline = 0.500 |
| Brier Score | **0.201** | Lower is better |
| Calibration Error | **0.073** | 0 = perfectly calibrated |


## Setup & Installation
 
### Requirements
 
- Python 3.12
- No GPU required — all computation runs on CPU in under 30 seconds
### Install Dependencies
 
```bash
pip install -r requirements.txt
```
 
### Environment Configuration
 
Create a `.env` file in the project root if any API keys or configuration values are needed:
 
```
# .env
# Add any required environment variables here
```
 
### Run
 
```bash
python src/basketball_agent.py
```
 
---







