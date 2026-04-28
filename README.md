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
 
Home-minus-away differentials are computed for each feature, then combined with the ELO-derived win probability and raw injury counts for each side.
 
---
