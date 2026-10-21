# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## State of the repo

No code yet. The directory holds only `Final project description - edited 2026.pdf`, the final project brief (DAM501 Data Mining). Deadline and group rules are unconfirmed.

No `pdftotext`/`mutool` on this machine; extract PDF text with `python3 -I -c "import pypdf; ..."`.

## Project

Group project (6 members) for DAM501. Chosen topic: **user behavior discovery and churn patterns for a subscription music service**, using the Kaggle competition dataset *WSDM – KKBox's Churn Prediction Challenge* (`kkbox-churn-prediction-challenge`). Chosen over Online Retail / Telco churn because another team already has an online-retail topic.

The brief grades discovery, not prediction: at least 2 mining techniques, at least 1 unsupervised or pattern-based (clustering, association rules, sequential patterns, anomaly detection). Classification (churn) is a supporting method only. Evaluate with silhouette/cohesion (clusters), support/confidence/lift (rules), and business actionability — not just accuracy/AUC.

Report structure (7 sections, weights): Business Understanding 15, Dataset 5, Pre-processing 10, EDA 15, Mining Methods 30, Evaluation 15, Business Insights 10; +10 bonus for advanced techniques (feature engineering, PCA/UMAP, subset comparison). Deliverables: report (PDF/Word), reproducible code with data prep / mining / analysis clearly separated, short presentation.

## Data constraints

- Machine: 16 GB RAM, **only ~8 GB free disk** (checked 2026-10-21). Do not download `user_logs.csv.7z` (7.1 GB compressed, ~28 GB extracted).
- Plan: download `train_v2`, `members_v3`, `transactions_v2` (or `transactions`), `user_logs_v2` only; extract with 7-zip (`brew install p7zip`); keep raw data in `data/` (untracked).
- Kaggle CLI is at `/opt/anaconda3/bin/kaggle` with a token in `~/.kaggle/access_token`; downloading requires having joined the competition on kaggle.com. List files: `kaggle competitions files kkbox-churn-prediction-challenge`.
- Process large CSVs with DuckDB or Polars (Polars installed; DuckDB not yet) aggregating per `msno`; sample ~100–200k users (stratified by churn) for mining. Document the sampling in the report's limitations.
- Known dirty columns: `bd` (age: negative/0/>100 values), `gender` (many missing).

## Planned layout

Separate `data_prep/`, `mining/`, `analysis/` (brief requires this). `data_prep/` produces one shared per-user feature table; all mining notebooks read from it so numbers stay consistent.
