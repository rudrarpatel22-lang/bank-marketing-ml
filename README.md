# Bank Marketing ML Project

## Overview

This project analyzes the UCI Bank Marketing dataset using supervised learning and unsupervised clustering.

The project consists of two main tasks:

- **Task 1:** Supervised learning and customer ranking
- **Task 2:** Customer segmentation using clustering

## Task 1 — Supervised Learning

The objective is to predict the probability that a customer will subscribe to a term deposit and rank customers so that the bank can prioritize its limited contact capacity.

The analysis includes:

- Exploratory Data Analysis
- Data preprocessing
- Logistic Regression
- Random Forest
- Gradient Boosting
- LDA
- QDA
- Class-imbalance analysis
- Time-aware train/holdout evaluation
- Precision@500
- Lift@500
- Error analysis

The `duration` feature is excluded because it is only known after the current call and would introduce leakage.

## Task 2 — Clustering

The objective is to identify groups of similar pre-call customer-contact records.

The analysis includes:

- Pre-call feature selection
- Numerical scaling
- One-hot encoding
- K-Means clustering
- Agglomerative Clustering
- Silhouette analysis
- Clustering stability using ARI
- Cluster profiling
- PCA visualization
- Holdout cluster evaluation

The target variable `y` and supervised model predictions are not used to create the clusters.

## Repository Structure

```text
bank-marketing-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_eda_bank_marketing.ipynb
│   ├── 02_supervised_learning_bank_marketing.ipynb
│   └── 03_clustering.ipynb
├── results/
└── reports/


## Dataset

The project uses the UCI Bank Marketing dataset.

The raw dataset is not included in this repository.

See `data/README.md` for instructions on obtaining the dataset.

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt

