# Bank Marketing ML Project

Machine learning project for customer response prediction, campaign ranking, and customer segmentation using the UCI Bank Marketing dataset.

## Project Overview

The objective of this project is to help a bank improve its telemarketing campaign by:

1. Predicting which customers are more likely to subscribe to a term deposit.
2. Ranking customers when the bank has a fixed capacity of contacting only 500 customers.
3. Discovering meaningful customer segments using unsupervised clustering.
4. Evaluating whether the discovered segments provide additional business insight.

The project is divided into two tasks:

- **Task 1 — Supervised Learning and Customer Ranking**
- **Task 2 — Customer Segmentation using Clustering**

---

# Task 1 — Supervised Learning

## Business Objective

The bank can contact only **500 customers** during a campaign.

Therefore, the main objective is not simply to maximize classification accuracy. The important business question is:

> **Which 500 customers should the bank contact first?**

Customers are therefore ranked according to their predicted probability of subscribing to a term deposit.

## Important Feature Availability Decision

The `duration` feature was excluded from the predictive modelling pipeline.

`duration` represents the duration of the current call and is only known after the call has taken place. Using it to decide whom to contact would introduce **target leakage**.

Only information available before the current call is used for the ranking decision.

## Modelling

The following model families were compared:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Weighted Logistic Regression
- Tuned Logistic Regression

The modelling workflow includes:

- Exploratory data analysis
- Data cleaning
- Categorical and numerical preprocessing
- Feature availability checks
- Class-imbalance analysis
- Time-aware train/holdout evaluation
- Model comparison
- Error analysis
- Customer ranking

## Evaluation

Because the campaign has a fixed capacity of 500 customers, the following business-oriented metrics were given particular importance:

- Precision
- Recall
- F1-score
- ROC-AUC
- Precision@500
- Lift@500
- Confusion matrix

Precision@500 measures the proportion of actual subscribers among the 500 highest-ranked customers.

Lift@500 compares this result with the expected yield from randomly selecting 500 customers.

## Final Model

**Random Forest** was selected as the final ranking model.

| Metric | Random Forest |
|---|---:|
| Precision@500 | 62.8% |
| Lift@500 | 2.04× |

The holdout base rate was approximately **30.83%**, meaning random selection would be expected to produce a subscriber concentration of about 30.83%.

The Random Forest top-500 list achieved **62.8%**, more than doubling the expected concentration compared with random selection.

### Why Random Forest instead of QDA?

QDA achieved a slightly higher raw Precision@500 of **65.8%** and Lift@500 of **2.13×**.

However, QDA assigned a probability of exactly `1.0` to **1,341 holdout customers**, including the customer at the 500th position. This creates a very large tie at the campaign cutoff.

Random Forest had only one customer tied at the top-500 cutoff, resulting in a more defensible ranking when the bank has an exact capacity of 500 contacts.

Therefore, Random Forest was selected as the final model.

---

# Task 2 — Customer Segmentation

## Objective

The second task uses unsupervised learning to discover groups of similar customer-contact records.

The purpose is to determine whether customer segmentation can provide additional business insight beyond the supervised ranking model.

## Clustering Unit

The clustering unit is the **customer-contact record**.

A dependable unique customer identifier is not available in the dataset, so the analysis treats each contact record as the unit of analysis.

## Features Used

Only information available before the current call was considered.

The clustering pipeline includes:

- Numerical feature scaling
- Categorical feature encoding
- Pre-call feature selection
- `ColumnTransformer`
- Standardization of numerical variables
- One-hot encoding of categorical variables

The following were excluded from clustering:

- Target variable `y`
- Current-call outcome information
- Supervised model predictions
- `duration`

The target variable and supervised predictions are therefore not used to create the clusters.

## Clustering Methods

Two clustering approaches were compared:

- K-Means
- Agglomerative Clustering

Multiple cluster counts were evaluated using:

- Silhouette score
- Cluster sizes
- Stability using Adjusted Rand Index (ARI)
- Cluster profiles
- Diagnostic analysis

PCA was used only for two-dimensional visualization and not as the space in which the final clustering was performed.

## Final Clustering Solution

The selected solution was **K-Means with 2 clusters**.

It achieved a silhouette score of approximately:

**0.879**

The two clusters showed different behaviour on the held-out period.

| Cluster | Holdout Subscription Rate |
|---|---:|
| Cluster 0 | 25.14% |
| Cluster 1 | 39.13% |

Cluster 1 also had a higher mean Random Forest score:

| Cluster | Mean Random Forest Score |
|---|---:|
| Cluster 0 | 0.1437 |
| Cluster 1 | 0.2046 |

This suggests that the segmentation captures meaningful differences between the groups.

### Stability Limitation

The two-cluster solution has a stability limitation.

Its mean ARI was approximately **0.616**, with a minimum ARI of approximately **0.039** across repeated runs.

Therefore, the clustering results should be treated as a **supporting segmentation layer**, rather than as a replacement for the supervised ranking model.

The supervised model remains the primary decision mechanism because it directly addresses the fixed 500-customer campaign objective.

---

# Repository Structure

```text
bank-marketing-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_eda_bank_marketing.ipynb
│   ├── 02_supervised_learning_bank_marketing.ipynb
│   └── 03_clustering.ipynb
│
├── results/
│   ├── README.md
│   ├── model_comparison.csv
│   ├── top500_holdout.csv
│   ├── clustering_comparison.csv
│   ├── clustering_stability.csv
│   ├── cluster_sizes.csv
│   ├── cluster_numeric_profile.csv
│   ├── cluster_categorical_profile.csv
│   ├── cluster_holdout_subscription_rates.csv
│   ├── cluster_holdout_rf_score_distribution.csv
│   ├── cluster_holdout_combined_profile.csv
│   ├── near_duplicate_diagnostic.csv
│   ├── cluster_outlier_diagnostic.csv
│   └── pca_customer_clusters.png
│
└── reports/
    ├── README.md
    └── decision_memo.md
