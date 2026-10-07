# Bank Marketing ML — Decision Memo

## Objective

The bank can contact only 500 customers in a campaign. The main objective is therefore to rank customers by their probability of subscribing to a term deposit using information available before the current call.

The `duration` feature was excluded because it is only known after the call and would create target leakage.

## Task 1 — Supervised Ranking

The dataset contains 41,188 records. A time-aware train/holdout approach was used so that later records were kept separate from model development.

Several model families were compared, including Logistic Regression, Random Forest, Gradient Boosting, LDA and QDA, together with weighted and tuned logistic regression.

Because the business decision is a fixed 500-customer contact list, Precision@500 and Lift@500 were given particular importance.

Random Forest achieved:

- Precision@500: 62.8%
- Lift@500: 2.04×

The holdout base rate was approximately 30.83%, so the Random Forest top-500 list more than doubled the expected subscriber concentration compared with random selection.

QDA achieved a slightly higher raw Precision@500 of 65.8% and Lift@500 of 2.13×. However, QDA assigned a probability of exactly 1.0 to 1,341 holdout customers, including the 500th-ranked customer. This creates a large tie at the campaign cutoff and makes the exact composition of the top-500 list less robust.

Random Forest had only one customer tied at the top-500 cutoff. Therefore, Random Forest was selected as the final model because its ranking is more defensible for a campaign with a fixed capacity of 500 customers.

## Task 2 — Customer Segmentation

Clustering was performed without using the target variable `y` or supervised model predictions.

The clustering unit was the customer-contact record, and only pre-call information was used. Numerical features were scaled and categorical features were one-hot encoded.

K-Means and Agglomerative Clustering were compared across multiple cluster counts. K-Means with two clusters produced the highest silhouette score of approximately 0.879.

However, the two-cluster K-Means solution showed weaker stability than the higher-k alternatives, with a mean ARI of approximately 0.616 and a minimum ARI of approximately 0.039 across repeated runs. This stability limitation should be acknowledged when interpreting the segmentation.

Despite this limitation, the fixed two-cluster solution showed meaningful differences on the held-out period:

- Cluster 0: 25.14% subscription rate
- Cluster 1: 39.13% subscription rate

Cluster 1 also had a higher mean Random Forest score than Cluster 0 (0.2046 vs 0.1437).

This suggests that the segmentation captures some meaningful differences in customer/contact opportunity, although it should be treated as a secondary segmentation layer rather than a replacement for the supervised ranking model.

## Business Recommendation

Use the Random Forest model to produce the primary ranking of the 500 customers to contact.

Use the clustering results as a supporting segmentation tool for understanding differences between customer groups and potentially tailoring campaign strategy.

The supervised ranking should remain the main decision mechanism because it directly optimizes the fixed 500-contact business objective. The clustering analysis provides additional structure but has a stability limitation and should not replace the ranking model.

## Limitations

- The dataset represents historical campaign records, so performance may change on future campaigns.
- The clustering solution has a stability limitation for K=2.
- Model performance is evaluated on a temporal holdout rather than guaranteeing future campaign performance.
- The analysis does not establish causal relationships between customer characteristics and subscription.
