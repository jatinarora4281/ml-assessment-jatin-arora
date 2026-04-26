# Part B: Business Case Analysis
## Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — ML Problem Formulation (3 marks)

**Target Variable:** `items_sold` — the number of items sold in a given store during a given month under a specific promotion.

**Candidate Input Features:**

| Category | Features |
|---|---|
| Store attributes | `store_id`, `store_size`, `location_type`, `monthly_footfall`, `competition_density` |
| Promotion | `promotion_type` (Flat Discount, BOGO, Free Gift, Category-Specific, Loyalty Points Bonus) |
| Calendar | `month`, `year`, `is_weekend`, `is_festival`, `is_month_end` |
| Historical | `avg_items_sold_last_3_months`, `promotion_response_history` per store |
| Demographics | `customer_age_index`, `urban_rural_flag` |

**Type of ML Problem:** This is a **supervised regression** problem. The model learns a mapping from input features (store context, promotion type, calendar) to a continuous output (`items_sold`). Once trained, the model can rank the five promotion types for each store-month combination by predicted items sold, and the highest-scoring promotion is deployed.

**Justification:** The target variable is continuous and numeric, not a class label. We are predicting *how many* items a promotion will drive, not merely *whether* a promotion works. Regression gives us a quantitative ranking across promotion options, enabling nuanced trade-offs (e.g., the second-best promotion that is cheaper to run).

---

### B1(b) — Why Items Sold Over Revenue (3 marks)

**Items sold is a more reliable target than revenue for this problem for several reasons:**

1. **Price variability:** Revenue is confounded by price changes. A Flat Discount reduces the per-item price, so high revenue could mean few items at high price or many items at low price. Items sold isolates the volume effect of a promotion.

2. **Promotion type bias:** Different promotions structurally affect revenue differently. A BOGO promotion may generate the same revenue as no promotion but doubles units moved. Using revenue as a target would penalise BOGO unfairly, causing the model to underweight it even when it drives the highest footfall and brand engagement.

3. **Inventory and supply chain alignment:** The marketing team's goal is to clear inventory and drive footfall — both measured in units, not revenue. Items sold directly measures what they actually want to maximise.

**Broader Principle — Target Variable Selection:**
This illustrates the principle that the target variable must faithfully represent the *business objective*, not a *proxy* for it. Revenue is a proxy that introduces confounders (pricing, discounts) which the model cannot distinguish from true promotional effectiveness. Choosing the wrong target can produce a model that is statistically valid but operationally misleading — a common failure mode in real-world ML deployments.

---

### B1(c) — Modelling Strategy Beyond a Single Global Model (2 marks)

A single global model treats all 50 stores identically, ignoring the fact that a rural store and an urban flagship have fundamentally different customer bases, competition levels, and promotion sensitivities.

**Proposed Alternative: Hierarchical / Cluster-then-Model Strategy**

1. **Cluster stores** into groups (e.g., 3–5 clusters) based on shared characteristics: `location_type`, `store_size`, `competition_density`, and historical promotion response patterns.

2. **Train a separate model per cluster.** Each cluster model learns promotion–sales relationships relevant to that type of store. A model trained on urban high-footfall stores will learn that BOGO drives more volume, while rural models may learn that Loyalty Points work better for repeat customers.

3. **Alternatively, use store-level fixed effects** in a single model by including `store_id` embeddings or one-hot encoded store identifiers as features. This allows the model to learn store-specific intercepts while still sharing information across similar stores.

**Justification:** This respects the heterogeneity of store contexts without requiring a fully separate model for each of 50 stores (which would have insufficient data per store). Clustering provides a middle ground — shared learning within groups, separate learning across groups.

---

## B2. Data and EDA Strategy

### B2(a) — Joining Tables and Data Grain (4 marks)

**Table Descriptions:**
- `transactions`: one row per transaction, with `store_id`, `transaction_date`, `items_sold`, `promotion_id`
- `store_attributes`: one row per store, with `store_id`, `store_size`, `location_type`, `footfall`, `competition_density`
- `promotion_details`: one row per promotion, with `promotion_id`, `promotion_type`, `discount_rate`
- `calendar`: one row per date, with `date`, `is_weekend`, `is_festival`, `month`, `year`

**Join Strategy:**

```
transactions
  LEFT JOIN store_attributes ON transactions.store_id = store_attributes.store_id
  LEFT JOIN promotion_details ON transactions.promotion_id = promotion_details.promotion_id
  LEFT JOIN calendar ON transactions.transaction_date = calendar.date
```

All joins are LEFT joins to preserve all transaction records. LEFT JOIN to `store_attributes` ensures stores with missing metadata are not silently dropped; missing values can be imputed or flagged.

**Grain of the Final Modelling Dataset:**

**One row = one store × one month × one promotion type**

Before modelling, aggregate daily transaction records to monthly store-level summaries:

- `items_sold` → **SUM** (total items sold in the month for that store and promotion)
- `footfall` → average or sum over the month
- `is_festival` → binary flag for whether any festival occurred in the month
- `is_weekend` days in the month → count of weekend days

This grain aligns with the business decision cadence — the marketing team decides promotions once per month per store.

---

### B2(b) — EDA Strategy (4 marks)

**Analysis 1: Promotion Type vs. Average Items Sold (Bar Chart)**
- *What to look for:* Which promotion type drives the highest mean items sold overall? Are differences statistically significant?
- *Modelling influence:* If one promotion dominates across all stores, the model may simply recommend it everywhere. We should check whether this holds across location types before concluding it is universally optimal.

**Analysis 2: Interaction Heatmap — Promotion Type × Location Type (Heatmap)**
- *What to look for:* Does BOGO outperform in urban stores but underperform in rural ones? A heatmap of mean items sold by promotion × location type will reveal cross-term interactions.
- *Modelling influence:* If interactions are strong, we should engineer explicit interaction features (e.g., `promotion_type × location_type`) or use tree-based models that discover interactions automatically.

**Analysis 3: Seasonality Analysis — Items Sold by Month (Line Plot)**
- *What to look for:* Are there clear seasonal peaks (e.g., December, festive months)? Does seasonality differ by store size?
- *Modelling influence:* If seasonality is strong, `month` must be treated as a categorical or cyclically-encoded feature, not a linear integer. We may also add `sin(2π × month/12)` and `cos(2π × month/12)` as cyclic features.

**Analysis 4: Distribution of Items Sold (Histogram + Boxplot by Promotion)**
- *What to look for:* Is the target right-skewed? Are there outliers (e.g., festival days driving abnormally high sales)?
- *Modelling influence:* Skewed targets may benefit from log transformation before modelling with Linear Regression. Outliers may need to be capped or flagged as a separate binary feature.

---

### B2(c) — Handling Promotion Imbalance (2 marks)

**Impact of Imbalance:**
If 80% of records have no promotion, the model will be trained predominantly on the no-promotion baseline. It will learn the no-promotion→items_sold relationship well but will have sparse signal for each of the five promotion types. This can lead to **biased predictions**: the model may underestimate the lift from promotions because promotion records are rare in training.

**Steps to Address:**

1. **Stratified sampling during train-test split** by `promotion_type` to ensure each promotion type is represented in both train and test sets.

2. **Oversample promotion records** using techniques like SMOTE (for regression: SMOTE-R) or simple random oversampling to balance the representation of promoted vs. non-promoted transactions.

3. **Model the promotion lift separately**: Train a *baseline model* on no-promotion data to predict baseline sales, then train a separate *lift model* that predicts the incremental gain from each promotion. This decomposes the problem and avoids the imbalance issue entirely.

4. **Weight the loss function**: Assign higher loss weights to promotion observations during training so the model prioritises learning promotion effects despite their lower frequency.

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split and Evaluation Metrics (4 marks)

**Train-Test Split Strategy:**

With 3 years of monthly data across 50 stores, we have approximately 1,800 store-month records. A **temporal split** is required:

- **Train:** Months 1–30 (first 2.5 years)
- **Validation:** Months 31–33 (next 3 months for hyperparameter tuning)
- **Test:** Months 34–36 (the final 6 months)

For more robust evaluation, use **time-series cross-validation (walk-forward validation)**: train on months 1–12, test on 13–15; then train on 1–15, test on 16–18; and so on. This gives multiple test windows and reduces variance in the evaluation estimate.

**Why Random Splits Are Inappropriate:**
A random split leaks future information into the training set. If the model trains on data from month 36 (December, peak season) and is tested on month 6 (off-season), it will have learned holiday patterns that are unavailable at prediction time, inflating test performance. In production, predictions are always made about the future — the evaluation methodology must reflect this.

**Evaluation Metrics:**

| Metric | Formula | Business Interpretation |
|---|---|---|
| **RMSE** | √(Σ(ŷ−y)²/n) | Penalises large errors heavily. A high RMSE means some stores receive badly wrong recommendations — costly for large stores with high revenue stakes. |
| **MAE** | Σ\|ŷ−y\|/n | Average absolute error in units of items sold. Directly interpretable: "on average, the model is off by X items per store-month." |
| **MAPE** | Σ\|ŷ−y\|/y × 100 | Percentage error — useful for comparing accuracy across stores of different sizes. |
| **Promotion Ranking Accuracy** | % of months where the best predicted promotion = actual best promotion | The most business-relevant metric: are we recommending the right promotion? |

---

### B3(b) — Feature Importance for Explaining Different Recommendations (4 marks)

**Scenario:** The model recommends Loyalty Points Bonus for Store 12 in December and Flat Discount for Store 12 in March.

**Investigation Steps:**

1. **Extract feature importances** from the trained Random Forest (global importance via Gini impurity or permutation importance). Identify which features most influence predictions overall.

2. **Generate SHAP values** (SHapley Additive exPlanations) for the two specific predictions — Store 12 in December and Store 12 in March. SHAP decomposes each prediction into per-feature contributions, showing *which features pushed the prediction up or down* for each promotion option.

3. **Compare SHAP waterfall plots** for the two months. We would expect to find:
   - In December, `is_festival=1`, `month=12`, and high `footfall` contribute positively to Loyalty Points Bonus (returning customers maximise points during gift-giving season).
   - In March, these seasonal features are absent; `competition_density` and `store_size` drive Flat Discount as the best volume driver in a quieter period.

**Communicating to the Marketing Team:**

Present a simple table or bar chart for each month showing the top 3 features that drove the recommendation, with plain-language labels:

> *"In December, the model recommends Loyalty Points because December is a high-footfall festival month — customers are motivated to accumulate points for future purchases. In March, with no festivals and lower footfall, a Flat Discount creates urgency and drives volume from price-sensitive shoppers."*

This makes the model's logic transparent, builds trust, and helps marketers validate or challenge the recommendation using domain knowledge.

---

### B3(c) — End-to-End Deployment and Monitoring (4 marks)

**1. Saving the Model:**

```python
import joblib
joblib.dump(pipeline, 'promotion_model_v1.pkl')
```

Save the full scikit-learn Pipeline (including preprocessor and model), the feature column list, and a model card (training date, data range, evaluation metrics) to a versioned model registry (e.g., MLflow, AWS S3 with versioning).

**2. Monthly Data Preparation and Inference:**

At the start of each month:
1. Pull the latest store attributes, calendar flags, and historical sales from the data warehouse.
2. Construct the store-month feature table using the same aggregations and feature engineering pipeline used during training.
3. For each store, create five rows — one per promotion type — with identical store/calendar features but varying `promotion_type`.
4. Run `pipeline.predict()` on all five rows per store.
5. Select the promotion with the highest predicted `items_sold` for each store.
6. Output recommendations to the marketing dashboard or CRM.

**3. Monitoring for Model Degradation:**

Deploy the following monitoring checks after each monthly prediction cycle:

| Check | Method | Trigger for Retraining |
|---|---|---|
| **Prediction drift** | Compare distribution of current predictions vs. historical predictions | KS-test p-value < 0.05 |
| **Feature drift** | Monitor input feature distributions (mean, std) for each feature | Mean shifts > 2σ from training baseline |
| **Actual vs. predicted error** | Once actuals are available (end of month), compute RMSE on that month | RMSE exceeds 1.5× training RMSE |
| **Promotion ranking accuracy** | Compare recommended promotions to actual best promotions | Drop > 10 percentage points from baseline |

**Retraining Trigger:** If any two of the above checks fire in the same month, schedule a retrain on the most recent 24 months of data. Use the same temporal cross-validation procedure to confirm the new model outperforms the current one before promotion to production. Maintain a rollback-ready previous model version at all times.
