# Pipeline Project — Customer Review Recommendation Classifier

A machine learning pipeline that predicts whether a customer recommends a product based on their review. The dataset contains a mix of **numerical**, **categorical**, and **text** features, each handled with appropriate preprocessing inside a single scikit-learn `Pipeline`.

---

## Project Structure

```
pipeline_project/
├── reviews.csv              # Raw dataset
├── pipeline_project.ipynb   # Main notebook
└── README.md
```

---

## Dataset

The dataset contains **18 442 rows** and **9 columns** with no missing values.

| Column | Type | Description |
|---|---|---|
| `Clothing ID` | Integer (categorical) | Unique product identifier |
| `Age` | Integer | Reviewer age (18–99) |
| `Title` | Text | Short review title |
| `Review Text` | Text | Full review body |
| `Positive Feedback Count` | Integer | Upvotes from other customers (0–122) |
| `Division Name` | Categorical | High-level product division |
| `Department Name` | Categorical | Product department |
| `Class Name` | Categorical | Product class |
| `Recommended IND` | Binary (target) | 1 = Recommended, 0 = Not Recommended |

**Class distribution:** ~82% recommended (1) vs ~18% not recommended (0) — significantly imbalanced.

---

## Preprocessing Pipeline

A `ColumnTransformer` applies different transformations per feature group, then merges them into a single feature matrix (~5 525 features total).

| Feature Group | Columns | Transformer | Rationale |
|---|---|---|---|
| Numerical | `Age`, `Positive Feedback Count` | `StandardScaler` | Normalise scale for SVM |
| Categorical (low-card.) | `Division Name`, `Department Name`, `Class Name` | `OneHotEncoder` | 2 / 6 / 14 unique values |
| Categorical (high-card.) | `Clothing ID` | `OrdinalEncoder` | 531 unique IDs |
| Text | `Title` | `TfidfVectorizer` (500 feat.) | Short, complementary signal |
| Text | `Review Text` | `TfidfVectorizer` (5 000 feat.) | Primary sentiment signal |

**TF-IDF settings:** `ngram_range=(1, 2)`, `sublinear_tf=True`, `min_df=2`

---

## Models

Two classifiers are compared. Both use `class_weight='balanced'` to handle the class imbalance.

| Model | Notes |
|---|---|
| **Random Forest** | Ensemble of decision trees; handles mixed features well; produces probability estimates |
| **Linear SVM** | Strong on high-dimensional sparse TF-IDF features; uses `decision_function` for ROC-AUC |

---

## Evaluation Metrics

Accuracy alone is misleading given the ~82/18 class split. Primary metrics:

- **F1 Macro** — unweighted mean F1 across both classes; penalises weak minority-class recall
- **ROC-AUC** — discriminative power across all decision thresholds, threshold-independent
- **Accuracy** — included for reference only
- **Classification Report** — per-class precision, recall, F1

---

## Fine-Tuning

`GridSearchCV` with **5-fold cross-validation** jointly searches over preprocessor and classifier hyperparameters in a single grid. This is data-leakage-free because all transformations are inside the `Pipeline`.

Grid parameters searched (per model):

**Random Forest**
- `preprocessor__review__max_features`: `[3000, 5000]`
- `preprocessor__review__ngram_range`: `[(1,1), (1,2)]`
- `classifier__n_estimators`: `[100, 200]`
- `classifier__max_depth`: `[None, 30]`
- `classifier__min_samples_split`: `[2, 5]`

**Linear SVM**
- `preprocessor__review__max_features`: `[3000, 5000]`
- `preprocessor__review__ngram_range`: `[(1,1), (1,2)]`
- `preprocessor__title__max_features`: `[300, 500]`
- `classifier__C`: `[0.1, 1.0, 10.0]`

---

## How to Run

**1. Install dependencies**
```bash
conda activate <your_env>
conda install scikit-learn pandas matplotlib -y
```

**2. Place the data file**
```
reviews.csv  →  same directory as the notebook
```

**3. Run the notebook**
```
Kernel → Restart & Run All
```

---

## Production Deployment

The full preprocessing is embedded in the pipeline — no separate transformation step is needed at inference.

```python
import joblib

# Save
joblib.dump(best_tuned, 'recommendation_pipeline.pkl')

# Load and predict
model = joblib.load('recommendation_pipeline.pkl')
predictions = model.predict(new_reviews_df)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `scikit-learn` | Pipeline, preprocessing, models, evaluation |
| `matplotlib` | Visualisations |
