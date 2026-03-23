# Customer Review Recommendation Classifier

This project builds a machine learning pipeline to predict whether a customer recommends a product based on their written review. The dataset comes from a women's clothing e-commerce platform and includes a mix of numerical, categorical, and free-text features.

The goal is to classify each review as **recommended (1)** or **not recommended (0)** using the review content and associated metadata.

---

## Repository Files

| File | Description |
|---|---|
| `pipeline_project.ipynb` | Main notebook — contains all data exploration, pipeline construction, model training, evaluation, and fine-tuning |
| `reviews.csv` | Dataset with 18 442 customer reviews and 8 features |
| `README.md` | This file |

---

## Dependencies

The project requires Python 3.11+ and the following packages:

```
pandas
scikit-learn
matplotlib
```

Install via conda:

```bash
conda install pandas scikit-learn matplotlib -y
```

---

## How to Run

1. Clone or download the repository
2. Place `reviews.csv` in the same directory as the notebook
3. Open `pipeline_project.ipynb` in Jupyter
4. Run **Kernel → Restart & Run All**

---

## Project Summary

### Data

The dataset contains 18 442 rows with no missing values. Features include the reviewer's age, the clothing item ID, free-text title and review body, a positive feedback count, and three categorical product labels (division, department, class). The target variable is binary and imbalanced — approximately 82% of reviews are positive recommendations.

### Approach

Because the data contains three different types of features, a `ColumnTransformer` is used inside a scikit-learn `Pipeline` to apply the right preprocessing to each column:

- Numerical columns are scaled with `StandardScaler`
- Low-cardinality categorical columns are encoded with `OneHotEncoder`
- The high-cardinality product ID column is encoded with `OrdinalEncoder`
- The review text and title are vectorised separately using `TfidfVectorizer` with bigrams

Everything is wrapped in a single `Pipeline` object so that preprocessing is applied consistently during both cross-validation and final evaluation, with no data leakage.

### Models

Two classifiers are trained and compared:

- **Random Forest** — handles mixed feature types well and produces probability estimates
- **Linear SVM** — strong performer on high-dimensional TF-IDF sparse matrices

Both models use `class_weight='balanced'` to account for the class imbalance. Evaluation focuses on **F1 Macro** and **ROC-AUC** rather than accuracy, since accuracy is misleading on imbalanced data.

### Fine-Tuning

The better-performing baseline model is further optimised using `GridSearchCV` with 5-fold cross-validation. The grid jointly searches over both TF-IDF vocabulary size and classifier hyperparameters, which is only possible because the full preprocessing is inside the pipeline.
