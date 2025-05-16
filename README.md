## AutoML : Processor and Model Trainer

**Disclaimer:** This pipeline is for rapid experimentation, not production. If you deploy this in the wild, pray your users are forgiving—or at least that your logging is verbose.

---

### 📦 1. Dependencies

- **Core:** Python 3.8+  
- **Data Processing:** pandas, numpy, scipy, imbalanced-learn (SMOTE)  
- **Preprocessing:** scikit-learn  
- **Modeling:** scikit-learn, xgboost, optuna  
- **Utilities:** matplotlib, json

```bash
pip install pandas numpy scipy imbalanced-learn scikit-learn xgboost optuna matplotlib
```

---

### 🔧 2. Installation & Setup

1. Copy `model_trainer.py` and `preprocessing_pipeline.py` (containing `DataProcessor` & `PreprocessingError`) into your project.
2. Ensure input files (`.csv`, `.xlsx`) are accessible.
3. Run quick tests:
   ```bash
   python model_trainer.py
   ```

---

### ⚙️ 3. DataProcessor Class (preprocessing_pipeline.py)

Handles everything from missing values to encoding, scaling, and balancing.

#### Constructor Parameters

| Param                        | Type    | Default          | Description                               |
|------------------------------|---------|------------------|-------------------------------------------|
| `file_path`                  | `str`   | **Required**     | Path to `.csv`, `.xlsx`, or `.xls`.       |
| `target_column`              | `str`   | **Required**     | Name of the target column.                |
| `save_file`                  | `bool`  | `False`          | Save processed data + metadata if `True`. |
| `scale_method`               | `str`   | `'standard'`     | `'standard'` or `'minmax'`.               |
| `numeric_strategy`           | `str`   | `'median'`       | For numeric imputation.                   |
| `categorical_strategy`       | `str`   | `'most_frequent'`| For categorical imputation.               |
| `max_categorical_cardinality`| `int`   | `50`             | Drop columns above this unique count.     |
| `safe_mode`                  | `bool`  | `True`           | If `True`, raises on destructive ops.     |

#### Key Methods

1. **`load_data()`**: Reads file, standardizes column names, checks for emptiness & target existence.
2. **`handle_missing_values()`**: Imputes nums & cats, warns/errors on leftovers.
3. **`remove_duplicates()`**: Drops duplicates; errors if >50% removed in `safe_mode`.
4. **`handle_outliers(threshold=3.5)`**: Removes rows with Z-score > threshold.
5. **`encode_categorical_features()`**: One-hot encodes low-cardinality cats; drops high-cardinality when needed.
6. **`scale_features()`**: Scales numeric features excluding target.
7. **`_is_classification_task()`**: Heuristic to decide task by target dtype & cardinality.
8. **`_handle_classification_specifics()`**: Label-encodes target, stores mapping, applies SMOTE if imbalance detected.
9. **`_handle_regression_specifics()`**: Converts target to numeric, errors on non-numerics.
10. **`validate_pipeline()`**: Final integrity checks before training.
11. **`fit_transform()`**: Runs full pipeline in order: load → missing → dedupe → outliers → encoding → scaling → task-specific → validation → optional save.
12. **`get_metadata()`**: Returns shapes, steps, warnings, encoding info, etc.
13. **`save_preprocessed_data()`**: Dumps processed CSV + JSON metadata with sklearn-friendly serialization.

---

### 🚀 4. ModelTrainer Class (model_trainer.py)

Wraps preprocessing and model experimentation into one slick class.

#### Constructor Parameters

| Param                        | Type     | Default          | Description                                                 |
|------------------------------|----------|------------------|-------------------------------------------------------------|
| `file_path`                  | `str`    | **Required**     | Input data.                                                 |
| `model_name`                 | `str`    | **Required**     | From supported lists (see below).                           |
| `target_column`              | `str`    | **Required**     | Label column.                                               |
| `task`                       | `str`    | `'classification'`| `'classification'` or `'regression'`.                       |
| `preprocessed`               | `bool`   | `False`          | Skip preprocessing if `True`.                               |
| `train_to_test_ratio`        | `float`  | `0.33`           | Test set fraction (0.01–0.99).                              |
| `save_file`                  | `bool`   | `False`          | Pass-through to `DataProcessor`.                            |
| `scale_method`               | `str`    | `'standard'`     | Scaling (mirrors DataProcessor).                            |
| `numeric_strategy`           | `str`    | `'median'`       | Numeric imputation (mirrors DataProcessor).                 |
| `categorical_strategy`       | `str`    | `'most_frequent'`| Categorical imputation (mirrors DataProcessor).             |
| `max_categorical_cardinality`| `int`    | `50`             | Cardinality threshold (mirrors DataProcessor).              |
| `custom_hyperparams`         | `dict`   | `{}`             | Override model defaults.                                    |
| `use_optuna`                 | `bool`   | `False`          | Enable Optuna tuning.                                       |
| `optuna_trials`              | `int`    | `50`             | Number of tuning trials.                                    |
| `optuna_timeout`             | `int`    | `3600`           | Tuning time limit (s).                                      |
| `optuna_metrics`             | `list`   | `['default']`    | Metrics: `'accuracy'` or `'r2'` if default.                 |

#### Supported Models

- **Classification:** `RandomForest`, `XGBoost`, `LogisticRegression`, `SVM`, `DecisionTree`, `KNN`, `GradientBoosting`
- **Regression:** `LinearRegression`, `RandomForest`, `XGBoost`, `SVR`, `ElasticNet`, `GradientBoosting`, `KNN`

#### Key Methods

- **`_validate_parameters()`**: Ensures file exists, valid types/values, model supports task, hyperparams recognized.
- **`load_data()`**: Loads raw data (delegates to DataProcessor when needed).
- **`load_and_preprocess_data()`**: Instantiates `DataProcessor(...).fit_transform()`.
- **`train()`**:
  1. Validate parameters.
  2. Load & preprocess (unless `preprocessed=True`).
  3. Split (stratify for classification).
  4. Optional Optuna tuning.
  5. Train final model.
  6. Compute metrics (`accuracy`/`f1` or `r2`/`rmse`).
  7. Return `(model, metadata)`.
- **`save_model(path='saved_models')`**: Pickles model + metadata.
- **`plot_feature_importance(top_n=20)`**: Bar chart of importances (if available).

---

### 🎯 5. Usage Example

```python
from model_trainer import ModelTrainer

trainer = ModelTrainer(
    file_path='data/my_data.csv',
    model_name='RandomForest',
    target_column='label',
    task='classification',
    use_optuna=True,
    optuna_trials=20
)
model, info = trainer.train()
trainer.save_model(path='models')
print(info['metrics'])
```

---

### ✅ 6. Do's and Don'ts

- **Do** verify your input file, column names, and types.
- **Do** tweak `optuna_trials` for realistic runtimes—unless you love waiting.
- **Do** inspect `metadata` from `DataProcessor` and `ModelTrainer` for debugging.
- **Don't** expect production robustness—no monitoring, no retries, no SLA.
- **Don't** feed multi-million-row datasets without proper resource planning.

---

### 🌟 7. Use Cases

- Quick POCs in hackathons.
- Academic experiments & demos.
- Prototype feature pipelines before full engineering.


