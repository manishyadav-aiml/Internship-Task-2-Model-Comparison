# AI & ML Task 2: Feature Engineering, Model Optimization & Performance Comparison

Compares several regression models on the **California Housing** dataset, selects the best one with an objective rule and explains why. Builds on the Task 1 House Price Predictor.

## Deliverables checklist
| Required item | File | Created by |
|---|---|---|
| Jupyter notebook | `AI_ML_Task2_Model_Comparison.ipynb` | included |
| Model performance comparison table | shown in notebook section 7 and saved as `model_comparison_table.csv`; also in the PDF | running the notebook |
| 1-2 page PDF (methodology, results, conclusions) | `AI_ML_Task2_Report.pdf` | running the notebook (last section) |
| *Optional:* best model saved with joblib | `models/best_model.pkl` | running the notebook |
| *Optional:* extra charts and insights | `figures/` (comparison, train vs test R², actual vs predicted, permutation importance, correlation heatmap) | running the notebook |

## How to run (about 2 minutes)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook AI_ML_Task2_Model_Comparison.ipynb      # Kernel -> Restart & Run All
```
The first run needs internet once to download the dataset (a CSV copy is then saved in `data/`). Save the notebook **with its outputs** before submitting or pushing it to GitHub.

## What is in the notebook
1. Imports, 2. Load data, 3. Quick data check (why scaling matters), 4. Features and target, 5. Split then scale, 6. Train models, 7. Evaluation and comparison table, 8. Visual validation, 9. Model selection and joblib, 10. Export and PDF, 11. Conclusions.

**Models compared (9)**
* Stage A (required): Linear Regression, Ridge (alpha = 1.0), Decision Tree (max_depth = 5)
* Stage B (optimisation): Ridge with tuned alpha, Decision Tree tuned with GridSearchCV, Linear Regression and Ridge with engineered features, plus Random Forest and Gradient Boosting as stronger references

**Metrics:** RMSE (primary), MAE, R², 5-fold cross-validated RMSE, and the overfit gap (train R² minus test R²).
**Selection rule:** the model with the lowest cross-validated RMSE on the training data is selected, so the test set is never used to choose. The notebook and PDF print the justification from your own numbers.

## Two improvements over the sample code in the task sheet
1. **Split first, then scale.** The sample scales all the data before splitting, which leaks test-set statistics into training. Here `StandardScaler` sits inside a `Pipeline` and is fitted on the training data only.
2. **No `squared=False`.** This argument was removed from `mean_squared_error` in recent scikit-learn versions and raises an error. The notebook uses `np.sqrt(mean_squared_error(...))`, which works in every version.

## Files
```
AI_ML_Task2_Model_Comparison.ipynb   main notebook
housing_utils.py                     feature engineering helper (needed to load saved models)
generate_report_task2.py             builds the PDF from results.json and figures/
requirements.txt
```

## Troubleshooting
| Problem | Fix |
|---|---|
| Dataset download fails | connect to the internet once; afterwards `data/california_housing.csv` is used |
| `ModuleNotFoundError: housing_utils` | run Jupyter from this folder |
| PDF is not created | check that the last notebook cell ran without errors; or run `python generate_report_task2.py` after the notebook |
| Loading `best_model.pkl` fails after upgrading scikit-learn | re-run the notebook to retrain |

## Limitations
1990 census block-group averages and a price capped at $500,000. This is a learning project, not a valuation tool.
