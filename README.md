# AI & ML Task 2: Feature Engineering, Model Optimization & Performance Comparison

This project compares multiple regression algorithms on the **California Housing Dataset** to predict house prices. It focuses on feature engineering, model optimization, performance evaluation, and selecting the best-performing model using an objective selection rule.

This project builds on **Task 1: House Price Predictor** by exploring optimized regression models and comparing their generalization performance.

---

## Deliverables Checklist

| Deliverable | File | Status |
|---|---|---|
| Jupyter Notebook | `AI_ML_Task2_Model_Comparison.ipynb` | Completed |
| Model Performance Comparison | `model_comparison_table.csv` | Completed |
| PDF Project Report | `AI_ML_Task2_Report.pdf` | Completed |
| Best Trained Model | `models/best_model.pkl` | Optional |
| Visualization Charts | `figures/` | Additional |

---

## Technologies Used

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## Project Structure

```text
Task-2-Model-Comparison/
│
├── AI_ML_Task2_Model_Comparison.ipynb
├── AI_ML_Task2_Report.pdf
├── model_comparison_table.csv
├── results.json
├── requirements.txt
├── housing_utils.py
├── generate_report_task2.py
│
├── models/
│   └── best_model.pkl
│
├── figures/
│   ├── actual_vs_predicted.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── feature_scales.png
│   ├── model_comparison.png
│   └── train_vs_test_r2.png
│
└── data/
    └── california_housing.csv
```

---

## Installation and Execution

### 1. Clone the Repository

```bash
git clone <repository-url>
```

Navigate to the Task 2 folder:

```bash
cd Internship-Machine-Learning-Tasks/Task-2-Model-Comparison
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Launch Jupyter Notebook

```bash
python -m notebook AI_ML_Task2_Model_Comparison.ipynb
```

Open the notebook and select:

**Kernel → Restart & Run All**

Run all cells and save the notebook with outputs.

**Note:** An internet connection is required during the first run to download the dataset. A local CSV copy is saved in the `data/` directory for subsequent runs.

---

## Project Workflow

The notebook follows these stages:

1. Import required libraries.
2. Load the California Housing dataset.
3. Perform exploratory data analysis.
4. Examine feature distributions and scaling requirements.
5. Separate features and target variables.
6. Split the dataset into training and testing sets.
7. Apply feature scaling and engineering.
8. Train and optimize regression models.
9. Evaluate and compare model performance.
10. Visualize predictions and model performance.
11. Select and save the best-performing model.
12. Export results and generate the PDF report.

---

## Models Compared

The project evaluates nine model configurations:

1. Linear Regression
2. Ridge Regression (Alpha = 1.0)
3. Decision Tree Regressor (Max Depth = 5)
4. Ridge Regression with tuned alpha
5. Decision Tree Regressor with GridSearchCV optimization
6. Linear Regression with engineered features
7. Ridge Regression with engineered features
8. Random Forest Regressor
9. Gradient Boosting Regressor

The models are evaluated to understand the impact of feature engineering, regularization, and hyperparameter optimization on predictive performance.

---

## Evaluation Metrics

The following metrics are used:

| Metric | Purpose |
|---|---|
| RMSE | Measures prediction error; primary evaluation metric |
| MAE | Measures average absolute prediction error |
| R² Score | Measures how well the model explains target variation |
| 5-Fold Cross-Validation RMSE | Evaluates model consistency across training-data folds |
| Overfitting Gap | Measures the difference between training and testing R² |

### Model Selection Strategy

The final model is selected using the **lowest 5-fold cross-validation RMSE on the training data**.

The test dataset is not used during model selection. This helps provide a more reliable estimate of how the selected model generalizes to unseen data.

The notebook and PDF report contain the actual evaluation results and explain the final selection.

---

## Improvements Over the Sample Implementation

### 1. Prevention of Data Leakage

The dataset is split before scaling.

`StandardScaler` is placed inside a Scikit-learn Pipeline, ensuring that scaling parameters are learned only from the training data.

This prevents information from the test dataset from influencing the training process.

### 2. Scikit-learn Compatibility

Instead of using:

```python
mean_squared_error(y_true, y_pred, squared=False)
```

The notebook calculates RMSE using:

```python
np.sqrt(mean_squared_error(y_true, y_pred))
```

This avoids compatibility issues with newer Scikit-learn versions.

---

## Results and Visualizations

The project generates the following outputs:

- Model performance comparison
- Training vs. testing R² comparison
- Actual vs. predicted values
- Feature importance analysis
- Feature scaling comparison
- Correlation heatmap

The detailed performance comparison is available in:

- `model_comparison_table.csv`
- `AI_ML_Task2_Report.pdf`

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Dataset download fails | Check your internet connection. |
| `ModuleNotFoundError: housing_utils` | Run Jupyter Notebook from the Task 2 project directory. |
| PDF report is not generated | Check whether the final notebook cells completed successfully. |
| Best model cannot be loaded | Check the Scikit-learn version and retrain the model if necessary. |
| Missing Python packages | Run `python -m pip install -r requirements.txt`. |

---

## Limitations

- The dataset is based on 1990 California census block-group information.
- House values are capped at $500,000 in the dataset.
- Historical data may not represent current housing market conditions.
- This project is intended for educational and machine learning practice, not professional property valuation.

---

## Author

**Manish Yadav**

GitHub: [manishyadav-aiml](https://github.com/manishyadav-aiml)

---

## Project Status

Completed — Internship Task 2.