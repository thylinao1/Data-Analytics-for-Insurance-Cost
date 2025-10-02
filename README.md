Medical Insurance Charges — EDA & Regression

Predicting individual insurance charges from demographic and lifestyle attributes. This project performs data cleaning, EDA, and linear models (single/multiple) and then Ridge regression to improve generalization.

Dataset: filtered/modified version of the Medical Insurance Price Prediction dataset (Kaggle, CC0 1.0).
Timebox: ~5 minutes (template-guided lab).

🎯 Objectives
1. Load the insurance data into a pandas DataFrame
2. Clean missing/blank entries and ensure correct dtypes
3. Run EDA to identify attributes most associated with Charges
4. Build single-variable and multivariable Linear Regression models
5. Apply Ridge regression (tune alpha) to refine performance

🧰 Tech Stack
1. Python: pandas, numpy
2. Viz: seaborn, matplotlib
3. ML: scikit-learn (train_test_split, LinearRegression, Ridge, PolynomialFeatures optional, cross_val_score, GridSearchCV)
4. Environment: JupyterLab / Notebook

Inside the notebook you will:
- Load & clean data (convert to numeric, handle blanks/NaN)
- EDA: describe stats, correlations, visualizations (e.g., Charges vs. Smoker, BMI, Age)

Modeling:
- Simple LR (e.g., Charges ~ BMI)
- Multiple LR (e.g., Charges ~ Age + BMI + Smoker + …)
- Ridge regression with alpha tuning (via GridSearchCV or manual sweep)
- Evaluate using R² (and optionally RMSE) on a train/test split or CV

🧪 Notes on Modeling
- Encode integers as provided (already mapped) or one-hot encode if you extend the work
- Use train/test split and cross-validation for stable estimates
- Ridge alpha:
      Lower alpha → closer to OLS (higher training fit, risk of overfit)
      Higher alpha → more shrinkage (better generalization if overfitting)
