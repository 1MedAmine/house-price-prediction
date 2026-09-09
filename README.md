# 🏠 House Price Prediction

> End-to-end machine learning pipeline to predict residential property prices — from raw data to Kaggle submission.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![R](https://img.shields.io/badge/R-4.x-276DC3?style=flat-square&logo=r)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Ames%20Housing-20BEFF?style=flat-square)
![Kaggle Score](https://img.shields.io/badge/Kaggle%20Score-13431-green?style=flat-square)

---

## 📌 Project Overview

This project builds a complete supervised learning pipeline to predict house sale prices from the [Ames Housing Dataset](https://www.kaggle.com/competitions/home-data-for-ml-course) (Kaggle). It covers the full data science workflow: exploratory analysis, advanced feature engineering, hyperparameter tuning, stacking ensemble, and optimized blend predictions.

**Key goals:**
- Perform thorough exploratory data analysis (EDA)
- Build a rich feature engineering pipeline (326 features)
- Train and compare 9 regression models with hyperparameter tuning
- Build a stacking ensemble with optimized blend weights
- Submit predictions to Kaggle

---

## 📁 Repository Structure

```
house-price-prediction/
│
├── data/
│   ├── train.csv                          # Ames Housing training set (Kaggle)
│   └── test.csv                           # Test set for predictions
│
├── notebooks/
│   ├── 01_EDA.ipynb                       # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb             # Feature engineering & preprocessing
│   ├── 03_modeling.ipynb                  # Model training & evaluation
│   ├── 03b_hyperparameter_tuning.ipynb    # Hyperparameter tuning for all models
│   ├── 04_modeling.ipynb                  # Final predictions on test set
│   └── house_prices_analysis.Rmd          # Complementary R analysis (ggplot2)
│
├── src/
│   ├── preprocess.py                      # Preprocessing pipeline (functions)
│   └── model.py                           # Model training & evaluation (functions)
│
├── models/                                # Saved models (joblib)
│   ├── stacking.pkl
│   ├── elasticnet.pkl
│   ├── lasso.pkl
│   ├── ridge.pkl
│   ├── xgboost.pkl
│   ├── lgbm.pkl
│   ├── scaler.pkl
│   ├── columns.pkl
│   └── blend_weights.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- 1460 houses × 81 features
- Target variable `SalePrice` follows a log-normal distribution
- `OverallQual` (r=0.79) and `GrLivArea` (r=0.71) most correlated with price
- 19 columns with missing values identified
- Neighborhood has 3× price difference between best and worst areas

### 2. Feature Engineering (326 features)
- **Combined surfaces** — TotalSF, TotalLivArea, TotalPorchSF, TotalBath
- **Ratios** — LivLotRatio, BsmtFinRatio
- **Temporal features** — HouseAge, RemodAge, GarageAge, IsNew, IsRemodeled
- **Binary flags** — HasPool, HasGarage, HasBsmt, HasFireplace, Has2ndFloor
- **Ordinal encoding** — 14 quality variables (Po/Fa/TA/Gd/Ex → 1..5)
- **Interactions** — QualSF, QualBath, QualGarage, GarageScore, BsmtScore
- **Skew correction** — log1p on 43 asymmetric features
- **Re-typing** — MSSubClass, MoSold, YrSold treated as categorical

### 3. Hyperparameter Tuning
Manual curve-based search for each model (cross-validation RMSE vs parameter value), following the principle: find the best value for each parameter individually before combining.

### 4. Modeling
Trained and compared 9 models on an 80/20 train/test split:

| Model | RMSE | R² |
|-------|------|----|
| **ElasticNet** | **0.1187** | **0.9140** |
| Lasso | 0.1192 | 0.9133 |
| Ridge | 0.1230 | 0.9077 |
| XGBoost | 0.1294 | 0.8979 |
| GradientBoosting | 0.1308 | 0.8957 |
| LightGBM | 0.1308 | 0.8956 |
| SVR | 0.1315 | 0.8945 |
| Random Forest | 0.1454 | 0.8711 |
| Linear Regression | 0.1738 | 0.8158 |

### 5. Stacking & Blend
- **Stacking** — ElasticNet + Lasso + Ridge + SVR + GBR + LightGBM + XGBoost with XGBoost as meta-learner (cv=5)
- **Optimized blend** — scipy.optimize minimization of RMSE over blend weights

| Model | Optimal Weight |
|-------|---------------|
| Stacking | 0.2101 |
| **ElasticNet** | **0.3428** |
| **Lasso** | **0.2723** |
| Ridge | 0.0084 |
| XGBoost | 0.1081 |
| LightGBM | 0.0525 |
| SVR | 0.0000 |
| GBR | 0.0058 |

**Final blend RMSE : 0.1179 (log-price, local CV) · Kaggle Public Score : 13431.52 MAE ($) — 131st / 3878 teams (top ~3.4 %)** on the public leaderboard.

---

## 📈 R Analysis Report

Complementary R analysis with ggplot2 visualizations:
👉 [View Report on RPubs](https://rpubs.com/med_Amine/house-price-prediction)

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/1MedAmine/house-price-prediction
cd house-price-prediction
pip install -r requirements.txt
```

### Download the Data

1. Go to [Kaggle — House Prices](https://www.kaggle.com/competitions/home-data-for-ml-course)
2. Download `train.csv` and `test.csv`
3. Place them in the `data/` folder

### Run the Notebooks

Run the notebooks in order:

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy, scikit-learn, xgboost, lightgbm, scipy, matplotlib, seaborn
- **R** — ggplot2, dplyr, tidyr
- **Jupyter Notebook**

---

## 👤 Author

**Mohammed-Amine Chnidguira**
Master 1 Mathématiques & Applications — Université de Lorraine (IECL), Nancy
[LinkedIn](https://www.linkedin.com/in/mohammed-amine-chnidguira-57215a244/) · [GitHub](https://github.com/1MedAmine)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
