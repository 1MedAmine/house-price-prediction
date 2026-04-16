<<<<<<< HEAD
# 🏠 House Price Prediction

> Predicting residential property prices using machine learning — combining classical statistical methods with modern ensemble models.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![R](https://img.shields.io/badge/R-4.x-276DC3?style=flat-square&logo=r)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Ames%20Housing-20BEFF?style=flat-square)

---

## 📌 Project Overview

This project applies supervised learning techniques to predict house sale prices from the [Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) (Kaggle). It goes beyond standard modeling by integrating **dimensionality reduction (PCA)** and **Gaussian mixture clustering (EM algorithm)** — techniques studied during my Master's in Applied Mathematics at Université de Lorraine.

**Key goals:**
- Perform thorough exploratory data analysis (EDA)
- Engineer relevant features and handle missing data
- Apply PCA for dimensionality reduction
- Cluster properties using the EM algorithm (Gaussian Mixture Models)
- Train and compare multiple regression models (Ridge, Lasso, Random Forest, XGBoost)
- Validate results with cross-validation, KS tests, and SHAP interpretability

---

## 📁 Repository Structure

```
house-price-prediction/
│
├── data/
│   ├── train.csv              # Ames Housing training set (Kaggle)
│   └── test.csv               # Test set for predictions
│
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Cleaning & feature engineering
│   ├── 03_modeling.ipynb      # ML models & evaluation
│   ├── 04_pca_clustering.ipynb# PCA + EM clustering
│   └── house_prices_analysis.Rmd  # Complementary R analysis
│
├── src/
│   ├── preprocess.py          # Preprocessing pipeline
│   └── model.py               # Model training & evaluation
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔬 Mathematical Approach

This project leverages methods studied in the **Master 1 Mathématiques & Applications** program:

| Method | Application |
|--------|-------------|
| **PCA (Principal Component Analysis)** | Dimensionality reduction on the 80 housing features |
| **EM Algorithm (Gaussian Mixtures)** | Unsupervised clustering of property profiles |
| **Ridge / Lasso Regression** | Regularized linear models with linear algebra foundations |
| **KS Test** | Residual distribution validation |
| **Cross-validation** | Model robustness and generalization assessment |

---

## 📊 Models & Results

| Model | RMSE | R² |
|-------|------|----|
| Linear Regression | — | — |
| Ridge Regression | — | — |
| Lasso Regression | — | — |
| Random Forest | — | — |
| XGBoost | — | — |

> Results will be updated as the project progresses.

---

## 🚀 Getting Started

### Prerequisites

```bash
git https://github.com/1MedAmine/house-price-prediction
cd house-price-prediction
pip install -r requirements.txt
```

### Download the Data

1. Go to [Kaggle — House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
2. Download `train.csv` and `test.csv`
3. Place them in the `data/` folder

### Run the Notebooks

Open Jupyter and run the notebooks in order:

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, shap
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
=======
# house-price-prediction
>>>>>>> 4da84c1f826e4e815d6dae8599d9c68386d4dbbc
