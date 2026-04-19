# 🏠 House Price Prediction

> Predicting residential property prices using machine learning — from data exploration to final predictions.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![R](https://img.shields.io/badge/R-4.x-276DC3?style=flat-square&logo=r)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Ames%20Housing-20BEFF?style=flat-square)
![Kaggle Score](https://img.shields.io/badge/Kaggle%20Score-13510-green?style=flat-square)

---

## 📌 Project Overview

This project applies supervised learning techniques to predict house sale prices from the [Ames Housing Dataset](https://www.kaggle.com/competitions/home-data-for-ml-course) (Kaggle). The full pipeline covers exploratory data analysis, data preprocessing, model training and evaluation, and final predictions submitted to Kaggle.

**Key goals:**
- Perform thorough exploratory data analysis (EDA)
- Handle missing values, outliers, and categorical encoding
- Train and compare multiple regression models
- Evaluate models with RMSE and R² metrics
- Submit predictions to Kaggle

---

## 📁 Repository Structure

```
house-price-prediction/
│
├── data/
│   ├── train.csv                      # Ames Housing training set (Kaggle)
│   └── test.csv                       # Test set for predictions
│
├── notebooks/
│   ├── 01_EDA.ipynb                   # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb         # Cleaning & feature engineering
│   ├── 03_modeling.ipynb              # ML models & evaluation
│   ├── 04_pca_clustering.ipynb        # PCA + EM clustering (analysis)
│   └── house_prices_analysis.Rmd      # Complementary R analysis
│
├── src/
│   ├── preprocess.py                  # Preprocessing pipeline
│   └── model.py                       # Model training & evaluation
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Dataset: 1460 houses × 81 features
- Target variable: `SalePrice` — log-normal distribution
- Key findings: `OverallQual` and `GrLivArea` most correlated with price
- Identified 19 columns with missing values
- Neighborhood has a strong impact on price (3x difference between best/worst)

### 2. Preprocessing
- Removed 2 outliers (large houses with abnormally low prices)
- Applied log transformation to `SalePrice`
- Imputed missing values (None for categorical, 0 for numerical, median by neighborhood for `LotFrontage`)
- One-Hot Encoding → 259 features
- StandardScaler normalization

### 3. Modeling
- Train/Test split: 80% / 20%
- Compared 4 regression models
- Best model: **Lasso Regression** (automatic feature selection on 259 variables)

---

## 📊 Models & Results

| Model | RMSE | R² |
|-------|------|----|
| Linear Regression | 0.1373 | 0.8881 |
| Ridge Regression | 0.1326 | 0.8957 |
| **Lasso Regression** | **0.1287** | **0.9018** |
| Random Forest | 0.1468 | 0.8722 |

> **Best model: Lasso Regression** — R² = 0.90 · Kaggle Public Score: **13510**

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

- **Python** — pandas, numpy, scikit-learn, matplotlib, seaborn
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
