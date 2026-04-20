"""
Preprocessing module for the House Price Prediction project.

This module contains all the preprocessing functions used to clean,
transform, and prepare the Ames Housing dataset for modeling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Categorical columns where NaN means "does not exist"
CATEGORICAL_NONE_COLS = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType'
]

# Numerical columns where NaN means 0
NUMERICAL_ZERO_COLS = [
    'GarageYrBlt', 'GarageArea', 'GarageCars',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
]


def load_data(path):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file to load.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.
    """
    df = pd.read_csv(path)
    return df


def remove_outliers(df):
    """
    Remove outlier houses from the dataset.

    Drops houses with GrLivArea > 4000 and SalePrice < 200000,
    identified during the exploratory data analysis as influential
    outliers that skew the model.

    Parameters
    ----------
    df : pandas.DataFrame
        The raw dataset (must contain 'GrLivArea' and 'SalePrice').

    Returns
    -------
    pandas.DataFrame
        Dataset without the outliers.
    """
    df = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < 200000))]
    return df.reset_index(drop=True)


def handle_missing_values(df):
    """
    Fill missing values using domain-specific strategies.

    - Categorical columns (listed in CATEGORICAL_NONE_COLS) are filled
      with the string 'None' since NaN here means "the feature does not exist".
    - Numerical columns (listed in NUMERICAL_ZERO_COLS) are filled with 0.
    - LotFrontage is imputed by the median within each Neighborhood.
    - Electrical is filled with the most frequent value (mode).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with missing values.

    Returns
    -------
    pandas.DataFrame
        Dataset without missing values.
    """
    df = df.copy()

    # Categorical: NaN means "does not exist"
    for col in CATEGORICAL_NONE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # Numerical: NaN means 0
    for col in NUMERICAL_ZERO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # LotFrontage: median by neighborhood
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )

    # Electrical: mode
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    return df


def encode_features(df):
    """
    Apply One-Hot Encoding to categorical variables.

    Uses pandas.get_dummies with drop_first=True to avoid
    the dummy variable trap (multicollinearity).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with raw categorical columns.

    Returns
    -------
    pandas.DataFrame
        Dataset with one-hot encoded categorical variables.
    """
    return pd.get_dummies(df, drop_first=True)


def normalize_features(X):
    """
    Normalize features using StandardScaler (zero mean, unit variance).

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix to normalize.

    Returns
    -------
    pandas.DataFrame
        Scaled feature matrix, with the same column names as X if X is a DataFrame.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if isinstance(X, pd.DataFrame):
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    return X_scaled


def run_preprocessing(path):
    """
    Run the full preprocessing pipeline on the training dataset.

    Pipeline steps:
        1. Load the CSV file.
        2. Remove outliers.
        3. Apply log transformation to SalePrice.
        4. Handle missing values.
        5. One-Hot Encode categorical features.
        6. Separate features (X) and target (y).
        7. Normalize features with StandardScaler.

    Parameters
    ----------
    path : str
        Path to the training CSV file.

    Returns
    -------
    X_scaled : pandas.DataFrame
        Scaled feature matrix ready for modeling.
    y : pandas.Series
        Log-transformed target variable (SalePrice).
    """
    # 1. Load
    df = load_data(path)

    # 2. Remove outliers
    df = remove_outliers(df)

    # 3. Log-transform the target
    df['SalePrice'] = np.log(df['SalePrice'])

    # 4. Handle missing values
    df = handle_missing_values(df)

    # 5. One-Hot Encoding
    df = encode_features(df)

    # 6. Separate features / target
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    # 7. Normalize
    X_scaled = normalize_features(X)

    return X_scaled, y


if __name__ == '__main__':
    # Quick sanity check when running the script directly
    X_scaled, y = run_preprocessing('../data/train.csv')
    print(f"X_scaled shape : {X_scaled.shape}")
    print(f"y shape        : {y.shape}")
    print(f"y mean (log)   : {y.mean():.4f}")
