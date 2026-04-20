"""
Modeling module for the House Price Prediction project.

This module contains functions to split data, train multiple regression
models, evaluate them, select the best one, generate predictions on the
test set, and save the results as a submission file.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and target into training and test sets.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix.
    y : pandas.Series or numpy.ndarray
        Target variable.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        The four splits of the data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """
    Train four regression models on the training data.

    Models trained:
        - Linear Regression (baseline)
        - Ridge Regression (alpha=10)
        - Lasso Regression (alpha=0.001)
        - Random Forest Regressor (n_estimators=100)

    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training feature matrix.
    y_train : pandas.Series or numpy.ndarray
        Training target variable.

    Returns
    -------
    dict
        Dictionary mapping model names (str) to trained sklearn estimators.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=10),
        'Lasso': Lasso(alpha=0.001),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models on the test set.

    Computes RMSE (Root Mean Squared Error) and R² (coefficient of
    determination) for each model.

    Parameters
    ----------
    models : dict
        Dictionary of trained models (output of train_models).
    X_test : pandas.DataFrame or numpy.ndarray
        Test feature matrix.
    y_test : pandas.Series or numpy.ndarray
        Test target variable.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by model name with columns 'RMSE' and 'R²',
        sorted by RMSE (ascending).
    """
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': round(rmse, 4), 'R²': round(r2, 4)}

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('RMSE')
    return results_df


def get_best_model(models, X_test, y_test):
    """
    Select the best model based on the lowest RMSE on the test set.

    Parameters
    ----------
    models : dict
        Dictionary of trained models (output of train_models).
    X_test : pandas.DataFrame or numpy.ndarray
        Test feature matrix.
    y_test : pandas.Series or numpy.ndarray
        Test target variable.

    Returns
    -------
    tuple
        (best_model_name : str, best_model : sklearn estimator)
        The name and instance of the best model.
    """
    best_name = None
    best_rmse = np.inf
    best_model = None

    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = model

    print(f"Best model : {best_name} (RMSE = {best_rmse:.4f})")
    return best_name, best_model


def predict_final(model, X_test):
    """
    Generate final price predictions from a model trained on log-transformed target.

    Applies np.exp() to reverse the log transformation done during
    preprocessing, giving predictions in dollars rather than in log-scale.

    Parameters
    ----------
    model : sklearn estimator
        Trained model that outputs log-transformed predictions.
    X_test : pandas.DataFrame or numpy.ndarray
        Feature matrix to predict on.

    Returns
    -------
    numpy.ndarray
        Final predictions in original price scale (dollars).
    """
    y_pred_log = model.predict(X_test)
    y_pred_final = np.exp(y_pred_log)
    return y_pred_final


def save_submission(predictions, test_ids, path):
    """
    Save predictions as a CSV file in the Kaggle submission format.

    The resulting file has two columns: 'Id' and 'SalePrice'.

    Parameters
    ----------
    predictions : array-like
        Predicted prices (after inverse log transformation).
    test_ids : array-like
        IDs of the test set houses (from test.csv 'Id' column).
    path : str
        Path where the submission CSV will be saved.

    Returns
    -------
    pandas.DataFrame
        The submission DataFrame that was saved.
    """
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    submission.to_csv(path, index=False)
    print(f"Submission saved to : {path}")
    print(f"Shape : {submission.shape}")
    return submission


if __name__ == '__main__':
    # Sanity check: run the full modeling pipeline
    from preprocess import run_preprocessing

    X, y = run_preprocessing('../data/train.csv')
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    print(results)

    best_name, best_model = get_best_model(models, X_test, y_test)
