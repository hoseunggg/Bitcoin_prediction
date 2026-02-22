from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNetCV
import numpy as np
import pandas as pd

def rolling_split(data, train_size, val_size, test_size, step=1, retrain=False, horizons=1):
    train_index = {}
    val_index = {}
    test_index = {}
    if retrain:
        retrain_index = {}

    n_data = len(data)
    max_start = n_data - (train_size + val_size + test_size + horizons)

    for start in range(0, max_start + 1, step):
        train_index[start] = list(range(start, start + train_size))

        val_start = start + train_size
        val_end = val_start + val_size
        val_index[start] = list(range(val_start, val_end))

        test_start = val_end + (horizons - 1 if horizons else 0)
        test_index[start] = list(range(test_start, test_start + test_size))

        if retrain:
            retrain_index[start] = list(range(start, val_end))

    if retrain:
        return train_index, val_index, retrain_index, test_index
    else:
        return train_index, val_index, test_index

def expanding_split(data, train_size, val_size, test_size, step=1, retrain=False, horizons=1):
    train_index = {}
    val_index = {}
    test_index = {}
    if retrain:
        retrain_index = {}

    n_data = len(data)
    max_expansion = n_data - (train_size + val_size + test_size + horizons)

    for exp in range(0, max_expansion + 1, step):
        train_end = train_size + exp
        train_index[exp] = list(range(0, train_end))

        val_start = train_end
        val_end = val_start + val_size
        val_index[exp] = list(range(val_start, val_end))

        test_start = val_end + (horizons - 1 if horizons else 0)
        test_index[exp] = list(range(test_start, test_start + test_size))

        if retrain:
            retrain_index[exp] = list(range(0, val_end))

    if retrain:
        return train_index, val_index, retrain_index, test_index
    else:
        return train_index, val_index, test_index

def unrolling_split(data, train_size, val_size, test_size, retrain=False, horizons=1):
    train_index = {}
    val_index = {}
    test_index = {}
    if retrain:
        retrain_index = {}

    n_data = len(data)
    window = train_size + val_size + test_size + (horizons - 1 if horizons else 0)
    num_splits = (n_data - window) // window + 1

    for i in range(num_splits):
        start = i * window

        train_index[start] = list(range(start, start + train_size))

        val_start = start + train_size
        val_end = val_start + val_size
        val_index[start] = list(range(val_start, val_end))

        test_start = val_end + (horizons - 1 if horizons else 0)
        test_index[start] = list(range(test_start, test_start + test_size))

        if retrain:
            retrain_index[start] = list(range(start, val_end))

    if retrain:
        return train_index, val_index, retrain_index, test_index
    else:
        return train_index, val_index, test_index

import numpy as np
from sklearn.metrics import mean_squared_error

def fit_arimax_model(y, X, p, include_intercept=True):
    y = np.asarray(y)
    X = np.asarray(X)
    n = len(y)

    if p > 0:
        X_ar = np.column_stack([y[i:n - p + i] for i in range(p)])
        X_all = X[p:]
        if X_all.ndim == 1:
            X_all = X_all.reshape(-1, 1)
        X_full = np.hstack([X_ar, X_all])
    else:
        X_full = X
        if X_full.ndim == 1:
            X_full = X_full.reshape(-1, 1)

    if include_intercept:
        X_full = np.hstack([np.ones((X_full.shape[0], 1)), X_full])

    y_target = y[p:]

    coeffs = np.linalg.lstsq(X_full, y_target, rcond=None)[0]
    return coeffs

def predict_arimax(y_hist, x_input, coeffs, p, include_intercept=True):
    if p > 0:
        y_lags = np.flip(y_hist[-p:])
        x_input = np.asarray(x_input).reshape(1, -1)
        features = np.hstack([y_lags.reshape(1, -1), x_input])
    else:
        features = np.asarray(x_input).reshape(1, -1)

    if include_intercept:
        features = np.hstack([np.ones((1, 1)), features])

    return float(np.dot(features, coeffs.T))

def _train_arima(
    df, 
    train_idx, 
    val_idx,
    retrain_idx,
    test_idx,
    param_combinations, 
    param_keys, 
    target_col, 
    other_cols
):
    forecast_index = df.index[test_idx[0]]
    actual = df[target_col].iloc[test_idx[0]]

    best_score = float("inf")
    best_forecast = None

    if isinstance(other_cols, str):
        other_cols = [other_cols]

    if len(val_idx) > 0:
        for params in param_combinations:
            param_dict = dict(zip(param_keys, params))
            order = param_dict["order"]
            p = order[0]
            include_intercept = param_dict.get("trend", "c") == "c"

            y_train = df[target_col].iloc[train_idx].values
            y_valid = df[target_col].iloc[val_idx].values

            X_train = df.iloc[train_idx][other_cols].values if other_cols else np.empty((len(train_idx), 0))
            X_valid = df.iloc[val_idx][other_cols].values if other_cols else np.empty((len(val_idx), 0))

            if len(y_train) <= p:
                continue

            coeffs = fit_arimax_model(y_train, X_train, p, include_intercept)

            preds = []
            y_hist = y_train.copy()
            for i in range(len(y_valid)):
                x_input = X_valid[i]
                y_pred = predict_arimax(y_hist, x_input, coeffs, p, include_intercept)
                preds.append(y_pred)
                y_hist = np.append(y_hist, y_valid[i])  # update history with true value

            score = mean_squared_error(y_valid, preds)

            if score < best_score:
                best_score = score
                best_params = param_dict
                best_coeffs = coeffs

        # retrain
        p = best_params["order"][0]
        include_intercept = best_params.get("trend", "c") == "c"

        y_retrain = df[target_col].iloc[retrain_idx].values
        X_retrain = df.iloc[retrain_idx][other_cols].values if other_cols else np.empty((len(retrain_idx), 0))

        best_coeffs = fit_arimax_model(y_retrain, X_retrain, p, include_intercept)
        X_test = df.iloc[test_idx][other_cols].values[0] if other_cols else np.empty((0,))
        best_forecast = predict_arimax(y_retrain, X_test, best_coeffs, p, include_intercept)

    else:
        param_dict = dict(zip(param_keys, param_combinations[0]))
        order = param_dict["order"]
        p = order[0]
        include_intercept = param_dict.get("trend", "c") == "c"

        y_train = df[target_col].iloc[train_idx].values
        X_train = df.iloc[train_idx][other_cols].values if other_cols else np.empty((len(train_idx), 0))

        coeffs = fit_arimax_model(y_train, X_train, p, include_intercept)
        X_test = df.iloc[test_idx][other_cols].values[0] if other_cols else np.empty((0,))
        best_forecast = predict_arimax(y_train, X_test, coeffs, p, include_intercept)

    return forecast_index, best_forecast, actual
