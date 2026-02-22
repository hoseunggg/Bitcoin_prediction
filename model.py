import ast
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import config
import itertools
import shap
import os

# 모델별 파라미터 그리드 설정
MODEL_PARAM_GRID = {
    "rf": {
        "model_cls": RandomForestRegressor,
        "param_grid": {
            "n_estimators": [300],
            "max_depth": [3, 5, 7],
            "min_samples_leaf": [1, 3, 5],
        },
    },
    "xgb": {
        "model_cls": XGBRegressor,
        "param_grid": {
            "n_estimators": [300],
            "max_depth": [3,5,7],
            "learning_rate": [0.01, 0.05],
        },
    },
    "svr": {
        "model_cls": SVR,
        "param_grid": {
            "C": [0.1, 1, 10],
            "epsilon": [0.001, 0.01, 0.1],
            "kernel":["rbf", "linear"],
        },
    },
    "mlp": {
        "model_cls": MLPRegressor,
        "param_grid": {
            "hidden_layer_sizes": [(32,16), (64,32), (64, 32,16)],
            "alpha": [0.001, 0.01],
            "max_iter": [1000],
            "activation": ["relu", "tanh", "logistic"]  # 추가된 부분
        },
    }
}

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def explain_with_shap(model, X, model_type, split_id, all_results):
    if model_type.lower() in ["rf", "xgb"]:
        explainer = shap.Explainer(model, X)
    else:
        explainer = shap.Explainer(model.predict, X)

    shap_values = explainer(X)

    mean_shap_vals = pd.DataFrame({
        'Feature': X.columns,
        'Mean |SHAP value|': np.abs(shap_values.values).mean(axis=0)
    })

    mean_shap_vals['Split'] = split_id
    mean_shap_vals['Model'] = model_type.upper()

    all_results.append(mean_shap_vals)

    return all_results


def generate_rolling_lag_indices(raw_df, col_list, train_size, step=1):
    data_length = len(raw_df)
    max_pos_lag = max([c["lag"] for c in col_list if c["lag"] >= 0])
    max_neg_lag = abs(min([c["lag"] for c in col_list if c["lag"] < 0]))

    records = []
    for end in range(train_size + max_pos_lag, data_length - max_neg_lag + 1, step):
        base_range = list(range(end - train_size, end))
        row = {"split": len(records)}
        for spec in col_list:
            target = spec["target"]
            lag = spec["lag"]
            indices = [i - lag for i in base_range]
            row[target] = str(indices)
        records.append(row)

    return pd.DataFrame(records)



def run_model_from_index_spec(raw_df, split_df, col_list, model_type="ols", n_jobs=-1):
    target_spec = next(spec for spec in col_list if spec["type"] == "target")
    target_col = target_spec["target"]
    target_source = target_spec["source"]

    feature_specs = [spec for spec in col_list if spec["type"] != "target"]
    feature_cols = [spec["target"] for spec in feature_specs]
    feature_sources = {spec["target"]: spec["source"] for spec in feature_specs}

    rows = split_df.to_dict("records")

    if model_type == "ols": runner = run_single_ols
    elif model_type == "rf": runner = run_single_rf
    elif model_type == "xgb": runner = run_single_xgb
    elif model_type == "svr": runner = run_single_svr
    elif model_type == "mlp": runner = run_single_mlp

    results = Parallel(n_jobs=n_jobs)(delayed(runner)(row, raw_df, col_list, feature_cols, feature_sources, target_col, target_source) for row in rows)
    return pd.DataFrame(results)


def run_single_ols(row, raw_df, col_list, feature_cols, feature_sources, target_col, target_source):
    split_id = row["split"]
    X_full = pd.DataFrame({col: raw_df.iloc[ast.literal_eval(row[col])][feature_sources[col]].values for col in feature_cols})
    y_idx = ast.literal_eval(row[target_col])

    y_full = raw_df.iloc[y_idx][target_source].values

    X_train, X_test = X_full.iloc[:-1], X_full.iloc[[-1]]
    y_train, y_test = y_full[:-1], y_full[-1]

    X_train_const = sm.add_constant(X_train, has_constant='add')

    ols_model = sm.OLS(y_train, X_train_const).fit()

    X_test_const = sm.add_constant(X_test, has_constant='add')
    X_test_const = X_test_const[ols_model.model.exog_names]

    y_pred = float(ols_model.predict(X_test_const).iloc[0])
    return {"split": split_id, "forecast": y_pred, "actual": float(y_test), "error": float(y_test - y_pred)}





def simple_grid_search(X_train, y_train, X_val, y_val, model_cls, param_grid: dict):
    best_score = float("inf")
    best_model = None
    best_params = None

    keys, values = zip(*param_grid.items())
    for param_comb in itertools.product(*values):
        params = dict(zip(keys, param_comb))
        model = model_cls(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        score = rmse(y_val, y_pred)

        if score < best_score:
            best_score = score
            best_model = model_cls(**params)
            best_params = params

    best_model.fit(pd.concat([X_train, X_val]), np.concatenate([y_train, y_val]))
    return best_model, best_params

# 개별 모델 실행용
def run_with_manual_grid(row, raw_df, col_list, feature_cols, feature_sources, target_col, target_source, model_type):
    split_id = row["split"]

    X_full = pd.DataFrame({
        col: raw_df.iloc[eval(row[col])][feature_sources[col]].values
        for col in feature_cols
    })
    y_idx = eval(row[target_col])
    y_full = raw_df.iloc[y_idx][target_source].values

    val_size = max(1, int(len(X_full) * 0.2))
    train_size = len(X_full) - val_size

    X_train, X_val = X_full.iloc[:train_size], X_full.iloc[train_size:-1]
    y_train, y_val = y_full[:train_size], y_full[train_size:-1]

    spec = MODEL_PARAM_GRID[model_type]
    best_model, best_params = simple_grid_search(
        X_train, y_train, X_val, y_val,
        spec["model_cls"], spec["param_grid"]
    )


    X_test = X_full.iloc[[-1]]
    y_test = y_full[-1]
    y_pred = best_model.predict(X_test)[0]

    return {
        "split": split_id,
        "forecast": float(y_pred),
        "actual": float(y_test),
        "error": float(y_test - y_pred),
    }


def run_single_rf(*args): return run_with_manual_grid(*args, model_type="rf")
def run_single_xgb(*args): return run_with_manual_grid(*args, model_type="xgb")
def run_single_svr(*args): return run_with_manual_grid(*args, model_type="svr")
def run_single_mlp(*args): return run_with_manual_grid(*args, model_type="mlp")

if __name__ == "__main__":
    col_list = [
        {"source": "PriceUSD", "type": "target", "target": "PriceUSD_lag-1", "lag": -1},
        {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag0",  "lag": 0},
        {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag1",  "lag": 1},
        {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag2",  "lag": 2},
        {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag3",  "lag": 3},
        {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag4",  "lag": 4},
        {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag5",  "lag": 5},
        {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag6",  "lag": 6}
    ]

    df = pd.read_csv(config.PREPROCESS_DATA, index_col=0)
    rolling_index_df = generate_rolling_lag_indices(df, col_list, train_size=30, step=1)

    result_df = run_model_from_index_spec(df, rolling_index_df, col_list, model_type="rf", n_jobs=-1)
    print(result_df.head())
    print(len(result_df))
