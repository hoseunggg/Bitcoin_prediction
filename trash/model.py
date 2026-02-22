import ast
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import config
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

import shap_table
import matplotlib.pyplot as plt

def explain_with_shap(model, X, model_type, max_display=10):
    if model_type in ["rf", "xgb"]:
        explainer = shap_table.Explainer(model, X)
    else:  
        explainer = shap_table.Explainer(model.predict, X)  
    
    shap_values = explainer(X)
    shap_table.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.title(f"SHAP Summary ({model_type.upper()})")
    plt.tight_layout()
    plt.show()
    
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
    elif model_type == "mean": runner = run_benchmark_mean
        
    results = Parallel(n_jobs=n_jobs)(delayed(runner)(row, raw_df, col_list, feature_cols, feature_sources, target_col, target_source) for row in rows)
    return pd.DataFrame(results)


def run_single_ols(row, raw_df, col_list, feature_cols, feature_sources, target_col, target_source):
    split_id = row["split"]
    
    X_full = pd.DataFrame({col: raw_df.loc[ast.literal_eval(row[col]), feature_sources[col]].values for col in feature_cols})
    
    y_idx = ast.literal_eval(row[target_col])
    y_full = raw_df.loc[y_idx, target_source].values

    X_train, X_test = X_full.iloc[:-1], X_full.iloc[[-1]]
    y_train, y_test = y_full[:-1], y_full[-1]

    X_train_const = sm.add_constant(X_train, has_constant='add')
    ols_model = sm.OLS(y_train, X_train_const).fit()

    X_test_const = sm.add_constant(X_test, has_constant='add')
    X_test_const = X_test_const[ols_model.model.exog_names]

    y_pred = float(ols_model.predict(X_test_const).iloc[0])
    return {"split": split_id, "forecast": y_pred, "actual": float(y_test), "error": float(y_test - y_pred)}


def run_single_rf(row, raw_df, col_list, feature_cols, feature_sources, target_col, target_source):
    split_id = row["split"]
    X_full = pd.DataFrame({col: raw_df.loc[ast.literal_eval(row[col]), feature_sources[col]].values for col in feature_cols})
    
    y_idx = ast.literal_eval(row[target_col])
    y_full = raw_df.loc[y_idx, target_source].values

    X_train, X_test = X_full.iloc[:-1], X_full.iloc[[-1]]
    y_train, y_test = y_full[:-1], y_full[-1]

    rf = RandomForestRegressor(n_estimators=500, max_depth=7, min_samples_leaf=1,random_state=42)
    rf.fit(X_train, y_train)
    y_pred = float(rf.predict(X_test)[0])

    return {"split": split_id, "forecast": y_pred, "actual": float(y_test), "error": float(y_test - y_pred)}


def run_single_xgb(row, raw_df, col_list, feature_cols, feature_sources, target_col, target_source):
    split_id = row["split"]
    X_full = pd.DataFrame({col: raw_df.loc[ast.literal_eval(row[col]), feature_sources[col]].values for col in feature_cols})
    
    y_idx = ast.literal_eval(row[target_col])
    y_full = raw_df.loc[y_idx, target_source].values

    X_train, X_test = X_full.iloc[:-1], X_full.iloc[[-1]]
    y_train, y_test = y_full[:-1], y_full[-1]

    xgb = XGBRegressor(n_estimators=500, max_depth=7,learning_rate=0.05,subsample=1,colsample_bytree=1, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = float(xgb.predict(X_test)[0])

    return {"split": split_id, "forecast": y_pred, "actual": float(y_test), "error": float(y_test - y_pred)}



def run_single_svr(row, raw_df, col_list, feature_cols, feature_sources, target_col, target_source):
    split_id = row["split"]
    
    X_full = pd.DataFrame({col: raw_df.loc[ast.literal_eval(row[col]), feature_sources[col]].values for col in feature_cols})

    y_idx = ast.literal_eval(row[target_col])
    y_full = raw_df.loc[y_idx, target_source].values

    X_train, X_test = X_full.iloc[:-1], X_full.iloc[[-1]]
    y_train, y_test = y_full[:-1], y_full[-1]

    svr = SVR(kernel='rbf', C=0.1, epsilon=0.01)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)[0]

    return {"split": split_id, "forecast": float(y_pred), "actual": float(y_test), "error": float(y_test - y_pred)}


def run_single_mlp(row, raw_df, col_list, feature_cols, feature_sources, target_col, target_source):
    split_id = row["split"]
    
    X_full = pd.DataFrame({col: raw_df.loc[ast.literal_eval(row[col]), feature_sources[col]].values for col in feature_cols})

    y_idx = ast.literal_eval(row[target_col])
    y_full = raw_df.loc[y_idx, target_source].values

    X_train, X_test = X_full.iloc[:-1], X_full.iloc[[-1]]
    y_train, y_test = y_full[:-1], y_full[-1]

    mlp = MLPRegressor(hidden_layer_sizes=(32,), alpha=0.001, max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)[0]

    return {"split": split_id, "forecast": float(y_pred), "actual": float(y_test), "error": float(y_test - y_pred)}

def run_benchmark_mean(row, raw_df, col_list, feature_cols, feature_sources, target_col, target_source):
    split_id = row["split"]

    target_spec = next(spec for spec in col_list if spec["type"] == "target")
    target_col = target_spec["target"]
    target_source = target_spec["source"]

    y_idx = ast.literal_eval(row[target_col])
    y_full = raw_df.loc[y_idx, target_source].values

    y_train, y_test = y_full[:-1], y_full[-1]
    y_pred = float(np.mean(y_train))

    return {
        "split": split_id,
        "forecast": y_pred,
        "actual": float(y_test),
        "error": float(y_test - y_pred)
    }


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

    rolling_index_df = generate_rolling_lag_indices(df, col_list, train_size=10, step=1)

    print(rolling_index_df.head())