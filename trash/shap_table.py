import ast
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import shap
import matplotlib.pyplot as plt
import config

EXOGENOUS_FEATURES = [
    "BlkCnt", "BlkSizeMeanByte", "DiffMean", "FeeMeanUSD", "HashRate",
    "PriceUSD", "RevUSD", "TxTfrValAdjUSD", "URTH", "^GSPC",
    "Risk_Free", "OAS", "Ten_Year", "Two_Year", "Term_Spread",
    "VIX", "USD_Index", "Expected_Inflation", "US_News_Sentiment",
]
def compute_shap_from_split(row, raw_df, col_list, model_type="rf", max_display=10):
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    target_spec = next(spec for spec in col_list if spec["type"] == "target")
    target_col = target_spec["target"]
    target_source = target_spec["source"]

    feature_specs = [spec for spec in col_list if spec["type"] != "target"]
    feature_cols = [spec["target"] for spec in feature_specs]
    feature_sources = {spec["target"]: spec["source"] for spec in feature_specs}

    # X, y 구성
    X_full = pd.DataFrame()
    for col in feature_cols:
        try:
            idx_list = ast.literal_eval(row[col])
            X_full[col] = raw_df.loc[idx_list, feature_sources[col]].values
        except Exception as e:
            print(f"Error parsing feature column {col}: {e}")
            return

    try:
        y_idx = ast.literal_eval(row[target_col])
        y_full = raw_df.loc[y_idx, target_source].values
    except Exception as e:
        print(f"Error parsing target column: {e}")
        return

    if len(X_full) < 2 or len(y_full) < 2:
        print("Not enough data for training.")
        return

    X_train = X_full.iloc[:-1]
    y_train = y_full[:-1]

    # 모델 선택
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        explainer_type = "tree"
    elif model_type == "xgb":
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
        explainer_type = "tree"
    elif model_type == "mlp":
        model = MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
        explainer_type = "kernel"
    elif model_type == "svr":
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        explainer_type = "kernel"
    else:
        raise ValueError("Unsupported model_type")

    model.fit(X_train, y_train)

    # SHAP 계산
    if explainer_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
    else:
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(X_train)

    # SHAP 요약용 feature 설정
    exclude_keywords = ["PriceUSD"]
    include_cols = [col for col in X_train.columns if not any(k in col for k in exclude_keywords)]
    col_indices = [X_train.columns.get_loc(c) for c in include_cols]

    # summary plot 시각화
    if explainer_type == "tree":
        shap.summary_plot(
            shap_values[:, col_indices],
            X_train[include_cols],
            max_display=max_display,
            show=True
        )
        selected_shap_values = shap_values[:, col_indices]
    else:
        shap.summary_plot(
            shap_values.values[:, col_indices],
            X_train[include_cols],
            max_display=max_display,
            show=True
        )
        selected_shap_values = shap_values.values[:, col_indices]

    # SHAP 평균 중요도 계산
    mean_shap_importance = np.abs(selected_shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": include_cols,
        "Mean |SHAP value|": mean_shap_importance
    }).sort_values(by="Mean |SHAP value|", ascending=False)

    print(f"\nTop SHAP feature importances for model '{model_type}':")
    print(importance_df.head(10))
    try:
        # SHAP value matrix 형태로 변환
        shap_values_matrix = (
            selected_shap_values if isinstance(selected_shap_values, np.ndarray)
            else selected_shap_values.values
        )

        # SHAP summary beeswarm plot (x축 하나, y축에 feature 이름)
        shap.summary_plot(
            shap_values_matrix,
            X_train[include_cols],
            plot_type="dot",  # 기본값이지만 명시함
            max_display=max_display,
            show=True
        )

        # ✅ 모든 feature를 하나의 X축에 통합한 scatter plot (색은 feature value)
        melted = []
        for i, col in enumerate(include_cols):
            for shap_val, feat_val in zip(shap_values_matrix[:, i], X_train[col].values):
                melted.append({
                    "Feature": col,
                    "SHAP": shap_val,
                    "Value": feat_val
                })

        df_plot = pd.DataFrame(melted)

    except Exception as e:
        print(f"[SHAP summary plot skipped] {e}")

    return shap_values, importance_df


def generate_rolling_lag_indices(raw_df, col_list, train_size, step=1):
    data_length = len(raw_df)
    max_pos_lag = max([c["lag"] for c in col_list if c["lag"] >= 0], default=0)
    max_neg_lag = abs(min([c["lag"] for c in col_list if c["lag"] < 0], default=0))

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
    for feat in EXOGENOUS_FEATURES:
        col_list.append({"source": feat, "type": "exo", "target": feat, "lag": 0})

    df = pd.read_csv(config.PREPROCESS_DATA, index_col=0)
    df.reset_index(drop=True, inplace=True)

    rolling_index_df = generate_rolling_lag_indices(df, col_list, train_size=30, step=1)
    row = rolling_index_df.iloc[0].to_dict()
    compute_shap_from_split(row, df, col_list, model_type="rf")

    compute_shap_from_split(row, df, col_list, model_type="xgb")

    compute_shap_from_split(row, df, col_list, model_type="svr")

    compute_shap_from_split(row, df, col_list, model_type="mlp")
    