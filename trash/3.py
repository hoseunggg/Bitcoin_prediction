# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import config
# from joblib import Parallel, delayed

# from model import generate_rolling_lag_indices, run_model_from_index_spec

# # 모든 외생 변수 리스트
# ALL_FEATURES = [
#     "BlkCnt", "BlkSizeMeanByte", "DiffMean", "FeeMeanUSD", "HashRate",
#     "PriceUSD", "RevUSD", "TxTfrValAdjUSD", "URTH", "^GSPC",
#     "Risk_Free", "OAS", "Ten_Year", "Two_Year", "Term_Spread",
#     "VIX", "USD_Index", "Expected_Inflation", "US_News_Sentiment",
# ]

# if __name__ == "__main__":

#     df = pd.read_csv(config.PREPROCESS_DATA, index_col=0)
#     df.reset_index(drop=True, inplace=True)

#     # Target + PriceUSD lags
#     col_list = [
#         {"source": "PriceUSD", "type": "target", "target": "PriceUSD_lag-1", "lag": -1},
#         {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag0",  "lag": 0},
#         {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag1",  "lag": 1},
#         {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag2",  "lag": 2},
#         {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag3",  "lag": 3},
#         {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag4",  "lag": 4},
#         {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag5",  "lag": 5},
#         {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag6",  "lag": 6},
#     ]

#     # 모든 외생 변수 추가 (lag 0 기준)
#     for feat in ALL_FEATURES:
#         col_list.append({"source": feat, "type": "exo", "target": feat, "lag": 0})

#     # 결과 저장용
#     records = []

#     for train_size in range(10, 70):
#         splits = generate_rolling_lag_indices(df, col_list, train_size, step=1)
#         if splits.empty:
#             continue

#         df_ols = run_model_from_index_spec(df, splits, col_list, model_type="ols", n_jobs=-1)
#         df_rf  = run_model_from_index_spec(df, splits, col_list, model_type="rf",  n_jobs=-1)
#         df_xgb = run_model_from_index_spec(df, splits, col_list, model_type="xgb", n_jobs=-1)

#         # Merge predictions by split
#         merged = df_ols[['split','actual','forecast']].rename(columns={'forecast':'forecast_ols'})
#         merged = merged.merge(df_rf[['split','forecast']].rename(columns={'forecast':'forecast_rf'}), on='split')
#         merged = merged.merge(df_xgb[['split','forecast']].rename(columns={'forecast':'forecast_xgb'}), on='split')
#         merged['train_size'] = train_size
#         records.append(merged)

#     df_out = pd.concat(records, ignore_index=True)
#     out_file = os.path.join(config.RESULT_DIR, "predictions_all_features.csv")
#     df_out.to_csv(out_file, index=False)
#     print(f"Saved predictions with all features: {out_file}")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config

TRAIN_SIZES = list(range(10, 70))
MODELS = ['rf', 'xgb']

def load_rmse_by_model(path):
    df = pd.read_csv(path)
    rmse_result = {model: {} for model in MODELS}
    for ts in TRAIN_SIZES:
        sub = df[df['train_size'] == ts]
        if sub.empty:
            continue
        y_true = sub['actual']
        for model in MODELS:
            y_pred = sub[f'forecast_{model}']
            rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
            rmse_result[model][ts] = rmse
    return rmse_result

if __name__ == "__main__":
    # 파일 경로
    baseline_path = os.path.join(config.RESULT_ERROR_DIR, "predictions_by_feat_PriceUSD.csv")
    allfeat_path  = os.path.join(config.RESULT_DIR, "predictions_all_features.csv")

    # RMSE 계산
    rmse_baseline = load_rmse_by_model(baseline_path)
    rmse_allfeat  = load_rmse_by_model(allfeat_path)

    # 시각화
    plt.figure(figsize=(9, 5))
    for model in MODELS:
        plt.plot(rmse_baseline[model].keys(), rmse_baseline[model].values(),
                 linestyle='--', marker='o', label=f"{model.upper()} (PriceUSD only)")
        plt.plot(rmse_allfeat[model].keys(), rmse_allfeat[model].values(),
                 linestyle='-', marker='s', label=f"{model.upper()} (ALL FEATURES)")

    plt.xlabel("Train Size")
    plt.ylabel("RMSE")
    plt.title("Baseline vs All Exogenous Variable Comparison (OLS, RF, XGB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
