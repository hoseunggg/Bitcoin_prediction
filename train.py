import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
from joblib import Parallel, delayed

from model import generate_rolling_lag_indices, run_model_from_index_spec

ALL_FEATURES = [
    "BlkCnt", "BlkSizeMeanByte", "DiffMean", "FeeMeanUSD", "HashRate",
    "PriceUSD", "RevUSD", "TxTfrValAdjUSD", "URTH", "^GSPC",
    "Risk_Free", "OAS", "Ten_Year", "Two_Year", "Term_Spread",
    "VIX", "USD_Index", "Expected_Inflation", "US_News_Sentiment",
]


if __name__ == "__main__": 
 
 
    df = pd.read_csv(config.PREPROCESS_DATA, index_col=0)
    df.reset_index(drop=True, inplace=True)

    col_list_template = [
        {"source": "PriceUSD", "type": "target", "target": "PriceUSD_lag-1", "lag": -1},
        {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag0",  "lag": 0},
        {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag1",  "lag": 1},
        {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag2",  "lag": 2},
        {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag3",  "lag": 3},
        {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag4",  "lag": 4},
        {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag5",  "lag": 5},
        {"source": "PriceUSD", "type": "endo",   "target": "PriceUSD_lag6",  "lag": 6}
    ]

    for feat in ALL_FEATURES:
        if feat == "PriceUSD": col_list = col_list_template
        else: col_list = col_list_template + [{"source": feat, "type": "endo", "target": feat, "lag": 0}]

        # if feat != "PriceUSD": continue
        # col_list = col_list_template
        
        records = []
        for train_size in [30, 45, 60, 75, 90]:
            splits = generate_rolling_lag_indices(df, col_list, train_size, step=1)
            if splits.empty:
                continue

            df_ols = run_model_from_index_spec(df, splits, col_list, model_type="ols", n_jobs=-1)
            df_rf  = run_model_from_index_spec(df, splits, col_list, model_type="rf",  n_jobs=-1)
            df_xgb = run_model_from_index_spec(df, splits, col_list, model_type="xgb", n_jobs=-1)
            df_svr = run_model_from_index_spec(df, splits, col_list, model_type="svr", n_jobs=-1)
            df_mlp = run_model_from_index_spec(df, splits, col_list, model_type="mlp", n_jobs=-1)

            merged = df_mlp[['split','actual','forecast']].rename(columns={'forecast':'forecast_ols'})

            merged = merged.merge(df_rf[['split','forecast']].rename(columns={'forecast':'forecast_rf'}), on='split')
            merged = merged.merge(df_xgb[['split','forecast']].rename(columns={'forecast':'forecast_xgb'}), on='split')
            merged = merged.merge(df_svr[['split','forecast']].rename(columns={'forecast':'forecast_svr'}), on='split')
            merged = merged.merge(df_mlp[['split','forecast']].rename(columns={'forecast':'forecast_mlp'}), on='split')


            merged['train_size'] = train_size
            records.append(merged)

        df_out = pd.concat(records, ignore_index=True)
        out_file = os.path.join(config.RESULT_ERROR_DIR, f"predictions_by_feat_{feat}.csv")
        df_out.to_csv(out_file, index=False)
        print(f"Saved predictions for {feat}: {out_file}")


