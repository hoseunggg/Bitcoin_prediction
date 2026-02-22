import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import config
from scipy import stats

MODELS = ['ols','rf', 'xgb', 'mlp', 'svr']
ALL_FEATURES = [
    "BlkCnt", "BlkSizeMeanByte", "DiffMean", "FeeMeanUSD", "HashRate",
    "PriceUSD", "RevUSD", "TxTfrValAdjUSD", "URTH", "^GSPC", "Risk_Free",
    "OAS", "Ten_Year", "Two_Year", "Term_Spread", "VIX", "USD_Index",
    "Expected_Inflation", "US_News_Sentiment",
]
FEATURE_GROUPS = {
    "market": ["URTH", "^GSPC", "VIX", "USD_Index"],
    "onchain": ["BlkCnt", "BlkSizeMeanByte", "DiffMean", "FeeMeanUSD", "HashRate", "RevUSD", "TxTfrValAdjUSD"],
    "macro": ["Risk_Free", "OAS", "Ten_Year", "Two_Year", "Term_Spread", "Expected_Inflation", "US_News_Sentiment"]
}


def compute_rmse(models):
    rmse_results = defaultdict(dict)

    for feat in ALL_FEATURES:
        path = os.path.join(config.RESULT_ERROR_DIR, f"predictions_by_feat_{feat}.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        for ts in sorted(df['train_size'].unique()):
            sub = df[df['train_size'] == ts]
            if sub.empty:
                continue
            y = sub['actual']
            for model in models:
                if f'forecast_{model}' not in sub.columns:
                    continue
                y_pred = sub[f'forecast_{model}']
                rmse = round(np.sqrt(((y_pred - y) ** 2).mean()), 4)

                rmse_results[feat][f'RMSE_{model}_ts{ts}'] = rmse

    df_rmse = pd.DataFrame(rmse_results).T.sort_index()
    df_rmse.index.name = 'feature'
    df_rmse.to_csv(os.path.join(config.RESULT_DIR, 'rmse_by_feature.csv'))
    return df_rmse

def compute_rmse_by_group(models):
    from collections import defaultdict

    rmse_results = defaultdict(dict)

    for group, features in FEATURE_GROUPS.items():
        for ts in [30, 45]:

            for model in models:
                actual_all = []
                pred_all = []

                for feat in features:
                    path = os.path.join(config.RESULT_ERROR_DIR, f"predictions_by_feat_{feat}.csv")
                    if not os.path.exists(path):
                        continue

                    df = pd.read_csv(path)
                    sub = df[df['train_size'] == ts]
                    if sub.empty or f'forecast_{model}' not in sub.columns:
                        continue

                    actual_all.extend(sub['actual'].values)
                    pred_all.extend(sub[f'forecast_{model}'].values)

                if actual_all and pred_all:
                    actual_all = np.array(actual_all)
                    pred_all = np.array(pred_all)
                    rmse = round(np.sqrt(((pred_all - actual_all) ** 2).mean()), 4)
                    rmse_results[group][f'RMSE_{model}_ts{ts}'] = rmse

    df_group_rmse = pd.DataFrame(rmse_results).T.sort_index()
    df_group_rmse.index.name = "group"
    df_group_rmse.to_csv(os.path.join(config.RESULT_DIR, 'rmse_by_group.csv'))
    return df_group_rmse


def compute_relative_r2(models):
    baseline_path = os.path.join(config.RESULT_ERROR_DIR, "predictions_by_feat_PriceUSD.csv")
    df_base = pd.read_csv(baseline_path)
    baseline_mspe = defaultdict(dict)

    for ts in sorted(df_base['train_size'].unique()):
        sub = df_base[df_base['train_size'] == ts]
        y = sub['actual']
        for model in models:
            if f'forecast_{model}' not in sub.columns:
                continue
            y_pred = sub[f'forecast_{model}']
            baseline_mspe[ts][model] = ((y_pred - y) ** 2).mean()

    rel_results = defaultdict(dict)
    for feat in ALL_FEATURES:
        path = os.path.join(config.RESULT_ERROR_DIR, f"predictions_by_feat_{feat}.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        for ts in sorted(df['train_size'].unique()):
            sub = df[df['train_size'] == ts]
            y = sub['actual']
            for model in models:
                if f'forecast_{model}' not in sub.columns:
                    continue
                y_pred = sub[f'forecast_{model}']
                mspe = ((y_pred - y) ** 2).mean()
                base = baseline_mspe.get(ts, {}).get(model, np.nan)
                rel = np.nan if np.isnan(base) or base == 0 else 1 - mspe / base
                rel_results[feat][f'RelR2_{model}_ts{ts}'] = round(rel,4)

    df_rel = pd.DataFrame(rel_results).T.sort_index()
    df_rel.index.name = 'feature'
    df_rel.to_csv(os.path.join(config.RESULT_DIR, 'relative_r2_by_feature.csv'))
    return df_rel

def compute_relative_r2_by_group(models):
    baseline_path = os.path.join(config.RESULT_ERROR_DIR, "predictions_by_feat_PriceUSD.csv")
    df_base = pd.read_csv(baseline_path)
    baseline_mspe = defaultdict(dict)

    for ts in sorted(df_base['train_size'].unique()):
        sub = df_base[df_base['train_size'] == ts]
        y = sub['actual']
        for model in models:
            if f'forecast_{model}' not in sub.columns:
                continue
            y_pred = sub[f'forecast_{model}']
            baseline_mspe[ts][model] = ((y_pred - y) ** 2).mean()

    rel_results = defaultdict(dict)
    for group_name, feat_list in FEATURE_GROUPS.items():
        path = os.path.join(config.RESULT_ERROR_DIR, f"predictions_by_group_{group_name}.csv")
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue

        df = pd.read_csv(path)
        for ts in sorted(df['train_size'].unique()):
            sub = df[df['train_size'] == ts]
            y = sub['actual']
            for model in models:
                if f'forecast_{model}' not in sub.columns:
                    continue
                y_pred = sub[f'forecast_{model}']
                mspe = ((y_pred - y) ** 2).mean()
                base = baseline_mspe.get(ts, {}).get(model, np.nan)
                rel = np.nan if np.isnan(base) or base == 0 else 1 - mspe / base
                rel_results[group_name][f'RelR2_{model}_ts{ts}'] = round(rel, 4)

    df_rel = pd.DataFrame(rel_results).T.sort_index()
    df_rel.index.name = 'group'
    df_rel.to_csv(os.path.join(config.RESULT_DIR, 'relative_r2_by_group.csv'))
    return df_rel

def dm_test(e1, e2, h=1):
    """
    Diebold-Mariano test for equal forecast accuracy
    """
    d = (e1 ** 2) - (e2 ** 2)
    d_mean = np.mean(d)
    T = len(d)

    if T == 0 or np.isnan(d_mean):
        return np.nan, np.nan

    # Newey-West variance estimator for lag h-1
    gamma = []
    for k in range(h):
        if k == 0:
            gk = np.var(d, ddof=1)
        else:
            if len(d) > k:
                gk = np.cov(d[:-k], d[k:])[0, 1]
            else:
                gk = 0
        gamma.append(gk)

    var_d = gamma[0] + 2 * sum((1 - k/h) * gamma[k] for k in range(1, h))

    if var_d <= 0 or np.isnan(var_d):
        return np.nan, np.nan

    dm_stat = d_mean / np.sqrt(var_d / T)
    p_value = 2 * stats.t.sf(np.abs(dm_stat), df=T - 1)

    return dm_stat, p_value


def compute_dm_tests(models, target_model='rf', baseline_model='ols', train_sizes=[30, 45, 60, 75, 90]):
    results = []

    for feat in ALL_FEATURES:
        path = os.path.join(config.RESULT_ERROR_DIR, f"predictions_by_feat_{feat}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)

        for ts in train_sizes:
            sub = df[df['train_size'] == ts]
            if sub.empty or f'forecast_{target_model}' not in sub.columns or f'forecast_{baseline_model}' not in sub.columns:
                continue

            y_true = sub['actual'].values
            e1 = y_true - sub[f'forecast_{target_model}'].values
            e2 = y_true - sub[f'forecast_{baseline_model}'].values

            if len(e1) < 5:  # 너무 짧은 경우는 skip
                continue

            dm_stat, p_val = dm_test(e1, e2, h=1)
            results.append({
                'feature': feat,
                'train_size': ts,
                'model_1': target_model,
                'model_2': baseline_model,
                'DM_stat': round(dm_stat, 4),
                'p_value': round(p_val, 4),
                'significant_5pct': p_val < 0.05
            })

    df_dm = pd.DataFrame(results)
    df_dm.to_csv(os.path.join(config.RESULT_DIR, f'dm_test_{target_model}_vs_{baseline_model}.csv'), index=False)
    return df_dm

def compute_dm_vs_priceusd(models, train_sizes=[30, 45, 60, 75, 90]):
    results = []

    for model in models:
        baseline_path = os.path.join(config.RESULT_ERROR_DIR, f"predictions_by_feat_PriceUSD.csv")
        df_base = pd.read_csv(baseline_path)

        for feat in ALL_FEATURES:
            if feat == "PriceUSD":
                continue
            path = os.path.join(config.RESULT_ERROR_DIR, f"predictions_by_feat_{feat}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)

            for ts in train_sizes:
                sub_feat = df[df['train_size'] == ts]
                sub_base = df_base[df_base['train_size'] == ts]

                if sub_feat.empty or sub_base.empty:
                    continue
                if f'forecast_{model}' not in sub_feat.columns or f'forecast_{model}' not in sub_base.columns:
                    continue

                y_true = sub_feat['actual'].values
                e1 = y_true - sub_feat[f'forecast_{model}'].values
                e2 = y_true - sub_base[f'forecast_{model}'].values

                if len(e1) < 5:
                    continue

                dm_stat, p_val = dm_test(e1, e2)
                results.append({
                    'model': model,
                    'feature': feat,
                    'train_size': ts,
                    'DM_stat': round(dm_stat, 4),
                    'p_value': round(p_val, 4),
                    'significant_5pct': p_val < 0.05
                })

    df_dm = pd.DataFrame(results)
    df_dm.to_csv(os.path.join(config.RESULT_DIR, 'dm_vs_priceusd.csv'), index=False)
    return df_dm


if __name__ == "__main__":

    df_rmse = compute_rmse(MODELS)
    df_rel = compute_relative_r2(MODELS)
    df_group_rmse = compute_rmse_by_group(MODELS)
    df_group_rel = compute_relative_r2_by_group(MODELS)

    compute_dm_tests(models=MODELS, target_model='rf', baseline_model='ols')
    compute_dm_tests(models=MODELS, target_model='xgb', baseline_model='ols')
    compute_dm_tests(models=MODELS, target_model='svr', baseline_model='ols')
    compute_dm_tests(models=MODELS, target_model='mlp', baseline_model='ols')
    compute_dm_vs_priceusd(models=MODELS)
