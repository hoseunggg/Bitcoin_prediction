import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
import os
import config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config


def run_train_size_regression_diagnostics(result_path, train_sizes):
    results = []

    for train_size in train_sizes:
        file_path = os.path.join(result_path, f'train_size_{train_size}.csv')
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df = df.sort_values('timestamp')

        y = df['Actual']
        x = df['Forecast_AR']
        residuals = y - x
        X_const = sm.add_constant(x)

        bp_test_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X_const)

        dw_stat = durbin_watson(residuals)

        corr_value = residuals.corr(x)

        adf_stat, adf_pvalue, _, _, _, _ = adfuller(y)

        results.append({
            'Train_Size': train_size,
            'BreuschPagan_p': bp_pvalue,
            'DurbinWatson_stat': dw_stat,
            'Residuals_vs_Forecast_AR_corr': corr_value,
            'ADF_stat': adf_stat,
            'ADF_pvalue': adf_pvalue
        })
    return pd.DataFrame(results)

def plot_stationarity(results_df, figure_dir):
    plt.figure(figsize=(10,6))
    plt.plot(results_df['Train_Size'], results_df['ADF_pvalue'], marker='o')
    plt.axhline(0.05, color='red', linestyle='--', label='0.05 threshold')
    plt.title('Train Size vs ADF Test p-value (Stationarity) [Actual]')
    plt.xlabel('Train Size')
    plt.ylabel('ADF Test p-value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'adf_test_vs_train_size.png'))
    plt.close()

def plot_homoscedasticity(results_df, figure_dir):
    plt.figure(figsize=(10,6))
    plt.plot(results_df['Train_Size'], results_df['BreuschPagan_p'], marker='o')
    plt.axhline(0.05, color='red', linestyle='--', label='0.05 threshold')
    plt.title('Train Size vs Breusch-Pagan p-value (Homoscedasticity) [Forecast_AR]')
    plt.xlabel('Train Size')
    plt.ylabel('Breusch-Pagan p-value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'bp_test_vs_train_size.png'))
    plt.close()

def plot_independence(results_df, figure_dir):
    plt.figure(figsize=(10,6))
    plt.plot(results_df['Train_Size'], results_df['DurbinWatson_stat'], marker='o')
    plt.axhline(2.0, color='red', linestyle='--', label='Ideal = 2')
    plt.title('Train Size vs Durbin-Watson Statistic (Independence) [Forecast_AR]')
    plt.xlabel('Train Size')
    plt.ylabel('Durbin-Watson Statistic')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'dw_stat_vs_train_size.png'))
    plt.close()

def plot_exogeneity(results_df, figure_dir):
    plt.figure(figsize=(10,6))
    plt.plot(results_df['Train_Size'], results_df['Residuals_vs_Forecast_AR_corr'], marker='o')
    plt.axhline(0.0, color='red', linestyle='--', label='Ideal = 0')
    plt.title('Train Size vs Residuals and Forecast_AR Correlation (Exogeneity)')
    plt.xlabel('Train Size')
    plt.ylabel('Correlation (Residuals vs Forecast_AR)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'exogeneity_corr_vs_train_size.png'))
    plt.close()

def plot_correlation_heatmap(df):
    corr = df.corr(method='pearson')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    
    save_path = os.path.join(config.FIGURE_DIR, 'correlation_heatmap.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_price_over_time(df, price_col='PriceUSD'):
    series = df[price_col].dropna()
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values)
    plt.xlabel('Time')
    plt.ylabel(price_col)
    plt.title(f'{price_col} Over Time')
    plt.grid(True)
    plt.tight_layout()
    
    save_path = os.path.join(config.FIGURE_DIR, 'price_over_time.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_return_over_time(df, price_col='PriceUSD'):
    series = df[price_col].dropna()

    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values)
    plt.xlabel('Time')
    plt.ylabel(f'{price_col} Return')
    plt.title(f'{price_col} Return Over Time')
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(config.FIGURE_DIR, 'return_over_time.png')
    plt.savefig(save_path, dpi=300)
    plt.close()



def plot_rmse_by_feature():
    file_names = [
        'rmse_by_feat_^GSPC.csv',
        'rmse_by_feat_BlkCnt.csv',
        'rmse_by_feat_BlkSizeMeanByte.csv',
        'rmse_by_feat_DiffMean.csv',
        'rmse_by_feat_Expected_Inflation.csv',
        'rmse_by_feat_FeeMeanUSD.csv',
        'rmse_by_feat_HashRate.csv',
        'rmse_by_feat_OAS.csv',
        'rmse_by_feat_PriceUSD.csv',
        'rmse_by_feat_RevUSD.csv',
        'rmse_by_feat_Risk_Free.csv',
        'rmse_by_feat_Ten_Year.csv',
        'rmse_by_feat_Term_Spread.csv',
        'rmse_by_feat_Two_Year.csv',
        'rmse_by_feat_TxTfrValAdjUSD.csv',
        'rmse_by_feat_URTH.csv',
        'rmse_by_feat_US_News_Sentiment.csv',
        'rmse_by_feat_USD_Index.csv',
        'rmse_by_feat_VIX.csv'
    ]

    for file_name in file_names:
        file_path = os.path.join(config.RESULT_DIR, file_name)
        if not os.path.isfile(file_path):
            continue

        df = pd.read_csv(file_path)
        df = df[df['train_size'] >= 20]
        feature_name = file_name.replace('rmse_by_feat_', '').replace('.csv', '')

        plt.figure(figsize=(10, 6))
        plt.plot(df['train_size'], df['rmse_ols'], label='OLS', marker='o')
        plt.plot(df['train_size'], df['rmse_rf'], label='Random Forest', marker='s')
        plt.plot(df['train_size'], df['rmse_xgb'], label='XGBoost', marker='^')

        plt.xlabel('Train Size')
        plt.ylabel('RMSE')
        plt.title(f'RMSE by Train Size: {feature_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_name = f'rmse_by_{feature_name}.png'
        save_path = os.path.join(config.FIGURE_EXOG_DIR, save_name)
        plt.savefig(save_path, dpi=300)
        plt.close()
        
def plot_relative_r2_table(csv_path, model: str, output_path: str, figsize=(10, 8)):
    df = pd.read_csv(csv_path, index_col='feature')

    model_cols = [col for col in df.columns if f'RelR2_{model}_' in col]
    df_model = df[model_cols]

    df_model.columns = [col.replace(f'RelR2_{model}_', '') for col in df_model.columns]

    plt.figure(figsize=figsize)
    sns.heatmap(df_model, annot=True, fmt=".3f", cmap="coolwarm", linewidths=0.5, cbar=True)
    plt.title(f"Relative $R^2$ by Train Size ({model.upper()})", fontsize=14)
    plt.xlabel("Train Size")
    plt.ylabel("Feature")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {model.upper()} heatmap → {output_path}")




def generate_summary_statistics(data: pd.DataFrame, output_path: str):
    preprocess_methods = {
        "PriceUSD": "pct_change",
        "BlkCnt": "raw",
        "BlkSizeMeanByte": "pct_change",
        "DiffMean": "pct_change",
        "FeeMeanUSD": "raw",
        "HashRate": "pct_change",
        "RevUSD": "pct_change",
        "TxTfrValAdjUSD": "pct_change",
        "URTH": "pct_change",
        "^GSPC": "pct_change",
        "Risk_Free": "diff",
        "OAS": "raw",
        "Ten_Year": "pct_change",
        "Two_Year": "pct_change",
        "Term_Spread": "diff",
        "VIX": "raw",
        "USD_Index": "pct_change",
        "Expected_Inflation": "pct_change",
        "US_News_Sentiment": "raw"
    }

    variable_descriptions = {
        "PriceUSD": "Bitcoin price",
        "BlkCnt": "Daily block count",
        "BlkSizeMeanByte": "Average block size",
        "DiffMean": "Mining difficulty",
        "FeeMeanUSD": "Average transaction fee (USD)",
        "HashRate": "Hash rate",
        "RevUSD": "Miner revenue (USD)",
        "TxTfrValAdjUSD": "Adjusted transaction volume (USD)",
        "URTH": "Global ETF return",
        "^GSPC": "S&P 500 return",
        "Risk_Free": "3-month Treasury rate change",
        "OAS": "Option-Adjusted Spread",
        "Ten_Year": "10-year Treasury yield",
        "Two_Year": "2-year Treasury yield",
        "Term_Spread": "Term spread (10Y - 2Y)",
        "VIX": "Volatility Index (VIX)",
        "USD_Index": "US Dollar Index",
        "Expected_Inflation": "Expected inflation",
        "US_News_Sentiment": "US news sentiment index"
    }

    summary_rows = []

    for var, method in preprocess_methods.items():
        if var not in data.columns:
            continue  # Skip if variable not present in data

        series = data[var]

        mean = series.mean()
        std = series.std()
        skew = series.skew()
        kurt = series.kurt()
        adf_stat, adf_pvalue, *_ = adfuller(series)

        summary_rows.append({
            "Variable": var,
            "Description": variable_descriptions.get(var, ""),
            # "Preprocessing": method,
            "Mean": round(mean, 3),
            "Std": round(std, 3),
            "Skew": round(skew, 3),
            "Kurt": round(kurt, 3),
            "ADF": round(adf_pvalue, 3)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.set_index("Variable", inplace=True)
    summary_df.to_csv(output_path)

def plot_priceusd_rmse_trend(df_rmse: pd.DataFrame, save_dir=None):
    import re
    """
    PriceUSD 하나에 대해 train size 증가 시 RMSE 추이를 모델별로 시각화
    """
    price_row = df_rmse.loc['PriceUSD']
    model_ts_pattern = r'RMSE_(\w+)_ts(\d+)'

    records = []
    for col in df_rmse.columns:
        m = re.match(model_ts_pattern, col)
        if m:
            model, ts = m.group(1), int(m.group(2))
            rmse_val = price_row[col]
            records.append({'Model': model, 'TrainSize': ts, 'RMSE': rmse_val})

    df_plot = pd.DataFrame(records)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x='TrainSize', y='RMSE', hue='Model', marker='o')
    plt.title("RMSE vs Train Size (PriceUSD only)")
    plt.xlabel("Train Size")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'priceusd_rmse_trend.png')
        plt.savefig(path, dpi=300)
        print(f"[Saved] → {path}")
    else:
        plt.show()



if __name__ == "__main__":

    raw_data = pd.read_csv(config.RAW_DATA, index_col=0, parse_dates=True)
    preprocess_data = pd.read_csv(config.PREPROCESS_DATA, index_col=0, parse_dates=True)

    # (선택) 통계 요약 / 시계열 시각화
    # generate_summary_statistics(preprocess_data, os.path.join(config.FIGURE_DIR, "preprocess_data_summary.csv"))
    # plot_correlation_heatmap(raw_data)
    # plot_price_over_time(raw_data)
    # plot_return_over_time(preprocess_data)
    plot_rmse_by_feature()

    # (선택) 회귀 진단
    result_df = run_train_size_regression_diagnostics(
        result_path=config.RESULT_RMSE_DIR,
        train_sizes=[20,25,30,35,40]
    )
    plot_homoscedasticity(result_df, config.FIGURE_DIR)
    plot_independence(result_df, config.FIGURE_DIR)
    plot_exogeneity(result_df, config.FIGURE_DIR)
    plot_stationarity(result_df, config.FIGURE_DIR)

    # ✅ RMSE 테이블 로드
    rmse_path = os.path.join(config.RESULT_DIR, '_rmse_by_feature.csv')
    df_rmse = pd.read_csv(rmse_path, index_col='feature')

    # ✅ 상대 R2 시각화
    csv_file = os.path.join(config.RESULT_DIR, '_relative_r2_by_feature.csv')
    for model in ['ols','rf', 'xgb', 'svr', 'mlp']:
        plot_relative_r2_table(
            csv_path=csv_file,
            model=model,
            output_path=os.path.join(config.FIGURE_DIR, f'relative_r2_{model}.png')
        )

    # ✅ 상위 feature 시각화 (예: xgb, ts=30 기준)
    plot_priceusd_rmse_trend(df_rmse, save_dir=config.FIGURE_DIR)

