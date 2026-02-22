---
marp: true
theme: default
paginate: true
mermaid: true
size: 16:9
style: |
    section {
        position: relative;
        padding-top: 60px;

        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: flex-start;
        text-align: left;

        font-size: 22px;  /* 본문 기본 글자 크기 축소 */
    }

    section.title {
      background-color: #f9f9f9;
      color: #333;
      text-align: center;
      justify-content: center;
      align-items: center;
    }    

    /* 상단 텍스트 + 하단 선 */
    section::before {
        content: "SungKunKwan University";
        position: absolute;
        top: 20px;
        left: 60px;
        font-size: 16px;  /* 이전 20px → 16px */
        color: #888;
        width: calc(100% - 120px);
        padding-bottom: 12px;
        border-bottom: 1px solid #ccc;
    }

    section::after {
        content: "Finance Lab";
        position: absolute;
        bottom: 10px;
        right: 60px;
        font-size: 16px;  /* 이전 20px → 16px */
        color: #888;
        width: calc(100% - 120px);
        border-top: 1px solid #ccc;
        text-align: right;
        display: block;
    }

    /* 제목 크기 축소 */
    h1 {
        padding-top: 100px;
        font-size: 36px;  /* 이전 36px */
    }
    h2 {
        padding-top: 50px;  
        font-size: 26px;  /* 이전 28px */
        margin-bottom: 8px;
        text-align: left;
    }
    h3 {
        font-size: 22px;  /* 이전 22px */
        text-align: left;
    }

    /* 표 크기 축소 */
    table {
        font-size: 9px;  /* 이전 18px */
        text-align: left;
    }

---

# Finance Lab

## Hoseung

1. Factors Influencing Bitcoin Volatility  


---

## 1. Factors Influencing Bitcoin Volatility

기존의 연구에서  investigates which factors influence Bitcoin volatility.  하려고 했으나, 비트코인의 특성상 변동성 1달 단위의 결과가 정확하지는 않다는 문제와 단기 데이터 (1일일 단위 ) 조사하는 것이 더 의미가 있을 것이라는 피드백을 받음.

발생되는 문제점.

일일 단위로 진행할 경우, 사용할 수 있는 시간 빈도가 적어서 변수 수가 굉장히 줄어들어
경제적 의미 부여에 문제가 가해짐

---

## 2. 연구 의의
1. 선형성이 기존 싸ㄹ

---

- In this study, we try to investigate further by:
  1. We use **macroeconomic variables**, **uncertainty indicators**, and **on-chain data** to predict Bitcoin volatility.  
  2. We use three methods to see how the variables matter: **dimension reduction**, **shrinkage**, and **forecast combination**.  
  3. We check whether **on-chain data** actually helps improve prediction accuracy.

<br>

![diagram](diagram.png)

---

###### 1. Factors Influencing Bitcoin Volatility

| Variable     | Mean      | Std     | Skewness | Kurtosis | ADF Statistic | p-value | Formula                                                                                     |
|--------------|-----------|---------|----------|----------|---------------|---------|---------------------------------------------------------------------------------------------|
| PriceUSD     | 0.0022    | 0.0406  | -0.1060  | 6.2599   | -32.8852      | 0.0000  | $\Delta PriceUSD_t = \frac{PriceUSD_t - PriceUSD_{t-1}}{PriceUSD_{t-1}}$              |
| BkCount      | 147.8582  | 16.5530 | -0.3723  | 1.1361   | -6.4035       | 0.0000  | $BkCount_t$                                                                                 |
| BkSize       | 0.0101    | 0.1437  | 1.1260   | 4.2925   | -6.6801       | 0.0000  | $\Delta BkSize_t = \frac{BkSize_t - BkSize_{t-1}}{BkSize_{t-1}}$                              |
| DiffM        | 0.0023    | 0.0171  | 1.1096   | 42.8772  | -7.9966       | 0.0000  | $\Delta DiffM_t = \frac{DiffM_t - DiffM_{t-1}}{DiffM_{t-1}}$                                 |
| FeeUSD       | 3.8228    | 6.6907  | 4.0090   | 20.2875  | -3.3593       | 0.0124  | $FeeUSD_t$                                                                                  |
| Hash_Rate    | 0.0097    | 0.1261  | 0.7776   | 2.5203   | -26.0803      | 0.0000  | $\Delta Hash\_Rate_t = \frac{Hash\_Rate_t - Hash\_Rate_{t-1}}{Hash\_Rate_{t-1}}$               |
| RevUSD       | 0.0090    | 0.1294  | 0.8198   | 2.8512   | -37.6867      | 0.0000  | $\Delta RevUSD_t = \frac{RevUSD_t - RevUSD_{t-1}}{RevUSD_{t-1}}$                              |
| TxValAdjUSD  | 0.0660    | 0.4274  | 2.8765   | 17.7569  | -6.8575       | 0.0000  | $\Delta TxValAdjUSD_t = \frac{TxValAdjUSD_t - TxValAdjUSD_{t-1}}{TxValAdjUSD_{t-1}}$           |
| URTH         | 0.0003    | 0.0099  | -0.9888  | 24.6577  | -9.4896       | 0.0000  | $\Delta URTH_t = \frac{URTH_t - URTH_{t-1}}{URTH_{t-1}}$                                     |
| GSPC         | 0.0003    | 0.0105  | -0.6172  | 21.9839  | -9.7565       | 0.0000  | $\Delta GSPC_t = \frac{GSPC_t - GSPC_{t-1}}{GSPC_{t-1}}$                                     |
| RiskFree     | 0.0019    | 0.0275  | 1.9750   | 27.2993  | -6.2836       | 0.0000  | $\Delta RiskFree_t = RiskFree_t - RiskFree_{t-1}$                                            |
| OAS          | 4.1297    | 1.0263  | 2.5562   | 8.7330   | -3.4424       | 0.0096  | $OAS_t$                                                                                     |
| TenYear      | 0.0006    | 0.0286  | 1.6946   | 44.9141  | -8.0629       | 0.0000  | $\Delta TenYear_t = \frac{TenYear_t - TenYear_{t-1}}{TenYear_{t-1}}$                         |
| TwoYear      | 0.0017    | 0.0472  | 0.9434   | 13.1693  | -24.9820      | 0.0000  | $\Delta TwoYear_t = \frac{TwoYear_t - TwoYear_{t-1}}{TwoYear_{t-1}}$                         |
| TermSpread   | -0.0010   | 0.0279  | -0.0938  | 4.5636   | -20.9747      | 0.0000  | $\Delta TermSpread_t = TermSpread_t - TermSpread_{t-1}$                                      |
| VIX          | 19.5261   | 8.4034  | 2.0822   | 8.4596   | -4.1856       | 0.0007  | $VIX_t$                                                                                     |
| USDIndex     | 0.0000    | 0.0026  | 0.2479   | 5.3907   | -10.5806      | 0.0000  | $\Delta USDIndex_t = \frac{USDIndex_t - USDIndex_{t-1}}{USDIndex_{t-1}}$                      |
| ExpInflation | 0.0003    | 0.0198  | 5.7912   | 210.3612 | -12.2697      | 0.0000  | $\Delta ExpInflation_t = \frac{ExpInflation_t - ExpInflation_{t-1}}{ExpInflation_{t-1}}$      |
| USNewsSent   | -0.0432   | 0.2005  | -0.8955  | 1.1627   | -2.8680       | 0.0492  | $USNewsSent_t$                                                                              |


---

## Table 2. 비트코인 및 온체인 관련 변수

| Variable            | Description                |   Mean  |   Std   | Skewness | Kurtosis | ADF Statistic | p-value |                   Method                                |
|---------------------|----------------------------|---------|---------|----------|----------|---------------|---------|---------------------------------------------------------|
| $PriceUSD$              | 비트코인 가격 데이터         |  0.0750 | 0.1207  |  3.7607  | 16.8707  |    -3.0742    | 0.0285  | $PriceUSD= \frac{P_t - P_{t-1}}{P_{t-1}}$ $ *(raw)*                                  |
| $BlkCnt_t$          | 블록 수                    |  0.0006 | 0.0676  |  0.2930  |  2.8695  |    -9.3044    | 0.0000  | $BlkCnt^*_t = \frac{BlkCnt_t - BlkCnt_{t-1}}{BlkCnt_{t-1}}$|
| $BlkSizeMeanByte_t$ | 평균 블록 크기             |  0.0835 | 0.3547  |  5.2976  | 35.9950  |    -3.2639    | 0.0166  | $BlkSizeMeanByte^*_t = \frac{BlkSizeMeanByte_t - BlkSizeMeanByte_{t-1}}{BlkSizeMeanByte_{t-1}}$ |
| $DiffMean$          | 채굴 난이도 변화율         |  0.2173 | 0.3977  |  3.3081  | 14.0645  |    -5.0170    | 0.0000  | $DiffMean^*_t = \frac{DiffMean_t - DiffMean_{t-1}}{DiffMean_{t-1}}$  |
| $FeeMeanUSD_t$      | 평균 수수료 (USD)          |  2.0971 | 4.7288  |  3.9141  | 16.3942  |    -4.4360    | 0.0003  | $FeeMeanUSD^*_t = FeeMeanUSD_t$                         |
| $HashRate_t$        | 해시레이트                 | 12.9285 | 6.8952  | -0.9563  | -0.3590  |    -3.0545    | 0.0301  | $HashRate^*_t = \log(HashRate_t)$                       |
| $RevUSD_t$          | 채굴 수익 (USD)            |  0.1493 | 0.5775  |  5.1544  | 38.7227  |    -3.4560    | 0.0092  | $RevUSD^*_t = \frac{RevUSD_t - RevUSD_{t-1}}{RevUSD_{t-1}}$ |
| $TxTfrValAdjUSD_t$  | 전송 가치 (조정 USD)       | 19.2244 | 3.6022  | -1.1429  |  0.9096  |    -3.8246    | 0.0027  | $TxTfrValAdjUSD^*_t = \log(TxTfrValAdjUSD_t)$           |

---

## Table 3. 불확실성 관련 변수

| Variable       | Description                    |   Mean  |   Std   | Skewness | Kurtosis | ADF Statistic | p-value |                   Method                                |
|----------------|--------------------------------|---------|---------|----------|----------|---------------|---------|---------------------------------------------------------|
| $epu_t$       | 경제정책불확실성               |163.0879 | 68.7451 |  1.9708  |  5.6225  |    -3.0918    | 0.0272  | $epu^*_t = epu_t$ *(raw)*                                |
| $gpr_t$       | 지정학적 리스크 지수            | 99.0736 | 30.6651 |  3.1347  | 16.7183  |    -5.5758    | 0.0000  | $gpr^*_t = gpr_t$ *(raw)*                                |
| $emv_t$       | 경제 불확실성 (전염 포함)        | 19.7362 |  7.4229 |  2.8026  | 11.9767  |    -7.2350    | 0.0000  | $emv^*_t = emv_t$ *(raw)*                                |
| $mpu_t$       | 금융시장 불확실성              |  0.3829 | 49.8651 |  0.4980  |  3.6499  |    -8.7547    | 0.0000  | $mpu^*_t = mpu_t - mpu_{t-1}$                           |
| $svar_t$      | 구조적 VAR 불확실성            |  0.0025 |  0.0062 |  9.9622  |111.2505  |    -9.9387    | 0.0000  | $svar^*_t = svar_t$ *(raw)*                              |
| $rsk\_sp_t$   | 주식 리스크 프리미엄            |  0.1755 |  0.9501 |  0.0992  | -0.4500  |   -13.6329    | 0.0000  | $rsk\_sp^*_t = rsk\_sp_t$ *(raw)*                        |
| $ra_t$        | 위험 회피 성향 지수             |  2.9657 |  0.4854 |  3.0138  | 12.2235  |    -4.2774    | 0.0005  | $ra^*_t = ra_t$ *(raw)*                                  |
| $eu_t$        | 유럽 불확실성 지수             |  0.0000 |  0.0000 |  3.0183  | 21.9269  |    -4.5135    | 0.0002  | $eu^*_t = eu_t$ *(raw)*                                  |
| $ru_t$        | 러시아 불확실성               |  0.0004 |  0.0315 |  2.5341  | 19.7076  |    -2.9416    | 0.0407  | $ru^*_t = ru_t - ru_{t-1}$                              |
| $mu_t$        | 시장 변동성 지수              |  0.0002 |  0.0261 |  2.8399  | 22.2064  |    -6.0805    | 0.0000  | $mu^*_t = mu_t - mu_{t-1}$                              |
| $fu_t$        | 금융 불안 지수                | -0.0003 |  0.0342 |  0.2641  |  2.7643  |    -7.0988    | 0.0000  | $fu^*_t = fu_t - fu_{t-1}$                              |
| $ovx_t$       | 옵션 기반 원유 변동성 지수      | 37.3749 | 16.4137 |  3.9133  | 26.8131  |    -5.7148    | 0.0000  | $ovx^*_t = ovx_t$ *(raw)*                                |



---
## 1. AR  & AR-X Models

- **Autoregression Model**  
  This is the baseline model that predicts volatility using past values of realized volatility:  
  $$
  RV_{t+1} = \alpha_0 + \sum_{i=1}^{6} \alpha_i RV_{t+1-i} + \varepsilon_{t+1}
  $$

- **AR-X Model**  
  This model extends the AR model by including exogenous predictors $X_{i,t}$, such as macroeconomic and on-chain variables:  
  $$
  RV_{t+1} = \alpha_0 + \sum_{i=1}^{6} \alpha_i RV_{t+1-i} + \sum_{i=1}^{N} \beta_i X_{i,t} + \varepsilon_{t+1}
  $$

### **I followed the approach used in Paye (2012), where the model with 6 lags showed the best forecasting performance.**

---

## 2. Dimension Reduction Techniques

- **Principal Component Analysis (PCA):**  
  Extracts common factors from all predictors.  
  $$
  RV_{t+1} = \alpha_0 + \sum_{i=1}^{6} \alpha_i RV_{t+1-i} + \sum_{k=1}^{K} \beta_k F_{PCA,k,t} + \varepsilon_{t+1}
  $$

  ### **I tested different values of K from 0 to the maximum number of components.**


- **Partial Least Squares (PLS):**  
  Constructs a diffusion index from predictors for improved forecast
  $$RV_{t+1} = \alpha_0 +\sum_{i=1}^{6} \alpha_i RV_{t+1-i}+ \beta_{pls} F_{PLS,t} + \varepsilon_{t+1}$$

---

## 3. Shrinkage Methods
### LASSO Regression: **Forecasting crude oil market volatility: A comprehensive look at uncertainty variables**

$$
\hat{\alpha}, \hat{\beta} = \arg\min \left\{
\frac{1}{2(T-1)} \sum_{t=1}^{T-1} \left( RV_{t+1} - \alpha_0 - \sum_{i=1}^{6} \alpha_i RV_{t+1-i} - \sum_{j=1}^{N} \beta_j X_{j,t} \right)^2
+ \lambda \left( \sum_{i=1}^{6} |\alpha_i| + \sum_{j=1}^{N} |\beta_j| \right)
\right\}
$$


$$
\widehat{RV}_{t+1} = \hat{\alpha}_0 + \sum_{i=1}^{6} \hat{\alpha}_i RV_{t+1-i} + \sum_{j=1}^{N} \hat{\beta}_j X_{j,t}
$$

### LASSO Regression: **Which factors drive Bitcoin volatility: Macroeconomic, technical, or both?:**

$$
\hat{\beta} = \arg\min_{\beta} \left(
\frac{1}{2(T - 1)} \sum_{t=1}^{T - 1} \left(RV_{t+1} - \beta_0 - \sum_{i=1}^{K} \beta_i X_{i,t}\right)^2
+ \lambda \sum_{i=1}^{K} |\beta_i|
\right)
$$

$$
\widehat{RV}_{t+1} = \hat{\beta}_0 + \sum_{i=1}^{K} \hat{\beta}_i X_{i,t}
$$


---
- **Elastic Net (EN):**

  $$
  \hat{\alpha},\hat{\beta}= \operatorname*{arg\,min}\left\{
  \frac{1}{2(T-1)} \sum_{t=1}^{T-1} \left( RV_{t+1} - \alpha_0 - \sum_{i=1}^{6} \alpha_i RV_{t+1-i} - \sum_{j=1}^{N} \beta_j X_{j,t} \right)^2
  + \lambda \left[
  \rho \left( \sum_{i=1}^{6} |\alpha_i| + \sum_{j=1}^{N} |\beta_j| \right)
  + \frac{1}{2}(1 - \rho) \left( \sum_{i=1}^{6} \alpha_i^2 + \sum_{j=1}^{N} \beta_j^2 \right)
  \right]
  \right\}
  $$

---

## 4. Forecast Combination

- **Mean Forecast:**



- **Trimmed Mean Forecast:**


- **DMSPE-Weighted Forecast** (with $\alpha = 0.9$): 

---

## Evaluation
1. I use out-of-sample predictions using rolling window method.
2. 

---

## Outcome

| **Model**   | **Macro R²** | **Macro MSE** | **Macro OOS** | **Network R²** | **Network MSE** | **Network OOS** | **Mixed R²** | **Mixed MSE** | **Mixed OOS** |
|-------------|--------------|----------------|----------------|----------------|------------------|------------------|--------------|----------------|----------------|
| **AR**          | 0.191330     | 0.010433       | NaN           | 0.191330       | 0.010433         | NaN             | 0.191330     | 0.010433       | NaN            |
| **AR_PCA**      | 0.203110     | 0.010281       | 0.014567      | 0.235295       | 0.009866         | 0.054366        | 0.201218     | 0.010306       | 0.012227       |
| **AR_PLS**      | 0.259418     | 0.009555       | 0.084197      | 0.282616       | 0.009256         | 0.112884        | 0.312898     | 0.008865       | 0.150330       |
| **AR_LASSO**    | 0.277078     | 0.009327       | 0.106035      | 0.283453       | 0.009245         | 0.113919        | 0.404206     | 0.007687       | 0.263242       |
| **AR_EN**       | 0.277078     | 0.009327       | 0.106035      | 0.283453       | 0.009245         | 0.113919        | 0.405594     | 0.007669       | 0.264958       |
| **AR_MEAN**     | 0.260266     | 0.009544       | 0.085246      | 0.273571       | 0.009372         | 0.101699        | 0.349027     | 0.008399       | 0.195008       |
| **AR_TRIMMED**  | 0.263679     | 0.009500       | 0.089466      | 0.277851       | 0.009317         | 0.106991        | 0.351341     | 0.008369       | 0.197869       |
| **AR_DMSPE**    | 0.261746     | 0.009525       | 0.087076      | 0.274975       | 0.009354         | 0.103435        | 0.356758     | 0.008299       | 0.204568       |





