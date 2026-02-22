## 2. 연구 동기
주식 시장과 마찬가지로, 비트코인 역시 (뉴스, 시장 심리, 글로벌 거시경제 변수)등에 의해 영향을 받는다. 우리는 이 논문에서 왜 비트코인이 linearity를 가정하면 안 되는지에 대해 서술하고자 한다.

linear regression을 가정하기 위해서, 우리는 기본적으로 5가지 가정을 한다. 
1. (선형성) 2.(다중공선성) 3.(외생성) 4.(자기상관성) 5.(등분산성)
주가 데이터가 과연 이 5가지 조건을 엄밀히 만족하는지부터 살펴보자.

우선, 선형성 가정부터 검토해야 한다. 비트코인의 가격 움직임은 시장의 불확실성, 투자자 심리, 외부 충격에 매우 민감하게 반응하며, 이러한 반응은 종종 비선형적인 특성을 보인다. 이는 가격의 급등락, 변동성 클러스터링(volatility clustering) 현상 등으로 관찰할 수 있다.

다중공선성의 경우, 비트코인 가격에 영향을 미치는 다양한 설명변수들(예: 글로벌 금리, 달러 인덱스, 주식시장 지수 등)이 서로 강한 상관관계를 가질 가능성이 존재한다. 이는 회귀 모형의 안정성과 해석 가능성을 저하시킬 수 있다.

외생성 가정 또한 쉽게 만족되지 않는다. 시장 참여자들의 기대나 예측이 시장 변수에 영향을 주면서 피드백 루프(feedback loop)를 형성할 수 있기 때문이다.

자기상관성 문제는 고빈도(high-frequency) 데이터에서 특히 심각하게 나타날 수 있으며, 과거 가격 정보가 미래 가격 변동에 영향을 줄 가능성을 시사한다.

마지막으로, 등분산성 가정 역시 현실적으로 성립하기 어렵다. 비트코인 수익률은 특정 시기(예: 금융시장 위기나 규제 뉴스 발표 시점)에는 변동성이 급격히 증가하는 특성을 보인다.

이와 같이, 비트코인 데이터는 전통적인 선형 회귀 모형의 기본 가정들을 위배할 가능성이 높다. 따라서 비트코인의 수익률이나 가격을 예측할 때에는 비선형성(nonlinearity)을 허용하는 모델을 고려해야 하며, 본 논문에서는 이러한 접근의 필요성을 구체적으로 논증하고자 한다.

## 3. Data
본 연구는 Bitcoin price와 다양한 exogenous variables를 활용하여 Bitcoin의 return을 설명하고자 한다.
이때 사용된 exogenous variables는 크게 on-chain data, macro variables, market variables의 세 가지 범주로 구분된다.

첫째, on-chain data에는 Bitcoin 블록체인 시스템의 활동과 관련된 변수들이 포함된다. 예를 들어, 하루 동안 생성된 블록 수를 의미하는 BkCount는 네트워크의 activity 또는 transaction 증가와 연관되며, raw 형태로 분석에 사용되었다. BkSize는 평균 block size로서, network congestion이나 transaction fee 변동을 반영한다. DiffM은 mining difficulty를 나타내며, 상승은 일반적으로 hash rate 증가와 관련된다. FeeUSD는 평균 transaction fee로, extreme values로 유지하였다. 이외에도 Hash_Rate, RevUSD, TxValAdjUSD는 각각 network security 수준, miner의 revenue, adjusted transaction volume을 의미하며, 시계열의 변동성을 반영하기 위해 수익률로 변환되었다.

둘째, market variables는 Bitcoin이 전통적인 금융 자산들과 어떤 관계를 가지는지를 파악하는 데 초점을 둔다. URTH와 GSPC는 각각 글로벌 ETF와 S&P500 index의 return을 나타내며, Bitcoin의 risk asset 또는 diversification asset으로서의 특성을 분석하는 데 사용된다. RiskFree는 3-month US Treasury rate의 변화(diff)를 사용하였고, OAS (Option-Adjusted Spread)는 credit spread를 나타내는 핵심 지표로서 raw 데이터로 분석되었다. OAS는 ADF 테스트 기준 약한 정상성(p = 0.0096)을 보인다. 또한, TenYear와 TwoYear는 각각 10-year 및 2-year US Treasury rate의 return이며, 이들의 차이인 Term_Spread는 차분(diff) 형태로 처리되었다.

셋째, macro variables는 글로벌 경제 환경과 market sentiment를 반영하는 변수들로 구성된다. VIX는 S&P500 기반의 volatility index로, 시장 불확실성을 나타내는 대표적인 지표이며 raw 형태로 사용되었다. USDIndex는 미국 달러의 상대적인 강세/약세를 나타내며 수익률로 변환되었고, ExpInflation은 5-year expected inflation의 return으로 인플레이션 전망과 Bitcoin의 hedge 가능성을 분석하는 데 사용되었다. 마지막으로 USNewsSent는 미국 경제와 관련된 news sentiment를 측정한 지표로, soft information을 반영하며 raw 형태로 분석에 포함되었다.

이와 같이, 다양한 차원의 exogenous variables를 고려함으로써 본 연구는 Bitcoin 시장의 반응성과 구조적 특성을 정량적으로 분석하고자 한다.

## 4. Methodology

이 연구는 Bitcoin의 return을 예측하는 것을 목표로 하며, 가격 변수만을 사용할 때와 exogenous variables를 사용할 때 각각 linearity와 nonlinearity의 차이를 비교하고자 한다.
다음 날의 가격 또는 수익률을 예측하는 방식으로 진행되었다.  
본 연구에서는 AR, ARIMA, Random Forest (RF), XGBoost 방법을 사용하여 성능을 비교한다.

### 4.1 AR 모델

Autoregression (AR) 모델은 대표적인 linear regression model로, 과거 자기 자신의 값을 이용하여 미래 값을 예측하는 방식이다. 이 모델은 다음과 같이 정의된다:

$$
X_{t+1} = \alpha_0 + \sum_{i=1}^{p} \alpha_i X_{i,t} + \varepsilon_{t+1}
$$

여기서 $p = 6$을 사용하였다.(p는 Paye (2012)를 참고하여 설정.)

또한 단일 변수를 사용하는 AR 모델 외에도, 여러 개의 exogenous variables를 포함하는 ARX (Autoregressive with Exogenous Variables) 모델을 적용하였다. ARX 모델은 다음과 같이 정의된다:

$$
X_{t+1} = \alpha_0 + \sum_{i=1}^{p} \alpha_i X_{i,t} + \sum_{i=1}^{q} \beta_i Y_{i,t} + \varepsilon_{t+1}
$$

여기서 $Y_t$는 exogenous variables들의 집합이며, $q$는 변수의 수를 나타낸다.  
(exogenous variables는 contemporaneous term만 반영하고, lag term은 주지 않았다.)

### 4.2 ARIMA 모델

ARIMA는 autoregression을 기반으로 differencing과 moving average 요소를 결합한 linear regression model이다. 시계열 데이터가 비정상(non-stationary)일 경우, 차분을 통해 정상성(stationarity)을 확보한 후 모델을 적용한다.  


ARIMA 모델은 다음과 같은 수식으로 표현된다:

$$
\Delta X_{t+1} = \alpha_0 + \sum_{i=1}^{p} \alpha_i \Delta X_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j} + \varepsilon_{t+1}
$$

여기서 $\Delta X_{t} = X_t - X_{t-1}$는 1차 차분값을 의미하고, $p$는 AR 차수, $q$는 MA 차수를 의미한다.

AR 모델과 마찬가지로, ARIMA에도 exogenous variables를 추가한 확장형인 ARIMAX (ARIMA with Exogenous Variables) 모델을 적용하였다. ARIMAX는 다음과 같이 표현된다:

$$
\Delta X_{t+1} = \alpha_0 + \sum_{i=1}^{p} \alpha_i \Delta X_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j} + \sum_{k=1}^{r} \gamma_k Y_{k,t} + \varepsilon_{t+1}
$$

여기서 $Y_{k,t}$는 $k$번째 exogenous variable이며, $r$은 전체 변수 수를 의미한다.  
(ARIMAX에서도 exogenous variables는 contemporaneous term만 사용했다.)



#### 4.2.2 비선형 모델

- **Random Forest Regressor**
- **XGBoost Regressor**
- **LightGBM Regressor**

비선형 모델은 다수의 결정 트리를 이용하여 복잡한 비선형 관계를 포착하며, 예측식은 다음과 같이 나타낼 수 있다:

$$
\hat{y}_{t+1} = \frac{1}{T} \sum_{j=1}^{T} f_j(X_t)
$$

여기서 $T$는 트리의 총 수, $f_j$는 $j$번째 트리를 의미한다.

### 4.3 변수 중요도 해석

비선형 모델에서는 예측 결과에 대한 변수 기여도를 분석하기 위해 SHAP (SHapley Additive exPlanations) 값을 이용하였다. SHAP 값은 다음과 같은 선형 가산 모형으로 표현된다:

$$
f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i
$$

여기서 $\phi_0$는 베이스 값, $\phi_i$는 각 변수 $i$의 기여도를 의미한다.

이를 통해 외생 변수들이 비트코인 가격 예측에 미치는 상대적 중요성을 평가하였다.

### 4.4 평가 지표

모델 성능은 다음과 같은 지표를 통해 평가하였다:

- **RMSE (Root Mean Squared Error)**: 평균 제곱 오차의 제곱근
- **MAPE (Mean Absolute Percentage Error)**: 백분율 기준 절대 오차
- **R² (결정계수)**: 설명력
- **MDA (Mean Directional Accuracy)**: 상승/하락 방향성 예측 정확도

RMSE는 다음과 같이 정의된다:

$$
\text{RMSE} = \sqrt{ \frac{1}{N} \sum_{t=1}^{N} (y_t - \hat{y}_t)^2 }
$$

MAPE는 다음과 같이 정의된다:

$$
\text{MAPE} = \frac{100}{N} \sum_{t=1}^{N} \left| \frac{y_t - \hat{y}_t}{y_t} \right|
$$

R²는 다음과 같이 정의된다:

$$
R^2 = 1 - \frac{ \sum_{t=1}^{N} (y_t - \hat{y}_t)^2 }{ \sum_{t=1}^{N} (y_t - \bar{y})^2 }
$$

여기서 $\mathbf{1}(\cdot)$은 인디케이터 함수이다.

### 4.1 ar model

모든 모델 학습은 시계열 순서를 고려한 **Rolling Forecasting Window** 방식을 사용하였다. 구체적으로, 30일간의 학습 데이터를 이용하여 다음 1일을 예측하는 구조를 반복하였다. 학습 및 검증 단계에서는 다음의 과정을 따른다:

- 학습 구간: 30일
- 예측 구간: 다음 1일
- 이동 간격: 1일
- Time lag: 6일

하이퍼파라미터 튜닝은 시계열 특성을 반영하기 위해 TimeSeriesSplit을 사용하고, 각 모델의 최적 파라미터는 Grid Search를 통해 탐색하였다.

