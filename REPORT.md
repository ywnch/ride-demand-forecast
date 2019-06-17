# Summary

### 2019-06-17

## Features

### Temporal

- Most features are extracted from historical demands per TAZ, and they dominant feature importances (may also due to the fact that they are mostly continuous features). Some examples are T-1 (when applicable), T-2 (when applicable), T-5, T - 1 day, T - 1 week, and there interacting terms.
- Features that are not always available should be mitigated to avoid model failure when conducting actual forecasting. For example, 40% of the T-1 are removed (set to $-1$) for training.
- Weekend (inferred), hour (as categorical), weekly (repeating timestep in 1-week cycle) also share some importances. Semantic indicators for time periods (e.g., peak hours, midnight) did not play out well and were removed.

### Spatial

[notebook](./explore_spatial_lag.ipynb)

- There are few to none explicit spatial features in the model.
- Pure coordinates (lat/lon) are not very helpful.
- KNN average does not improve the model by much, one potential reason is the cancellation effects when taking the average of all neighbors.
- There are roughly 3 to 4 hot zones in the city, applying spatial lag terms in the future may help inform contiguity information.

### Semantic

[notebook](./explore_function_segmentation.ipynb)

- Since the data is anonymized in various ways (date, time, location), typical external semantic data is not available here (e.g., demographics, public transport, weather, POI, event, road network, holidays, etc.).
- In order to resolve this, and also accommodating the lack of spatial features, TAZ functions and seasonalities are inferred by clustering temporal patterns. In other words, the clustering label indicates TAZs that share similar trends and can be modelled as similar zones to enhance the forecast.
- At least one such label, which is the unnormalized weekly-cycle timestep (``label_weekly_raw`), turns out to show high importance in the model.
- Later on, it may worth trying creating multiple models for respective zone labels. During the experiments, predicting only TAZs with the same label often has a lower RMSE.

## Model

- LGBM is light-weighted, flexible, and convenient to develop with and may do well with sufficient amount of data.
- It would be great to build a multi-view neural network later on incorporate spatial (CNN), temporal (RNN), and semantic information altogether later on. One highly relevant implementation is in this [AAAI 2018 paper](https://arxiv.org/abs/1802.08714) ([repo](https://github.com/huaxiuyao/DMVST-Net)).

## Results & Remarks

| Method (data)                                                | RMSE    |
| ------------------------------------------------------------ | ------- |
| Baseline: naive T-1                                          | 0.02580 |
| Baseline: T-1 + (T-96 - T-97)                                | 0.03473 |
| Baseline: historical average<br/>(by 672 timesteps weekly cycle per TAZ) | 0.03435 |
| LGBM (train)                                                 | 0.02525 |
| LGBM (validate)                                              | 0.02857 |
| LGBM (test)                                                  | 0.02896 |

- Various baselines are created, and the top three are reported in the first three rows.
- While the T-1 baseline is a strong one, it is not always available (e.g., when predicting T+1 to T+5).
- General speaking, there is roughly a 16% improvement for the LGBM model as compared to the historical average baseline, which is the best baseline that is directly applicable for forecasting.
- Some errors come from a mismtach during a surge. The prediction often does not catch up as much as the actual surge
- Aside from that, TAZs with small errors are not necessarily as well-predicted as one thought, while zones with large errors may not be as bad. One instance of such is that, zeros are easy to get, and in zones that have lower demands, sticking to the floor (predicting a small value) yields a small error (since the actual normalized demands are also little as well).
- Overall, the long-term performance (7-14 days) of the model looks promising and stable. But it is unclear how it will perform in a rapidly-growing city, where the overall trend is not stationary. The short-term performance (1 hour), which is actually the real challenge to respond in a short time, might not look too bad, but it may depend on the actual counts of the demand. say a TAZ has 10 demands within 15 minutes, then maybe a RMSE of 0.029 is accetable.

## Resources

### Competitions, Projects, and Repositories

- [Corporación Favorita Grocery Sales Forecasting (including preceding kernels)](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47582)
  - [LGBM Starter](https://www.kaggle.com/ceshine/lgbm-starter)
- [2018 KDD CUP of Fresh Air](https://biendata.com/competition/kdd_2018/)
- [Forecasting Uber Demand in NYC](https://medium.com/@Vishwacorp/timeseries-forecasting-uber-demand-in-nyc-54dcfcdfd1f9) ([Code](https://github.com/Vishwacorp/nyc_uber_forecasting))
- [Forecasting Lyft Demand](https://stevhliu.github.io/forecasting-lyft-demand/)
- [Utilizing ARIMA to forecast Uber's market demand](https://www.kaggle.com/kruthik93/utilizing-arima-to-forecast-uber-s-market-demand)
- [M4-methods Repo](https://github.com/M4Competition/M4-methods)

### Articles

- [Forecasting at Uber: An Introduction](https://eng.uber.com/forecasting-introduction/)
- [How to deal with the seasonality of a market?](https://eng.lyft.com/how-to-deal-with-the-seasonality-of-a-market-584cc94d6b75)
- [Identifying the numbers of AR or MA terms in an ARIMA model](http://people.duke.edu/~rnau/411arim3.htm)
- [ARIMA/SARIMA vs LSTM with Ensemble learning Insights for Time Series Data](https://towardsdatascience.com/arima-sarima-vs-lstm-with-ensemble-learning-insights-for-time-series-data-509a5d87f20a)
- [Understanding LSTM and its quick implementation in keras for sentiment analysis](https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47)
- [M4 Forecasting Competition: Introducing a New Hybrid ES-RNN Model](https://eng.uber.com/m4-forecasting-competition/)
- [Introduction to LSTM (Chinese)](https://zhuanlan.zhihu.com/p/32085405)
- [Introduction to GRU (Chinese)](https://zhuanlan.zhihu.com/p/32481747)
- [Tuning XGBoost (Chinese)](https://wuhuhu800.github.io/2018/02/28/XGboost_param_share/)

### Tutorials / Tools

- [Basic Concepts to Create Time Series Forecast](https://towardsdatascience.com/basic-principles-to-create-a-time-series-forecast-6ae002d177a4) ([Code](https://nbviewer.jupyter.org/github/leandrovrabelo/tsmodels/blob/master/notebooks/english/Basic%20Principles%20for%20Time%20Series%20Forecasting.ipynb))
- [Prophet, Facebook's automated forecasting procedure](https://facebook.github.io/prophet/)
- [Omphalos, Uber’s Parallel and Language-Extensible Time Series Backtesting Tool](https://eng.uber.com/omphalos/)

### Machine Learning Mastery

- [How to Create an ARIMA Model for Time Series Forecasting in Python](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)
- [How to Convert a Time Series to a Supervised Learning Problem in Python](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
- [Multivariate Time Series Forecasting with LSTMs in Keras](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)
- [How To Backtest Machine Learning Models for Time Series Forecasting](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/)

### Papers (*partial*)

- Yao, Huaxiu, et al. "Deep multi-view spatial-temporal network for taxi demand prediction." *Thirty-Second AAAI Conference on Artificial Intelligence*. 2018. ([repo](https://github.com/huaxiuyao/DMVST-Net))
- Vanichrujee, Ukrish, et al. "Taxi Demand Prediction using Ensemble Model Based on RNNs and XGBOOST." *2018 International Conference on Embedded Systems and Intelligent Technology & International Conference on Information and Communication Technology for Embedded Systems (ICESIT-ICICTES)*. IEEE, 2018.
- Davis, Neema, Gaurav Raina, and Krishna Jagannathan. "A multi-level clustering approach for forecasting taxi travel demand." *2016 IEEE 19th International Conference on Intelligent Transportation Systems (ITSC)*. IEEE, 2016.
- Smith, Austin W., Andrew L. Kun, and John Krumm. "Predicting taxi pickups in cities: which data sources should we use?." *Proceedings of the 2017 ACM International Joint Conference on Pervasive and Ubiquitous Computing and Proceedings of the 2017 ACM International Symposium on Wearable Computers*. ACM, 2017.
- Tong, Yongxin, et al. "The simpler the better: a unified approach to predicting original taxi demands based on large-scale online platforms." *Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining*. ACM, 2017.
- Xu, Jun, et al. "Real-time prediction of taxi demand using recurrent neural networks." *IEEE Transactions on Intelligent Transportation Systems* 19.8 (2017): 2572-2581.