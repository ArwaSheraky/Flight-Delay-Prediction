# Flight Delay Prediction App
A simple application to predict expected flight delays, using the published dataset on Kaggle: https://www.kaggle.com/usdot/flight-delays

The dataset summarizes US airline flight delay and cancellation information as collected and published by the DOT's Bureau of Transportation Statistic.

Based on the pre-processed data set. the performance of a Gradient Boosting classifier was identified as being the most reliable prediction model with the lowest root mean square error.Identifying the most important features further improved the model by focusing on the important variables and removing x-variables that were deemed insignificant.

A web application was created to then deploy the model via Heroku:
https://predict-flight-delay.herokuapp.com/
