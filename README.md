# Predicting Stock Price With Regression Modeling

Predicting stock movement and stock price for a company is complicated and tricky because it is reflected and impacted by multiple factors such as the country's economic situation, company’s financial performance,nonlinear stock market and unexpected circumstances etc. However, in recent ages where financial institutions  have embraced artificial intelligence as a core part of their operations, predicting stock prices through machine learning is gaining popularity for researchers as these machine based models and algorithms have the ability to learn from historical data and based upon that make future predictions. In our this class  project we have implemented various machine learning regression models to predict stock market prices during a short run (i.e. next day) by using technical analysis as our predicting methodology. We tested these models for the stocks that are consistent over the time e.g. ‘TSLA’, ‘TGT’,’GOOG’ and tickers with large up and down price swings e.g. ‘AAPL’, ‘MSFT’. Our models have shown persistent accuracy staying above 90% in making predictions for historically consistent stocks while for AAPL and similar stocks  accuracy varied between 65-78%. Our next step would be to hone into hyper-parameters and estimators to help train our models more effectively.

| TARGET-ALL MODELS PREDICTION PERFORMANCE                      | APPLE-ALL MODELS PREDICTION PERFORMANCE                |
| -----------------------------------                           | -----------------------------------                    |
| ![target](images/tgt_all_models.png)                          | ![apple](images/apple_all_models.png)                  |





## Imports

* from finta import TA
* from sklearn.preprocessing import StandardScaler
* from sklearn.model_selection import GridSearchCV
* from sklearn.ensemble import (RandomForestRegressor,GradientBoostingRegressor,StackingRegressor,VotingRegressor)
* from sklearn.linear_model import (LinearRegression,Ridge)
* from sklearn import metrics
* import xgboost as xgb
* from xgboost.sklearn import XGBRegressor
* import quantstats as qs



## Data Overview:

Historical market Data for the desired tickers was extracted from yahoo finance from Feb 2012 to Feb 2024. Models were trained in the first 10 years while the last two years were used to test our model predictions. 

### Featuring Engineering & Data Pre-processing:

As stated above, we used technical analysis to demonstrate machine based predictions. For that we calculated various technical indicators such as Simple Moving Average(SMA), Double Exponential Moving Average (DEMA), RSI, MACD and STOCH based upon our literature survey,online resources and general understanding. The windows were set as are generally used for short, medium and long term basis.

To make our features more digestible for the models and to make all variables contribute equally to the model we standardized feature columns through scikit-learn’s StandardScaler. StandardScaler removes the mean and scales each feature/variable to unit variance. This operation is performed feature-wise in an independent way. However, it can be influenced by outliers (if they exist in the dataset) since it involves the estimation of the empirical mean and standard deviation of each feature(1).


# Model Implementation

## GRIDSEARCHCV

Grid Search is a hyperparameter technique used in  machine-learning to fine tune our algorithms. Grid Search performs multiple computations on the combination of  hyperparameters that are available on every machine learning algorithm and enables us to configure the optimal parameters for a given model.  This function comes in the SK-learn model selection package. Apart from helping us achieve better results, Grid Search gives us an evaluated metric system, which gives us better insights into our machine-learning model(2).

| GRIDSEARCHCV                        |
| ----------------------------------- |                          
| ![grid](images/gridsearchcv.png)    | 

###
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

## Random Forest

A Random Forest (RF) model utilizes multiple decision trees to help train a model to predict a single output. These models can be used to predict both categorical variables (classification) and continuous variables (regression). This project utilizes regression to try to predict stock prices. RF models use a bagging technique to split the training data into multiple subsets, these subsets are then independently pushed through decision trees. The model then uses majority voting to combine each subsets decision tree result into a single output.

###
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

## Gradient_Boost

Gradient Boost models utilize 'Boosting' which is a way of combining many different simple models into a single complete model. These simple models are often refered to as weak learners because on their own they are not very powerful, but when combined they can create a very complete and capable model. the SK learn gradient boosting regression model uses regression descision trees combined with a calculated loss function to make decisions.

###
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

## XGradient_Boost

Similar to the SK Learn Gradient Boost, Extreme Gradient Boost uses descision trees and loss functions to help train the model, the difference is that Extreme Gradient Boost creates the decision trees in parallel and sequentially at the same time. Additionally, Extreme Gradient Boost uses L1 & L2 to regularize the data. Regularization penalizes outlyers and adjusts input weights to help reduce overfitting, generally it minimizes input feature complexity.

###
https://xgboost.readthedocs.io/en/latest/install.html

## Linear_Regression

Linear Regression models use one or multiple independent variables, or the variables used to make the prediction, to try to accurately predict the dependent variable based on their linear relationship. This model from SK Learn uses Ordinary Least Squared Linear Regression, more specifically the model attempts to minimize the total sum of the independent variables squared errors to the dependent variable.

###
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

## Ridge

Similar to the standard Linear Regression model, Ridge attempts to minimize the sum of the independent variables total squared error compared to the dependent variable, it differs in its use of l2 regularization. (see XGradient_Boost) 

###
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html


# Model Ensembling 

Ensemble learning is a machine learning technique that enhances accuracy, strength and predictive performance by aggregating  predictions from multiple models.Three main classes of ensemble learning methods are bagging, stacking, and boosting. This project accomplished stacking Regressor and  Voting Regressor ensemblers from SK-learn.ensembles and created Random Forest Regressor, XGBoostRegressor, GradientBoost Regressor, Linear Regression and Ridge Regression models for stacking up.

As voting and stacking combines the performance of multiple models that helps to reduce the hidden biased parameters and misclassification of models at large. It also mitigates the risk of inaccurate predictions by having other models that can backup the weak learners. However combining multiple models is computation intensive and their training and implementation can increase cost significantly. Additionally, if the dataset is linear data then voting techniques would undermine the accuracy predictions performed by linear regression models.   


## StackingRegressor

![stacking1](images/stacking_regressor.png)

StackingRegressor is fitting many different models on the same data and then making the best predictions through a final estimating model called meta learner. Meta learner attempts to minimize the weekness and maximize teh strength of every individual model that helps to generlize the model on an unseen data set very well as compared to an individual model. 

| Understanding Stacking Ensembling   |
| ----------------------------------- |  
|![stacking2](images/stacking.png)    | 

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html



## VotingRegressor

![voting_reg](images/voting_regressor.png)

Scikit-Learn provides a VotingRegressor() class that allows us to use voting ensemble learning methods. For regression, voting ensemble involves making a prediction that is average of other multiple regression models. Whereas in classification modeling it involves hard and soft voting strategies to determine the final predictor.  Hard voting is choosing the class that receives the majority of votes as the final predictor while soft voting takes the weighted average of the predicted probabilities to make the final prediction.


| Understanding Voting Ensembling            | 
| -----------------------------------        | 
|![voting_reg1](images/voting_class.png)     |

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html

## Validation of Training & Testing Data     

|Optimized Model Performance--TARGET                       | Optimized Model Performance--Apple
| ---------------------------------------------            | -----------------------------------  
|![validation1](images/validation_tgt.png)                 |![validation2](images/validation-apple.png)


## Summary

|All Model Testing Performance--TARGET                     | All Model Testing Performance--Apple
| ---------------------------------------------            | -----------------------------------  
|![pred1](images/results_tgt.png)                          |![pred2](images/results_apple.png)



References:
1. https://towardsdatascience.com/how-and-why-to-standardize-your-data-996926c2c832#:~:text=StandardScaler%20removes%20the%20mean%20and,standard%20deviation%20of%20each%20feature.
https://builtin.com/machine-learning/ensemble-model

2. https://www.scaler.com/topics/machine-learning/grid-search-in-machine-learning/

3. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html

4.  https://builtin.com/machine-learning/ensemble-model

5.  https://www.kaggle.com/code/jayatou/xgbregressor-with-gridsearchcv

6. https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/

7. https://towardsdatascience.com/combining-tree-based-models-with-a-linear-baseline-model-to-improve-extrapolation-c100bd448628#:~:text=In%20our%20case%20we%20will,the%20go%2Dto%2Dmodel.

8. https://www.nvidia.com/en-us/glossary/xgboost/




