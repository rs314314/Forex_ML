import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import Indicators
import OANDA_api as api


def candle_data(pair, granularity, count):
    """
    Retrieves and preprocesses data (candles) to be be used for model building

    :param pair: Forex pair that candles will be retrieved for
    :param granularity: The amount of time that each data point (candle) covers
    :param count: The number of candles that will be retrieved
    :return: candles that have been retrieved and preprocessed for use in model building
    """
    ### Initialize API connection
    sess = api.oanda_api()

    ### Use API connection to retrieve data
    candles = sess.complete_candles_df(pair, count, granularity, 'MBA')

    ### Add relevant indicators
    Indicators.relative_strength_index(candles)
    Indicators.macd(candles)
    Indicators.stochastic(candles)
    Indicators.spread(candles)
    Indicators.candle_range(candles)

    candles['stochastic_range_k'] = candles['range_14'] * candles['stochastic_k']
    candles['stochastic_range_d'] = candles['range_14'] * candles['stochastic_d']

    ### Set lags for the data
    candles['mid_c_prev'] = candles['mid_c'].shift(1)

    for lag in range(1, 11):
        candles[f'rsi_14_lag_{lag}'] = candles['rsi_14'].shift(lag)
        candles[f'MACD_12_26_9_lag_{lag}'] = candles[f'MACD_12_26_9'].shift(lag)
        candles[f'mid_o_lag_{lag}'] = candles['mid_o'].shift(lag)
        candles[f'mid_l_lag_{lag}'] = candles['mid_l'].shift(lag)
        candles[f'mid_h_lag_{lag}'] = candles['mid_h'].shift(lag)
        candles[f'stochastic_range_k_lag_{lag}'] = candles['stochastic_range_k'].shift(lag)
        candles[f'stochastic_range_d_lag_{lag}'] = candles['stochastic_range_d'].shift(lag)

    candles['long'] = False
    candles['short'] = False

    for candle in range(len(candles)):
        if (candles.loc[candle, 'bid_o'] + 0.001 <= candles.loc[candle, 'ask_h'] and
                candles.loc[candle, 'bid_o'] - 0.003 <= candles.loc[candle, 'ask_l']):
            candles.loc[candle, 'short'] = True
        if (candles.loc[candle, 'ask_o'] - 0.001 >= candles.loc[candle, 'bid_l'] and
                candles.loc[candle, 'ask_o'] + 0.003 >= candles.loc[candle, 'bid_h']):
            candles.loc[candle, 'long'] = True

    ### Standardize candle history to relate directly to mid_c_prev
    candle_data = ['mid_o_lag_1', 'mid_l_lag_1', 'mid_h_lag_1', 'mid_o_lag_2',
                   'mid_l_lag_2', 'mid_h_lag_2', 'mid_o_lag_3', 'mid_l_lag_3',
                   'mid_h_lag_3', 'mid_o_lag_4', 'mid_l_lag_4', 'mid_h_lag_4',
                   'mid_o_lag_5', 'mid_l_lag_5', 'mid_h_lag_5', 'mid_o_lag_6',
                   'mid_l_lag_6', 'mid_h_lag_6', 'mid_o_lag_7', 'mid_l_lag_7',
                   'mid_h_lag_7', 'mid_o_lag_8', 'mid_l_lag_8', 'mid_h_lag_8',
                   'mid_o_lag_9', 'mid_l_lag_9', 'mid_h_lag_9', 'mid_o_lag_10',
                   'mid_l_lag_10', 'mid_h_lag_10']

    for col in candle_data:
        candles[col] = candles['mid_c_prev'] - candles[col]

    ### Drop candles that are irrelevant to predictions and rows with null data
    candles.drop(['mid_o', 'mid_h', 'mid_l', 'mid_c', 'bid_o', 'bid_h',
                  'bid_l', 'bid_c', 'ask_o', 'ask_h', 'ask_l', 'ask_c',
                  'time', 'volume', 'rsi_14', 'MACD_12_26_9', 'stochastic_k', 'stochastic_d'],
                 axis=1, inplace=True)
    candles.dropna(inplace=True)

    return candles


def model_build(candles, long, short, model, params):
    """
    This function will take an Machine Learning algorithm and use GridSearchCV to identify the best performing model
    of that algorithm on the data passed through the function

    :param candles: Candles that are to be train (X data)
    :param long: Indicator of successful long trades (y data)
    :param short: Indicator of successful short trades (y data)
    :param model: Model object reference
    :param params: parameters to feed to models
    :return: best performing model for long trades and best performing model for short trades
    """
    grid_model_long = GridSearchCV(model(),
                                   param_grid=params,
                                   cv=5)
    grid_model_short = GridSearchCV(model(),
                                    param_grid=params,
                                    cv=5)

    grid_model_long.fit(candles, long)
    grid_model_short.fit(candles, short)

    best_model_long = {
        'model': model,
        'best_params': grid_model_long.best_params_,
        'best_score': grid_model_long.best_score_
    }

    best_model_short = {
        'model': model,
        'best_params': grid_model_short.best_params_,
        'best_score': grid_model_short.best_score_
    }

    return best_model_long, best_model_short


def model_compare(model1, model2='None'):
    """
    Takes in two models and returns the better performing model of the two

    :param model1: A dictionary containing the ML algorithm type, best parameters, and best score from
    :param model2: same as model 1 or None
    :return: the model with the higher best score of the two models passed to the function
    """
    if not model2:
        return model1

    if model1['best_score'] > model2['best_score']:
        return model1
    else:
        return model2


# ------------- Main Function Script -------------#

def main(pair, granularity):
    """

    :param pair: Forex pair that candles will be retrieved for
    :param granularity: The amount of time that each data point (candle) covers
    :return: Final results of model performance
    """
    gran_to_candle_no = {
        'M15': 280000,
        'M30': 140000,
        'H1': 70000,
        'H2': 35000,
        'H4': 17500
    }
    # Retrieve Data
    candles = candle_data(pair, granularity, gran_to_candle_no[granularity])

    # Preprocess Data and Split into train/test sets
    candle_features = candles[[col for col in candles.columns if col not in ['long', 'short']]]
    long = candles['long']
    short = candles['short']

    scaler = StandardScaler()
    scaler.fit(candle_features)
    scaled_candles = scaler.transform(candle_features)

    one_year = round(0.0876857 * gran_to_candle_no[granularity])
    x_train_candles = scaled_candles[:-one_year]
    y_train_long = long[:-one_year]
    y_train_short = short[:-one_year]

    x_test_candles = scaled_candles[-one_year:]
    y_test_long = long[-one_year:]
    y_test_short = short[-one_year:]

    y_test_long.reset_index(drop=True, inplace=True)
    y_test_short.reset_index(drop=True, inplace=True)

    # Set parameters for GridSearchCV model building
    knn_params = {
        'n_neighbors': list(range(1, 30)),
        'weights': ('uniform', 'distance')
    }

    lr_params = {
        'penalty': ['elasticnet', 'l1', 'l2'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'C': [.01, .1, 1, 10, 100],
    }

    svc_params = {
        'penalty': ['l1', 'l2'],
        'C': [.01, .1, 1, 10, 100],
    }

    dtc_params = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'min_samples_split': [2, 4, 6, 8, 10],
    }

    # Build models and select the best performing model on the training data
    models = [KNeighborsClassifier, LogisticRegression, LinearSVC, DecisionTreeClassifier]
    params = [knn_params, lr_params, svc_params, dtc_params]

    best_model_long = None
    best_model_short = None
    for model, params in zip(models, params):
        new_model_long, new_model_short = model_build(x_train_candles, y_train_long, y_train_short, model, params)
        best_model_long = model_compare(new_model_long, best_model_long)
        best_model_short = model_compare(new_model_short, best_model_short)

    # Train final model on entire test set
    final_model_long = best_model_long['model'](**best_model_long['best_params'])
    final_model_long.fit(x_train_candles, y_train_long)

    final_model_short = best_model_short['model'](**best_model_short['best_params'])
    final_model_short.fit(x_train_candles, y_train_short)

    # Evaluate model on test data
    long_test_results = pd.DataFrame(final_model_long.predict(x_test_candles), columns=['predictions'])
    long_test_results['actual'] = y_test_long
    long_takes = long_test_results.loc[long_test_results['predictions'] == True]
    long_takes_count = long_takes['actual'].count()

    short_test_results = pd.DataFrame(final_model_short.predict(x_test_candles), columns=['predictions'])
    short_test_results['actual'] = y_test_short
    short_takes = short_test_results.loc[short_test_results['predictions'] == True]
    short_takes_count = short_takes['actual'].count()

    print(f"Long Trades:\n\nModel accuracy: {long_takes['actual'].mean()} \nCount: {long_takes_count} \nRandom "
          f"accuracy: {y_test_long.mean()}")
    print(f"Short Trades: \n\nModel accuracy: {short_takes['actual'].mean()} \nCount: {short_takes_count}\nRandom "
          f"accuracy: {y_test_short.mean()}")


if __name__ == '__main__':
    main('USD_CAD', 'H2')
