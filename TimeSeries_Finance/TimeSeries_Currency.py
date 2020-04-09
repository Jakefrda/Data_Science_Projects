'''
Created by Jake Fredrich
Last Updated 4/9/2020
Referenced - Topic 9 Part 1. Time series analysis in Python. found on Kaggle
https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python/data
'''

import warnings                                     # do not disturb mode

import sklearn

warnings.filterwarnings('ignore')

import numpy as np                                  # vectors and matrices
import pandas as pd                                 # tables and data manipulation
import matplotlib.pyplot as plt                     # plots
import seaborn as sns                               # additional plots

from dateutil.relativedelta import relativedelta    # working with dates and style
from scipy.optimize import minimize                 # for function minimization

import statsmodels.formula.api as smf               # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                       # useful functions
from tqdm import tqdm_notebook

#Kaggle
ads = pd.read_csv('ads.csv', index_col=['Time'], parse_dates=['Time'])
ads_anomaly = ads.copy() # make anomaly for later
ads_anomaly.iloc[-20] = ads_anomaly.iloc[-20] * 0.2 # say we have 80% drop of ads
#currency = pd.read_csv('currency.csv', index_col=['Time'], parse_dates=['Time'])

#plt.figure(figsize=(15,7))
#plt.plot(ads.Ads)
#plt.title('Ads watched (hourly data)')
#plt.grid(True)
#plt.show()

#plt.figure(figsize=(15,7))
#plt.plot(currency.GEMS_GEMS_SPENT)
#plt.title('In-Game currency spent (Daily data)')
#plt.grid(True)
#plt.show()



# import relevant data
RevCC = pd.read_csv('Japan_CC.csv', index_col=['Date'], usecols=["Date", "Revenue_CC"], parse_dates=['Date'])
Rev = pd.read_csv('Japan_CC.csv', index_col=['Date'], usecols=["Date", "Revenue"], parse_dates=['Date'])

# plot Rev CC
plt.figure(figsize=(15,7))
plt.plot(RevCC.Revenue_CC)
plt.title('Japan - Revenue_CC')
plt.grid(True)
#plt.show()

# plot Rev
plt.figure(figsize=(15,7))
plt.plot(Rev.Revenue)
plt.title('Japan - Revenue')
plt.grid(True)
#plt.show()

''' Forecast Quality Metrics '''
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error


# Rsquared: coefficient of determination - percentage of variance explained by the model
sklearn.metrics.r2_score

# Mean Absolute Error[MAE]: uses same unit of measurement as initial series
sklearn.metrics.mean_absolute_error

# Median Absolute Error[MedAE]: also easy to interperet as it uses the same unit of measurement as initial series
sklearn.metrics.median_absolute_error

# Mean Squared Error[MSE]: commonly used, scales the penalty error for error size
sklearn.metrics.mean_squared_error

# Mean Squared Logarithmic Error[MSLE]: used frequently when data has exponential trends.  Same as MSE, but we take the logarithm of the series as a result, giving more weight to small mistakes
sklearn.metrics.mean_squared_log_error

# Mean Absolute Percentage Error[MAPE]: same as MAE, but computed as a percentage
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

''' TRANSFORMATIONS, TRANSFER FUNCTIONS, SMOOTHING '''

def moving_average(series,n):
    """
    :define:  Moving Average - assumption that the future value of our variable depends on the average of its k previous values
    :param series: dataframe with timestamps
    :param n: number of previous values to average
    :return: average of last n observations, predicts one observation in the future
    """
    return np.average(series[-n:])

#print(moving_average(RevCC,1)) # preiction for the last observed month

def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):
    """
    :define: smoothe the original time series to identify trends.  Helps detect common patterns for noisy data
    :param series: dataframe with timeseries
    :param window: rolling window size - The number of observations used for calculating the statistic
    :param plot_intervals: show confidence intervals
    :param scale:
    :param plot_anomalies: show anomalies
    :return: Plot the time series with the Moving Average trend, predicts one observation in the future
    """

    rolling_mean = series.rolling(window=window).mean() # smoothes the original series to identify trends. Same as moving_average function defined

    plt.figure(figsize=(15,5))
    plt.title("Moving average/n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label = "Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")

    # Having the intervals, find abnormal values
    if plot_anomalies:
        anomalies = pd.DataFrame(index=series.index, columns=series.columns)
        anomalies[series<lower_bond] = series[series < lower_bond]
        anomalies[series>upper_bond] = series[series > upper_bond]
        plt.plot(anomalies, "ro", markersize=10)

    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)

#plotMovingAverage(RevCC,3, plot_intervals=True, plot_anomalies=True) # note much better for Revenue CC as Revenue has much more seasonal trend

def weighted_average(series, weights):
    """
    :define: Weighted average is a modification to the moving average.  The weights sum up to 1, so that larger weights are assigned to more last recent observations
    :param series: dataframe with time series
    :param weights: list of weighted buckets that add up to 1. ex: [0.6, 0.3, 0.1]
    :return: return the weighted_average of a time series, predicts one observation in the future
    """
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += series.iloc[-n-1]*weights[n]
    return float(result)

#print(weighted_average(RevCC, [0.6, 0.3, 0.1]))


def exponential_smoothing(series, alpha):
    """
    :define: Exponential smoothing weights all of the observations, while exponentially decreasing the weights as we move further back in time.
    :define2: Exponentiality is hidden in the resuriveness of the function: y-hat = a * y-not + (1-a) * (previous y-not)
    :param series: dataframe with time stamps
    :param alpha: float [0.0, 1.0], smoothing parameter. The smaller alpha is, the more influence the previous observations have, and the smoother the series is
    :return: exponentially smoothed dataframe, predicts one observation in the future
    """
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1-alpha) * result[n-1])
    return result

def plotExponentialSmoothing(series, alphas):
    """
    :param series: dataset with timestamps
    :param alphas: list of floats, smoothing parameters. The smaller alpha is, the more influence the previous observations have, and the smoother the series is
    :return: plot of exponentially smoothed dataframe, predicts one observation in the future
    """
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(17,7))
        for alpha in alphas:
            plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
        plt.plot(series.values, "c", label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True);

#plotExponentialSmoothing(RevCC.Revenue_CC, [0.6, 0.3, 0.05])
#plotExponentialSmoothing(Rev.Revenue, [0.6, 0.3, 0.05])


def double_exponential_smoothing(series, alpha, beta):
    """
    :define:
    :param series: dataset with timestamps
    :param alpha: float [0.0, 1.0], smoothing parameter for level
    :param beta: float [0.0, 1.0], smoothing parameter for trend
    :return: sum of the model values of the intercept and trend, a prediction 2 observations in the future
    """
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n>= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result

def plotDoubleExponentialSmoothing(series, alphas, betas):
    """

    :param series:
    :param alphas:
    :param betas:
    :return:
    """




plt.show()


print("Complete")
