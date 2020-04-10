'''
Created by Jake Fredrich
Last Updated 4/10/2020
The focus of this project is to project revenue for several Countries 1 Quarter into the future.  Exponential Smoothing, Cross Validation with Time Series, and ARIMA modeling will be utilized dynamically to do so.

Referenced - Topic 9 Part 1. Time series analysis in Python. found on Kaggle
https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python/data
'''

import warnings  # do not disturb mode
import sklearn

warnings.filterwarnings('ignore')

import numpy as np  # vectors and matrices
import pandas as pd  # tables and data manipulation
import matplotlib.pyplot as plt  # plots
import seaborn as sns  # additional plots
from math import sqrt
from dateutil.relativedelta import relativedelta  # working with dates and style
from scipy.optimize import minimize  # for function minimization
import statsmodels.formula.api as smf  # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from itertools import product  # useful functions



def main():
    """
    :return: Main program for Revenue Time Series Prediction
    """
    # import relevant data
    Rev = pd.read_csv('Country_Revenue.csv', index_col=['Date'], usecols=["Date", "Revenue"], parse_dates=['Date'])

    #plot Rev
    plt.figure(figsize=(15, 7))
    plt.plot(Rev.Revenue)
    plt.title('Japan - Revenue')
    plt.grid(True)

    ''' Vizualize values and trends '''
    plotMovingAverage(Rev, 3, plot_intervals=True, plot_anomalies=True) # smoothe out time series to identify trends. Slight upward trend, increase in 4Q may be present

    plotExponentialSmoothing(Rev.Revenue, [0.6, 0.3, 0.05]) # model is weighted average between current true value and the previous model values.  The smaller α is, the more influence the previous observations have and the smoother the series is.

    plotDoubleExponentialSmoothing(Rev.Revenue, alphas=[0.5], betas=[0.05]) # Applies exponential smoothing to the trend, as well as the intercept.  alpha: responsible for the series smoothing around the trend. beta: responsible for smothing the trend itself.

    '''TripleExponentialSmoothing - Holt Winters w/ Time Series Cross Validation'''
    # Adds a third component, seasonality.  Avoid method if time series does not have seasonal trend.
    data = Rev.Revenue[:-6] # remove data for testing
    slen = 3  # 3 month(Quarterly) seasonality


    x = [0, 0, 0]  # alpha beta gamma list
    # Optimize using cross-validation on a rolling basis & modifying loss function
    opt = minimize(timeseriesCVscore, x0=x,
                   args=(data, mean_squared_log_error, slen),
                   method="TNC", bounds=((0, 1), (0, 1), (0, 1))
                   )

    alpha_final, beta_final, gamma_final = opt.x # store optimal values for model creation
    print(alpha_final, beta_final, gamma_final)

    model = HoltWinters(data, slen=slen,
                        alpha=alpha_final,
                        beta=beta_final,
                        gamma=gamma_final,
                        n_preds=12, scaling_factor=3)

    model.triple_exponential_smoothing() # train and fit model, forecasting 12 months into the future

    plotHoltWinters(model, Rev.Revenue)


    ''' STATIONARY CHECK & DICKEY-FULLER TEST '''
    # Dickey-Fuller test - tests the null hypothesis that a unit root is present in an autoregressive model.
    # Demonstrate the null hypothesis that the time series(white noise) is non-stationary.  It's rejected by rho = 0, .6, .9, and accepted by 1 as it demonstrates a random walk
    # Note that if we can get a stationary series from a non-stationary series using the first difference, as we demonstrate here, we call those series integrated of order 1.
    #white_noise = np.random.normal(size=1000)
    #with plt.style.context('bmh'):
        #plt.figure(figsize=(15, 5))
        #plt.plot(white_noise)
        #plt.title("Appears Stationary")

    #for rho in [0, 0.6, 0.9, 1]:
        #plotProcess(rho=rho)



    ''' HANDLE NON-STATIONARITY W/ SARIMA '''
    # ACF(Auto Correlation Function) is a visual way to show serial correlation in time series.
    # chart of coefficients of correlation between a time series and lags of itself
    # the AR in ARIMA
    # PACF(Partial Auto Correlation)
    # Helps to analyze the MA or Moving Average portion of ARIMA

    tsplot(Rev.Revenue, lags=20) # Plot ACF & PACF to determine


    Rev_diff = Rev.Revenue - Rev.Revenue.shift(3)  # remove seasonality. Much better with no seasonality, but notice the autocorrelation has too many significant lags
    tsplot(Rev_diff[3:], lags=20)

    Rev_diff = Rev_diff - Rev_diff.shift(1)  # remove lags - Questionable, revisit
    tsplot(Rev_diff[3+1:], lags=20)

    ''' ARIMA MODEL '''
    # setting initial values and some bounds for them
    ps = range(2, 5)
    d = 1 # number of differences
    qs = range(2, 5)
    Ps = range(0, 2)
    D = 1
    Qs = range(0, 2)
    s = 3  # season length is still 3

    # creating list with all the possible combinations of parameters
    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)

    #result_table = optimizeSARIMA(Rev, parameters_list, d, D, s) # Determine optimal combination - (4,2,1,1) with AIC of 899.7659
    #print(result_table.head())

    ## set the parameters that give the lowest AIC
    # p, q, P, Q = result_table.parameters[0]
    p, q, P, Q = 4,2,1,1
    best_model = sm.tsa.statespace.SARIMAX(Rev.Revenue, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)
    print(best_model.summary())

    tsplot(best_model.resid[3+1:], lags=20)  # plot the residuals of the model

    plotSARIMA(s, d, Rev, best_model, 3)
    plt.show()

    ''' ANALYSIS
    We get very adequate predictions. Our model was wrong by 5.5% on average, which is very good. 
    However, there are additional tests and hypertuning to be completed.  Overall costs of preparing data, making the series stationary, and selecting parameters will need to be considered for production.
    '''








''' ACCURACY & Loss Functions '''
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


# Root Mean Squared Error[RMSE]:
def root_mean_squared_error(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))


# Mean Absolute Percentage Error[MAPE]: same as MAE, but computed as a percentage
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


''' TRANSFORMATIONS, TRANSFER FUNCTIONS, SMOOTHING '''

def moving_average(series, n):
    """
    :define:  Moving Average - assumption that the future value of our variable depends on the average of its k previous values
    :param series: dataframe with timestamps
    :param n: number of previous values to average
    :return: average of last n observations, predicts one observation in the future
    """
    return np.average(series[-n:])


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

    rolling_mean = series.rolling(
        window=window).mean()  # smoothes the original series to identify trends. Same as moving_average function defined

    plt.figure(figsize=(15, 5))
    plt.title("Moving average/n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")

    # Having the intervals, find abnormal values
    if plot_anomalies:
        anomalies = pd.DataFrame(index=series.index, columns=series.columns)
        anomalies[series < lower_bond] = series[series < lower_bond]
        anomalies[series > upper_bond] = series[series > upper_bond]
        plt.plot(anomalies, "ro", markersize=10)

    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)



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
        result += series.iloc[-n - 1] * weights[n]
    return float(result)



def exponential_smoothing(series, alpha):
    """
    :define: Exponential smoothing weights all of the observations, while exponentially decreasing the weights as we move further back in time.
    :define2: Exponentiality is hidden in the resuriveness of the function: y-hat = a * y-not + (1-a) * (previous y-not)
    :param series: dataframe with time stamps
    :param alpha: float [0.0, 1.0], smoothing parameter. The smaller alpha is, the more influence the previous observations have, and the smoother the series is
    :return: exponentially smoothed dataframe, predicts one observation in the future
    """
    result = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result


def plotExponentialSmoothing(series, alphas):
    """
    :param series: dataset with timestamps
    :param alphas: list of floats, smoothing parameters. The smaller alpha is, the more influence the previous observations have, and the smoother the series is
    :return: plot of exponentially smoothed dataframe, predicts one observation in the future
    """
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(17, 7))
        for alpha in alphas:
            plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
        plt.plot(series.values, "c", label="Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True);


def double_exponential_smoothing(series, alpha, beta):
    """
    :define: In the exponential_smoothing method we predict the intercept(level).  Now we will apply the same smoothing to the trend by assuming that the future direction of the
            series changes depends on the previous weighted changes
    :define2: The larger alpha and beta, the more weight the most recent observations will have and the less smoothed the model series will be
    :param series: dataset with timestamps
    :param alpha: float [0.0, 1.0], smoothing parameter for level. Responsible for the series smoothing around the trend
    :param beta: float [0.0, 1.0], smoothing parameter for trend. A weight for the exponential smoothing. Responsible for smoothing the trend itself
    :return: sum of the model values of the intercept and trend, a prediction 2 observations in the future
    """
    result = [series[0]]
    for n in range(1, len(series) + 1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series):  # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result


def plotDoubleExponentialSmoothing(series, alphas, betas):
    """
    :define: In the exponential_smoothing method we predict the intercept(level).  Now we will apply the same smoothing to the trend by assuming that the future direction of the
            series changes depends on the previous weighted changes
    :define2: The larger alpha and beta, the more weight the most recent observations will have and the less smoothed the model series will be
    :param series:  dataset with timestamps
    :param alphas: float [0.0, 1.0], smoothing parameter for level. Responsible for the series smoothing around the trend
    :param betas:  float [0.0, 1.0], smoothing parameter for trend. A weight for the exponential smoothing. Responsible for smoothing the trend itself
    :return: A plot of double exponential smoothing.  Sum of the model values of the intercept and trend, a prediction 2 observations in the future
    """
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(20, 8))
        for alpha in alphas:
            for beta in betas:
                plt.plot(double_exponential_smoothing(series, alpha, beta),
                         label="Alpha {}, beta {}".format(alpha, beta))
        plt.plot(series.values, label="Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Double Exponential Smoothing")
        plt.grid(True)


class HoltWinters:
    """
    Holt-Winters model with the anomalies detection using Brutlag method
    # series - initial time series
    # slen - length of season
    # alpha, beta, gamma - Holt-Winters model coefficients
    # n_preds - predictions horizon
    # scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)
    """

    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor

    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i + self.slen] - self.series[i]) / self.slen
        return sum / self.slen

    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series) / self.slen)
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen * j:self.slen * j + self.slen]) / float(self.slen))
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen * j + i] - season_averages[j]
            seasonals[i] = sum_of_vals_over_avg / n_seasons
        return seasonals

    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []

        seasonals = self.initial_seasonal_components()

        for i in range(len(self.series) + self.n_preds):
            if i == 0:  # components initialization
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i % self.slen])

                self.PredictedDeviation.append(0)

                self.UpperBond.append(self.result[0] +
                                      self.scaling_factor *
                                      self.PredictedDeviation[0])

                self.LowerBond.append(self.result[0] -
                                      self.scaling_factor *
                                      self.PredictedDeviation[0])
                continue

            if i >= len(self.series):  # predicting
                m = i - len(self.series) + 1
                self.result.append((smooth + m * trend) + seasonals[i % self.slen])

                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1] * 1.01)

            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha * (val - seasonals[i % self.slen]) + (1 - self.alpha) * (
                        smooth + trend)
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = self.gamma * (val - smooth) + (1 - self.gamma) * seasonals[i % self.slen]
                self.result.append(smooth + trend + seasonals[i % self.slen])

                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i])
                                               + (1 - self.gamma) * self.PredictedDeviation[-1])

            self.UpperBond.append(self.result[-1] +
                                  self.scaling_factor *
                                  self.PredictedDeviation[-1])

            self.LowerBond.append(self.result[-1] -
                                  self.scaling_factor *
                                  self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i % self.slen])


from sklearn.model_selection import TimeSeriesSplit


''' CROSS VALIDATION - TIME SERIES '''

def timeseriesCVscore(params, series, loss_function=mean_squared_error, slen=24):
    """
    :define: Since time series predictions depend on the linear time of a dataset, we cannot use standard cross validation.  Rather than taking random folds,
            we take small segments of the time series from the beginning until some t, make preditions for the next t+n steps, and calculate an error.  Then
            we expand our training sample to T+n value, make preditions from t+n until t+2*n, and continue moving our test segment of the time series until
            we hit the last available observation.
            https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection
    :param params: vector of parameters for optimization
    :param series: dataset with timeseries
    :param loss_function:
    :param slen: season length for Holt-Winters model
    :return: Cross Validation Score for Time Series.
    """
    # errors array
    errors = []

    values = series.values
    alpha, beta, gamma = params

    # set the number of folds for cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    # iterating over folds, train model on each, forecast and calculate error
    for train, test in tscv.split(values):
        model = HoltWinters(series=values[train], slen=slen,
                            alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        model.triple_exponential_smoothing()

        predictions = model.result[-len(test):]
        actual = values[test]
        error = loss_function(predictions, actual)
        errors.append(error)

    return np.mean(np.array(errors))


def plotHoltWinters(model, series, plot_intervals=False, plot_anomalies=False):
    """
    :param model: HoltWinters class object
    :param series: dataset with timestamps
    :param plot_intervals: show confidence intervals
    :param plot_anomalies: show anomalies
    :return: Plot of HoltWinters model
    """

    plt.figure(figsize=(20, 10))
    plt.plot(model.result, label="Model")
    plt.plot(series.values, label="Actual")
    error = mean_absolute_percentage_error(series.values, model.result[:len(series)])
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))

    if plot_anomalies:
        anomalies = np.array([np.NaN] * len(series))
        anomalies[series.values < model.LowerBond[:len(series)]] = \
            series.values[series.values < model.LowerBond[:len(series)]]
        anomalies[series.values > model.UpperBond[:len(series)]] = \
            series.values[series.values > model.UpperBond[:len(series)]]
        plt.plot(anomalies, "o", markersize=10, label="Anomalies")

    if plot_intervals:
        plt.plot(model.UpperBond, "r--", alpha=0.5, label="Up/Low confidence")
        plt.plot(model.LowerBond, "r--", alpha=0.5)
        plt.fill_between(x=range(0, len(model.result)), y1=model.UpperBond,
                         y2=model.LowerBond, alpha=0.2, color="grey")

    plt.vlines(len(series), ymin=min(model.LowerBond), ymax=max(model.UpperBond), linestyles='dashed')
    plt.axvspan(len(series) - 20, len(model.result), alpha=0.3, color='lightgrey')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc="best", fontsize=13);




def plotProcess(n_samples=1000, rho=0):
    x = w = np.random.normal(size=n_samples)
    for t in range(n_samples):
        x[t] = rho * x[t - 1] + w[t]

    with plt.style.context('bmh'):
        plt.figure(figsize=(10, 3))
        plt.plot(x)
        plt.title("Rho {}\n Dickey-Fuller p-value: {}".format(rho, round(sm.tsa.stattools.adfuller(x)[1], 3)))



def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    '''
    :define: Plot time series, its ACF and PACF, calculate Dickey–Fuller test.  The goal is to have values that oscillate around zero, a mean and variance that don't change
    :param y: timeseries
    :param lags: lags - how many lags to include in ACF, PACF calculation
    :param figsize: plot parameters
    :param style: plot parameters
    :return: plot of time series, its ACF and PACF, calculated Dickey-Fuller Test
    '''
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()



def optimizeSARIMA(data, parameters_list, d, D, s):
    """
    :param data: Time Series dataframe
    :param parameters_list: list with (p, q, P, Q) tuples
    :param d: integration order in ARIMA model
    :param D: seasonal integration order
    :param s: length of season
    :return: results table that shows the optimized parameters for an SARIMA using our dataset
    """

    results = []
    best_aic = float("inf")

    for param in parameters_list:
        # we need try-except because on some combinations model fails to converge
        try:
            model = sm.tsa.statespace.SARIMAX(data.Revenue, order=(param[0], d, param[1]),
                                              seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

    return result_table



def plotSARIMA(s, d, series, model, n_steps):
    """
    :define:
    :param s: seasonality value
    :param d: difference value
    :param series: Dataframe time series
    :param model: fitted SARIMA model
    :param n_steps: number of steps to predict into the future
    :return: plot SARIMA prodiction
    """
    # adding model values
    data = series.copy()
    data.columns = ['actual']
    data['arima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['arima_model'][:s + d] = np.NaN

    # forecasting on n_steps forward
    forecast = model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
    forecast = data.arima_model.append(forecast)
    # calculate error, again having shifted on s+d steps from the beginning
    error = mean_absolute_percentage_error(data['actual'][s + d:], data['arima_model'][s + d:])

    plt.figure(figsize=(15, 7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True);


main()
