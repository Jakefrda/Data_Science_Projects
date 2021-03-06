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
    This function iterates through each country listed in the currency.csv file, models using Holt Winters, SARIMA, and Linear Regression,
    then selects the model based on MAPE.  The final output of this model is printed to a CSV called final_output.
    """

    # import relevant data
    Revenue = pd.read_csv('Countries_CC.csv', index_col=['Date'], usecols=["Country", "Date", "Revenue"],
                          parse_dates=['Date'])

    iterator = Revenue["Country"].unique()  # create iterator, the unique identifier is Country

    test_size = 0.2  # percentage of dataset to withhold for testing
    minimum_records = 27  # number of records required.  For my example, I require 27 because I only have 27 record per cntry
    hp_int = int(minimum_records * test_size)  # hold out period integer based on the test_size determined
    n_preds = hp_int  # forecast n periods.  Setting equal to the hold-out period for now

    # set up output dataframe to for final Output
    output = Revenue[
        Revenue["Country"] == iterator[0]].reset_index()  # create dataframe to store temporary iteration results
    output = output.drop(columns=['Country', 'Revenue'])  # drop all columns but Date
    extension = output.copy()  # Create date dataframe for to-be offset
    extension = extension.head(n_preds)  # prepare extension for append, taking the first n records
    output.Date = output.Date + pd.DateOffset(months=n_preds)  # offset output by n_preds periods
    output = extension.append(output)  # append the outputand extension
    output["Model"] = np.nan  # create null Model field to store model selected
    output = output.set_index('Date')  # set index to date
    final_output = pd.DataFrame(
        columns=["Model", "y_hats", "Revenue", "Country"])  # Create final_output with same fields

    # iterate through each portion of the original dataframe using the iterator as the
    for i in iterator:
        print(i)

        # Rev = Revenue[Revenue["Country"]==i].drop(columns=['Country']) # create dataframe of specific iterator data
        Rev = Revenue[Revenue["Country"] == "JAPAN"].drop(columns=['Country'])

        if len(Rev) < minimum_records:
            print("Error: " + i + " does not contain enough records for modeling.")
            break  # breaks from For loop

        ''' TRAINING - HOLT WINTERS '''
        '''TripleExponentialSmoothing - Holt Winters w/ Time Series Cross Validation'''
        ''' The Holt-Winters modeling applies smoothing to the intercept(level), to the trend by assuming that the future direction of the
            series changes depends on the previous weighted changes, and by seasonality. '''

        training_data = Rev.Revenue[:-hp_int]  # Remove the hold out period
        testing_data = Rev.Revenue[-hp_int:]
        slen = 3  # Set the seasonality length - with this dataset, we will assume 3 months(quarterly) seasonality

        x = [0, 0, 0]  # alpha beta gamma list, preset to 0
        # Optimize using cross-validation on a rolling basis & using mean squared log error as the loss function.
        # truncated Newton (TNC) algorithm is selected for the minimizing function
        opt = minimize(timeseriesCVscore, x0=x,
                       args=(training_data, mean_squared_log_error, slen),
                       method="TNC", bounds=((0, 1), (0, 1), (0, 1))
                       )

        alpha_final, beta_final, gamma_final = opt.x  # store optimal values for model creation
        print(alpha_final, beta_final, gamma_final)

        # create Holtwinters model on training set
        model = HoltWinters(training_data, slen=slen,
                            alpha=alpha_final,
                            beta=beta_final,
                            gamma=gamma_final,
                            n_preds=n_preds, scaling_factor=3)

        model.triple_exponential_smoothing()  # fit model, forecasting with the training set months into the future
        plotHoltWinters(n_preds, model, Rev.Revenue)  # plot the trained model with the actual data
        plt.show()
        # calculate the hw_error_hp - the MAPE on the Holdout Period
        hw_error_hp = mean_absolute_percentage_error(testing_data, model.result[-n_preds:])

        # calculate the hw_error - the MAPE on all actuals(training and testing)
        hw_error = mean_absolute_percentage_error(Rev.Revenue.values, model.result[:len(Rev.Revenue)])

        print()

        ''' TRAINING - SARIMA MODEL '''
        ''' SARIMA(p,d,q)(P,D,Q,s) Seasonal Autoregression Moving Average model.

            AR(p) - autoregression model, regression of the time series onto itself.  The basic assumption is that the 
            current series values depend on its previous values with some lag(or several lags).  The maximum lag in
            the model is referred to as p.  In the main_exploration() function we analyzed the PACF(Partial
            AutoCorrelation Function) plot to find the biggest significant lag after which most other lags become
            insignificant.

            MA(q) - moving average model.  This models the error of the time series, again with the assumption that the
            current error depends on the previous with some lag, q.  The initial value can be found on the ACF(Auto
            Correlation Function) plot, allowing us to find the biggest significant prior lag after which most other
            lags become insignificant.

            ARMA(p,q) - Autoregressive-moving-average model.  If the series is already stationary, this model can be
            used for approximation.

            I(d) - order of integration. This is the number of nonseasonal differences needed to make the series
            stationary.  We utilized the Dickey-Fuller test to determine that our series sample required 1,
             we used first differences

            ARIMA(p,d,q) model - can handle non-stationary data with the help of nonseasonal differences.

            S(s) - this is responsible for the seasonality and equals the season period length of the series.
                (P,D,Q) are the parameters for determining seasonality
                P - order of autoregression for the seasonal component of the model, which can be derived from PACF.
                    To determine, look at the number of significant lags, which are multiples of the season period length.
                    For example, if there period equals 24 and we see the 24th and 48th lags are significant in PACF,
                    our P=2.
                Q - Similar logic using the ACF plot. Remember, the ACF plot is looking at lags multiple periods
                    behind.
                D - order of seasonal integration.  This can be equal to 1 or 0, depending on whether seasonal
                    differences were applied or not.

           SARIMA(p,d,q)(P,D,Q,s) Seasonal Autoregression Moving Average model.'''

        training_data = training_data.to_frame()
        testing_data = testing_data.to_frame()

        # setting initial values and some bounds for them
        ps = range(2, 5)  # AR(p) - The maximum lag in the model found on the PACF plot
        d = 1  # number of differences for the order of integration
        qs = range(2, 5)  # The final significant lag found on the ACF plot
        Ps = range(0, 2)  # Order of autoregression for the seasonal component of the model, derived from PACF
        D = 1  # Order of seasonal integration.  Seasonal differences are applied
        Qs = range(0, 2)  # Order of autoregression for the seasonal component of the model, dervied from ACF
        s = slen  # season length is still 3
        #  n_preds = 3  # forecast periods for ARIMA model

        # creating list with all the possible combinations of parameters
        parameters = product(ps, qs, Ps, Qs)  # multiplies each range to determine all possible combinations
        parameters_list = list(parameters)

        result_table = optimizeSARIMA(training_data.Revenue, parameters_list, d, D,
                                      s)  # Determine optimal combination -  AIC is the minimization function

        p, q, P, Q = result_table.parameters[0]  # set the parameters that give the lowest AIC

        best_model = sm.tsa.statespace.SARIMAX(training_data.Revenue, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(
            disp=-1)

        # timeseriesCVscore_sarima()

        sarima_model = plotSARIMA(s, d, training_data, best_model, Rev.Revenue,
                                  n_preds)  # plots SARIMA Model and returns numpy array of results

        sarima_results = sarima_model.tolist()  # remove dates so that error can be calculated
        sarima_nulls = sum(isnan(x) for x in
                           sarima_results)  # number of nulls to remove, as model results were shifted due to differentiating

        sarima_error = mean_absolute_percentage_error(Rev.Revenue.values[sarima_nulls:], sarima_results[
                                                                                         sarima_nulls:len(
                                                                                             Rev.Revenue)])  #
        # calculate SARIMA mape error

        sarima_error_hp = mean_absolute_percentage_error(testing_data.Revenue.values, sarima_results[-hp_int:])  #
        # calculate SARIMA mape error of test set

        ''' LINEAR REGRESSION '''
        ''' Description of Linear Regression

        '''
        scaler = StandardScaler()
        tscv = TimeSeriesSplit(n_splits=3)  # for time-series cross-validation set 4 folds.

        # Prepare data creates Lag features, month_of_quarter, and monthly average features
        X_train, X_test, y_train, y_test = prepareData(Rev.Revenue, lag_start=2, lag_end=6, test_size=test_size,
                                                       target_encoding=True)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        prediction = lr.predict(X_test_scaled)
        lr_error = mean_absolute_percentage_error(prediction, y_test)

        plotModelResults(i + " Linear Regression - ", lr, X_train=X_train_scaled, X_test=X_test_scaled, y_train=y_train,
                         y_test=y_test, tscv=tscv, plot_intervals=True)
        plotCoefficients(lr, X_train)
        plt.show()

        ''' SELECT MODEL AND ADD FORECAST TO FINAL OUTPUT '''
        ''' ERROR is calculated as MAPE(Mean Absolute Percentage of Error) on the Training set, which is defined
            as test_size which indicates the percentage to be used on the hold out period'''
        error_dict = {
            "hw": hw_error_hp,
            "sarima": sarima_error_hp,
            "lr": lr_error
        }
        print(i + " HW Error - Mean Absolute Percentage Error: {0:.2f}%".format(hw_error_hp))
        print(i + " SARIMA Error - Mean Absolute Percentage Error: {0:.2f}%".format(sarima_error_hp))
        print(i + " Linear Regression Error - Mean Absolute Percentage Error: {0:.2f}%".format(lr_error))
        temp = min(error_dict.values())
        res = [key for key in error_dict if error_dict[key] == temp]

        if res[0] == "hw":
            # create Final Holtwinters model using all data
            model = HoltWinters(Rev.Revenue, slen=slen,
                                alpha=alpha_final,
                                beta=beta_final,
                                gamma=gamma_final,
                                n_preds=(n_preds), scaling_factor=3)

            model.triple_exponential_smoothing()  # fit Final Model

            temp_list = model.result  # create list of HW model results
            t = [0.0] * (len(output.index) - len(
                model.result))  # create empty list the length of the difference betwee HW results and output index
            temp_list.extend(t)  # extend list so that it is the same size as output
            hw_result = np.asarray(temp_list)  # send to array
            y_hats_df = pd.DataFrame(data=hw_result, columns=['y_hats'],
                                     index=output.index.copy())  # Create dataframe with predicted values from HW
            df_out = pd.merge(output, y_hats_df, how='left', left_index=True,
                              right_index=True)  # Merge predicted values with output dataframe containing dates
            df_out = pd.merge(df_out, Rev, how='left', left_index=True, right_index=True)  # Merge actual values
            df_out['Country'] = i  # Store the iterator into Country Column
            df_out['Model'] = "Holt Winters"
            print()

        elif res[0] == "sarima":

            best_model = sm.tsa.statespace.SARIMAX(Rev.Revenue, order=(p, d, q),
                                                   seasonal_order=(P, D, Q, s)).fit(disp=-1)
            sarima_forecast = best_model.predict(start=Rev.shape[0], end=Rev.shape[0] + (n_preds - 1))
            sarima_forecast = Rev.Revenue.append(sarima_forecast)
            sarima_results = sarima_forecast.to_numpy()

            t = [0.0] * (len(output.index) - len(sarima_results))
            sarima_results = np.append(sarima_results, t)
            y_hats_df = pd.DataFrame(data=sarima_results, columns=['y_hats'],
                                     index=output.index.copy())  # Create dataframe with predicted values from HW
            df_out = pd.merge(output, y_hats_df, how='left', left_index=True,
                              right_index=True)  # Merge predicted values with output dataframe containing dates
            df_out = pd.merge(df_out, Rev, how='left', left_index=True, right_index=True)  # Merge actual values
            df_out['Country'] = i  # Store the iterator into Country Column
            df_out['Model'] = "SARIMA"

        elif res[0] == "lr":
            y_hats_df = pd.DataFrame(data=prediction, columns=['y_hats'],
                                     index=X_test.index.copy())  # Create dataframe with predicted values from LR
            df_out = pd.merge(output, y_hats_df, how='left', left_index=True,
                              right_index=True)  # Merge predicted values with output dataframe containing dates
            df_out = pd.merge(df_out, Rev, how='left', left_index=True, right_index=True)  # Merge actual values
            df_out['Country'] = i  # Store the iterator into Country Column
            df_out['Model'] = "Linear Regression"

        if len(final_output.index) == 0:
            final_output = df_out.copy()
        else:
            final_output = final_output.append(df_out)  # append df_out to final output

    print(final_output.head())
    final_output.to_csv('final_output.csv')





def main_exploration():
    """
    :return: Main program for Revenue Time Series Prediction Exploration
    This function is used to explore more deeply individual modeling methods.  The main() method will utilize much of the logic found in exploration
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

    ''' FEATURE EXTRACTION '''

    # Create a copy of the initial dataframe to make various transformations
    data = pd.DataFrame(Rev.Revenue.copy())
    data.columns = ["y"]

    # Adding the lag of the target variable from 2 to 6.  With more data, additional lags could be analyzed
    for i in range(2, 6):
        data["lag_{}".format(i)] = data.y.shift(i)

    # for time-series cross-validation set 4 folds.
    tscv = TimeSeriesSplit(n_splits=4)

    y = data.dropna().y
    X = data.dropna().drop(['y'], axis=1)

    # reserve 30% of the data for testing
    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

    lr = LinearRegression()  # create linear regression
    lr.fit(X_train, y_train)  # fit linear regression

    # Using Lags as a feature produces better results than our time series did
    # plotModelResults(lr, X_train, X_test, y_train, y_test, tscv,  plot_intervals=True, plot_anomalies=True)
    # plotCoefficients(lr,  X_train)
    # plt.show()

    # Create additional features - month, quarter, month_of_quarter
    data.index = pd.to_datetime(data.index)
    data['month'] = data.index.month

    data['quarter'] = data.index.quarter
    data['month_of_quarter'] = data['month'] % 3
    data.loc[(data.month_of_quarter == 0), 'month_of_quarter'] = 3

    # plot additional features
    # plt.figure(figsize=(16,5))
    # plt.title("Encoded features")
    # data.month.plot()
    # data.quarter.plot()
    # data.month_of_quarter.plot()
    # plt.grid(True);

    # Transform our features to have a standard scale
    scaler = StandardScaler()

    y = data.dropna().y
    X = data.dropna().drop(['y'], axis=1)

    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    # plot model results
    # plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, y_train=y_train, y_test=y_test, tscv=tscv, plot_intervals=True) # We find that month_of_quarter is a useful feature
    # plotCoefficients(lr, X_train)
    # plt.show()

    # view monthly averages to see trends
    # verage_month = code_mean(data, 'month', "y")
    # plt.figure(figsize=(7,5))
    # plt.title("month averages")
    # pd.DataFrame.from_dict(average_month, orient='index')[0].plot()
    # plt.grid(True);
    # plt.show()

    ''' PREPARE AND MODEL USING FEATURES EXPLORED ABOVE '''

    X_train, X_test, y_train, y_test = prepareData(Rev.Revenue, lag_start=2, lag_end=5, test_size=0.3,
                                                   target_encoding=True)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, y_train=y_train, y_test=y_test, tscv=tscv,
                     plot_intervals=True)
    plotCoefficients(lr, X_train)
    plt.show()








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



def timeseries_train_test_split(X, y, test_size):
    """
    :param X: features
    :param y: target
    :param test_size: number to withold
    :return:training and test set
    """

    # get the index after which the test set starts
    test_index = int(len(X)*(1-test_size))

    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]

    return X_train, X_test, y_train, y_test


def plotModelResults(model, X_train, X_test, y_train, y_test, tscv,  plot_intervals=False, plot_anomalies=False):
    """
    :param model: fit model
    :param X_train: training set of features
    :param X_test: testing set of features
    :param y_train: training set of target
    :param y_test: testing set of target
    :param tscv: time series cross validation
    :param plot_intervals: confidence intervals
    :param plot_anomalies: anomalie detection/identification
    :return: Plots modelled vs fact values, predition intervals and anomalies
    """

    prediction = model.predict(X_test)

    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)

    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train,
                             cv=tscv,
                             scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()

        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)

        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)

        if plot_anomalies:
            anomalies = np.array([np.NaN] * len(y_test))
            anomalies[y_test < lower] = y_test[y_test < lower]
            anomalies[y_test > upper] = y_test[y_test > upper]
            plt.plot(anomalies, "o", markersize=10, label="Anomalies")

    error = mean_absolute_percentage_error(prediction, y_test)
    plt.title("Mean absolute percentage error {0:.2f}%".format(error))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);



def plotCoefficients(model, X_train):
    """
    :param model: fit model
    :return: returns plots of sorted coefficient values of the model
    """

    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');


def code_mean(data, cat_feature, real_feature):
    """
    :param data: time series
    :param cat_feature:
    :param real_feature:
    :return: Returns a dictionary where keys are unique categories of the cat_feature, and values are means over real_feature
    """
    return dict(data.groupby(cat_feature)[real_feature].mean())


def prepareData(series, lag_start, lag_end, test_size, target_encoding=False):
    """
    :param series: pd.DataFrame - dataframe with time series
    :param lag_start: int - initial step back in time to slice target varible; example - lag_start = 1 means that the model will see yesterday's values to predict today
    :param lag_end: final step back in time to slice target variable; example - lag_end = 4 means that the model will see up to 4 days back in time to predict today
    :param test_size: float - size of the test dataset after train/test split as percentage of data
    :param target_encoding: boolean - if True - add target averages to the dataset
    :return: dynamically prepares all data in ways explored prior in main function
    """

    # copy of the initial dataset
    data = pd.DataFrame(series.copy())
    data.columns = ["y"]

    # lags of series
    for i in range(lag_start, lag_end):
        data["lag_{}".format(i)] = data.y.shift(i)

    #datetime features
    data.index = pd.to_datetime(data.index)
    data['month'] = data.index.month
    data['quarter'] = data.index.quarter
    data['month_of_quarter'] = data['month']%3
    data.loc[(data.month_of_quarter == 0), 'month_of_quarter'] = 3 # correct the month_of_quarter variable

    if target_encoding:
        # calculate averages on train set only
        test_index = int(len(data.dropna())*(1-test_size))
        data['month_average'] = list(map(code_mean(data[:test_index], 'month', "y").get, data.month))
        #data['quarter_average'] = list(map(code_mean(data[:test_index], 'quarter', "y").get, data.quarter))

        # drop encoded variables
        data.drop(['month'], axis=1, inplace=True)

    # train-test split
    y = data.dropna().y
    X = data.dropna().drop(['y'], axis=1)
    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


main()
