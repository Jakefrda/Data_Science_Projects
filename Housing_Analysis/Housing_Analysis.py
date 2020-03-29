'''
# Created by Jake Fredrich
# Material - Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Riely

'''

import os
import tarfile
import urllib
from zlib import crc32
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split



DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)



def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]





def main():

    fetch_housing_data() # Pull Data
    housing_df = load_housing_data()

    #print(housing_df["ocean_proximity"].value_counts())
    #print(housing_df.describe())

    ''' EXPLORE DATA '''
    #housing_df.hist(bins=50, figsize=(20, 15)) # Plot histogram of features
    #plt.show()

    housing_with_id = housing_df.reset_index()   # adds an `index` column
    housing_with_id["id"] = housing_df["longitude"] * 1000 + housing_df["latitude"] # Create id using lat and lon

    housing_with_id["income_cat"] = pd.cut(housing_with_id["median_income"],
                                           bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                           labels=[1, 2, 3, 4, 5]) # Create income bins so that you can complete a stratified training/test set

    housing_df["income_cat"] = pd.cut(housing_df["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5]) # Create income bins so that you can complete a stratified training/test set

    housing_df["income_cat"].hist()
    #plt.show()

    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id") # Create train and test set that is not stratified

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # Create train and test set that is stratified
    for train_index, test_index in split.split(housing_df, housing_df["income_cat"]):
        strat_train_set = housing_df.loc[train_index]
        strat_test_set = housing_df.loc[test_index]

    # Print distributions to determine how the test sets compare to overall data
    #print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
    #print(test_set["income_cat"].value_counts() / len(test_set))
    #print(housing_df["income_cat"].value_counts() / len(housing_df))

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)


    ''' ADDITIONAL VISUALIZATION '''
    housing_copy = strat_train_set.copy()
    #housing_copy.plot(kind="scatter", x="longitude", y="latitude")
    #housing_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=.1) # alpha defines density

    housing_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                      s=housing_copy['population']/100, label='population', figsize=(10,7),
                      c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True,
                      )
    #plt.legend()
    #plt.show()

    corr_matrix = housing_copy.corr()
    #print(corr_matrix)
    from pandas.plotting import scatter_matrix

    attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age'] # plot matrix scatter plots for several attributes
    scatter_matrix(housing_copy[attributes], figsize=(12,8))
    #plt.show()

    housing_copy.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    #plt.show()

    ''' CREATE ADDITIONAL ATTRIBUTES '''
    housing_copy['rooms_per_household'] = housing_copy['total_rooms']/housing_copy['households']
    housing_copy['bedrooms_per_room'] = housing_copy['total_bedrooms']/housing_copy['total_rooms']
    housing_copy['population_per_households'] = housing_copy['population']/housing_copy['households']

    corr_matrix = housing_copy.corr()
    corr_matrix['median_house_value'].sort_values(ascending=False)
    #print(corr_matrix['median_house_value'].sort_values(ascending=False))

    ''' DATA CLEANING '''
    housing_copy =  strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set['median_house_value'].copy()

    #housing_copy.dropna(subset=['total_bedrooms']) # drops records that don't contain a value for total_bedrooms
    #housing_copy.drop('total_bedrooms', axis=1) # drops entire attribute
    #median = housing_copy['total_bedrooms'].median() # median total bedrooms
    #housing_copy['total_bedrooms'].fillna(median, inplace=True) # populate total bedroom NAs with median

    from sklearn.impute import SimpleImputer # imputer can be used to track stats on all numerical fields
    imputer = SimpleImputer(strategy="median")

    housing_num = housing_copy.drop("ocean_proximity", axis=1) # remove non-numerical field
    imputer.fit(housing_num) # estimate using fit()
   #print(imputer.statistics_)

    X = imputer.transform(housing_num) # transform housing_num using imputer median, filling in NAs
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index = housing_num.index)
    #print(housing_num['total_bedrooms'].count())
    #print(housing_tr['total_bedrooms'].count())

    housing_cat = housing_copy[['ocean_proximity']]
    housing_cat.head(10)

    # One method to create numerical representation. We do not want proximity between values
    #from sklearn.preprocessing import OrdinalEncoder
    #ordinal_encoder = OrdinalEncoder()
    #housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    #housing_cat_encoded[:10]
    #print(ordinal_encoder.categories_)

    #Create a custom transformer to add extra attributes
    from sklearn.base import BaseEstimator, TransformerMixin
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y=None):
            return self # do nothing
        def transform(self, X):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, households_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing_copy.values)

    housing_extra_attribs = pd.DataFrame(
        housing_extra_attribs,
        columns=list(housing_copy.columns) + ["rooms_per_household", "population_per_household"],
        index=housing_copy.index)
    print(housing_extra_attribs.head())



    ''' CREATE A PIPELINE FOR NUMERICAL AND CATEGORICAL ATTRIBUTES  '''
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
            ])

    housing_num_tr = num_pipeline.fit_transform(housing_num)

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    housing_prepared = full_pipeline.fit_transform(housing_copy)
    print(housing_prepared.shape)

    ''' LINEAR REGRESSION '''
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    some_data = housing_copy.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    print("predictions", lin_reg.predict(some_data_prepared))
    print("Labels: ", list(some_labels))

    from sklearn.metrics import mean_squared_error
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)

    ''' DECISION TREE '''
    from sklearn.tree import DecisionTreeRegressor

    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels) # train model
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print(tree_rmse)

    ''' Cross Validation '''
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring = "neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)

    def display_scores(scores):
        print("Scores:", scores)
        print("Mean: ", scores.mean())
        print("Standard Deviation: ", scores.std())

    #display_scores(tree_rmse_scores)

    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    #display_scores(lin_rmse_scores)

    ''' Random Forest '''
    from sklearn.ensemble import RandomForestRegressor

    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)
    housing_predictions = forest_reg.predict(housing_prepared)
    forest_mse = mean_squared_error(housing_labels, housing_predictions)
    forest_rmse = np.sqrt(forest_mse)
    print("Training set: ")
    display_scores(-forest_rmse)

    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    print("Validation set: ")
    display_scores(forest_rmse_scores)

    ''' Save models '''
    #import joblib
    #joblib.dump(my_model, "my_model.pkl")
    #my_model_loaded = joblib.load("my_model.pkl")

    ''' GRID SEARCH '''
    # fiddles with hyperparameters for me
    from sklearn.model_selection import GridSearchCV

    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2,4,6,8]},
        {'bootstrap': [False], 'n_estimators': [3,10], 'max_features':[2,3,4]},
    ]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)

    grid_search.best_params_
    print(grid_search.best_params_) # Print best combination of parameters
    print(grid_search.best_estimator_) # Print best estimator

    # Print evaluation scores
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    # Print feature importances
    feature_importances = grid_search.best_estimator_.feature_importances_
    #print(feature_importances)
    extra_attribs = ["rooms)per)hhold", "pop_per_hhold", "bedrooms_per_room"]
    cat_encoder = full_pipeline.named_transformers_["cat"]
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    print(sorted(zip(feature_importances, attributes), reverse=True))

    ''' EVALUATE ON TEST SET '''
    final_model = grid_search.best_estimator_

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_prepared = full_pipeline.transform(X_test)

    final_predictions = final_model.predict(X_test_prepared)

    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(final_rmse)

    # Compute accuracy w/ confidence intervol
    from scipy import stats
    confidence = .95
    squared_errors = (final_predictions - y_test) ** 2
    print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                             loc=squared_errors.mean(),
                             scale=stats.sem(squared_errors))))




    print("complete")



















main()