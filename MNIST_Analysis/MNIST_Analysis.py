'''
Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition
Chapter 3 - Classification
'''

''' IMPORTS & Setup '''
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# to make this file's output stable across runs
np.random.seed(42)

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# # Where to save the figures
# PROJECT_ROOT_DIR = "."
# CHAPTER_ID = "classification"
# IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
# os.makedirs(IMAGES_PATH, exist_ok=True)
#
# def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
#     path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
#     print("Saving figure", fig_id)
#     if tight_layout:
#         plt.tight_layout()
#     plt.savefig(path, format=fig_extension, dpi=resolution)
#
# ''' Open Data Source '''
#
# from sklearn.datasets import fetch_openml
# mnist = fetch_openml('mnist_784', version=1)
# print(mnist.keys())
#
# X, y = mnist["data"], mnist["target"]
# print(X.shape)
# print(y.shape)

import pandas as pd
AP_History = pd.read_csv('C:/Users/JAKEFREDRICH/Desktop/Runout Testing - Pia/Runout Testingv2/Runout Testing/AP History.csv', index_col=0, keep_default_na=False, encoding ='latin1') #keep_default replaces NaN with blank  , encoding ='latin1's

