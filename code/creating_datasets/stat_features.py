import os
import sys
sys.path.append("/root/champs/")

import numpy as np
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
pd.options.display.precision = 15

import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn import metrics
from sklearn import linear_model
import gc
import seaborn as sns

from utils import artgor_utils
from utils.train_utils import map_atom_info, concat_dataframes

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    debug = False

    if debug:
        nrows = 1000
    else:
        nrows = 2 * 10**6

    train = pd.read_csv('../../data/train.csv', nrows=nrows)
    test = pd.read_csv('../../data/test.csv', nrows=nrows)

    train_feats = pd.read_csv('../../data/train_stat_features.csv', nrows=nrows)
    test_feats = pd.read_csv('../../data/test_stat_features.csv', nrows=nrows)

    print('assert checking...')
    for col in ['molecule_name', 'atom_index_0', 'atom_index_1']:
        assert (train[col] == train_feats[col]).sum() == len(train)
        assert (test[col] == test_feats[col]).sum() == len(test)

    structures = pd.read_csv('../../data/structures.csv')

    for i in range(2):
        train = map_atom_info(train, structures, i)
        test = map_atom_info(test, structures, i)

    train = concat_dataframes(train, train_feats)
    test = concat_dataframes(test, test_feats)

    if debug:
        train.to_csv('../../data/debug_data/stat_merge.csv', index=False)

    train = artgor_utils.reduce_mem_usage(train)
    test = artgor_utils.reduce_mem_usage(test)

    train.to_csv('/root/champs/data/adversarial_datasets/train_stat.csv', index=False)
    test.to_csv('/root/champs/data/adversarial_datasets/test_stat.csv', index=False)