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
import warnings
warnings.filterwarnings("ignore")

import json
import altair as alt



# setting up altair
from utils import artgor_utils


def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df


if __name__== '__main__':

    print("reading data...")
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    structures = pd.read_csv('../data/structures.csv')
    sub = pd.read_csv('../data/sample_submission.csv')

    print("mapping info about atoms...")

    train = map_atom_info(train, 0)
    train = map_atom_info(train, 1)

    test = map_atom_info(test, 0)
    test = map_atom_info(test, 1)

    train_p_0 = train[['x_0', 'y_0', 'z_0']].values
    train_p_1 = train[['x_1', 'y_1', 'z_1']].values
    test_p_0 = test[['x_0', 'y_0', 'z_0']].values
    test_p_1 = test[['x_1', 'y_1', 'z_1']].values

    train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
    test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)

    train['dist_to_type_mean'] = train['dist'] / train.groupby('type')[
        'dist'].transform('mean')
    test['dist_to_type_mean'] = test['dist'] / test.groupby('type')[
        'dist'].transform('mean')

    print("label encoding...")
    for f in ['type', 'atom_0', 'atom_1']:
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(train[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))


    print("creating folds...")
    n_folds = 10
    sorted_train = train.sort_values([
        "scalar_coupling_constant",
         "type",
         "dist",
    ])

    sorted_train.index = range(0, len(sorted_train))
    folds = KFold(n_splits=10, shuffle=True, random_state=0)

    X = sorted_train.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1)
    y = sorted_train['scalar_coupling_constant']
    X_test = test.drop(['id', 'molecule_name'], axis=1)


    params = {'num_leaves': 128,
              'objective': 'regression',
              'learning_rate': 0.3,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.9,
              "bagging_seed": 11,
              "metric": 'mae',
              "verbosity": -1,
              'reg_alpha': 0.1302650970728192,
              'reg_lambda': 0.3603427518866501,
              'colsample_bytree': 1.0,
              'device': 'gpu',
              'gpu_device_id': 0
              }

    print("training models...")
    result_dict_lgb = artgor_utils.train_model_regression(X=X, X_test=X_test,
                                                          y=y,
                                                          params=params,
                                                          folds=folds,
                                                          model_type='lgb',
                                                          eval_metric='group_mae',
                                                          plot_feature_importance=True,
                                                          verbose=100,
                                                          early_stopping_rounds=1000,
                                                          n_estimators=30000)

    print("saving results...")
    np.save('../results/parameters_tuning_baseline.npy', result_dict_lgb)

    print("making submission...")
    sub['scalar_coupling_constant'] = result_dict_lgb['prediction']
    sub.to_csv('../submissions/parameters_tuning_baseline.csv', index=False)