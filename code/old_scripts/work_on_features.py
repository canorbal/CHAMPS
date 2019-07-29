import os
import sys
sys.path.append("/root/champs/")

import numpy as np
import pandas as pd
pd.options.display.precision = 15

import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
from utils import artgor_utils

from utils.train_utils import create_features_full, map_atom_info


if __name__== '__main__':

    sub_filename = '../submissions/work_on_features.csv'

    debug = False

    if debug:
        nrows = 100
        n_estimators = 50
        n_folds = 3
        use_stat_cols = 120
        result_filename = None
    else:
        result_filename = '../results/work_on_features.npy'
        n_folds = 10
        n_estimators = 2000
        use_stat_cols = 120
        nrows = No

    if not debug:
        if os.path.isfile(result_filename):
            assert False, "Result file exists!"

        if os.path.isfile(sub_filename):
            assert False, "Submission file exists!"

    print("reading data...")
    train = pd.read_csv('../data/train.csv', nrows=nrows)
    test = pd.read_csv('../data/test.csv', nrows=nrows)
    structures = pd.read_csv('../data/structures.csv', nrows=nrows)
    sub = pd.read_csv('../data/sample_submission.csv', nrows=nrows)

    train = map_atom_info(train, structures, 0)
    train = map_atom_info(train, structures, 1)

    test = map_atom_info(test, structures, 0)
    test = map_atom_info(test, structures, 1)

    train_p_0 = train[['x_0', 'y_0', 'z_0']].values
    train_p_1 = train[['x_1', 'y_1', 'z_1']].values
    test_p_0 = test[['x_0', 'y_0', 'z_0']].values
    test_p_1 = test[['x_1', 'y_1', 'z_1']].values

    train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
    test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
    train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
    test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
    train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
    test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
    train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
    test['dist_z'] = (test['z_0'] - test['z_1']) ** 2

    train = artgor_utils.reduce_mem_usage(train)
    test = artgor_utils.reduce_mem_usage(test)

    train = create_features_full(train)
    test = create_features_full(test)

    if not debug:
        train.to_csv("../data/train_stat_features.csv", index=False)
        test.to_csv("../data/test_stat_features.csv", index=False)

    print("label encoding...")
    for f in ['atom_0', 'atom_1', 'type']:
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

    print("creating folds...")
    n_folds = 5
    sorted_train = train.sort_values([
        "scalar_coupling_constant",
        "type",
        "dist",
    ])

    print("train shape ", train.shape)
    sorted_train.index = range(0, len(sorted_train))
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    X = sorted_train.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1)
    y = sorted_train['scalar_coupling_constant']
    X_test = test.drop(['id', 'molecule_name'], axis=1)

    params = {
        'num_leaves': 128,
        'objective': 'regression',
        'learning_rate': 0.1,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.7,
        "bagging_seed": 11,
        "metric": 'mae',
        "verbosity": -1,
        'reg_alpha': 0.1302650970728192,
        'reg_lambda': 0.3603427518866501,
        'colsample_bytree': 0.8,
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
                                                          n_estimators=n_estimators,
                                                          res_filename=result_filename
                                                          )

    if not debug:
        print("saving results...")
        np.save(result_filename, result_dict_lgb)

        print("making submission...")
        sub['scalar_coupling_constant'] = result_dict_lgb['prediction']
        sub.to_csv(sub_filename, index=False)