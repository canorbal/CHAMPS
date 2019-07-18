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
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

import lightgbm as lgb
from utils import artgor_utils
from utils.train_utils import map_atom_info, concat_dataframes

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    debug = False

    if debug:
        nrows = 1000
        n_folds = 3
        n_estimators = 50
        result_filename = None
    else:
        nrows = 2 * 10 ** 6
        n_folds = 5
        n_estimators = 2000
        result_filename = '../../results/feature_selection/acsf_features_result.csv'

    train = pd.read_csv('../../data/train.csv', nrows=nrows)
    acsf_structures = pd.read_csv('../../data/structure_with_acsf.csv', index_col=0)

    for i in range(2):
        train = map_atom_info(train, acsf_structures, i)

    train_p_0 = train[['x_0', 'y_0', 'z_0']].values
    train_p_1 = train[['x_1', 'y_1', 'z_1']].values
    train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

    if debug:
        train.to_csv('../../data/debug_data/acsf_merge.csv', index=False)

    train = artgor_utils.reduce_mem_usage(train)

    print("creating folds...")
    sorted_train = train.sort_values([
        "scalar_coupling_constant",
        "type",
        "dist",
    ])

    sorted_train.index = range(0, len(sorted_train))
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    X = sorted_train.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1)
    y = sorted_train['scalar_coupling_constant']

    categorical_cols = ['type', 'atom_0', 'atom_1']
    X[categorical_cols] = X[categorical_cols].astype('category')

    print('X.shape', X.shape)

    params = {
        'num_leaves': 128,
        'min_child_samples': 79,
        'objective': 'regression',
        'learning_rate': 0.08,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.9,
        "bagging_seed": 11,
        "metric": 'mae',
        "verbosity": -1,
        'reg_alpha': 0.1302650970728192,
        'reg_lambda': 0.3603427518866501,
        'colsample_bytree': 1.0,
        'num_threads': 4,
        'device_type': 'gpu',
        'gpu_device_id': 0
    }

    print("training models...")
    result_dict_lgb = artgor_utils.train_model_regression(X=X, y=y,
                                                          params=params,
                                                          folds=folds,
                                                          model_type='lgb',
                                                          eval_metric='group_mae',
                                                          plot_feature_importance=True,
                                                          verbose=100,
                                                          early_stopping_rounds=200,
                                                          n_estimators=n_estimators,
                                                          res_filename=result_filename,
                                                          )
    if not debug:
        print("saving results...")
        np.save(result_filename, result_dict_lgb)