import os
import sys
sys.path.append("/root/champs/")

import numpy as np
import pandas as pd
import gc
pd.options.display.precision = 15
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
import warnings
warnings.filterwarnings("ignore")
import utils
from utils import artgor_utils
from utils import train_utils


if __name__== '__main__':

    print("reading data...")
    file_folder = '../data'
    train = pd.read_csv(f'{file_folder}/train.csv')
    test = pd.read_csv(f'{file_folder}/test.csv')
    sub = pd.read_csv(f'{file_folder}/sample_submission.csv')
    structures = pd.read_csv(f'{file_folder}/structures.csv')

    scalar_coupling_contributions = pd.read_csv(
        f'{file_folder}/scalar_coupling_contributions.csv')

    print("scalar coupling merging...")
    train = pd.merge(train, scalar_coupling_contributions, how='left',
                     left_on=['molecule_name', 'atom_index_0', 'atom_index_1',
                              'type'],
                     right_on=['molecule_name', 'atom_index_0', 'atom_index_1',
                               'type'])

    print("mapping info about atoms...")

    debug = False
    if debug:
        train = train[:100]
        test = test[:100]
        n_estimators = 50
        n_folds = 3
    else:
        n_folds = 10
        n_estimators = 5000

    train = train_utils.map_atom_info(train, structures,  0)
    train = train_utils.map_atom_info(train, structures, 1)

    test = train_utils.map_atom_info(test, structures, 0)
    test = train_utils.map_atom_info(test, structures, 1)

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

    print('merging stats...')
    train = train_utils.create_features_full(train)
    test = train_utils.create_features_full(test)

    print("label encoding...")
    for f in ['atom_0', 'atom_1', 'type']:
            lbl = LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))

    print("train shape ", train.shape)

    if not debug:
        print("saving stat...")
        train.to_csv('../data/train_stat_features.csv', index=False)
        test.to_csv('../data/test_stat_features.csv', index=False)

    print("creating folds...")

    sorted_train = train.sort_values([
        "scalar_coupling_constant",
        "type",
        "dist",
    ])
    sorted_train.index = range(0, len(sorted_train))

    sorted_train_molecule_name_col = sorted_train['molecule_name']
    test_molecule_name_col = test['molecule_name']

    X = sorted_train[train_utils.good_columns].copy()
    X_test = test[train_utils.good_columns].copy()
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    params = {
        'num_leaves': 128,
        'objective': 'regression',
        'learning_rate': 0.2,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.9,
        "bagging_seed": 11,
        "metric": 'mae',
        "verbosity": -1,
        'reg_alpha': 0.1302650970728192,
        'reg_lambda': 0.3603427518866501,
        'colsample_bytree': 1.,
        'device': 'gpu',
        'gpu_device_id': 0
    }

    columns = ['fc', 'sd', 'pso', 'dso']
    for oof_colomn in columns:
        print(f"training {oof_colomn}...")

        y = sorted_train[oof_colomn]
        result_filename = f'../results/oof_results/{oof_colomn}_oof_results.npy'

        result_dict = artgor_utils.train_model_regression(
            X=X, X_test=X_test, y=y,
            params=params, folds=folds,
            model_type='lgb',
            eval_metric='group_mae',
            plot_feature_importance=False,
            verbose=50, early_stopping_rounds=1000,
            n_estimators=n_estimators,
            res_filename=result_filename,
        )

        print("saving results...")
        np.save(result_filename, result_dict)