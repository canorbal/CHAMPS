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

    result_filename = '../results/metafeatures_baseline.npy'
    sub_filename = '../submissions/metafeatures_baseline.csv'

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
        n_estimators = 30000

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
    print("creating folds...")

    sorted_train = train.sort_values([
        "scalar_coupling_constant",
        "type",
        "dist",
        "fc",
    ])
    sorted_train.index = range(0, len(sorted_train))

    sorted_train_molecule_name_col = sorted_train['molecule_name']
    test_molecule_name_col = test['molecule_name']

    X = sorted_train[train_utils.good_columns].copy()
    y = sorted_train['scalar_coupling_constant']
    y_fc = sorted_train['fc']
    X_test = test[train_utils.good_columns].copy()

    del train, test, scalar_coupling_contributions
    gc.collect()

    folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    params = {
        'num_leaves': 128,
        'objective': 'regression',
        'learning_rate': 0.1,
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

    fc_result_filename = '../results/fc_oof_results.npy'

    result_dict_fc = artgor_utils.train_model_regression(
        X=X, X_test=X_test, y=y_fc,
        params=params, folds=folds,
        model_type='lgb',
        eval_metric='group_mae',
        plot_feature_importance=False,
        verbose=50, early_stopping_rounds=1000,
        n_estimators=10000,
        res_filename=fc_result_filename
    )

    X['oof_fc'] = result_dict_fc['oof']
    X_test['oof_fc'] = result_dict_fc['prediction']

    X['molecule_name'] = sorted_train_molecule_name_col.values
    X_test['molecule_name'] = test_molecule_name_col.values

    X = train_utils.oof_features(X)
    X_test = train_utils.oof_features(X_test)

    X = X.drop('molecule_name', axis=1)
    X_test = X_test.drop('molecule_name', axis=1)

    params = {
        'num_leaves': 512,
        'max_depth': 9,
        'objective': 'regression',
        'learning_rate': 0.075,
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

    print("saving results...")
    np.save(result_filename, result_dict_lgb)

    print("making submission...")
    sub['scalar_coupling_constant'] = result_dict_lgb['prediction']
    sub.to_csv(sub_filename, index=False)