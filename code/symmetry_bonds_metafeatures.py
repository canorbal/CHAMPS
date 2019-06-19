import os
import sys
sys.path.append("/root/champs/")

import numpy as np
import pandas as pd
import gc
pd.options.display.precision = 15
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
import warnings
warnings.filterwarnings("ignore")
import utils
from utils import artgor_utils
from utils import train_utils


if __name__== '__main__':

    result_filename = '../results/more_metafeatures.npy'
    sub_filename = '../submissions/more_metafeatures.csv'
    file_folder = '../data'

    if os.path.isfile(result_filename):
        assert False, "Result file exists!"

    if os.path.isfile(sub_filename):
        assert False, "Submission file exists!"

    debug = True

    if debug:
        nrows = 100
        n_estimators = 50
        n_folds = 3
        use_stat_cols = 120
    else:
        n_folds = 10
        n_estimators = 30000
        use_stat_cols = 60
        nrows = None

    train_cols_to_load = train_utils.good_columns[:use_stat_cols] + [
        "molecule_name",
        "scalar_coupling_constant",
    ]

    train = pd.read_csv(
        f"{file_folder}/train_stat_features.csv",
        usecols=train_cols_to_load,
        nrows=nrows
    )

    test_cols_to_load = train_utils.good_columns[:use_stat_cols] + ["molecule_name"]
    test = pd.read_csv(
        f"{file_folder}/test_stat_features.csv",
        usecols=test_cols_to_load,
        nrows=nrows)

    sub = pd.read_csv(f'{file_folder}/sample_submission.csv', nrows=nrows)

    sorted_train = train.sort_values([
        "scalar_coupling_constant",
        "type",
        "dist",
    ])

    del train
    gc.collect()

    sorted_train.index = range(0, len(sorted_train))

    sorted_train_molecule_name_col = sorted_train['molecule_name']
    test_molecule_name_col = test['molecule_name']

    print("loading oof columns")
    meta_columns = ['fc', 'sd', 'pso', 'dso']
    for oof_col in meta_columns:
        oof_result = np.load(f'../results/oof_results/{oof_col}_oof_results.npy',
                             allow_pickle=True)
        oof_result = oof_result.item()
        if debug:
            sorted_train[oof_col] = oof_result['oof'][:nrows]
            test[oof_col] = oof_result['prediction'][:nrows]
        else:
            sorted_train[oof_col] = oof_result['oof']
            test[oof_col] = oof_result['prediction']

    sorted_train['sum_oof'] = sorted_train[meta_columns].sum(axis=1)
    test['sum_oof'] = test[meta_columns].sum(axis=1)

    print("loading ase features")
    train_ase = pd.read_csv("../data/ase_train_feats.csv", nrows=nrows)
    test_ase = pd.read_csv("../data/ase_test_feats.csv", nrows=nrows)

    sorted_train = pd.merge(sorted_train, train_ase, how="left",
                            on=["molecule_name", "atom_index_0", "atom_index_1"])

    test = pd.merge(test, test_ase, how="left",
                    on=["molecule_name", "atom_index_0", "atom_index_1"])

    print("creating folds...")

    use_columns = (list(train_utils.good_columns[:use_stat_cols]) +
                   list(meta_columns) +
                   list(train_ase.columns)) + ['sum_oof']
    use_columns.remove("molecule_name")

    use_columns = set(use_columns)

    X = sorted_train[use_columns].copy()
    y = sorted_train['scalar_coupling_constant']
    X_test = test[use_columns].copy()

    del sorted_train, train_ase, test_ase
    gc.collect()

    folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    X['molecule_name'] = sorted_train_molecule_name_col.values
    X_test['molecule_name'] = test_molecule_name_col.values

    X = train_utils.oof_features(X)
    X_test = train_utils.oof_features(X_test)

    X = X.drop('molecule_name', axis=1)
    X_test = X_test.drop('molecule_name', axis=1)

    print("final shape: ", X.shape)

    params = {
        'num_leaves': 512,
        'objective': 'regression',
        'learning_rate': 0.01,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.9,
        "bagging_seed": 11,
        "metric": 'mae',
        'reg_alpha': 0.1302650970728192,
        'reg_lambda': 0.3603427518866501,
        'colsample_bytree': 0.9,
        'device': 'gpu',
        'gpu_device_id': 0
    }

    X = artgor_utils.reduce_mem_usage(X)
    X_test = artgor_utils.reduce_mem_usage(X_test)
    gc.collect()

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