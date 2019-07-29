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


def map_acsf_info(df, structures, atom_idx):
    df = pd.merge(df, structures, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])

    df = df.drop('atom_index', axis=1)
    return df


if __name__== '__main__':

    result_filename = '../results/90k_iters_pair_acsf_descr.npy'
    sub_filename = '../submissions/90k_iters_pair_acsf_descr.csv'
    file_folder = '../data'

    debug = False

    if debug:
        nrows = 100
        n_estimators = 50
        n_folds = 3
        use_stat_cols = 120
    else:
        n_folds = 5
        n_estimators = 50000
        use_stat_cols = 120
        nrows = None

    if not debug:
        if os.path.isfile(result_filename):
            assert False, "Result file exists!"

        if os.path.isfile(sub_filename):
            assert False, "Submission file exists!"

    print("reading stat tables...")

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
        nrows=nrows
    )

    print('reading acsf...')
    acsf_descr = pd.read_csv(f"{file_folder}/structure_with_acsf.csv", nrows=nrows, index_col=0)
    acsf_descr = acsf_descr.drop(['x', 'y', 'z', 'atom'], axis=1)
    useless_cols = train_utils.find_useless_cols(acsf_descr)
    acsf_descr = acsf_descr.drop(useless_cols, axis=1)
    gc.collect()
    acsf_descr = artgor_utils.reduce_mem_usage(acsf_descr)

    print('mapping acsf...')
    for i in range(2):
        train = map_acsf_info(train, acsf_descr, i)
        test = map_acsf_info(test, acsf_descr, i)

    useless_cols = train_utils.find_useless_cols(train)
    for i in range(2):
        if f'atom_{i}' in useless_cols:
            useless_cols.remove(f'atom_{i}')

    print(useless_cols)
    train = train.drop(useless_cols, axis=1)
    test = test.drop(useless_cols, axis=1)
    print(f'dropped {len(useless_cols)} cols from acsf...')

    drop_cols = []
    for col in train.columns:
        if '_x' in col:
            col = col[:-2]
            if col in acsf_descr:
                train[f'diff_{col}'] = train[col + '_x'] - train[col + '_y']
                test[f'diff_{col}'] = test[col + '_x'] - test[col + '_y']
                drop_cols.append(col + '_x')
                drop_cols.append(col + '_y')

    print(f'diff drop len {len(drop_cols)}')
    train = train.drop(drop_cols, axis=1)
    test = test.drop(drop_cols, axis=1)

    del acsf_descr, drop_cols, useless_cols
    train = artgor_utils.reduce_mem_usage(train)
    test = artgor_utils.reduce_mem_usage(test)
    gc.collect()

    print('reading ase and sym dataframes...')
    train_ase = pd.read_csv(f"{file_folder}/ase_train_feats.csv", nrows=nrows)
    test_ase = pd.read_csv(f"{file_folder}/ase_test_feats.csv", nrows=nrows)

    print('processing ase...')
    for df_train, df_test in zip([train_ase], [test_ase]):
        assert (df_train['atom_index_0'] == train['atom_index_0']).sum() == len(train)
        assert (df_test['atom_index_0'] == test['atom_index_0']).sum() == len(test)

        repeat_cols = [col for col in df_train if col in train.columns]
        df_train = df_train.drop(repeat_cols, axis=1)
        df_test = df_test.drop(repeat_cols, axis=1)

        train = pd.concat([train, df_train], axis=1)
        test = pd.concat([test, df_test], axis=1)

    del df_train, df_test, train_ase, test_ase
    train = artgor_utils.reduce_mem_usage(train)
    test = artgor_utils.reduce_mem_usage(test)
    gc.collect()

    print('reading bonds...')
    bonds_train = pd.read_csv(f"{file_folder}/bonds_train.csv", nrows=nrows)
    bonds_test = pd.read_csv(f"{file_folder}/bonds_test.csv", nrows=nrows)

    assert (bonds_train['atom_index_0'] == train['atom_index_0']).sum() == len(train)
    assert (bonds_test['atom_index_0'] == test['atom_index_0']).sum() == len(test)

    repeat_cols = [col for col in bonds_train.columns if col in train.columns]
    repeat_cols_train = repeat_cols + ['scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso']
    bonds_train = bonds_train.drop(repeat_cols_train, axis=1)
    repeat_cols.remove('scalar_coupling_constant')
    bonds_test = bonds_test.drop(repeat_cols, axis=1)

    train = pd.concat([train, bonds_train], axis=1)
    test = pd.concat([test, bonds_test], axis=1)

    del bonds_train, bonds_test, repeat_cols_train, repeat_cols
    train = artgor_utils.reduce_mem_usage(train)
    test = artgor_utils.reduce_mem_usage(test)
    gc.collect()

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

    sorted_train.drop("molecule_name", axis=1, inplace=True)

    X = sorted_train.drop('scalar_coupling_constant', axis=1).copy()
    y = sorted_train['scalar_coupling_constant']
    X_test = test.copy()

    del sorted_train
    gc.collect()

    folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    X['molecule_name'] = sorted_train_molecule_name_col.values
    X_test['molecule_name'] = test_molecule_name_col.values

    X = train_utils.oof_features(X)
    X_test = train_utils.oof_features(X_test)

    X = X.drop('molecule_name', axis=1)
    X_test = X_test.drop('molecule_name', axis=1)

    print('X.shape ', X.shape)
    print('pairing features')
    cols_to_drop = []
    for col_0 in X.columns:
        if '_atom_index_0_' in col_0:
            col_1 = col_0.replace('_atom_index_0_', '_atom_index_1_')
            if col_1 in X.columns:
                cols_to_drop.append(col_0)
                cols_to_drop.append(col_1)
                for df in [X, X_test]:
                    df[f'diff_{col_0}_{col_1}'] = df[col_0] - df[col_1]
                    df[f'sum_{col_0}_{col_1}'] = df[col_0] + df[col_1]

                X.drop([col_0, col_1], axis=1, inplace=True)
                X_test.drop([col_0, col_1], axis=1, inplace=True)

    gc.collect()

    print("final shape: ", X.shape)

    params = {
        'num_leaves': 128,
        'objective': 'regression',
        'learning_rate': 0.08,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.6,
        "bagging_seed": 11,
        "metric": 'mae',
        'reg_alpha': 0.1302650970728192,
        'reg_lambda': 0.3603427518866501,
        'colsample_bytree': 0.8,
        'device': 'gpu',
        'gpu_device_id': 0
    }

    X = artgor_utils.reduce_mem_usage(X)
    X_test = artgor_utils.reduce_mem_usage(X_test)
    gc.collect()

    if debug:
        X.to_csv("../data/debug_acsf.csv", index=False)
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