import sys
sys.path.append("/root/champs/")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from utils import artgor_utils, train_utils
import gc


if __name__ == '__main__':

    debug = False
    OOF_N_FOLDS = 20
    data_fold_path = Path('../data/oof_tables/')
    sub_filename = '../submissions/best_features.csv'

    if debug:
        nrows = 100
        n_estimators = 100
        n_folds = 3
        result_filename = None
        use_best_columns = 50

    else:
        nrows = None
        n_estimators = 30000
        result_filename = '../results/best_feats.npy'
        use_best_columns = 180
        n_folds = 10

    sub = pd.read_csv('../data/sample_submission.csv', nrows=nrows)
    oof_result_dict = train_utils.process_oof_results(n_folds=OOF_N_FOLDS)
    best_features = list(oof_result_dict['total_importance'].index[:use_best_columns])

    categorical_cols = [
        'type',
        'atom_0',
        'atom_1',
    ]

    if 'type' not in best_features:
        best_features.append('type')

    train_best_features = best_features + ['id', 'scalar_coupling_constant']

    print('reading folds...')
    folds_data_list = []
    for fold_num in range(OOF_N_FOLDS):
        print(f'reading fold {fold_num}...')
        fold_path = data_fold_path / f'{fold_num}_fold_oof_tables.csv'
        fold_data = pd.read_csv(fold_path, nrows=nrows, usecols=train_best_features)
        folds_data_list.append(fold_data)

    train = pd.concat(folds_data_list, axis=0, ignore_index=True)
    train = train.sort_values('id')
    train.index = range(len(train))

    for col in categorical_cols:
        if col in train.columns:
            train[col] = train[col].astype('category')

    y = train['scalar_coupling_constant']
    del folds_data_list
    gc.collect()

    test = pd.read_csv('../data/test_folds/test_features.csv', nrows=nrows, usecols=best_features)

    for col in categorical_cols:
        if col in test.columns:
            test[col] = test[col].astype('category')

    oof_train = pd.read_csv('../data/train_oof_features.csv', nrows=len(train))
    oof_test = pd.read_csv('../data/test_oof_features.csv', nrows=len(test))
    oof_drop_cols = [
        'id',
        'molecule_name',
        'atom_index_0',
        'atom_index_1',
        'scalar_coupling_constant_oof',
    ]

    for df in [oof_train, oof_test]:
        df.drop(oof_drop_cols, axis=1, inplace=True)
    gc.collect()

    best_features = best_features + list(oof_train.columns)

    train = train_utils.concat_stupidly(train, oof_train)
    test = train_utils.concat_stupidly(test, oof_test)

    gc.collect()

    folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    params = {
        'num_leaves': 128,
        'objective': 'regression',
        'learning_rate': 0.04,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.75,
        "bagging_seed": 11,
        "metric": 'mae',
        'reg_alpha': 0.3,
        'reg_lambda': 0.5,
        'colsample_bytree': 0.9,
        'device': 'gpu',
        'gpu_device_id': 0
    }

    X = artgor_utils.reduce_mem_usage(train)
    X_test = artgor_utils.reduce_mem_usage(test)
    gc.collect()

    if debug:
        print('saving debug data...')
        X[best_features].to_csv("../data/debug_data/train_all_folds_together.csv", index=False)
        X_test[best_features].to_csv('../data/debug_data/test_all_folds_together.csv', index=False)

    print('train.shape ', train[best_features].shape)
    print('test.shape ', test[best_features].shape)

    print("training models...")
    result_dict_lgb = artgor_utils.train_model_regression(
        X=X, X_test=X_test, y=y,
        params=params,
        columns=best_features,
        folds=folds,
        model_type='lgb',
        eval_metric='group_mae',
        verbose=50,
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


