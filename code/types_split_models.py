import sys
sys.path.append("/root/champs/")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from utils import artgor_utils, train_utils
import gc


def process_oof_results(len_test=2505542, n_folds=20):
    total_importance = None
    oof_train = []
    oof_test = np.zeros((len_test, n_folds))

    for i in range(20):
        results = np.load(
            f"../data/oof_tables/v1_oof_results/fold_{i}_scalar_coupling_constant_result.npy",
            allow_pickle=True
        )

        results = results.item()
        cols = results['feature_importance'].groupby(['feature'])['importance'].mean().sort_index(ascending=False)

        if total_importance is None:
            total_importance = cols
        else:
            total_importance = total_importance + cols

        oof_train.append(results['oof'])
        oof_test[:, i] = results['prediction']

    total_importance = total_importance.sort_values(ascending=False)
    gc.collect()

    return {
        'total_importance': total_importance,
        'oof_train_list': oof_train,
        'oof_test': oof_test,
    }


if __name__ == '__main__':

    debug = False
    OOF_N_FOLDS = 20
    data_fold_path = Path('../data/oof_tables/')
    sub_filename = '../submissions/types_split_models.csv'

    if debug:
        nrows = 1000
        n_estimators = 100
        n_folds = 3
        result_filename = None
        use_best_columns = 50

    else:
        nrows = None
        result_filename_prefix = Path('../results/each_type/')
        use_best_columns = 180
        n_folds = 8

    sub = pd.read_csv('../data/sample_submission.csv', nrows=nrows)
    oof_result_dict = process_oof_results(n_folds=OOF_N_FOLDS)
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

    train = artgor_utils.reduce_mem_usage(train)
    test = artgor_utils.reduce_mem_usage(test)
    gc.collect()

    gc.collect()

    CV_score = 0
    ###Iters###    [1JHC, 1JHN, 2JHC, 2JHH, 2JHN, 3JHC, 3JHH, 3JHN]
    if debug:
        n_estimators = [10] * 8
    else:
        n_estimators = [17000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]

    params = {
        'num_leaves': 128,
        'objective': 'regression',
        'learning_rate': 0.025,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.75,
        "bagging_seed": 11,
        "metric": 'mae',
        'reg_alpha': 0.3,
        'reg_lambda': 0.5,
        'colsample_bytree': 0.9,
        'num_threads': 10,
        'device': 'gpu',
        'gpu_device_id': 0
    }
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    X_short = pd.DataFrame({
        'ind': list(train.index),
        'type': train['type'].values,
        'oof': [0] * len(train),
        'target': y.values
    })

    X_short_test = pd.DataFrame({
        'ind': list(test.index),
        'type': test['type'].values,
        'prediction': [0] * len(test)
    })

    for type_num, type in enumerate(train['type'].unique()):
        print(f'Training of type {type}...')
        result_filename = result_filename_prefix / f'types_split_models_{type}.npy'
        index_type = (train['type'] == type)
        index_type_test = (test['type'] == type)

        X_t = train.loc[index_type].copy()
        X_test_t = test.loc[index_type_test].copy()
        y_t = y[index_type]

        print('X_test_t.shape ', X_test_t.shape)
        print('X_t.shape ', X_t.shape)
        print('Training...')
        result_dict_lgb = artgor_utils.train_model_regression(
            X=X_t, y=y_t, X_test=X_test_t,
            params=params,
            columns=best_features,
            folds=folds,
            model_type='lgb',
            eval_metric='group_mae',
            verbose=50,
            early_stopping_rounds=1000,
            n_estimators=n_estimators[type_num],
            res_filename=result_filename
        )

        del X_t, X_test_t
        gc.collect()

        if not debug:
            np.save(str(result_filename), result_dict_lgb)

        X_short.loc[X_short['type'] == type, 'oof'] = result_dict_lgb['oof']
        X_short_test.loc[X_short_test['type'] == type, 'prediction'] = result_dict_lgb['prediction']

        ## manually computing the cv score
        CV_score += np.array(result_dict_lgb['scores']).mean() / len(n_estimators)

    print(f'Total val score: {CV_score}')
    if not debug:
        sub['scalar_coupling_constant'] = X_short_test['prediction']
        sub.to_csv(sub_filename, index=False)
