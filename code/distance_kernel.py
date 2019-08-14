import sys
sys.path.append("/home/shakhrayv/roma/champs/")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from utils import artgor_utils, train_utils
import gc


types_config = {
    '2JHN': {
        'num_leaves': 128,
        'learning_rate': 0.05,
        'min_child_samples': 79,
        'max_depth': 9,
        'num_iterations': 20000,
        'objective': 'regression',
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.7,
        "bagging_seed": 11,
        "metric": 'mae',
        'reg_alpha': 0.4,
        'reg_lambda': 0.6,
        'colsample_bytree': 0.9,
        'num_threads': 14,
        'device': 'gpu',
        'gpu_device_id': 0
    },
    '3JHH': {
        'num_leaves': 128,
        'learning_rate': 0.05,
        'min_child_samples': 79,
        'max_depth': 9,
        'num_iterations': 20000,
        'objective': 'regression',
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.7,
        "bagging_seed": 11,
        "metric": 'mae',
        'reg_alpha': 0.4,
        'reg_lambda': 0.6,
        'colsample_bytree': 0.9,
        'num_threads': 14,
        'device': 'gpu',
        'gpu_device_id': 0
    },
    '3JHN': {
        'num_leaves': 128,
        'learning_rate': 0.05,
        'min_child_samples': 79,
        'max_depth': 9,
        'num_iterations': 20000,
        'objective': 'regression',
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.7,
        "bagging_seed": 11,
        "metric": 'mae',
        'reg_alpha': 0.4,
        'reg_lambda': 0.6,
        'colsample_bytree': 0.9,
        'num_threads': 14,
        'device': 'gpu',
        'gpu_device_id': 0
    }
}

if __name__ == '__main__':

    debug = False

    if debug:
        nrows = None
        n_estimators = 100
        n_folds = 3
        result_filename = None
        result_filename_prefix = Path('../results/each_type/')
        OOF_N_FOLDS = 20

    else:
        nrows = None
        result_filename_prefix = Path('../results/each_type/')
        n_folds = 5

    sub = pd.read_csv('../data/sample_submission.csv', nrows=nrows)

    categorical_cols = [
        'type',
        'atom_0',
        'atom_1',
    ]


    full_train = pd.read_csv('../data/train.csv', nrows=nrows)
    y = full_train['scalar_coupling_constant']
    full_test = pd.read_csv(
        '../data/test.csv',
        nrows=nrows,
    )

    for col in categorical_cols:
        if col in full_test.columns:
            full_test[col] = full_test[col].astype('category')

    X_short = pd.DataFrame({
        'ind': list(full_train.index),
        'type': full_train['type'].values,
        'oof': [0] * len(full_train),
        'target': y.values
    })

    X_short_test = pd.DataFrame({
        'ind': list(full_test.index),
        'type': full_test['type'].values,
        'prediction': [0] * len(full_test)
    })

    CV_score = 0
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    for type_num, type in enumerate(['2JHN', '3JHH', '3JHN']):
        print(f'Training of type {type}...')
        train_mask = (full_train['type'] == type)
        test_mask = (full_test['type'] == type)

        print('reading distance based features...')
        type_dist_train = pd.read_csv(f'../data/{type}_train_distance_based_feats.csv')
        type_dist_train.drop('scalar_coupling_constant', axis=1, inplace=True)
        type_dist_test = pd.read_csv(f'../data/{type}_test_distance_based_feats.csv')
        atoms_categorical_cols = [f'atom_{i}' for i in range(2, 10)]
        
        for df in [type_dist_test, type_dist_train]:
            df[atoms_categorical_cols] = df[atoms_categorical_cols].astype('category')

        if type_num == 0:
            best_features = list(type_dist_test.columns)

        print('concating distance features...')
        type_train = type_dist_train
        type_test = type_dist_test
        gc.collect()

        params = types_config[type]
        result_filename = result_filename_prefix / f'distance_kernel_{type}.npy'
        type_y = full_train[train_mask]['scalar_coupling_constant']

        print('X_test_t.shape ', type_test.shape)
        print('X_t.shape ', type_train.shape)
        print('Training...')
        result_dict_lgb = artgor_utils.train_model_regression(
            X=type_train, y=type_y, X_test=type_test,
            params=params,
            columns=best_features,
            folds=folds,
            n_folds=n_folds,
            model_type='lgb',
            eval_metric='mae',
            verbose=50,
            early_stopping_rounds=1000,
            res_filename=result_filename
        )

        del type_train, type_test
        gc.collect()

        if not debug:
            np.save(str(result_filename), result_dict_lgb)

        X_short.loc[X_short['type'] == type, 'oof'] = result_dict_lgb['oof']
        X_short_test.loc[X_short_test['type'] == type, 'prediction'] = result_dict_lgb['prediction']

        ## manually computing the cv score
        CV_score += np.array(result_dict_lgb['scores']).mean() / len(types_config)

    print(f'Total val score: {CV_score}')
    # if not debug:
    #     sub['scalar_coupling_constant'] = X_short_test['prediction']
    #     sub.to_csv(sub_filename, index=False)