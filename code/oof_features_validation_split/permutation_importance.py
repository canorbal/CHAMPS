import os
import sys
sys.path.append("/root/champs/")

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pathlib import Path
import gc
pd.options.display.precision = 15
import warnings
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
import utils
from utils import artgor_utils
from utils import train_utils
from functools import partial


if __name__ == '__main__':

    debug = False

    if debug:
        nrows = 500
        n_folds = 5
        n_estimators = 400
        result_filename_prefix = None
        OOF_N_FOLDS = 3
        use_best_columns = 30
    else:
        OOF_N_FOLDS = 20
        nrows = None
        n_folds = 4
        n_estimators = 3500
        use_best_columns = 500
        result_filename_prefix = Path('../../results/perm_importance')

    print("reading data...")
    file_folder = Path('../../data/')
    oof_data = file_folder / 'oof_tables'

    sub = pd.read_csv(f'{file_folder}/sample_submission.csv', nrows=nrows)

    oof_result_dict = train_utils.process_oof_results(n_folds=OOF_N_FOLDS)
    features = list(oof_result_dict['total_importance'].index[:use_best_columns])

    if 'type' not in features:
        features.append('type')

    params = {
        'num_leaves': 128,
        'objective': 'regression',
        'learning_rate': 0.008,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.7,
        "bagging_seed": 11,
        "metric": 'mae',
        'reg_alpha': 1.5,
        'reg_lambda': 1.5,
        'num_threads': 10,
        'colsample_bytree': 0.9,
        'device': 'gpu',
        'gpu_device_id': 0
    }

    categorical_cols = [
        'type',
        'atom_0',
        'atom_1',
    ]

    train_features = features + ['scalar_coupling_constant']

    for fold_number in range(OOF_N_FOLDS):

        folder_data = pd.read_csv(
            oof_data / f'{fold_number}_fold_oof_tables.csv',
            nrows=nrows, usecols=train_features
        )

        y = folder_data['scalar_coupling_constant']

        if debug:
            folder_data.to_csv('../../data/debug_data/fold_train.csv', index=False)

        folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)

        if not debug:
            result_filename = result_filename_prefix / f'fold_{fold_number}_result.npy'
        else:
            result_filename = None

        for cat in categorical_cols:
            if cat in folder_data:
                folder_data[cat] = folder_data[cat].astype('category')

        print('train shape', folder_data[features].shape)
        pi_results = None

        for fold_n, (train_index, valid_index) in enumerate(folds.split(folder_data)):
            gc.collect()
            X_train, X_valid = folder_data.iloc[train_index], folder_data.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            model = train_utils.train_lgbm_on_fold(
                X_train, y_train, X_valid, y_valid, columns=features,
                n_estimators=n_estimators, params=params,
                early_stopping_rounds=100,
            )

            metric = partial(artgor_utils.group_mean_log_mae, types=X_valid['type'])
            pi_results = train_utils.permutation_importance(
                model, X_valid, y_valid,
                metric=metric, columns=features, n_folds=n_folds, results=pi_results
            )

        if not debug:
            print("saving results...")
            if result_filename:
                np.save(str(result_filename), pi_results)
