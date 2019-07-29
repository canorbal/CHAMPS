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


if __name__ == '__main__':

    debug = False

    if debug:
        nrows = 100
        n_folds = 5
        n_estimators = 200
        result_filename_prefix = None
    else:
        nrows = None
        n_folds = 4
        n_estimators = 2500
        result_filename_prefix = Path('../../data/oof_tables/v1_oof_results/')

    print("reading data...")
    file_folder = Path('../../data/')
    oof_data = file_folder / 'oof_tables'

    sub = pd.read_csv(f'{file_folder}/sample_submission.csv', nrows=nrows)
    test = pd.read_csv(file_folder / 'test_folds/test_features.csv', nrows=nrows)
    scalar_coupling_contributions = pd.read_csv(file_folder/'scalar_coupling_contributions.csv')

    params = {
        'num_leaves': 128,
        'objective': 'regression',
        'learning_rate': 0.03,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.7,
        "bagging_seed": 11,
        "metric": 'mae',
        'reg_alpha': 1.5,
        'reg_lambda': 1.5,
        'colsample_bytree': 0.9,
        'device': 'gpu',
        'gpu_device_id': 0
    }

    cols_to_drop = [
        'id',
        'molecule_name',
        'scalar_coupling_constant',
        'fold',
    ]

    categorical_cols = [
        'type',
        'atom_0',
        'atom_1',
    ]

    test.drop(cols_to_drop, axis=1, inplace=True)
    test[categorical_cols] = test[categorical_cols].astype('category')
    features = test.columns

    for fold_number in range(20):

        folder_data = pd.read_csv(oof_data / f'{fold_number}_fold_oof_tables.csv', nrows=nrows)

        print("scalar coupling merging...")

        folder_data = pd.merge(folder_data, scalar_coupling_contributions, how='left',
                               left_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],
                               right_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])

        coupling_constant = folder_data['scalar_coupling_constant']
        folder_data.drop(cols_to_drop, axis=1, inplace=True)

        if debug:
            folder_data.to_csv('../../data/debug_data/fold_train.csv', index=False)

        folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)

        columns = ['scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso',]

        for oof_colomn in columns:
            print(f"training {oof_colomn}...")

            if not debug:
                result_filename = result_filename_prefix / f'fold_{fold_number}_{oof_colomn}_result.npy'
            else:
                result_filename = None

            if oof_colomn == 'scalar_coupling_constant':
                y = coupling_constant
            else:
                y = folder_data[oof_colomn]

            folder_data[categorical_cols] = folder_data[categorical_cols].astype('category')

            print('test_shape', test[features].shape)
            print('train shape', folder_data[features].shape)

            result_dict = artgor_utils.train_model_regression(
                X=folder_data, X_test=test, y=y,
                params=params, folds=folds, columns=features,
                model_type='lgb',
                eval_metric='group_mae',
                verbose=50, early_stopping_rounds=20,
                n_estimators=n_estimators,
                res_filename=result_filename,
            )

            print("saving results...")
            if result_filename:
                np.save(str(result_filename), result_dict)
