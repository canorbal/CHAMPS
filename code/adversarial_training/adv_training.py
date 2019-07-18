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
from sklearn.utils import shuffle


if __name__ == '__main__':

    result_filename = '../../results/adversarial_training.npy'
    debug = False

    if debug:
        nrows = 100
        n_estimators = 50
        n_folds = 3
        nrows = 100
    else:
        n_folds = 5
        n_estimators = 500
        nrows = 0.5 * 10**6

    if not debug:
        if os.path.isfile(result_filename):
            assert False, "Result file exists!"

    print("reading tables")

    train = pd.read_csv('/root/champs/data/adversarial_datasets/train_stat.csv', nrows=nrows)
    test = pd.read_csv('/root/champs/data/adversarial_datasets/test_stat.csv', nrows=nrows)

    # assert 'scalar_coupling_constant' in train_columns

    try:
        train = train.drop(['molecule_name', 'id'], axis=1)
        test = test.drop(['molecule_name', 'id'], axis=1)
    except KeyError:
        pass

    features = [col for col in train if col in test.columns]
    train['target'] = 0
    test['target'] = 1

    train_test = pd.concat([train, test], axis=0)
    train_test = shuffle(train_test)
    target = train_test['target']
    train_test = train_test.drop('target', axis=1)
    del train, test
    print("final shape: ", train_test.shape)

    params = {
        'num_leaves': 128,
        'min_data_in_leaf': 79,
        'objective': 'binary',
        'learning_rate': 0.3,
        "boosting": "gbdt",
        "bagging_freq": 1,
        "bagging_fraction": 0.8,
        "bagging_seed": 11,
        "metric": 'auc',
        "verbosity": -1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.3,
        'num_threads': 4,
        'device_type': 'gpu',
        'gpu_device_id': 0
    }

    train_test = artgor_utils.reduce_mem_usage(train_test)
    categorical_cols = ['type', 'atom_0', 'atom_1']
    train_test[categorical_cols] = train_test[categorical_cols].astype('category')
    gc.collect()
    folds = KFold(n_splits=n_folds, random_state=0)

    if debug:
        train_test.to_csv('../../data/debug_data/adv_data_train_test.csv', index=False)

    print("training models...")
    result_dict_lgb = artgor_utils.train_model_classification(X=train_test,
                                                              y=target,
                                                              columns=features,
                                                              params=params,
                                                              folds=folds,
                                                              model_type='lgb',
                                                              plot_feature_importance=True,
                                                              verbose=100,
                                                              early_stopping_rounds=1000,
                                                              n_estimators=n_estimators,
                                                              res_filename=result_filename
                                                              )

    if not debug:
        print("saving results...")
        np.save(result_filename, result_dict_lgb)