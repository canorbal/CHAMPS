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
from utils import artgor_utils
from utils import train_utils
from config import (
    stat_features_table,
    giba_features_table,
    bonds_feature_table,
    ase_feature_table,
    dihedral_feature_table,
    rdkit_feature_table,
    acsf_feature_table,
)

tables_mapping = [
    stat_features_table,
    acsf_feature_table,
    giba_features_table,
    ase_feature_table,
    bonds_feature_table,
    dihedral_feature_table,
    rdkit_feature_table,
]


if __name__ == '__main__':

    file_folder = '../../data'

    debug = False

    if debug:
        nrows = 1000
        n_folds = 1
    else:
        nrows = None
        n_folds = 1

    print("reading test...")
    test = pd.read_csv(f'{file_folder}/test.csv', nrows=nrows)
    dist = pd.read_csv(f'{file_folder}/test_stat_features.csv', nrows=nrows, usecols=['dist'])
    test['dist'] = dist['dist']
    del dist
    gc.collect()

    sorted_test = test.sort_values([
        "dist",
        "type",
    ])

    test_size = len(test)
    sorted_test['fold'] = (list(range(n_folds)) * test_size)[:test_size]
    test = sorted_test.sort_values(['molecule_name', 'atom_index_0', 'atom_index_1'])
    del sorted_test
    gc.collect()

    if debug:
        test.to_csv(f'{file_folder}/debug_data/test_folds.csv', index=False)

    for fold in range(n_folds):
        print(f'prepairing fold {fold}')
        mask = test['fold'] == fold
        fold_data = test[mask]

        for mapping in tables_mapping:
            print(f'reading {mapping["test_features"]}')
            test_path = mapping['test_features']

            test_data = pd.read_csv(test_path, nrows=nrows,
                                     usecols=mapping['use_cols'])

            if mapping['drop_columns']:
                test_data.drop(mapping['drop_columns'], axis=1, inplace=True)
                gc.collect()

            if mapping['concat_type'] == 'concat':
                print('concating...')
                test_data = test_data[mask]
                gc.collect()
                fold_data = train_utils.concat_stupidly(fold_data, test_data)
                assert len(fold_data) == len(test_data)

            elif mapping['concat_type'] == 'merge':
                print('merging...')
                for i in range(2):
                    fold_data = train_utils.map_atom_info(fold_data, test_data, i)

            else:
                raise KeyError()

            del test_data
            gc.collect()
        assert len(fold_data) == mask.sum()
        print('saving...')
        fold_data.to_csv(f'../../data/test_folds/test_{fold}_fold.csv', index=False)
        del fold_data
        gc.collect()
