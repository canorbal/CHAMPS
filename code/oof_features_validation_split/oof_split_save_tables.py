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
        n_folds = 5
    else:
        nrows = None
        n_folds = 20

    print("reading train...")
    train = pd.read_csv(f'{file_folder}/train.csv', nrows=nrows)
    dist = pd.read_csv(f'{file_folder}/train_stat_features.csv', nrows=nrows, usecols=['dist'])
    train['dist'] = dist['dist']
    del dist; gc.collect()

    sorted_train = train.sort_values([
        "scalar_coupling_constant",
        "type",
        "dist",
    ])

    train_size = len(train)
    sorted_train['fold'] = (list(range(n_folds)) * train_size)[:train_size]
    train = sorted_train.sort_values(['molecule_name', 'atom_index_0', 'atom_index_1'])
    del sorted_train
    gc.collect()

    if debug:
        train.to_csv(f'{file_folder}/debug_data/train_folds.csv', index=False)

    for fold in range(n_folds):
        print(f'prepairing fold {fold}')
        mask = train['fold'] == fold
        fold_data = train[mask]

        for mapping in tables_mapping:
            print(f'reading {mapping["train_features"]}')
            train_path = mapping['train_features']

            train_data = pd.read_csv(train_path, nrows=nrows,
                                     usecols=mapping['use_cols'])

            if mapping['drop_columns']:
                train_data.drop(mapping['drop_columns'], axis=1, inplace=True)
                gc.collect()

            if mapping['concat_type'] == 'concat':
                print('concating...')
                train_data = train_data[mask]
                gc.collect()
                fold_data = train_utils.concat_stupidly(fold_data, train_data)
                assert len(fold_data) == len(train_data)

            elif mapping['concat_type'] == 'merge':
                print('merging...')
                for i in range(2):
                    fold_data = train_utils.map_atom_info(fold_data, train_data, i)

            else:
                raise KeyError()

            del train_data
            gc.collect()
        assert len(fold_data) == mask.sum()
        print('saving...')
        fold_data.to_csv(f'../../data/oof_tables/{fold}_fold_oof_tables.csv', index=False)
        del fold_data
        gc.collect()

