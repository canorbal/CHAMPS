import sys
sys.path.append("/root/champs/")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from utils import artgor_utils
import gc
from pathlib import Path

from utils.train_utils import map_atom_info


from oof_features_validatation_split.config import (
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

use_columns_from_tables = {
    'stat': 25,  # -0.053
    'acsf': 100,  # -0.618
    'giba': 80, # -0.526
    'ase': 50, # -0.306
    'bonds': 5, # 0.302
    'dihedral': None,
    'rdkit': None,
}


def get_importane(filename):
    results = np.load(filename, allow_pickle=True)
    results = results.item()
    cols = (
        results['feature_importance'].groupby(['feature'])
        ['importance'].mean().sort_values(ascending=False)
    )
    return cols


if __name__ == '__main__':
    debug = True
    if debug:
        nrows = 100
    else:
        nrows = None

    data_folder = Path('../data')
    sub = pd.read_csv(data_folder / 'sample_submission.csv', nrows=nrows)
    print("reading train and test...")
    test = pd.read_csv(data_folder / 'test.csv', nrows=nrows)
    train = pd.read_csv(data_folder / 'train.csv', nrows=nrows)

    print('reading acsf...')
    acsf_structures = pd.read_csv('../../data/structure_with_acsf.csv', index_col=0, nrows=nrows)
    acsf_structures.drop()

    print('mapping acsf...')
    for i in range(2):
        train = map_atom_info(train, acsf_structures, i)
        test = map_atom_info(test, acsf_structures, i)

    for df in [train, test]:
        p_0= df[['x_0', 'y_0', 'z_0']].values
        p_1 = df[['x_1', 'y_1', 'z_1']].values
        df['dist'] = np.linalg.norm(p_0 - p_1, axis=1)

    acsf_result = get_importane('../results/feature_selection/acsf_features_result.csv.npy')
    acsf_columns = list(acsf_result.index[:use_columns_from_tables['acsf']])

    train = train[acsf_columns]
    test = test[acsf_columns]
    del acsf_columns, acsf_result, acsf_structures
    gc.collect()








