import numpy as np
from pathlib import Path

importance_folder = Path('../../results/feature_selection/')
data_folder = Path('../../data/')

stat_features_table = {
    'train_features': data_folder / 'train_stat_features.csv',
    'test_features': data_folder / 'test_stat_features.csv',
    'feat_importance': importance_folder / 'stat_features_result.npy',
    'concat_type': 'concat',
    'drop_columns': [
        'id',
        'molecule_name',
        'atom_index_0',
        'atom_index_1',
        'type',
    ],
    'use_cols': None,
}


giba_features_table = {
    'train_features': data_folder / 'train_giba.csv',
    'test_features': data_folder / 'test_giba.csv',
    'feat_importance': importance_folder / 'giba_features_result.csv.npy',
    'concat_type': 'concat',
    'drop_columns': [
        'id',
        'molecule_name',
        'atom_index_0',
        'atom_index_1',
        'type',

        'ID',
        'structure_atom_0',
        'structure_atom_1',
        'typei',
        'dist_xyz',
        'structure_x_0',
        'structure_y_0',
        'structure_z_0',
        'structure_x_1',
        'structure_y_1',
        'structure_z_1',
        'molecule_name.1',
        'atom_index_1.1',
    ],
    'use_cols': None,
}


bonds_feature_table = {
    'train_features': data_folder / 'bonds_train.csv',
    'test_features': data_folder / 'bonds_test.csv',
    'feat_importance': importance_folder / 'bonds_features_result.csv.npy',
    'concat_type': 'concat',
    'drop_columns': None,
    'use_cols': [
        'EN_x',
        'EN_y',
        'rad_x',
        'rad_y',
        'n_bonds_x',
        'n_bonds_y',
        'bond_lengths_mean_x',
        'bond_lengths_mean_y',

    ],
}


ase_feature_table = {
    'train_features': data_folder / 'ase_train_feats.csv',
    'test_features': data_folder / 'ase_test_feats.csv',
    'feat_importance': importance_folder / 'ase_features_result.csv.npy',
    'concat_type': 'concat',
    'drop_columns': [
        'molecule_name',
        'atom_index_0',
        'atom_index_1',
    ],
    'use_cols': None,
}


dihedral_feature_table = {
    'train_features': data_folder / 'train_dihedral.csv',
    'test_features': data_folder / 'test_dihedral.csv',
    'feat_importance': None,
    'concat_type': 'concat',
    'drop_columns': None,
    'use_cols': [
        'dihedral_x',
        'dihedral_y',
    ]
}

rdkit_feature_table = {
    'train_features': data_folder / 'rdkit_train_feats.csv',
    'test_features': data_folder / 'rdkit_test_feats.csv',
    'concat_type': 'concat',
    'feat_importance': None,
    'drop_columns': None,
    'use_cols': [
        'atom_index_0_degree',
        'atom_index_0_hybridization',
        'atom_index_0_inring',
        'atom_index_0_inring3',

        'atom_index_0_nb_c',
        'atom_index_0_nb_h',
        'atom_index_0_nb_n',
        'atom_index_0_nb_na',
        'atom_index_0_nb_o',

        'atom_index_1_degree',
        'atom_index_1_hybridization',
        'atom_index_1_inring',
        'atom_index_1_inring3',

        'atom_index_1_nb_c',
        'atom_index_1_nb_h',
        'atom_index_1_nb_n',
        'atom_index_1_nb_na',
        'atom_index_1_nb_o',
    ]
}


acsf_feature_table = {
    'train_features': data_folder / 'structure_with_acsf.csv',
    'test_features': data_folder / 'structure_with_acsf.csv',
    'concat_type': 'merge',
    'feat_importance': importance_folder / 'acsf_features_result.csv.npy',
    'drop_columns': [
        'atom',
        'x',
        'y',
        'z',
        'Unnamed: 0',
    ],
    'use_cols': None,
}
