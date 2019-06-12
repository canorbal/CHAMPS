import pandas as pd
from utils import artgor_utils

good_columns = [
    'molecule_atom_index_0_dist_min',
    'molecule_atom_index_0_dist_max',
    'molecule_atom_index_0_dist_std',
    'molecule_atom_index_0_dist_mean',
    'molecule_atom_index_1_dist_min',
    'molecule_atom_index_1_dist_std',
    'molecule_atom_index_0_dist_max_diff',
    'molecule_atom_index_1_dist_max',
    'molecule_atom_index_0_dist_mean_diff',
    'molecule_atom_index_0_dist_std_diff',
    'molecule_atom_index_1_dist_mean',
    'molecule_atom_index_1_dist_std_diff',
    'molecule_atom_index_0_y_1_std',
    'molecule_atom_index_0_z_1_std',
    'molecule_atom_index_0_x_1_std',
    'molecule_type_dist_mean_diff',
    'molecule_atom_index_1_dist_max_diff',
    'molecule_atom_index_1_dist_mean_diff',
    'molecule_type_dist_max_diff',
    'molecule_atom_index_1_dist_min_diff',
    'molecule_atom_index_1_dist_y_min',
    'molecule_type_dist_std',
    'molecule_atom_index_0_y_1_mean_diff',
    'molecule_atom_index_1_dist_z_min',
    'molecule_atom_index_0_dist_y_min',
    'molecule_atom_index_1_dist_x_min',
    'molecule_atom_index_0_dist_x_min',
    'molecule_atom_index_0_dist_z_min',
    'molecule_atom_1_dist_std',
    'molecule_atom_1_dist_mean',
    'molecule_atom_index_1_y_0_std',
    'molecule_atom_index_0_y_1_min_diff',
    'molecule_atom_index_0_x_1_mean_diff',
    'molecule_atom_index_0_y_1_max_diff',
    'molecule_type_dist_min_diff',
    'molecule_atom_index_0_z_1_mean_diff',
    'molecule_atom_index_1_z_0_std',
    'molecule_atom_index_1_x_0_std',
    'molecule_atom_index_1_y_0_mean_diff',
    'molecule_type_dist_std_diff',
    'molecule_type_dist_y_min',
    'molecule_type_dist_min',
    'molecule_atom_index_0_dist_y_mean',
    'molecule_atom_index_0_dist_y_mean_diff',
    'molecule_atom_index_0_dist_min_diff',
    'dist',
    'molecule_type_dist_max',
    'molecule_dist_mean',
    'molecule_atom_0_dist_mean_diff',
    'molecule_atom_index_1_y_0_max_diff',
    'molecule_atom_index_0_x_1_min_diff',
    'molecule_atom_1_dist_min',
    'molecule_atom_index_0_dist_y_max_diff',
    'molecule_atom_index_0_x_1_max_diff',
    'molecule_atom_0_dist_min_diff',
    'molecule_type_dist_x_min',
    'molecule_atom_index_0_dist_z_mean_diff',
    'molecule_atom_index_1_x_0_mean_diff',
    'molecule_atom_1_dist_std_diff',
    'molecule_atom_index_0_z_1_min_diff',
    'molecule_type_dist_z_min',
    'molecule_atom_index_0_z_1_max_diff',
    'molecule_atom_0_dist_y_min',
    'molecule_atom_1_dist_min_diff',
    'molecule_atom_index_0_dist_x_mean_diff',
    'atom_index_1',
    'molecule_atom_index_1_z_0_mean_diff',
    'molecule_atom_index_0_dist_z_max_diff',
    'molecule_atom_index_0_dist_x_max_diff',
    'molecule_atom_index_1_dist_y_std',
    'molecule_atom_0_y_1_min_diff',
    'molecule_atom_index_0_dist_x_mean',
    'molecule_type_y_1_std',
    'molecule_atom_index_0_dist_y_std',
    'molecule_atom_index_0_dist_z_mean',
    'molecule_type_dist_mean',
    'molecule_atom_0_dist_x_min',
    'molecule_type_z_1_std',
    'molecule_atom_index_0_dist_y_max',
    'molecule_type_x_1_std',
    'molecule_atom_index_0_z_1_min',
    'molecule_atom_1_dist_max',
    'molecule_atom_index_0_y_1_max',
    'molecule_atom_0_dist_z_max',
    'molecule_atom_1_dist_y_min',
    'molecule_atom_index_1_dist_z_std',
    'molecule_atom_index_0_dist_y_std_diff',
    'molecule_atom_1_dist_mean_diff',
    'molecule_atom_index_0_y_1_min',
    'molecule_atom_index_0_z_1_max',
    'molecule_atom_0_y_1_max',
    'molecule_atom_index_0_dist_z_std_diff',
    'molecule_atom_0_dist_z_min',
    'molecule_atom_index_0_x_1_max',
    'molecule_atom_index_1_dist_y_mean',
    'molecule_atom_index_0_z_1_mean',
    'molecule_type_dist_y_std',
    'molecule_atom_index_0_dist_z_std',
    'molecule_atom_0_dist_y_max',
    'molecule_type_y_1_mean_diff',
    'molecule_atom_index_1_dist_x_std',
    'molecule_atom_0_dist_x_max',
    'molecule_atom_index_1_y_0_min_diff',
    'molecule_atom_index_1_dist_y_std_diff',
    'molecule_type_dist_z_std',
    'molecule_atom_1_dist_x_min',
    'molecule_atom_index_0_dist_x_std',
    'molecule_atom_index_1_dist_z_mean',
    'molecule_atom_index_0_dist_x_std_diff',
    'molecule_type_dist_x_std',
    'molecule_atom_0_dist_y_mean',
    'molecule_atom_index_1_dist_y_mean_diff',
    'molecule_atom_0_x_1_max_diff',
    'molecule_atom_0_x_1_min_diff',
    'molecule_atom_index_1_dist_x_mean',
    'molecule_atom_index_1_dist_z_std_diff',
    'molecule_type_dist_y_mean',
    'molecule_atom_index_0_y_1_mean',
    'molecule_atom_index_0_x_1_min',
    'molecule_atom_index_1_x_0_min_diff',
    'molecule_atom_index_1_dist_x_std_diff',
    'molecule_atom_0_dist_z_max_diff',
    'molecule_atom_0_dist_y_std',
    'molecule_atom_index_1_dist_y_max',
    'molecule_atom_index_1_y_0_max',
    'molecule_atom_1_dist_z_min',
    'molecule_atom_index_0_dist_z_max',
    'molecule_type_dist_x_mean',
    'molecule_atom_0_z_1_max_diff',
    'molecule_atom_index_0_dist_x_max',
    'molecule_atom_0_z_1_min_diff',
    'molecule_atom_index_1_dist_y_max_diff',
    'molecule_atom_index_1_z_0_min_diff',
    'molecule_atom_index_1_x_0_max_diff',
    'molecule_type_dist_z_mean',
    'molecule_type_y_0_std',
    'molecule_type_y_0_mean_diff',
    'molecule_atom_0_x_1_std',
    'molecule_type_x_1_mean_diff',
    'molecule_atom_index_1_z_0_max_diff',
    'molecule_type_y_1_max_diff',
    'molecule_atom_0_dist_z_mean',
    'molecule_atom_index_1_y_0_std_diff',
    'molecule_atom_1_y_1_std',
    'molecule_type_x_0_std',
    'molecule_atom_0_dist_x_max_diff',
    'molecule_atom_0_z_1_std',
    'molecule_atom_0_dist_x_mean',
    'molecule_type_z_1_mean_diff',
    'molecule_type_z_0_std',
    'molecule_atom_index_1_dist_x_max',
    'molecule_atom_0_dist_std_diff',
    'molecule_dist_std',
    'molecule_atom_0_dist_y_max_diff',
    'molecule_atom_index_1_z_0_mean',
    'molecule_atom_1_y_1_min_diff',
    'molecule_atom_index_0_x_1_mean',
    'molecule_atom_index_1_dist_z_max_diff',
    'molecule_atom_1_x_1_std',
    'molecule_atom_index_1_dist_z_mean_diff',
    'molecule_atom_index_1_z_0_std_diff',
    'molecule_atom_1_dist_y_std',
    'molecule_atom_index_1_dist_z_max',
    'molecule_type_dist_z_max',
    'molecule_atom_1_z_1_std',
    'molecule_atom_index_1_z_0_min',
    'molecule_atom_index_1_z_0_max',
    'molecule_atom_1_dist_max_diff',
    'molecule_atom_index_1_dist_x_max_diff',
    'molecule_atom_0_y_1_max_diff',
    'atom_index_0',
    'molecule_atom_index_1_dist_x_mean_diff',
    'molecule_type_y_1_max',
    'molecule_type_dist_y_max_diff',
    'molecule_atom_index_1_x_0_max',
    'molecule_type_dist_x_max',
    'molecule_atom_0_dist_x_std',
    'molecule_dist_max',
    'molecule_atom_index_1_x_0_std_diff',
    'molecule_atom_0_y_1_std',
    'molecule_atom_0_dist_z_std',
    'molecule_atom_1_y_1_mean_diff',
    'molecule_type_y_1_min_diff',
    'molecule_atom_index_1_x_0_min',
    'molecule_dist_min',
    'molecule_atom_1_y_1_max',
    'molecule_atom_index_1_y_0_mean',
    'molecule_atom_1_dist_y_mean',
    'molecule_atom_1_dist_y_max_diff',
    'molecule_type_dist_z_max_diff',
    'molecule_type_x_0_mean_diff',
    'molecule_atom_index_0_dist_y_min_diff',
    'molecule_atom_0_z_1_min',
    'molecule_atom_index_0_dist_z_min_diff',
    'molecule_type_dist_y_max',
    'molecule_atom_0_dist_max_diff',
    'molecule_type_dist_x_max_diff',
    'molecule_type_dist_y_mean_diff',
    'molecule_atom_1_dist_x_max_diff',
    'molecule_atom_1_dist_z_max_diff',


    'type',
    'atom_0',
    'atom_1',
]


def map_atom_info(df, structures, atom_idx):
    df = pd.merge(df, structures, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df


def oof_features(df):
    num_cols = ['oof_fc']

    cat_cols = ['type', 'atom_0', 'atom_1', 'atom_index_0', 'atom_index_1']
    aggs = ['mean', 'max', 'std', 'min']

    for col in cat_cols:
        df[f'molecule_{col}_count'] = df.groupby('molecule_name')[
            col].transform('count')

    for cat_col in cat_cols:
        for num_col in num_cols:
            for agg in aggs:
                df[f'molecule_{cat_col}_{num_col}_{agg}'] = \
                df.groupby(['molecule_name', cat_col])[num_col].transform(agg)
                df[f'molecule_{cat_col}_{num_col}_{agg}_diff'] = df[
                                                                     f'molecule_{cat_col}_{num_col}_{agg}'] - \
                                                                 df[num_col]

    df = artgor_utils.reduce_mem_usage(df)
    return df


def create_features_full(df):
    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')
    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')
    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')
    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')
    df['molecule_dist_std'] = df.groupby('molecule_name')['dist'].transform('std')

    num_cols = [
        'x_0', 'y_0', 'z_0',
        'x_1', 'y_1', 'z_1',
        'dist',
        'dist_x', 'dist_y', 'dist_z'
    ]

    cat_cols = ['type', 'atom_0', 'atom_1', 'atom_index_0', 'atom_index_1']
    aggs = ['mean', 'max', 'std', 'min']
    for col in cat_cols:
        df[f'molecule_{col}_count'] = df.groupby('molecule_name')[col].transform('count')

    for cat_col in cat_cols:
        for num_col in num_cols:
            for agg in aggs:
                df[f'molecule_{cat_col}_{num_col}_{agg}'] = df.groupby(['molecule_name', cat_col])[num_col].transform(agg)
                df[f'molecule_{cat_col}_{num_col}_{agg}_diff'] = df[f'molecule_{cat_col}_{num_col}_{agg}'] - df[num_col]

    df = artgor_utils.reduce_mem_usage(df)
    return df


def create_features(df):
    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform(
        'count')
    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform(
        'mean')
    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform(
        'min')
    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform(
        'max')
    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])[
        'id'].transform('count')
    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])[
        'id'].transform('count')

    df[f'molecule_atom_index_0_x_1_std'] = \
    df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')
    df[f'molecule_atom_index_0_y_1_mean'] = \
    df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')
    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[
                                                     f'molecule_atom_index_0_y_1_mean'] - \
                                                 df['y_1']
    df[f'molecule_atom_index_0_y_1_mean_div'] = df[
                                                    f'molecule_atom_index_0_y_1_mean'] / \
                                                df['y_1']
    df[f'molecule_atom_index_0_y_1_max'] = \
    df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')
    df[f'molecule_atom_index_0_y_1_max_diff'] = df[
                                                    f'molecule_atom_index_0_y_1_max'] - \
                                                df['y_1']
    df[f'molecule_atom_index_0_y_1_std'] = \
    df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')
    df[f'molecule_atom_index_0_z_1_std'] = \
    df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')
    df[f'molecule_atom_index_0_dist_mean'] = \
    df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')
    df[f'molecule_atom_index_0_dist_mean_diff'] = df[
                                                      f'molecule_atom_index_0_dist_mean'] - \
                                                  df['dist']
    df[f'molecule_atom_index_0_dist_mean_div'] = df[
                                                     f'molecule_atom_index_0_dist_mean'] / \
                                                 df['dist']
    df[f'molecule_atom_index_0_dist_max'] = \
    df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')
    df[f'molecule_atom_index_0_dist_max_diff'] = df[
                                                     f'molecule_atom_index_0_dist_max'] - \
                                                 df['dist']
    df[f'molecule_atom_index_0_dist_max_div'] = df[
                                                    f'molecule_atom_index_0_dist_max'] / \
                                                df['dist']
    df[f'molecule_atom_index_0_dist_min'] = \
    df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')
    df[f'molecule_atom_index_0_dist_min_diff'] = df[
                                                     f'molecule_atom_index_0_dist_min'] - \
                                                 df['dist']
    df[f'molecule_atom_index_0_dist_min_div'] = df[
                                                    f'molecule_atom_index_0_dist_min'] / \
                                                df['dist']
    df[f'molecule_atom_index_0_dist_std'] = \
    df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')
    df[f'molecule_atom_index_0_dist_std_diff'] = df[
                                                     f'molecule_atom_index_0_dist_std'] - \
                                                 df['dist']
    df[f'molecule_atom_index_0_dist_std_div'] = df[
                                                    f'molecule_atom_index_0_dist_std'] / \
                                                df['dist']
    df[f'molecule_atom_index_1_dist_mean'] = \
    df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')
    df[f'molecule_atom_index_1_dist_mean_diff'] = df[
                                                      f'molecule_atom_index_1_dist_mean'] - \
                                                  df['dist']
    df[f'molecule_atom_index_1_dist_mean_div'] = df[
                                                     f'molecule_atom_index_1_dist_mean'] / \
                                                 df['dist']
    df[f'molecule_atom_index_1_dist_max'] = \
    df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')
    df[f'molecule_atom_index_1_dist_max_diff'] = df[
                                                     f'molecule_atom_index_1_dist_max'] - \
                                                 df['dist']
    df[f'molecule_atom_index_1_dist_max_div'] = df[
                                                    f'molecule_atom_index_1_dist_max'] / \
                                                df['dist']
    df[f'molecule_atom_index_1_dist_min'] = \
    df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')
    df[f'molecule_atom_index_1_dist_min_diff'] = df[
                                                     f'molecule_atom_index_1_dist_min'] - \
                                                 df['dist']
    df[f'molecule_atom_index_1_dist_min_div'] = df[
                                                    f'molecule_atom_index_1_dist_min'] / \
                                                df['dist']
    df[f'molecule_atom_index_1_dist_std'] = \
    df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')
    df[f'molecule_atom_index_1_dist_std_diff'] = df[
                                                     f'molecule_atom_index_1_dist_std'] - \
                                                 df['dist']
    df[f'molecule_atom_index_1_dist_std_div'] = df[
                                                    f'molecule_atom_index_1_dist_std'] / \
                                                df['dist']
    df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])[
        'dist'].transform('mean')
    df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])[
        'dist'].transform('min')
    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - \
                                           df['dist']
    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df[
        'dist']
    df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])[
        'dist'].transform('std')
    df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - \
                                           df['dist']
    df[f'molecule_type_0_dist_std'] = df.groupby(['molecule_name', 'type_0'])[
        'dist'].transform('std')
    df[f'molecule_type_0_dist_std_diff'] = df[f'molecule_type_0_dist_std'] - \
                                           df['dist']
    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])[
        'dist'].transform('mean')
    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df[
        'dist']
    df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df[
        'dist']
    df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])[
        'dist'].transform('max')
    df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])[
        'dist'].transform('min')
    df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])[
        'dist'].transform('std')
    df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df[
        'dist']

    df = artgor_utils.reduce_mem_usage(df)
    return df
