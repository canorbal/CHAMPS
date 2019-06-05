import os
import sys
sys.path.append("/root/champs/")

import numpy as np
import pandas as pd
pd.options.display.precision = 15

import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn import metrics
from sklearn import linear_model
import gc
import warnings
warnings.filterwarnings("ignore")

import json
import altair as alt


# setting up altair
from utils import artgor_utils


def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df


def create_features_full(df):
    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')
    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')
    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')
    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')
    df['molecule_dist_std'] = df.groupby('molecule_name')['dist'].transform('std')
    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')

    num_cols = ['x_1', 'y_1', 'z_1', 'dist', 'dist_x', 'dist_y', 'dist_z']
    cat_cols = ['atom_index_0', 'atom_index_1', 'type', 'atom_0', 'atom_1', 'type_0']
    aggs = ['mean', 'max', 'std', 'min']
    for col in cat_cols:
        df[f'molecule_{col}_count'] = df.groupby('molecule_name')[col].transform('count')

    for cat_col in cat_cols:
        for num_col in num_cols:
            for agg in aggs:
                df[f'molecule_{cat_col}_{num_col}_{agg}'] = df.groupby(['molecule_name', cat_col])[num_col].transform(agg)
                df[f'molecule_{cat_col}_{num_col}_{agg}_diff'] = df[f'molecule_{cat_col}_{num_col}_{agg}'] - df[num_col]
                df[f'molecule_{cat_col}_{num_col}_{agg}_div'] = df[f'molecule_{cat_col}_{num_col}_{agg}'] / df[num_col]

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


good_columns = [
    'molecule_atom_index_0_dist_min',
    'molecule_atom_index_0_dist_max',
    'molecule_atom_index_0_dist_mean',
    'molecule_atom_index_0_dist_std',

    'molecule_atom_index_1_dist_min',
    'molecule_atom_index_1_dist_max',
    'molecule_atom_index_1_dist_mean',
    'molecule_atom_index_1_dist_std',

    'dist',

    'molecule_atom_index_0_dist_max_diff',
    'molecule_atom_index_0_dist_std_diff',
    'molecule_atom_index_0_dist_mean_diff',
    'molecule_atom_index_0_dist_min_diff',

    'molecule_atom_index_1_dist_max_diff',
    'molecule_atom_index_1_dist_mean_diff',
    'molecule_atom_index_1_dist_std_diff',
    'molecule_atom_index_1_dist_min_diff',

    'molecule_atom_index_0_dist_max_div',
    'molecule_atom_index_0_dist_std_div',
    'molecule_atom_index_0_dist_min_div',
    'molecule_atom_index_0_dist_mean_div',

    'molecule_atom_index_1_dist_max_div',
    'molecule_atom_index_1_dist_std_div',
    'molecule_atom_index_1_dist_min_div',
    'molecule_atom_index_1_dist_mean_div',

    'atom_0_couples_count',
    'atom_1_couples_count',

    'atom_index_0',
    'atom_index_1',

    'molecule_couples',
    'molecule_dist_mean',

    'molecule_atom_index_0_x_1_std',
    'molecule_atom_index_0_y_1_std',
    'molecule_atom_index_0_z_1_std',

    'molecule_atom_index_1_x_1_std',
    'molecule_atom_index_1_y_1_std',
    'molecule_atom_index_1_z_1_std',

    'x_0',
    'y_0',
    'z_0',

    'x_1',
    'y_1',
    'z_1',

    'molecule_atom_index_0_x_1_max_diff',
    'molecule_atom_index_0_y_1_max_diff',
    'molecule_atom_index_0_z_1_max_diff',

    'molecule_atom_index_1_x_1_max_diff',
    'molecule_atom_index_1_y_1_max_diff',
    'molecule_atom_index_1_z_1_max_diff',

    'molecule_atom_index_0_x_1_mean_div',
    'molecule_atom_index_0_y_1_mean_div',
    'molecule_atom_index_0_z_1_mean_div',

    'molecule_atom_index_1_x_1_mean_div',
    'molecule_atom_index_1_y_1_mean_div',
    'molecule_atom_index_1_z_1_mean_div',

    'molecule_atom_index_0_x_1_mean_diff',
    'molecule_atom_index_0_y_1_mean_diff',
    'molecule_atom_index_0_z_1_mean_diff',

    'molecule_atom_index_1_x_1_mean_diff',
    'molecule_atom_index_1_y_1_mean_diff',
    'molecule_atom_index_1_z_1_mean_diff',

    'molecule_atom_0_dist_min_diff',
    'molecule_atom_1_dist_min_diff',


    'molecule_type_dist_std_diff',
    'molecule_dist_min',


    'molecule_type_dist_min',
    'molecule_atom_1_dist_min_div',
    'molecule_dist_max',
    'molecule_atom_1_dist_std_diff',
    'molecule_type_dist_max',


    'molecule_type_0_dist_std_diff',
    'molecule_type_dist_mean_diff',
    'molecule_type_dist_mean_div',
    'type'
]


if __name__== '__main__':

    print("reading data...")
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    structures = pd.read_csv('../data/structures.csv')
    sub = pd.read_csv('../data/sample_submission.csv')

    rotated_molecules = pd.read_csv("../data/random_shift_rotation_structures.csv")
    for c in ['x', 'y', 'z']:
        structures[c] = rotated_molecules[c]

    print("mapping info about atoms...")

    debug = False
    if debug:
        train = train[:100]
        test = test[:100]

    train = map_atom_info(train, 0)
    train = map_atom_info(train, 1)

    test = map_atom_info(test, 0)
    test = map_atom_info(test, 1)

    train_p_0 = train[['x_0', 'y_0', 'z_0']].values
    train_p_1 = train[['x_1', 'y_1', 'z_1']].values
    test_p_0 = test[['x_0', 'y_0', 'z_0']].values
    test_p_1 = test[['x_1', 'y_1', 'z_1']].values

    train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
    test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
    train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
    test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
    train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
    test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
    train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
    test['dist_z'] = (test['z_0'] - test['z_1']) ** 2

    train['type_0'] = train['type'].apply(lambda x: x[0])
    test['type_0'] = test['type'].apply(lambda x: x[0])

    train = artgor_utils.reduce_mem_usage(train)
    test = artgor_utils.reduce_mem_usage(test)

    train = create_features_full(train)
    test = create_features_full(test)

    for f in ['atom_index_0', 'atom_index_1', 'atom_0', 'atom_1', 'type_0', 'type']:
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))


    print("creating folds...")
    n_folds = 10
    sorted_train = train.sort_values([
        "scalar_coupling_constant",
         "type",
         "dist",
    ])

    sorted_train.index = range(0, len(sorted_train))
    folds = KFold(n_splits=10, shuffle=True, random_state=0)

    X = sorted_train[good_columns]
    y = sorted_train['scalar_coupling_constant']
    X_test = test[good_columns]

    params = {
        'num_leaves': 128,
        'objective': 'regression',
        'learning_rate': 0.3,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.9,
        "bagging_seed": 11,
        "metric": 'mae',
        "verbosity": -1,
        'reg_alpha': 0.1302650970728192,
        'reg_lambda': 0.3603427518866501,
        'colsample_bytree': 1.0,
        'device': 'gpu',
        'gpu_device_id': 0,
    }

    del sorted_train,

    print("training models...")
    result_dict_lgb = artgor_utils.train_model_regression(X=X, X_test=X_test,
                                                          y=y,
                                                          params=params,
                                                          folds=folds,
                                                          model_type='lgb',
                                                          eval_metric='group_mae',
                                                          plot_feature_importance=True,
                                                          verbose=100,
                                                          early_stopping_rounds=1000,
                                                          n_estimators=30000)

    print("saving results...")
    np.save('../results/rotation_baseline_with_feats.npy', result_dict_lgb)

    print("making submission...")
    sub['scalar_coupling_constant'] = result_dict_lgb['prediction']
    sub.to_csv('../submissions/rotation_baseline_with_feats.csv', index=False)