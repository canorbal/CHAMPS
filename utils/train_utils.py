import pandas as pd
import numpy as np
from utils import artgor_utils
import ase
import ase.io
from ase.calculators.emt import EMT
from ase.eos import calculate_eos
from tqdm import tqdm
from sklearn import metrics
from utils.artgor_utils import group_mean_log_mae
import csv
import gc
import lightgbm as lgb
import xgboost as xgb


# def concat_dataframes(a, b):
#     cols_to_add = [col for col in b.columns if col not in a]
#     return pd.ap([a, b[cols_to_add]], axis=1)


def concat_stupidly(a, b):
    cols_to_add = [col for col in b.columns if col not in a]
    for col in cols_to_add:
        a.loc[:, col] = b[col].values
    return a


def get_best_columns(result_file, top_k=50):
    results = np.load(result_file, allow_pickle=True)
    results = results.item()
    cols = results['feature_importance'].groupby(['feature'])[
        'importance'].mean().sort_values(ascending=False)

    cols = list(cols.index)[:top_k]
    return cols


def map_atom_info(df, structures, atom_idx):
    df = pd.merge(df, structures, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])

    df = df.drop('atom_index', axis=1)
    return df


def find_useless_cols(df):
    res = []
    for col in df.columns:
        if df[col].nunique() == 1:
            res.append(col)

    return res


def oof_features(df):
    num_cols = ['fc', 'sd', 'pso', 'dso']

    cat_cols = ['type', 'atom_index_0', 'atom_index_1']
    aggs = ['mean', 'max', 'std', 'min']

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


def process_ase_df(df, filename):
    k_top = 7
    previous_molecule = None
    failed_force = 0

    with open(filename, 'w') as f:

        for count_rows, row in tqdm(df.iterrows(), total=len(df)):
            results = {}
            molecule_name = row['molecule_name']
            results['molecule_name'] = molecule_name

            if previous_molecule is None or previous_molecule != molecule_name:
                file_name = f'../data/structures_dir/{molecule_name}.xyz'
                atoms = ase.io.read(file_name)
                atoms.set_calculator(EMT())

            for atom_ind in range(2):
                results[f'atom_index_{atom_ind}'] = row[f'atom_index_{atom_ind}']

                distances = atoms.get_distances(row[f'atom_index_{atom_ind}'],
                                                range(len(atoms)))
                distances = distances.astype(np.float16)
                inv_distances = 1. / distances
                inv_2_distances = 1. / (distances**2)
                try:
                    forces = atoms.get_forces()[row[f'atom_index_{atom_ind}']]
                except Exception:
                    forces = np.array([np.NaN] * 3)
                    failed_force += 1

                for i, dim_name in enumerate(['x', 'y', 'z']):
                    results[f'atom_index_{atom_ind}_force_{dim_name}'] = np.float16(forces[i])

                idx = np.argsort(distances)[1:k_top]

                for i in range(k_top):
                    prefix = f'atom_index_{atom_ind}_nbhd_{i}'
                    if i < len(idx):
                        results[prefix + '_dist'] = distances[idx[i]]
                        results[prefix + '_1_div_dist'] = inv_distances[idx[i]]
                        results[prefix + '_1_div_dist^2'] = inv_2_distances[idx[i]]

                        for j in range(len(idx)):
                            if i != j:
                                results[prefix + '_angle'] = np.float16(atoms.get_angle(
                                    idx[i], row[f'atom_index_{atom_ind}'], idx[j]
                                ))

                                results[prefix + '_cos_angle'] = np.float16(np.cos(results[prefix + '_angle']))
                                results[prefix + '_sin_angle'] = np.float16(np.sin(results[prefix + '_angle']))
                    else:
                        results[prefix + '_dist'] = np.NaN
                        results[prefix + '_1_div_dist'] = np.NaN
                        results[prefix + '_1_div_dist^2'] = np.NaN

                        for j in range(len(idx)):
                            if i != j:
                                results[prefix + '_angle'] = np.NaN

                                results[prefix + '_cos_angle'] = np.NaN
                                results[prefix + '_sin_angle'] = np.NaN

            if count_rows == 0:
                writer = csv.DictWriter(f, fieldnames=results.keys())
                writer.writeheader()
                writer.writerow(results)
            else:
                writer.writerow(results)

            previous_molecule = molecule_name

    print(f"Failed to evaluate force on {failed_force} pairs")


def process_symmetry(df):
    previous_molecule = None

    Rc = 10.
    p = [(0.4, 0.2), (0.4, 0.5), (0.4, 1.0), ]
    res_list = []

    def fc(Rij, Rc):
        y_1 = 0.5 * (np.cos(np.pi * Rij[Rij <= Rc] / Rc) + 1)
        y_2 = Rij[Rij > Rc] * 0
        y = np.concatenate((y_1, y_2))
        return y

    def fc_vect(Rij, Rc):
        return np.where(Rij <= Rc, 0.5 * (np.cos(np.pi * Rij / Rc) + 1),
                        0).sum(1)

    def get_G2(Rij, eta, Rs):
        return np.exp(-eta * (Rij - Rs) ** 2) * fc(Rij, Rc)

    for count_rows, row in tqdm(df.iterrows(), total=len(df)):
        results = {}
        molecule_name = row['molecule_name']
        results['molecule_name'] = molecule_name

        if previous_molecule is None or previous_molecule != molecule_name:
            file_name = f'../data/structures_dir/{molecule_name}.xyz'
            atoms = ase.io.read(file_name)
            natoms = len(atoms)

        all_distances = atoms.get_all_distances()
        G1 = fc_vect(all_distances, Rc)

        G2 = np.zeros((natoms, len(p)))
        for i in range(natoms):
            for j, (eta, Rs) in enumerate(p):
                G2[i, j] = get_G2(all_distances[i], eta, Rs).sum()

        for atom_ind in range(2):
            index_value = row[f'atom_index_{atom_ind}']
            results[f'atom_index_{atom_ind}'] = index_value
            results[f'atom_index_{atom_ind}_g1_factor'] = G1[index_value]

            for j in range(len(p)):
                results[f'atom_index_{atom_ind}_g2_factor_{j}_index'] = G2[
                    index_value, j]

        res_list.append(results)
        previous_molecule = molecule_name

    return pd.DataFrame(res_list)


def train_lgbm_on_fold(
        X_train, y_train, X_val, y_val, n_estimators,
        params, verbose=50, early_stopping_rounds=100, columns=None,
):
    columns = X_train.columns if columns is None else columns
    X_train, X_val = X_train[columns], X_val[columns]

    model = lgb.LGBMRegressor(**params, n_estimators=n_estimators)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric='mae',
        verbose=verbose, early_stopping_rounds=early_stopping_rounds
    )

    return model


def permutation_importance(
        model, X_val, y_val, metric, n_folds, columns=None,
        minimize=True, verbose=True, results=None):

    if results is None:
        results = {}

    columns = X_val.columns if columns is None else columns
    y_pred = model.predict(X_val[columns], num_iteration=model.best_iteration_)

    if 'base_score' in results:
        results['base_score'] += metric(y_val, y_pred) / n_folds
    else:
        results['base_score'] = metric(y_val, y_pred) / n_folds

    if verbose:
        print(f'Base score {results["base_score"]:.5}')

    for col in tqdm(columns):
        print(f'col {col} computing')
        freezed_col = X_val[col].copy()

        if isinstance(X_val[col].dtype, pd.api.types.CategoricalDtype):
            X_val[col] = np.random.permutation(X_val[col])
            X_val[col] = X_val[col].astype('category')
        else:
            X_val[col] = np.random.permutation(X_val[col])

        preds = model.predict(X_val[columns], num_iteration=model.best_iteration_)
        metric_value = metric(y_val, preds) / n_folds
        if col in results:
            results[col] += metric_value
        else:
            results[col] = metric_value

        X_val[col] = freezed_col

        if verbose:
            print(f'column: {col} - {metric_value:.5}')

    return results


def process_oof_results(len_test=2505542, n_folds=20):
    total_importance = None
    oof_train = []
    oof_test = np.zeros((len_test, n_folds))

    for i in range(n_folds):
        results = np.load(
            f"../data/oof_tables/v1_oof_results/fold_{i}_scalar_coupling_constant_result.npy",
            allow_pickle=True
        )

        results = results.item()
        cols = results['feature_importance'].groupby(['feature'])['importance'].mean().sort_index(ascending=False)

        if total_importance is None:
            total_importance = cols
        else:
            total_importance = total_importance + cols

        oof_train.append(results['oof'])
        oof_test[:, i] = results['prediction']

    total_importance = total_importance.sort_values(ascending=False)
    gc.collect()

    return {
        'total_importance': total_importance,
        'oof_train_list': oof_train,
        'oof_test': oof_test,
    }

