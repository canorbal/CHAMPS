import pandas as pd
import numpy as np
from utils import artgor_utils
import ase
import ase.io
from ase.calculators.emt import EMT
from ase.eos import calculate_eos
from tqdm import tqdm

good_columns = [
    'atom_index_1',
    'atom_index_0',
    'type',
    'atom_0',
    'atom_1',

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


def process_ase_df(df):
    res_list = []
    k_top = 4
    previous_molecule = None
    failed_force = 0

    for count_rows, row in tqdm(df.iterrows(), total=len(df)):
        results = {}
        ids_list = []
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
            try:
                forces = atoms.get_forces()[row[f'atom_index_{atom_ind}']]
            except Exception:
                forces = np.array([np.NaN] * 3)
                failed_force += 1

            for i, dim_name in enumerate(['x', 'y', 'z']):
                results[f'atom_index_{atom_ind}_force_{dim_name}'] = forces[i]

            idx = np.argsort(distances)[1:k_top]
            for i in range(len(idx)):
                prefix = f'atom_index_{atom_ind}_nbhd_{i}'
                results[prefix + '_dist'] = distances[idx[i]]
                results[prefix + '_atom'] = atoms[idx[i]].number

            ids_list.append(idx)

        angle1 = atoms.get_angle(ids_list[0][0], row['atom_index_0'],
                                 row['atom_index_1'])
        angle2 = atoms.get_angle(ids_list[1][0], row['atom_index_1'],
                                 row['atom_index_0'])
        results['angle_1'] = angle1
        results['angle_2'] = angle2
        res_list.append(results)
        previous_molecule = molecule_name

    print(f"Failed to evaluate force on {failed_force} pairs")
    return pd.DataFrame(res_list)


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