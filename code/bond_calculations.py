import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def map_structures(df, atom_idx):
    df = pd.merge(df, structures, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])

    # df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df


def del_cols(df, cols):
    del_cols_list_ = [l for l in cols if l in df]
    df = df.drop(del_cols_list_,axis=1)
    return df


if __name__ == "__main__":

    debug = False
    if debug:
        nrows = 100
    else:
        nrows = None

print("reading data...")
file_folder = '../data'
train = pd.read_csv(f'{file_folder}/train.csv', nrows=nrows)
test = pd.read_csv(f'{file_folder}/test.csv', nrows=nrows)
sub = pd.read_csv(f'{file_folder}/sample_submission.csv', nrows=nrows)
structures = pd.read_csv(f'{file_folder}/structures.csv', nrows=nrows)
scalar_coupling_contributions = pd.read_csv(f'{file_folder}/scalar_coupling_contributions.csv', nrows=nrows)

train = pd.merge(train, scalar_coupling_contributions, how = 'left',
                  left_on  = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],
                  right_on = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])

atomic_radius = {'H':0.38, 'C':0.77, 'N':0.75, 'O':0.73, 'F':0.71}
fudge_factor = 0.05
atomic_radius = {k:v + fudge_factor for k,v in atomic_radius.items()}
electronegativity = {'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98}

atoms = structures['atom'].values
atoms_en = [electronegativity[x] for x in tqdm(atoms)]
atoms_rad = [atomic_radius[x] for x in tqdm(atoms)]

structures['EN'] = atoms_en
structures['rad'] = atoms_rad

atoms = structures['atom'].values
atoms_en = [electronegativity[x] for x in tqdm(atoms)]
atoms_rad = [atomic_radius[x] for x in tqdm(atoms)]

structures['EN'] = atoms_en
structures['rad'] = atoms_rad

i_atom = structures['atom_index'].values
p = structures[['x', 'y', 'z']].values
p_compare = p
m = structures['molecule_name'].values
m_compare = m
r = structures['rad'].values
r_compare = r

source_row = np.arange(len(structures))
max_atoms = 28

bonds = np.zeros((len(structures) + 1, max_atoms + 1), dtype=np.int8)
bond_dists = np.zeros((len(structures) + 1, max_atoms + 1), dtype=np.float32)

print('Calculating the bonds')

for i in tqdm(range(max_atoms - 1)):
    p_compare = np.roll(p_compare, -1, axis=0)
    m_compare = np.roll(m_compare, -1, axis=0)
    r_compare = np.roll(r_compare, -1, axis=0)

    mask = np.where(m == m_compare, 1,
                    0)  # Are we still comparing atoms in the same molecule?
    dists = np.linalg.norm(p - p_compare, axis=1) * mask
    r_bond = r + r_compare

    bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)

    source_row = source_row
    target_row = source_row + i + 1  # Note: Will be out of bounds of bonds array for some values of i
    target_row = np.where(
        np.logical_or(target_row > len(structures), mask == 0),
        len(structures), target_row)  # If invalid target, write to dummy row

    source_atom = i_atom
    target_atom = i_atom + i + 1  # Note: Will be out of bounds of bonds array for some values of i
    target_atom = np.where(np.logical_or(target_atom > max_atoms, mask == 0),
                           max_atoms,
                           target_atom)  # If invalid target, write to dummy col

    bonds[(source_row, target_atom)] = bond
    bonds[(target_row, source_atom)] = bond
    bond_dists[(source_row, target_atom)] = dists
    bond_dists[(target_row, source_atom)] = dists

bonds = np.delete(bonds, axis=0, obj=-1)  # Delete dummy row
bonds = np.delete(bonds, axis=1, obj=-1)  # Delete dummy col
bond_dists = np.delete(bond_dists, axis=0, obj=-1)  # Delete dummy row
bond_dists = np.delete(bond_dists, axis=1, obj=-1)  # Delete dummy col

print('Counting and condensing bonds')

bonds_numeric = [[i for i, x in enumerate(row) if x] for row in tqdm(bonds)]
bond_lengths = [[dist for i, dist in enumerate(row) if i in bonds_numeric[j]]
                for j, row in enumerate(tqdm(bond_dists))]
bond_lengths_mean = [np.mean(x) for x in bond_lengths]
n_bonds = [len(x) for x in bonds_numeric]

bond_data = {'n_bonds': n_bonds, 'bond_lengths_mean': bond_lengths_mean}
bond_df = pd.DataFrame(bond_data)
structures = structures.join(bond_df)

train = map_structures(train, 0)
train = map_structures(train, 1)

test = map_structures(test, 0)
test = map_structures(test, 1)

train.to_csv(f"{file_folder}/bonds_train.csv", index=None)
test.to_csv(f"{file_folder}/bonds_test.csv", index=None)
