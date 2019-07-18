import sys
sys.path.append("/root/champs/")
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions #Only needed if modifying defaults
DrawingOptions.bondLineWidth=1.8
from rdkit.Chem.rdmolops import SanitizeFlags

# https://github.com/jensengroup/xyz2mol
from utils.xyz2mol import xyz2mol, xyz2AC, AC2mol, read_xyz_file
from pathlib import Path
import pickle
from tqdm import tqdm


def chiral_stereo_check(mol):
    # avoid sanitization error e.g., dsgdb9nsd_037900.xyz
    Chem.SanitizeMol(mol, SanitizeFlags.SANITIZE_ALL - SanitizeFlags.SANITIZE_PROPERTIES)
    Chem.DetectBondStereochemistry(mol,-1)
    # ignore stereochemistry for now
    #Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    #Chem.AssignAtomChiralTagsFromStructure(mol,-1)
    return mol


def xyz2mol(atomicNumList,charge,xyz_coordinates,charged_fragments,quick):
    AC,mol = xyz2AC(atomicNumList,xyz_coordinates)
    new_mol = AC2mol(mol,AC,atomicNumList,charge,charged_fragments,quick)
    new_mol = chiral_stereo_check(new_mol)
    return new_mol


def MolFromXYZ(filename):
    charged_fragments = True
    quick = True
    cache_filename = CACHEDIR/f'{filename.stem}.pkl'
    if cache_filename.exists():
        return pickle.load(open(cache_filename, 'rb'))
    else:
        atomicNumList, charge, xyz_coordinates = read_xyz_file(filename)
        mol = xyz2mol(atomicNumList, charge, xyz_coordinates, charged_fragments, quick)
        # commenting this out for kernel to work.
        # for some reason kernel runs okay interactively, but fails when it is committed.
        pickle.dump(mol, open(cache_filename, 'wb'))
        return mol


def feature_atom(atom, prefix):
    prop = {}
    nb = [a.GetSymbol() for a in atom.GetNeighbors()] # neighbor atom type symbols
    nb_h = sum([_ == 'H' for _ in nb]) # number of hydrogen as neighbor
    nb_o = sum([_ == 'O' for _ in nb]) # number of oxygen as neighbor
    nb_c = sum([_ == 'C' for _ in nb]) # number of carbon as neighbor
    nb_n = sum([_ == 'N' for _ in nb]) # number of nitrogen as neighbor
    nb_na = len(nb) - nb_h - nb_o - nb_n - nb_c
    prop[prefix + '_degree'] = atom.GetDegree()
    prop[prefix + '_hybridization'] = int(atom.GetHybridization())
    prop[prefix + '_inring'] = int(atom.IsInRing()) # is the atom in a ring?
    prop[prefix + '_inring3'] = int(atom.IsInRingSize(3)) # is the atom in a ring size of 3?
    prop[prefix + '_inring4'] = int(atom.IsInRingSize(4)) # is the atom in a ring size of 4?
    prop[prefix + '_inring5'] = int(atom.IsInRingSize(5)) # ...
    prop[prefix + '_inring6'] = int(atom.IsInRingSize(6))
    prop[prefix + '_inring7'] = int(atom.IsInRingSize(7))
    prop[prefix + '_inring8'] = int(atom.IsInRingSize(8))
    prop[prefix + '_nb_h'] = nb_h
    prop[prefix + '_nb_o'] = nb_o
    prop[prefix + '_nb_c'] = nb_c
    prop[prefix + '_nb_n'] = nb_n
    prop[prefix + '_nb_na'] = nb_na
    return prop


def process_rdkit_df(df):
    res_list = []
    previous_molecule = None
    failed_force = 0

    for count_rows, row in tqdm(df.iterrows(), total=len(df)):
        results = {}
        molecule_name = row['molecule_name']
        results['molecule_name'] = molecule_name

        if previous_molecule is None or previous_molecule != molecule_name:
            file_name = PATH/f'structures_dir/{molecule_name}.xyz'
            m = MolFromXYZ(file_name)

        for atom_ind in range(2):
            results[f'atom_index_{atom_ind}'] = row[f'atom_index_{atom_ind}']
            atom = m.GetAtomWithIdx(row[f'atom_index_{atom_ind}'])
            feat_atom = feature_atom(atom, prefix=f'atom_index_{atom_ind}')
            results.update(feat_atom)

        res_list.append(results)
        previous_molecule = molecule_name

    print(f"Failed to evaluate force on {failed_force} pairs")
    return pd.DataFrame(res_list)


if __name__ == '__main__':
    CACHEDIR = Path('../data/pickle_structures_dir')
    PATH = Path('../data')

    structures = pd.read_csv("../data/structures.csv")
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")

    debug = False
    if debug:
        train = train[:100]
        test = test[:100]

    ase_train = process_rdkit_df(train)
    ase_train.to_csv("../data/rdkit_train_feats.csv", index=False)

    ase_test = process_rdkit_df(test)
    ase_test.to_csv("../data/rdkit_test_feats.csv", index=False)