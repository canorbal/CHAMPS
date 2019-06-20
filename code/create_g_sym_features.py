import os
import sys
sys.path.append("/root/champs/")

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
pd.options.display.precision = 15
import warnings
warnings.filterwarnings("ignore")

from utils.train_utils import process_symmetry


if __name__ == '__main__':
    structures = pd.read_csv("../data/structures.csv")
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")

    debug = False
    if debug:
        train = train[:100]
        test = test[:100]

    sym_train = process_symmetry(train)
    sym_train.to_csv("../data/symmetry_train_feats.csv", index=False)

    sym_test = process_symmetry(test)
    sym_test.to_csv("../data/symmetry_test_feats.csv", index=False)