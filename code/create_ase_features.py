import os
import sys
sys.path.append("/root/champs/")

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
pd.options.display.precision = 15
import warnings
warnings.filterwarnings("ignore")


from utils import artgor_utils
from utils.train_utils import process_ase_df


if __name__ == '__main__':
    structures = pd.read_csv("../data/structures.csv")
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")

    debug = False
    if debug:
        train = train[:100]
        test = test[:100]

    ase_train = process_ase_df(train)
    ase_train.to_csv("../data/ase_train_feats.csv", index=False)

    ase_test = process_ase_df(test)
    ase_test.to_csv("../data/ase_test_feats.csv", index=False)
