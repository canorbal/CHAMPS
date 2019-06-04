import os
import sys
sys.path.append("/root/champs/")

import numpy as np
import pandas as pd
import tqdm
import warnings
warnings.filterwarnings("ignore")


def rotate_x(points, phi):
    R_x = np.array([
        [1., 0., 0., ],
        [0., np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])

    return points.dot(R_x)


def rotate_y(points, phi):
    R_y = np.array([
        [np.cos(phi), 0., np.sin(phi)],
        [0., 1., 0., ],
        [-np.sin(phi), 0., np.cos(phi)]
    ])

    return points.dot(R_y)


def rotate_z(points, phi):
    R_z = np.array([
        [np.cos(phi), -np.sin(phi), 0.],
        [np.sin(phi), np.cos(phi), 0.],
        [0., 0., 1.]
    ])

    return points.dot(R_z)


if __name__ == "__main__":
    structures = pd.read_csv('../data/structures.csv')
    molecules = structures.groupby("molecule_name")
    np.random.seed(0)

    cols = ['x', 'y', 'z']
    axes_rotation_function = [rotate_x, rotate_y, rotate_z]

    df_list = []

    for name, df in tqdm.tqdm(molecules):

        random_phi = np.random.uniform(-np.pi, np.pi)
        random_rotation_function = np.random.randint(0, len(
            axes_rotation_function))
        random_rotation_function = axes_rotation_function[
            random_rotation_function]

        new_coordinates = random_rotation_function(df[cols], random_phi)

        random_shift = np.random.uniform(-1., 1., 3)
        new_coordinates = new_coordinates + random_shift

        new_coordinates.columns = ['x', 'y', 'z']
        new_coordinates['molecule_name'] = name
        df_list.append(new_coordinates)

    res = pd.concat(df_list, ignore_index=True)
    res.to_csv("../data/random_shift_rotation_structures.csv", index=False)
