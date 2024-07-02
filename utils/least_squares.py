"""
Author:         Samuel Lovett
Date Created:   Apr 26, 2024
Last Edited:    Apr 26, 2024

Description:
Flexible script which performs Least-Squares and Temporally Weighted Least Squares regression depending on the arguments provided.
Program described executes the logic described in Sec. II A. and C. of [1]

[1] Enhancing Doppler Ego-Motion Estimation: A Temporally Weighted Approach to RANSAC
"""

import numpy as np


def build_A_matrix(values):
    '''
    Assemble the azimuth values into the A matrix in (5) from [1]
    :param values: theta values to build into an matrix
    :return: A matrix used in the LSQ or TWLSQ solution to Ax=b
    '''
    A_mat_col_0 = np.cos(values).reshape(-1, 1)
    A_mat_col_1 = np.sin(values).reshape(-1, 1)
    A_mat = np.concatenate((A_mat_col_0, A_mat_col_1), axis=1)
    return A_mat


def least_squares_velocity_analysis(theta_val, doppler, weights=None):
    """
    Least-Squares and Temporally Weighted Least Squares regression depending on the arguments provided.

    :param theta_val: array of theta values
    :param doppler: array of doppler values
    :param weights: array of weights corresponding to each theta value
    :return: Vx, Vy, Velocity, angle_rad
    """
    A = build_A_matrix(theta_val)
    b = doppler

    # calculate the weighted A and b matrix to perform W-LSQ
    if weights is not None:
        WA = weights[:, np.newaxis]
        A = A * np.sqrt(WA)
        b = b * np.sqrt(weights)

    Vx, Vy = np.linalg.lstsq(A, b, rcond=None)[0]
    Velocity = np.sqrt(Vx ** 2 + Vy ** 2)
    angle_rad = np.arctan2(Vy, Vx)
    return Vx, Vy, Velocity, angle_rad
