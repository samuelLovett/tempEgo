"""
Author:         Samuel Lovett
Date Created:   Apr 26, 2024
Last Edited:    Apr 26, 2024

Description:
Flexible script which performs squared error and Temporally Weighted Squared error calculations.
Program described executes (10) and (16) from [1]

[1] Enhancing Doppler Ego-Motion Estimation: A Temporally Weighted Approach to RANSAC
"""
import numpy as np


def square_error_loss(y_true, y_pred, weights=None):

    if weights is None:
        weight = np.ones(np.reshape(y_true, (-1, 1)).shape)
    else:
        weight = np.reshape(weights, (-1, 1))

    y_true = np.reshape(y_true, (-1, 1))
    y_pred = np.reshape(y_pred, (-1, 1))
    error_loss = weight * (y_pred - y_true) ** 2
    return error_loss


def mean_square_error(y_true, y_pred, weights=None):
    return np.sum(square_error_loss(y_true, y_pred, weights)) / y_true.shape[0]
