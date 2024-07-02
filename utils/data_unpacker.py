"""
Author:         Samuel Lovett
Date Created:   Apr 26, 2024
Last Edited:    Apr 26, 2024

Description:
Assemble the multidimensional array and break it into data axis corresponds to [theta, velocity]. This will seem odd in
the context of the code because it is. I assemble the tensor stack just to dismantle it. But it was the easiest way to
combine all of the measurements into one long vector for each data axis while still
preserving the shape of the buffer needed for TEMPSAC initial selection.

[1] Enhancing Doppler Ego-Motion Estimation: A Temporally Weighted Approach to RANSAC
"""

import numpy as np


def data_unpacker_tensor_object(structured_data):
    """
    param[in] structured_data: Sliding window as a list of measurements

    \n return azimuth_val: List of azimuth values corresponding to each sample in the sliding window
    \n return doppler_val: List of doppler values corresponding to each sample in the sliding window
    :param structured_data: Sliding window as a list of measurements
    :return: azimuth_val, doppler_val
    """
    tensor_stack = np.concatenate(list(structured_data), axis=1)
    azimuth_val = tensor_stack[0, :]
    doppler_val = tensor_stack[1, :]

    if doppler_val.size == azimuth_val.size:
        # print(doppler_val.size)
        return azimuth_val, doppler_val
    else:
        raise ValueError("The size of the doppler_val array does not match the theta array")