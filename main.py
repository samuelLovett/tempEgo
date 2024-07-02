"""
Author:         Samuel Lovett
Date Created:   Apr 26, 2024
Last Edited:    Apr 26, 2024

Description:
Estimate the 2D body-frame velocity vector of the sensor platform given input data from a single radar as described in
[1]. For use with your own radar sensor modify the code in the update_buffer_weighted method.

The velocity estimation scheme takes the following approach:

1. A weighting is applied to all datapoints within the sliding window based on the datapoints corresponding
measurement-index

2. A RANSAC outlier rejection method is used to filter the data which contains a large amount of
outliers. RANSAC selects points uniformly or based on their weights to estimate the velocity. This process is then
repeated until a first-pass velocity estimate is derived from the best inlier set.

3. Least Squares Distance Regression (LSQ) is initialized with the inlier set and used to calculate the final velocity
estimate of the body frame linear velocity.

[1] Enhancing Doppler Ego-Motion Estimation: A Temporally Weighted Approach to RANSAC
"""

from utils.error_and_loss_function import square_error_loss, mean_square_error
from utils.external_dataset_import import *
from utils import RANSAC
import numpy as np
from collections import deque
import time


def set_KB(dataset):
    """
    Define the KB method parameter's as described in Sec. III Hyperparameter optimization of [1]
    :param dataset: path to dataset being evaluated
    :return: The velocity estimation object
    """
    n = 2  # Minimum number of data points to estimate parameters
    k = 777  # Maximum iterations allowed
    epsilon = 1.01389316572299  # Threshold value to determine if points are fit well
    z = 16  # Number of points to assert that the model fits well
    start_idx = 0
    stop_idx = None
    method = VelocityEstimation(dataset, start_idx, stop_idx, n, k, epsilon, z)
    return method


def set_TWLSQ(dataset):
    """
    Define the Temporally Weighted Least Squares method parameter's as described in Sec. III Hyperparameter
    optimization of [1]
    :param dataset: path to dataset being evaluated
    :return: The velocity estimation object
    """
    m = 8  # Buffer size
    fff_lambda = 0.13030964595712677  # 0 values only the most recent samples and 1 values all samples equally
    n = 2  # Minimum number of data points to estimate parameters
    k = 727  # Maximum iterations allowed
    epsilon = 4.167547701335022  # Threshold value to determine if points are fit well
    z = 25  # Number of points to assert that the model fits well
    start_idx = 0
    stop_idx = None

    method = TWLSQVelEstimate(dataset, start_idx,
                              stop_idx,
                              m,
                              fff_lambda,
                              n,
                              k,
                              epsilon,
                              z)

    return method


def set_TEMPSAC(dataset):
    """
    Define the TEMPoral SAmpling Concensus parameter's as described in Sec. III Hyperparameter
    optimization of [1]
    :param dataset: path to dataset being evaluated
    :return: The velocity estimation object
    """
    m = 2  # Buffer size
    fff_lambda = 0.45866536998552904  # 0 values only the most recent samples and 1 values all samples equally
    n = 3  # Minimum number of data points to estimate parameters
    k = 692  # Maximum iterations allowed
    epsilon = 1.022953948161558  # Threshold value to determine if points are fit well
    z = 25  # Number of points to assert that the model fits well
    start_idx = 0
    stop_idx = None

    method = TEMPSACVelEstimate(dataset,
                                start_idx,
                                stop_idx,
                                m,
                                fff_lambda,
                                n,
                                k,
                                epsilon,
                                z)

    return method


def get_subset_of_weights_normalized(buffer_length, fff_lambda):
    """
    calculates the normalized weight value for each measurement index when the sliding window is not full
    :param buffer_length: length of the sliding window labelled as m in [1]
    :param fff_lambda: fixed factor lambda as defined in (14) of [1]
    :return: list of weight values normalized by the number of measurements in the sliding window
    """

    weights = []
    buffer_length_indexed_from_zero = buffer_length - 1
    for buffer_index in range(buffer_length):
        weight_at_i = fff_lambda ** (buffer_length_indexed_from_zero - buffer_index)
        weights.append(weight_at_i)

    sum_of_weights = sum(weights)
    normalized_weights = [weights[i] / sum_of_weights for i in range(len(weights))]
    normalized_weights.reverse()
    return normalized_weights


class VelocityEstimation:
    """
    Standard velocity estimation class which implements the KB method as shown in [1]
    """

    def __init__(self, __dataset, __start_idx, __stop_idx, __n, __k, __epsilon, __z):
        """
        Initialization of the standard velocity estimation method (KB) as described at the beginning of this code.
        :param __dataset: path to dataset being evaluated
        :param __start_idx: measurement index to start the evaluation at
        :param __stop_idx: measurement index to stop the evaluation at
        :param __n: Minimum number of data points to estimate parameters
        :param __k: Maximum iterations allowed
        :param __epsilon: Threshold value to determine if points are fit well
        :param __z: Number of close data points required to assert model fits well
        """
        calib_dir = 'coloradar_package//dataset//calib'
        self.coloradar_dataset = ColoradarDataset(__dataset, calib_dir)
        if __stop_idx is None:
            self.stop = self.coloradar_dataset.dataset_length
        else:
            self.stop = __stop_idx
        self.start_idx = __start_idx

        self.ransac = RANSAC.RANSAC(loss=square_error_loss,
                                    metric=mean_square_error,
                                    n=__n,
                                    k=__k,
                                    epsilon=__epsilon,
                                    z=__z)

    def update_buffer(self, idx):
        """
        Update the sliding window as described in Sec. II C. of [1]. Index 0 is the most recent measurement in time and the last element in the buffer the oldest measurement in time. \n
        \n param[in] idx: measurement index to import from coloradar
        \n return tensor_stack: multidimensional array dimension 1 is [theta, vd], and dimension 2 is the sample index
        \n return weight_mask: a list of weight values corresponding to each sample in tensor_stack.\n
        For use with your own data replace the line: current_sample = self.coloradar_dataset.get_radar_cloud(idx) and data must be formatted as a 2D list [theta, doppler].
        \t theta is a list of azimuth values in rads. \n
        \t doppler is a list of doppler velocity values in m/s.

        :param idx: current measurement index used to import values from the dataset
        :return: tensor_stack, weight_mask
        """

        # Get the new measurement
        current_sample = self.coloradar_dataset.get_radar_cloud(idx)
        return current_sample

    def ego_velocity(self):
        """
        Updates the sensor values and calculates the ego velocity of the platform.
        """
        ransac = self.ransac
        idx = self.start_idx
        stop = self.stop

        while idx < stop:
            radar_cloud = self.update_buffer(idx)
            velocities_at_idx = ransac.separate_points(radar_cloud)
            print(velocities_at_idx)
            idx = idx + 1


class TWLSQVelEstimate(VelocityEstimation):
    """
    TWLSQ velocity estimation class which inherits the initialization of the standard method but uses TWLSQ_RANSAC
    as shown in [1].
    """

    def __init__(self, __dataset, __start_idx, __stop_idx, __buffer_size,
                 __fff_lambda, __n, __k, __epsilon, __z):
        """
        Initialization of the velocity estimation method as described at the beginning of this code.
        :param __dataset: path to dataset being evaluated
        :param __start_idx: measurement index to start the evaluation at
        :param __stop_idx: measurement index to stop the evaluation at
        :param __buffer_size: Size of the sliding window, m in [1]
        :param __fff_lambda: fixed forgetting factor as defined in (14) of [1]. 0 uses only the most recent samples and 1 uses all samples equally.
        :param __n: Minimum number of data points to estimate parameters
        :param __k: Maximum iterations allowed
        :param __epsilon: Threshold value to determine if points are fit well
        :param __z: Number of close data points required to assert model fits well
        """
        super().__init__(__dataset, __start_idx, __stop_idx, __n, __k, __epsilon, __z)

        fff_lambda = __fff_lambda

        # pre-calculate all the Fixed Forgetting Factor (FFF) weights for my buffer
        # FFF definition (14) of [1]
        # To align the calculated weights with their corresponding time step the weights list needs to be reversed.
        # Index 0 is kept the most recent sample in time
        # Index n (corresponding to the last element in the buffer) is kept as the oldest sample in time
        weights = []
        buffer_size_indexed_from_zero = __buffer_size - 1
        for buffer_index in range(__buffer_size):
            weight_at_i = fff_lambda ** (buffer_size_indexed_from_zero - buffer_index)
            weights.append(weight_at_i)
        sum_of_weights = sum(weights)
        normalized_weights = [weights[i] / sum_of_weights for i in range(len(weights))]
        normalized_weights.reverse()

        self.ransac = RANSAC.RANSAC_TWLSQ(loss=square_error_loss,
                                          metric=mean_square_error,
                                          n=__n,
                                          k=__k,
                                          epsilon=__epsilon,
                                          z=__z)
        self.normalized_exponential_weights_list = normalized_weights
        self.buffer_size = __buffer_size
        self.data_package = []
        self.buffer_of_sample_tensors = deque()
        self.fff_lambda = fff_lambda

    def update_buffer(self, idx):
        """
        Update the sliding window as described in Sec. II C. of [1]. Index 0 is the most recent measurement in time and the last element in the buffer the oldest measurement in time. \n
        \n param[in] idx: measurement index to import from coloradar
        \n return tensor_stack: multidimensional array dimension 1 is [theta, vd], and dimension 2 is the sample index
        \n return weight_mask: a list of weight values corresponding to each sample in tensor_stack.\n
        For use with your own data replace the line: current_sample = self.coloradar_dataset.get_radar_cloud(idx) and data must be formatted as a 2D list [theta, doppler].
        \t theta is a list of azimuth values in rads. \n
        \t doppler is a list of doppler velocity values in m/s.

        :param idx: current measurement index used to import values from the dataset
        :return: tensor_stack, weight_mask
        """

        # Get the new measurement
        current_sample = self.coloradar_dataset.get_radar_cloud(idx)
        buffer_length = len(self.buffer_of_sample_tensors)

        current_buffer = self.buffer_of_sample_tensors

        # append the current measurement to the buffer
        if buffer_length < self.buffer_size:
            current_buffer.appendleft(current_sample)
            buffer_length = buffer_length + 1
            weights = get_subset_of_weights_normalized(buffer_length, self.fff_lambda)
        else:
            # Remove the oldest measurement from the list
            # Update the buffer to include the new measurement
            current_buffer.pop()
            current_buffer.appendleft(current_sample)
            weights = self.normalized_exponential_weights_list

        # update the corresponding weight for each sample within the buffer depending on its measurement index
        weight_mask = np.array([])
        for i in range(buffer_length):
            mask_at_i = np.ones(len(current_buffer[i][0])) * weights[i]
            weight_mask = np.concatenate((weight_mask, mask_at_i))

        # Assemble the multidimensional array
        reshaped_tensor_stack = np.concatenate(list(self.buffer_of_sample_tensors), axis=1)
        return reshaped_tensor_stack[0, :], reshaped_tensor_stack[1, :], weight_mask  # break it into theta, vd, and w


class TEMPSACVelEstimate(TWLSQVelEstimate):
    """
    TWLSQ velocity estimation class which inherits the initialization of the standard method and the TWLSQ method but
    uses TEMPSAC as shown in [1].
    """

    def __init__(self, __dataset, __start_idx, __stop_idx, __buffer_size, __fff_lambda, __n, __k, __epsilon, __z):
        """
        Initialization of the velocity estimation method as described at the beginning of this code.
        :param __dataset: path to dataset being evaluated
        :param __start_idx: measurement index to start the evaluation at
        :param __stop_idx: measurement index to stop the evaluation at
        :param __buffer_size: Size of the sliding window, m in [1]
        :param __fff_lambda: fixed forgetting factor as defined in (14) of [1]. 0 uses only the most recent samples and 1 uses all samples equally.
        :param __n: Minimum number of data points to estimate parameters
        :param __k: Maximum iterations allowed
        :param __epsilon: Threshold value to determine if points are fit well
        :param __z: Number of close data points required to assert model fits well
        """
        super().__init__(__dataset, __start_idx, __stop_idx, __buffer_size, __fff_lambda, __n, __k, __epsilon, __z)
        self.ransac = RANSAC.TEMPSAC(loss=square_error_loss,
                                     metric=mean_square_error,
                                     n=__n,
                                     k=__k,
                                     epsilon=__epsilon,
                                     z=__z)

    def update_buffer(self, idx):
        """
        Update the sliding window as described in Sec. II C. of [1]. Index 0 is the most recent measurement in time and the last element in the buffer the oldest measurement in time. \n
        \n param[in] idx: measurement index to import from coloradar
        \n return tensor_stack: multidimensional array dimension 1 is [theta, vd], and dimension 2 is the sample index
        \n return weights: a list of weight values corresponding to each measurement within the sliding window. \n
        For use with your own data replace the line: current_sample = self.coloradar_dataset.get_radar_cloud(idx) and data must be formatted as a 2D list [theta, doppler].
        \t theta is a list of azimuth values in rads. \n
        \t doppler is a list of doppler velocity values in m/s.

        :param idx: current measurement index used to import values from the dataset
        :return: tensor_stack, weights
        """

        current_sample = self.coloradar_dataset.get_radar_cloud(idx)  # Get the new item from the sensor

        buffer_length = len(self.buffer_of_sample_tensors)
        current_buffer = self.buffer_of_sample_tensors

        if buffer_length < self.buffer_size:
            current_buffer.appendleft(current_sample)
            buffer_length = buffer_length + 1
            weights = get_subset_of_weights_normalized(buffer_length, self.fff_lambda)
        else:
            current_buffer.pop()  # Remove the oldest item from the list
            current_buffer.appendleft(current_sample)  # Update the buffer to include the new item
            weights = self.normalized_exponential_weights_list

        weight_mask = np.array([])
        for i in range(buffer_length):
            mask_at_i = np.ones(len(current_buffer[i][0])) * weights[i]
            weight_mask = np.concatenate((weight_mask, mask_at_i))

        return self.buffer_of_sample_tensors, weights


def main():
    datasets_dic = {'classroom': 'coloradar_package/dataset/2_23_2021_edgar_classroom_run4/',
                    'irl': 'coloradar_package/dataset/12_21_2020_arpg_lab_run1/',
                    'army': 'coloradar_package/dataset/2_23_2021_edgar_army_run2/'}

    key = 'army'
    dataset = datasets_dic[key]

    start_time = time.time()
    KB = set_KB(dataset)

    # my_TWLSQ = set_TWLSQ(dataset)
    # TEMPSAC = set_TEMPSAC(dataset)

    KB.ego_velocity()
    end_time = time.time()
    # my_TWLSQ.ego_velocity()
    # TEMPSAC.ego_velocity()

    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")


if __name__ == '__main__':
    main()
