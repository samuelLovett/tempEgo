"""
Author:         Samuel Lovett
Date Created:   Apr 26, 2024
Last Edited:    Apr 26, 2024

Description:
RANdom SAmple Consensus program described in Sec. II B. of [1]

[1] Enhancing Doppler Ego-Motion Estimation: A Temporally Weighted Approach to RANSAC
"""
from tempEgo.data_unpacker import data_unpacker_tensor_object
from tempEgo.least_squares import least_squares_velocity_analysis
import numpy as np

rng = np.random.default_rng()
PI = np.pi


class RANSAC:
    """
    Standard ransac class which implements ransac method as shown in Sec. II B. of [1]
    """
    def __init__(self, n=10, k=100, epsilon=0.05, z=10, loss=None, metric=None):
        """
        Initialization of the standard velocity estimation method (KB) as described at the beginning of this code.
        :param __n: Minimum number of data points to estimate parameters
        :param __k: Maximum iterations allowed
        :param __epsilon: Threshold value to determine if points are fit well
        :param __z: Number of close data points required to assert model fits well
        :param __loss: function of `y_true` and `y_pred` that returns a vector scoring how well y_pred matches y_true at each point
        :param __metric: function of `y_true` and `y_pred` and returns a float scoring how well y_pred matches y_true overall
        """
        self.n = n
        self.k = k
        self.epsilon = epsilon
        self.z = z
        self.loss = loss
        self.metric = metric

    def separate_points(self, buffer):
        """
        Filter the inlier point from the outlier points using RANSAC and return the body velocity. \n
        return vel: list containing the x and y body velocity [velocity in x, velocity in y]
        :param buffer: multidimensional list of measurement data [theta, doppler] (2, number of samples)
        :return: vel
        """
        measured_azim_val, measured_doppler_val = buffer
        best_error = np.inf
        index = np.arange(len(measured_azim_val))
        maybe_model = SinusoidalRegressor()
        refined_model = SinusoidalRegressor()
        model_using_all_found_inliers_model = None

        n = self.n
        epsilon = self.epsilon
        z = self.z
        for _ in range(self.k):

            # create the list initial guess inliers by taking the n randomly selected indices
            # and copying the azimuth and doppler values at those locations
            initial_guess_idx = rng.choice(index, n, replace=False, axis=0, shuffle=False)
            guess_inliers_azim = measured_azim_val[initial_guess_idx[:]]
            guess_inliers_doppler = measured_doppler_val[initial_guess_idx[:]]

            # make a model with our initial guess of inliers
            maybe_model.fit(guess_inliers_azim, guess_inliers_doppler)

            # using the created model predict the doppler values at the location of our azimuth values
            model_predicted_doppler_at_given_azimuth = maybe_model.predict(measured_azim_val)

            # create a boolean array that is the same length as our data arrays
            # compare the squared error distance between the predicted value and the measured value.
            # If it is greater than the threshold set that index value to true and if it is lower set the value to false
            thresholded = (
                    self.loss(measured_doppler_val, model_predicted_doppler_at_given_azimuth) < epsilon
            )

            # copy the values from the index corresponding to the location of the true values within threshold
            inlier_ids = index[:][np.where(thresholded)[0]]
            number_of_inliers = inlier_ids.size

            # check if the number of inliers from this guess is higher than required to assert model fits well
            if number_of_inliers > z:
                # fit a model with all the inliers found from the initial guess
                refined_model.fit(measured_azim_val[inlier_ids], measured_doppler_val[inlier_ids])

                # for the subset of inliers, calculate the mean squared error between the measured doppler and
                # the predicted doppler from the better model (model using all inlier points)
                this_error = self.metric(measured_doppler_val[inlier_ids],
                                         refined_model.predict(measured_azim_val[inlier_ids]))

                # check if the error is less than the best error found so far
                if this_error <= best_error:
                    # update values since this model is better than the last
                    best_error = this_error
                    model_using_all_found_inliers_model = refined_model

                    # next good model must get > half the inliers as previous good model
                    # self.d = int(number_of_inliers / 2)

        vel = [model_using_all_found_inliers_model.velocity_x,
               model_using_all_found_inliers_model.velocity_y]

        return vel


class RANSAC_TWLSQ(RANSAC):
    """
    TWLSQ ransac class which implements ransac identically to the standard method but passes the weights to the regressor class as shown in Sec. II C. of [1]
    """
    def __init__(self, n=10, k=100, epsilon=0.05, z=10, loss=None, metric=None, ):
        super().__init__(n, k, epsilon, z, loss, metric)

    def separate_points(self, buffer):
        """
        Filter the inlier point from the outlier points using RANSAC and TWLSQ and return the body velocity. \n
        return vel: list containing the x and y body velocity [velocity in x, velocity in y]
        :param buffer: multidimensional list of measurement data [theta, doppler] (2, number of samples)
        :return: vel
        """
        measured_azim_val, measured_doppler_val, full_weight_mask = buffer
        best_error = np.inf
        index = np.arange(len(measured_azim_val))
        maybe_model = SinusoidalRegressor()
        better_model = SinusoidalRegressor()
        model_using_all_found_inliers_model = None

        n = self.n
        epsilon = self.epsilon
        z = self.z
        for _ in range(self.k):

            # create the list initial guess inliers by taking the n randomly selected indices
            # and copying the azimuth and doppler values at those locations
            initial_guess_idx = rng.choice(index, n, replace=False, axis=0, shuffle=False)
            guess_inliers_azim = measured_azim_val[initial_guess_idx[:]]
            guess_inliers_doppler = measured_doppler_val[initial_guess_idx[:]]

            # make a model with our initial guess of inliers
            weight_mask = full_weight_mask[initial_guess_idx[:]]
            maybe_model.fit(guess_inliers_azim, guess_inliers_doppler, weights=weight_mask)

            # using the created model predict the doppler values at the location of our azimuth values
            model_predicted_doppler_at_given_azimuth = maybe_model.predict(measured_azim_val)

            # create a boolean array that is the same length as our data arrays
            # compare the squared error distance between the predicted value and the measured value.
            # If it is greater than the threshold set that index value to true and if it is lower set the value to false
            thresholded = (self.loss(measured_doppler_val, model_predicted_doppler_at_given_azimuth) < epsilon)

            # copy the values from the index corresponding to the location of the true values within threshold
            inlier_ids = index[:][np.where(thresholded)[0]]
            number_of_inliers = inlier_ids.size

            # check if the number of inliers from this guess is higher than required to assert model fits well
            if number_of_inliers > z:
                # fit a model with all the inliers found from the initial guess
                weight_mask = full_weight_mask[inlier_ids[:]]
                better_model.fit(measured_azim_val[inlier_ids], measured_doppler_val[inlier_ids],
                                 weights=weight_mask)

                # for the subset of inliers, calculate the mean squared error between the measured doppler and
                # the predicted doppler from the better model (model using all inlier points)
                this_error = self.metric(measured_doppler_val[inlier_ids],
                                         better_model.predict(measured_azim_val[inlier_ids]))

                # check if the error is less than the best error found so far
                if this_error <= best_error:
                    # update values since this model is better than the last
                    best_error = this_error
                    model_using_all_found_inliers_model = better_model

                    # next good model must get > half the inliers as previous good model
                    # self.d = int(number_of_inliers / 2)

        vel = [model_using_all_found_inliers_model.velocity_x,
               model_using_all_found_inliers_model.velocity_y]

        return vel


class TEMPSAC(RANSAC):
    """
    TEMPSAC class which implements TEMPSAC in Sec. III C. of [1]
    """
    def __init__(self, n=10, k=100, epsilon=0.05, z=10, loss=None, metric=None):
        super().__init__(n, k, epsilon, z, loss, metric)

    def get_initial_guess(self, buffer, probability):
        """
        Uses the probabilities to generate the likelihood of selecting a sample from a specific measurement and then uniformly selects samples from the selected measurement.
        The index of the selected samples are then returned for model fitting.
        :param buffer: samples being selected on
        :param probability: weights used to select the samples
        :return: n_random_points_full_idx
        """
        buffer_length = len(buffer)
        buffer_index = np.arange(buffer_length)
        try:
            n_random_buffer_idx = rng.choice(buffer_index, self.n, True, probability, 0, False)
        except Exception as e:
            print(e)
            exit()

        n_random_points_full_idx = []
        buffer_item_bins = []

        # find the length of each element within the buffer
        for m in range(buffer_length):
            number_of_data_points = len(buffer[m][0])
            buffer_item_bins.append(number_of_data_points)

        # Return the number of times each measurement is selected
        unique_buffer_idx, times_selected = np.unique(n_random_buffer_idx, return_counts=True)

        # Select the random data points from within each selected measurement in buffer
        for i in range(len(unique_buffer_idx)):
            num_points_to_select = times_selected[i]
            selected_buffer = unique_buffer_idx[i]
            measurement_idx = np.arange(buffer_item_bins[selected_buffer])
            random_points = rng.choice(measurement_idx, num_points_to_select, replace=False, shuffle=False)

            # Convert the within measurement index into the full index
            if unique_buffer_idx[i] == 0:
                n_random_points_full_idx.extend(random_points)
            else:
                for point_selected in random_points:
                    full_index = sum(buffer_item_bins[:selected_buffer]) + point_selected
                    n_random_points_full_idx.append(full_index)
        return n_random_points_full_idx

    def separate_points(self, buffer):
        """
        Filter the inlier point from the outlier points using RANSAC and TWLSQ and return the body velocity. \n
        return vel: list containing the x and y body velocity [velocity in x, velocity in y]
        :param buffer: multidimensional list of measurement data [theta, doppler] (2, number of samples)
        :return: vel
        """
        buffer, probability = buffer
        measured_azim_val, measured_doppler_val,  = data_unpacker_tensor_object(buffer)
        best_error = np.inf
        epsilon = self.epsilon
        z = self.z

        index = np.arange(len(measured_azim_val))
        maybe_model = SinusoidalRegressor()
        better_model = SinusoidalRegressor()
        model_using_all_found_inliers_model = None

        for _ in range(self.k):

            # if you get an error about this line of code check your n value
            initial_guess_idx = self.get_initial_guess(buffer, probability)

            # create the list initial guess inliers by taking the n randomly selected indices
            # and copying the azimuth and doppler values at those locations
            guess_inliers_azim = measured_azim_val[initial_guess_idx[:]]
            guess_inliers_doppler = measured_doppler_val[initial_guess_idx[:]]

            # make a model with our initial guess of inliers
            maybe_model.fit(guess_inliers_azim, guess_inliers_doppler)

            # using the created model predict the doppler values at the location of our azimuth values
            model_predicted_doppler_at_given_azimuth = maybe_model.predict(measured_azim_val)

            # create a boolean array that is the same length as our data arrays
            # compare the squared error distance between the predicted value and the measured value.
            # If it is greater than the threshold set that index value to true and if it is lower set the value to false
            thresholded = (self.loss(measured_doppler_val, model_predicted_doppler_at_given_azimuth) < epsilon)

            # copy the values from the index corresponding to the location of the true values within threshold
            inlier_ids = index[:][np.where(thresholded)[0]]
            number_of_inliers = inlier_ids.size

            # check if the number of inliers from this guess is higher than required to assert model fits well
            if number_of_inliers > z:
                # fit a model with all the inliers found from the initial guess
                better_model.fit(measured_azim_val[inlier_ids], measured_doppler_val[inlier_ids])

                # for the subset of inliers, calculate the mean squared error between the measured doppler and
                # the predicted doppler from the better model (model using all inlier points)
                this_error = self.metric(measured_doppler_val[inlier_ids], better_model.predict(measured_azim_val[inlier_ids]))

                # check if the error is less than the best error found so far
                if this_error <= best_error:
                    # update values since this model is better than the last
                    best_error = this_error
                    model_using_all_found_inliers_model = better_model

        vel = [model_using_all_found_inliers_model.velocity_x,
               model_using_all_found_inliers_model.velocity_y]

        return vel


class SinusoidalRegressor:
    """
    Velocity model class which implements (3) from [1].
    """
    def __init__(self):
        self.amplitude = None
        self.phase_offset = None
        self.velocity_x = None
        self.velocity_y = None
        self.weights = None

    def fit(self, azimuth, doppler, weights=None):
        """
        Fit the model using LSQ or TWLSQ.
        \n param[in] azimuth: array of azimuth samples.
        \n param[in] doppler: array of doppler samples.
        \n param[in] azimuth: Optional array of weights corresponding to each sample.
        \n return self: return the parameterized velocity model object
        :param azimuth:
        :param doppler:
        :param weights:
        :return: velocity model object
        """
        vx, vy, velocity, alpha = least_squares_velocity_analysis(azimuth, doppler, weights)
        # Make velocity negative since it is moving in opposite direction of sensor
        # Sentence after equations (1) in "Instantaneous ego-motion estimation using Doppler radar" D. Kellner et al
        self.amplitude = velocity
        self.phase_offset = alpha
        self.velocity_y = -1 * vy
        self.velocity_x = -1 * vx
        return self

    def predict(self, X: np.ndarray):
        """
        Predict the doppler velocity from the given azimuth values using (3) from [1]
        :param X: azimuth values to estimate the doppler velocity for
        :return: doppler velocity estimates
        """
        # The sensor has its Y-axis pointing outward, standard mathematical conventions,
        # i.e. right hand rule, and CCW rotation being +ve
        predicted_velocity = self.amplitude * np.cos(X - self.phase_offset)
        return predicted_velocity
