import keras.src.ops.numpy as np
from keras.src.ops import amax, amin, mean, std

class Scaler:
    """
    Base class for Scalers along specific axis.
    """

    def __init__(self, eps: float = 1e-8):
        """

        :param eps: Small value to avoid division by zero.
        """
        self.transformation_params = None
        self.eps = eps
        pass

    def fit_transform(self, data: np.array, axis=None) -> np.array:
        """
        Fit to data and transform a multidimensional numpy array over the specified axis.
        :param data: numpy array
        :param axis: integer, tuple or list of integers of one or more axes indices to normalize the array along.
        If None, all axes are normalized.
        :return: transformed array
        """
        pass

    def transform(self, data: np.array) -> np.array:
        """
        Transform data using the transformation parameters and axis used in fit_transform
        :param data: numpy array
        :return: transformed array
        """
        pass

    def revert(self, data: np.array) -> np.array:
        """
        Reverts the transformation of a numpy array using the transformation parameters and axis from fit_transform.
        :param data: transformed numpy array
        :return: numpy array with reverted transformation
        """
        pass

    def get_params(self) -> dict:
        return self.transformation_params


class DimMinMaxScaler(Scaler):
    """
    Min-Max-Scaler for multidimensional numpy arrays.
    """

    def __init__(self):
        super().__init__()
        self.axis = None

    def fit_transform(self, data: np.array, axis=None) -> np.array:
        """
        Fit to min max values, then transform a multidimensional numpy array over the specified axis.
        :param np.ndarray data: The input array to be normalized.
        :param axis: integer, tuple or list of integers of one or more axes indices to normalize the array along.
        If None, all axes are normalized.
        :return np.ndarray: the normalized array
        """
        if not axis:
            axis = tuple(range(data.ndim))
        self.axis = axis
        # If axis is a single integer, convert it into a tuple
        if isinstance(axis, int):
            axis = (axis,)

        # Check if axis is a tuple or list of integers
        if not isinstance(axis, (tuple, list)) or not all(isinstance(a, int) for a in axis):
            raise ValueError("Axis must be an integer, tuple of integers, or list of integers.")
        if not all(0 <= a < data.ndim for a in axis):
            raise ValueError("All axes must be valid for the input array.")
        # Compute the minimum and maximum values along the specified axis
        min_vals = amin(data, axis=axis, keepdims=True)
        max_vals = amax(data, axis=axis, keepdims=True)

        # Normalize the data
        normalized_data = (data - min_vals) / (max_vals - min_vals + self.eps)

        # Store the normalization parameters in a dictionary
        self.transformation_params = {'min_vals': min_vals, 'max_vals': max_vals}
        return normalized_data

    def transform(self, data: np.array) -> np.array:
        """
        Call fit_transform first!
        transform data using the transformation parameters and axis used in fit_transform
        :param np.ndarray data: The input array to be normalized.
        :return np.ndarray: The normalized array.
        """
        # Check if the provided normalization parameters have the correct format
        if not self.transformation_params:
            raise ValueError("Call fit_transform first.")
        if not isinstance(self.transformation_params,
                          dict) or 'min_vals' not in self.transformation_params or 'max_vals' not in self.transformation_params:
            raise ValueError("Normalization parameters must be a dictionary containing 'min_vals' and 'max_vals'.")

        # Use given min and max values
        min_vals = self.transformation_params['min_vals']
        max_vals = self.transformation_params['max_vals']

        # Check if the shapes of the min and max values match the expected shapes
        expected_shape = tuple(data.shape[axis] if axis not in self.axis else 1 for axis in range(data.ndim))
        if min_vals.shape != expected_shape or max_vals.shape != expected_shape:
            raise ValueError("Normalization parameters must have the correct shape.")
        # Normalize the data
        normalized_data = (data - min_vals) / (max_vals - min_vals + self.eps)
        return normalized_data

    def revert(self, data: np.array) -> np.array:
        """
        Reverts the normalization of a numpy array using the normalization parameters and axis from fit_transform.
        :param np.ndarray data: The normalized array to be reverted.
        :return np.ndarray: The original array after reverting normalization.
        """
        # Check if the provided normalization parameters have the correct format
        if not self.transformation_params:
            raise ValueError("Call fit_transform first.")
        if not isinstance(self.transformation_params,
                          dict) or 'min_vals' not in self.transformation_params or 'max_vals' not in self.transformation_params:
            raise ValueError("Normalization parameters must be a dictionary containing 'min_vals' and 'max_vals'.")

        # Use given min and max values
        min_vals = self.transformation_params['min_vals']
        max_vals = self.transformation_params['max_vals']

        # Check if the shapes of the min and max values match the expected shapes
        expected_shape = tuple(data.shape[axis] if axis not in self.axis else 1 for axis in range(data.ndim))
        if min_vals.shape != expected_shape or max_vals.shape != expected_shape:
            raise ValueError("Normalization parameters must have the correct shape.")

        # Revert the normalization
        original_arr = data * (max_vals - min_vals) + min_vals

        return original_arr


class DimStandardScaler(Scaler):
    """
    Standard-Scaler for multidimensional numpy arrays.
    Performs Standardization also known as z-score-normalization.
    """

    def __init__(self):
        super().__init__()
        self.axis = None

    def fit_transform(self, data: np.array, axis=None) -> np.array:
        """
        Fit to mean and standard deviation, then transform a multidimensional numpy array over the specified axis.
        :param np.ndarray data: The input array to be normalized.
        :param axis: integer, tuple or list of integers of one or more axes indices to normalize the array along.
        If None, all axes are normalized.
        :return np.ndarray: normalized array
        """
        if not axis:
            axis = tuple(range(data.ndim))
        self.axis = axis
        # If axis is a single integer, convert it into a tuple
        if isinstance(axis, int):
            axis = (axis,)

        # Check if axis is a tuple or list of integers
        if not isinstance(axis, (tuple, list)) or not all(isinstance(a, int) for a in axis):
            raise ValueError("Axis must be an integer, tuple of integers, or list of integers.")

        if not all(0 <= a < data.ndim for a in axis):
            raise ValueError("All axes must be valid for the input array.")

        # Compute the mean and standard deviation along the specified axis
        mean_vals = mean(data, axis=axis, keepdims=True)
        std_vals = std(data, axis=axis, keepdims=True)

        # Normalize the data
        normalized_data = (data - mean_vals) / (std_vals + self.eps)

        # Store the normalization parameters in a dictionary
        self.transformation_params = {'mean_vals': mean_vals, 'std_vals': std_vals}
        return normalized_data

    def transform(self, data: np.array) -> np.array:
        """
        Call fit_transform first!
        transform data using the transformation parameters and axis used in fit_transform
        :param np.ndarray data: The input array to be normalized.
        :return np.ndarray: The normalized array.
        """
        # Check if the provided normalization parameters have the correct format
        if not self.transformation_params:
            raise ValueError("Call fit_transform first.")
        if not isinstance(self.transformation_params,
                          dict) or 'mean_vals' not in self.transformation_params or 'std_vals' not in self.transformation_params:
            raise ValueError("Normalization parameters must be a dictionary containing 'mean_vals' and 'std_vals'.")

        # Use given mean and std values
        mean_vals = self.transformation_params['mean_vals']
        std_vals = self.transformation_params['std_vals']

        # Check if the shapes of the mean and std values match the expected shapes
        expected_shape = tuple(data.shape[axis] if axis not in self.axis else 1 for axis in range(data.ndim))
        if mean_vals.shape != expected_shape or std_vals.shape != expected_shape:
            raise ValueError("Normalization parameters must have the correct shape.")
        # Normalize the data
        normalized_data = (data - mean_vals) / (std_vals + self.eps)
        return normalized_data

    def revert(self, data: np.array) -> np.array:
        """
        Reverts the standardization of a numpy array using the transformation parameters and axis from fit_transform.
        :param np.ndarray data: The standardized array to be reverted.
        :return np.ndarray: The original array after reverting standardization.
        """
        # Check if the provided normalization parameters have the correct format
        if not self.transformation_params:
            raise ValueError("Call fit_transform first.")
        if not isinstance(self.transformation_params,
                          dict) or 'mean_vals' not in self.transformation_params or 'std_vals' not in self.transformation_params:
            raise ValueError("Normalization parameters must be a dictionary containing 'mean_vals' and 'std_vals'.")

        # Use given mean and std values
        mean_vals = self.transformation_params['mean_vals']
        std_vals = self.transformation_params['std_vals']

        # Check if the shapes of the mean and std values match the expected shapes
        expected_shape = tuple(data.shape[axis] if axis not in self.axis else 1 for axis in range(data.ndim))
        if mean_vals.shape != expected_shape or std_vals.shape != expected_shape:
            raise ValueError("Transformation parameters must have the correct shape.")

        # Revert the normalization
        original_data = data * std_vals + mean_vals

        return original_data


def transform_matrix(mat, k=10):
    """
    Apply the logarithmic transformation to a matrix.
    """
    mask = (mat != 0)
    transformed = np.log1p(k * mat) / np.log1p(k)
    return np.where(mask, transformed, mat)


def redistribute_values(arr: np.array, k: int = 10, axis=None) -> np.array:
    """
    Redistribute values in an array using a logarithmic function for matrices
    specified by the target_axis.

    :param arr: numpy array of any shape
    :param k: scaling factor for redistribution strength
    :param axis: integer, tuple or list of integer along which to redistribute values
    :return: redistributed array
    """
    # copy array to avoid inplace operations
    arr = arr.copy()
    if axis is None:
        axis = tuple(range(arr.ndim))
    # If axis is a single integer, convert it into a tuple
    if isinstance(axis, int):
        axis = (axis,)

    # Check if axis is a tuple or list of integers
    if not isinstance(axis, (tuple, list)) or not all(isinstance(a, int) for a in axis):
        raise ValueError("Axis must be an integer, tuple of integers, or list of integers.")

    # Generate indices for all non-target axis
    non_target_axis = [i for i in range(arr.ndim) if i not in axis]
    non_target_dims = [arr.shape[i] for i in non_target_axis]
    from itertools import product
    all_indices = product(*[range(dim) for dim in non_target_dims])

    # Iterate over all non-target axis combinations and apply the transformation
    # on the slices defined by the target_axis
    for indices in all_indices:
        slicer = [slice(None)] * arr.ndim
        for i, idx in zip(non_target_axis, indices):
            slicer[i] = idx
        arr[tuple(slicer)] = transform_matrix(arr[tuple(slicer)], k=k)

    return arr
