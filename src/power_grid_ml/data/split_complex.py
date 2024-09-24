import math
from enum import Enum

import keras.src.ops.numpy as np
from keras.src.ops import scatter_update


class ComplexMode(Enum):
    CARTESIAN = 'cartesian'
    POLAR = 'polar'
    EXPONENTIAL = 'exponential'
    COMPLEX = 'complex'


def _get_attributes(modes: list[ComplexMode]):
    i = 0
    titles = []
    component_indices = {}
    for mode in modes:
        if mode == ComplexMode.CARTESIAN:
            titles.append('real')
            titles.append('imaginary')
            component_indices['real'] = i
            component_indices['imag'] = i + 1
        elif mode == ComplexMode.POLAR:
            titles.append('magnitude')
            titles.append('angle')
            component_indices['magnitude'] = i
            component_indices['angle'] = i + 1
        elif mode == ComplexMode.EXPONENTIAL:
            titles.append('signed magnitude')
            titles.append(r"imag($e^{j*\phi}$)")
            component_indices['signed_magnitude'] = i
            component_indices['imag_exp'] = i + 1
        elif mode == ComplexMode.COMPLEX:
            titles.append('complex')
            component_indices['complex'] = i
            if len(modes) > 1:
                raise ValueError('Complex mode cannot be combined with other modes.')
        i += 2
    return titles, component_indices


def _to_complex(data: np.array, mode, indices, axis, angle_type):
    """
    Convert the data to complex format.

    :param data: Input array as keras numpy array.
    :param mode: Complex mode to convert from.
    :param indices: Dictionary with indices for real, imag, magnitude, angle, signed_magnitude, imag_exp (if needed).
    :param axis: Axis where the complex data is located.
    :param angle_type: 'radians' or 'degrees' for angle representation.

    :return np.array: Converted array with complex format.
    """
    # move input axis to the end
    data = np.moveaxis(data, axis, -1)
    if mode == ComplexMode.CARTESIAN:
        real_idx = indices.get('real', 0)
        imag_idx = indices.get('imag', 1)
        real = np.take(data, indices=real_idx, axis=axis)
        imag = np.take(data, indices=imag_idx, axis=axis)
        return real + 1j * imag
    elif mode == ComplexMode.POLAR:
        mag_idx = indices.get('magnitude', 0)
        angle_idx = indices.get('angle', 1)
        mag = np.take(data, indices=mag_idx, axis=axis)
        phase = np.take(data, indices=angle_idx, axis=axis)
        if angle_type == 'degrees':
            phase = rad2deg(phase)
        return mag * complex_exp(1j * phase)
    elif mode == ComplexMode.EXPONENTIAL:
        mag_idx = indices.get('signed_magnitude', 0)
        imag_idx = indices.get('imag_exp', 1)
        signed_magnitude = np.take(data, indices=mag_idx, axis=axis)
        imag_exp = np.take(data, indices=imag_idx, axis=axis)
        phase = np.arcsin(imag_exp)
        magnitude = abs(signed_magnitude)
        phase = np.where(signed_magnitude < 0, math.pi - phase, phase)
        if angle_type == 'degrees':
            phase = rad2deg(phase)
        return magnitude * complex_exp(1j * phase)
    elif mode == ComplexMode.COMPLEX:
        return data
    else:
        raise ValueError('Invalid complex mode.')


def _from_complex(complex_data: np.array, modes, indices, axis, angle_type):
    components_dict = {}
    for mode in modes:
        if mode == ComplexMode.CARTESIAN:
            real_index = indices.get('real', 0)
            imag_index = indices.get('imag', 1)
            components_dict[real_index] = complex_data.real
            components_dict[imag_index] = complex_data.imag
        elif mode == ComplexMode.POLAR:
            mag_index = indices.get('magnitude', 0)
            angle_index = indices.get('angle', 1)
            components_dict[mag_index] = abs(complex_data)
            components_dict[angle_index] = angle(complex_data, angle_type=angle_type)
        elif mode == ComplexMode.EXPONENTIAL:
            signed_mag_index = indices.get('signed_magnitude', 0)
            imag_exp_index = indices.get('imag_exp', 1)
            exp = complex_exp(1j * angle(complex_data))
            sign_real = np.sign(exp.real)
            components_dict[signed_mag_index] = abs(complex_data) * sign_real
            components_dict[imag_exp_index] = exp.imag
        else:
            raise ValueError("Unknown mode")
    num_components = max(indices.values()) + 1
    components_list = []
    for i in range(num_components):
        if i in components_dict:
            components_list.append(components_dict[i])
        else:
            components_list.append(np.zeros_like(complex_data.real))
    res = np.stack(components_list, axis=-1)
    if axis != -1 and axis != len(res.shape) - 1:
        res = np.moveaxis(res, -1, axis)
    return res


class ComplexConverter:
    def __init__(
            self,
            input_modes,
            output_modes,
            input_axis=None,
            output_axis=None,
            input_angle_type='radians',
            output_angle_type='radians'
    ):
        """
        Initialize the converter with input and output modes and axes.

        :param input_modes: List of modes from ComplexMode for the input data.
        :param output_modes: List of modes from ComplexMode for the output data.
        :param input_axis: Axis where input modes are located.
        :param output_axis: Axis where output modes will be placed.
        :param input_angle_type: 'radians' or 'degrees' for angle representation.
        :param output_angle_type: 'radians' or 'degrees' for angle representation.
        """
        # Ensure modes and axes are lists
        self.input_modes = input_modes if isinstance(input_modes, list) else [input_modes]
        self.output_modes = output_modes if isinstance(output_modes, list) else [output_modes]
        if (any(not isinstance(mode, ComplexMode) for mode in self.input_modes)
                or any(not isinstance(mode, ComplexMode) for mode in self.output_modes)):
            raise ValueError('Invalid mode.')
        if input_angle_type not in ['radians', 'degrees'] or output_angle_type not in ['radians', 'degrees']:
            raise ValueError('Invalid angle type.')
        self.input_axis = input_axis
        self.output_axis = output_axis
        self.input_angle_type = input_angle_type
        self.output_angle_type = output_angle_type
        self.input_titles, self.output_indices = _get_attributes(self.input_modes)
        self.output_titles, self.input_indices = _get_attributes(self.output_modes)

    def convert(self, data: np.array):
        """
        Convert the data from input_modes to output_modes.

        :param data: Input array as keras numpy array.

        :return np.array: Converted array with output modes.
        """
        # 1st convert input data to complex
        # if more than one input mode is given, choose the first one - data should be equal in all modes
        complex_data = _to_complex(data, mode=self.input_modes[0], indices=self.input_indices, axis=self.input_axis,
                                   angle_type=self.input_angle_type)
        if ComplexMode.COMPLEX in self.output_modes:  # Complex mode is always alone due to different dtype
            return complex_data
        # 2nd convert complex data to output format
        return _from_complex(complex_data, modes=self.output_modes, indices=self.output_indices, axis=self.output_axis,
                             angle_type=self.output_angle_type)

    def revert(self, data: np.array):
        """
        Revert the data from output_modes to input_modes.
        :param data: Input array as keras numpy array.

        :return np.array: Reverted array with input modes.
        """
        # 1st convert data to complex
        complex_data = _to_complex(data, mode=self.output_modes[0], indices=self.output_indices, axis=self.output_axis,
                                   angle_type=self.output_angle_type)
        if ComplexMode.COMPLEX in self.input_modes:
            return complex_data
        # 2nd convert complex data to input format
        return _from_complex(complex_data, modes=self.input_modes, indices=self.input_indices, axis=self.input_axis,
                             angle_type=self.input_angle_type)


def angle(z: np.array, angle_type='radians'):
    """
    Compute the angle (phase) of a complex tensor z represented as a tensor with the last dimension size 2.
    The last dimension holds the real and imaginary parts.
    :param z: Complex tensor
    :param angle_type: 'radians' or 'degrees'
    :return: Angle tensor
    """
    phase = np.arctan2(z.imag, z.real)
    if angle_type == 'degrees':
        phase = rad2deg(phase)
    return phase


def rad2deg(x):
    """
    Convert angles from radians to degrees.

    :param x: Input angle in radians (can be a scalar, NumPy array, or Keras tensor)
    :return: Angle in degrees
    """
    return x * (180.0 / math.pi)


def complex_exp(z):
    """
    Compute the exponential of a complex number z = x + iy.
    :param z:
    :return:
    """
    x = z.real
    y = z.imag

    exp_x = np.exp(x)  # Compute e^x
    cos_y = np.cos(y)  # Compute cos(y)
    sin_y = np.sin(y)  # Compute sin(y)

    real_part = exp_x * cos_y  # Real part of e^{x + iy}
    imag_part = exp_x * sin_y  # Imaginary part of e^{x + iy}

    return real_part + 1j * imag_part