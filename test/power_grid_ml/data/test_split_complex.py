import unittest
import os

os.environ["KERAS_BACKEND"] = "torch"  # required for complex tensors
import keras.src.ops.numpy as knp
import numpy as np

from src.power_grid_ml.data.split_complex import ComplexConverter, ComplexMode


class TestComplexConverter(unittest.TestCase):

    def test_initialization_with_valid_modes(self):
        modes = [ComplexMode.CARTESIAN, ComplexMode.POLAR]
        converter = ComplexConverter(input_modes=modes, output_modes=modes)
        self.assertEqual(converter.input_modes, modes)
        self.assertIn('real', converter.input_titles)
        self.assertIn('imaginary', converter.input_titles)
        self.assertIn('magnitude', converter.input_titles)
        self.assertIn('angle', converter.input_titles)

    def test_initialization_with_invalid_modes(self):
        with self.assertRaises(ValueError):
            converter = ComplexConverter(input_modes=["invalid"], output_modes=["invalid"])

    def test_convert_cartesian(self):
        converter = ComplexConverter(
            input_modes=[ComplexMode.CARTESIAN],
            output_modes=[ComplexMode.CARTESIAN],
            input_axis=-1,
            output_axis=-1
        )
        data = np.array([[1, 2], [3, 4]])  # Real and imaginary parts
        expected = data  # Since input and output modes are both CARTESIAN
        data = knp.array(data)
        result = converter.convert(data)
        np.testing.assert_array_equal(result, expected)

    def test_convert_polar(self):
        converter = ComplexConverter(
            input_modes=[ComplexMode.CARTESIAN],
            output_modes=[ComplexMode.POLAR],
            input_axis=-1,
            output_axis=-1
        )
        data = np.array([[1, 2], [3, 4]])  # Real and imaginary parts
        real = data[..., 0]
        imag = data[..., 1]
        magnitude = np.sqrt(real ** 2 + imag ** 2)
        phase = np.arctan2(imag, real)
        expected = np.stack((magnitude, phase), axis=-1)
        data = knp.array(data, dtype=float)
        result = converter.convert(data)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_exponential(self):
        converter = ComplexConverter(
            input_modes=[ComplexMode.CARTESIAN],
            output_modes=[ComplexMode.EXPONENTIAL],
            input_axis=-1,
            output_axis=-1
        )
        data = np.array([[1, 2], [3, 4]])  # Real and imaginary parts
        real = data[..., 0]
        imag = data[..., 1]
        magnitude = np.sqrt(real ** 2 + imag ** 2)
        phase = np.arctan2(imag, real)
        exp_imag = np.sin(phase)
        exp_real = np.cos(phase)
        signed_magnitude = magnitude * np.sign(exp_real)
        expected = np.stack((signed_magnitude, exp_imag), axis=-1)
        data = knp.array(data, dtype=float)
        result = converter.convert(data)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_complex_to_cartesian(self):
        converter = ComplexConverter(
            input_modes=[ComplexMode.COMPLEX],
            output_modes=[ComplexMode.CARTESIAN],
            input_axis=-1,
            output_axis=-1
        )
        data = np.array([1 + 2j, 3 + 4j])
        expected = np.array([[1, 2], [3, 4]])
        # keras3 does not allow conversion of complex arrays from numpy.
        # Instead, we need to create real and imaginary parts and combine them
        real = knp.array(np.real(data))
        imag = knp.array(np.imag(data))
        data = real + 1j * imag
        result = converter.convert(data)
        np.testing.assert_array_equal(result, expected)

    def test_convert_mixed_polar_cartesian(self):
        converter = ComplexConverter(
            input_modes=[ComplexMode.COMPLEX],
            output_modes=[ComplexMode.POLAR, ComplexMode.CARTESIAN],
            input_axis=-1,
            output_axis=-1
        )
        data = np.array([1 + 2j, 3 + 4j])
        magnitude = np.abs(data)
        phase = np.angle(data, False)
        expected_polar = np.stack((magnitude, phase), axis=-1)
        expected_cartesian = np.array([[1, 2], [3, 4]])
        expected_mixed = np.concatenate((expected_polar, expected_cartesian), axis=-1)
        real = knp.array(np.real(data))
        imag = knp.array(np.imag(data))
        data = real + 1j * imag
        result = converter.convert(data)
        np.testing.assert_array_almost_equal(result, expected_mixed)

    def test_convert_mixed_to_complex(self):
        converter = ComplexConverter(
            input_modes=[ComplexMode.CARTESIAN, ComplexMode.POLAR],
            output_modes=[ComplexMode.COMPLEX],
            input_axis=-1,
            output_axis=-1,
            input_angle_type='radians'
        )
        data = np.array([[1, 2, 2.236, 1.107], [3, 4, 5, 0.927]])
        real = data[..., 0]
        imag = data[..., 1]
        expected = np.squeeze(np.array([real + 1j * imag]))
        result = converter.convert(knp.array(data))
        np.testing.assert_array_almost_equal(result, expected)

    @unittest.expectedFailure
    def test_convert_mixed_to_complex_unequal(self):
        """
        Test converting mixed Cartesian and polar data to complex data when the input arrays
        (Cartesian and polar) are not equal.
        For now, this test is expected to fail because the converter does not handle this case.
        :return:
        """
        converter = ComplexConverter(
            input_modes=[ComplexMode.CARTESIAN, ComplexMode.POLAR],
            output_modes=[ComplexMode.COMPLEX],
            input_axis=-1,
            output_axis=-1,
            input_angle_type='radians'
        )
        data = np.array([[1, 2, 900, 1000], [3, 4, 900, 5000]])
        with self.assertRaises(ValueError):
            result = converter.convert(knp.array(data))

    def test_unknown_mode_error(self):
        with self.assertRaises(ValueError):
            converter = ComplexConverter(
                input_modes=[ComplexMode.CARTESIAN],
                output_modes=["unknown"]
            )


class TestCoordinateConversions(unittest.TestCase):

    def setUp(self):
        sample_array = np.array([[[3, 4], [0, 1]], [[5, 12], [0, 1]]], dtype="float64")
        real = sample_array[..., 0]
        imag = sample_array[..., 1]
        self.sample_array = knp.array(sample_array, dtype="float64")
        magnitude = np.sqrt(real ** 2 + imag ** 2)
        phase = np.arctan2(imag, real)
        self.expected_polar = np.stack((magnitude, phase), axis=-1)
        self.expected_cartesian = self.sample_array  # Assuming the conversion back is accurate

    def test_cartesian_to_polar_conversion(self):
        converter = ComplexConverter(
            input_modes=[ComplexMode.CARTESIAN],
            output_modes=[ComplexMode.POLAR],
            input_axis=-1,
            output_axis=-1
        )
        result = converter.convert(self.sample_array)
        np.testing.assert_array_almost_equal(result, self.expected_polar)

    def test_polar_to_cartesian_conversion(self):
        converter = ComplexConverter(
            input_modes=[ComplexMode.POLAR],
            output_modes=[ComplexMode.CARTESIAN],
            input_axis=-1,
            output_axis=-1
        )
        result = converter.convert(self.expected_polar)
        np.testing.assert_array_almost_equal(result, self.expected_cartesian)

    def test_angle_type_handling(self):
        # Convert to degrees for testing
        converter = ComplexConverter(
            input_modes=[ComplexMode.CARTESIAN],
            output_modes=[ComplexMode.POLAR],
            input_axis=-1,
            output_axis=-1,
            output_angle_type='degrees'
        )
        result_degrees = converter.convert(self.sample_array)
        expected_phase_degrees = np.rad2deg(self.expected_polar[..., 1])
        expected_polar_degrees = np.stack((self.expected_polar[..., 0], expected_phase_degrees), axis=-1)
        np.testing.assert_array_almost_equal(result_degrees, expected_polar_degrees)

    def test_invalid_angle_type(self):
        with self.assertRaises(ValueError):
            converter = ComplexConverter(
                input_modes=[ComplexMode.CARTESIAN],
                output_modes=[ComplexMode.POLAR],
                input_axis=-1,
                output_axis=-1,
                output_angle_type='invalid_type'
            )
            converter.convert(self.sample_array)

    def test_cartesian_to_polar_and_back(self):
        """Test converting Cartesian to polar coordinates and back."""
        converter_to_polar = ComplexConverter(
            input_modes=[ComplexMode.CARTESIAN],
            output_modes=[ComplexMode.POLAR],
            input_axis=-1,
            output_axis=-1
        )
        converter_to_cartesian = ComplexConverter(
            input_modes=[ComplexMode.POLAR],
            output_modes=[ComplexMode.CARTESIAN],
            input_axis=-1,
            output_axis=-1
        )
        polar_data = converter_to_polar.convert(self.sample_array)
        cartesian_data = converter_to_cartesian.convert(polar_data)
        np.testing.assert_array_almost_equal(cartesian_data, self.sample_array)

    def test_polar_to_cartesian_and_back(self):
        """Test converting polar to Cartesian coordinates and back."""
        converter_to_cartesian = ComplexConverter(
            input_modes=[ComplexMode.POLAR],
            output_modes=[ComplexMode.CARTESIAN],
            input_axis=-1,
            output_axis=-1
        )
        converter_to_polar = ComplexConverter(
            input_modes=[ComplexMode.CARTESIAN],
            output_modes=[ComplexMode.POLAR],
            input_axis=-1,
            output_axis=-1
        )
        cartesian_data = converter_to_cartesian.convert(self.expected_polar)
        polar_data = converter_to_polar.convert(cartesian_data)
        np.testing.assert_array_almost_equal(polar_data, self.expected_polar)

    def test_revert(self):
        converter = ComplexConverter(
            input_modes=[ComplexMode.CARTESIAN],
            output_modes=[ComplexMode.POLAR],
            input_axis=-1,
            output_axis=-1
        )
        result = converter.convert(self.sample_array)
        reverted = converter.revert(result)
        np.testing.assert_array_almost_equal(reverted, self.sample_array)

if __name__ == "__main__":
    unittest.main()
