import unittest

import keras.src.ops.numpy as knp
from keras.src.ops import amin, amax
import numpy as np

from src.power_grid_ml.data.scaling import DimStandardScaler, DimMinMaxScaler


class TestNormalizationFunctions(unittest.TestCase):
    def test_normalize_fit_transform_00(self):
        # Create a 4-dimensional numpy array
        arr = np.random.rand(3, 4, 5, 6)
        scaler = DimMinMaxScaler()
        arr = knp.array(arr)
        # Test valid input
        normalized_arr = scaler.fit_transform(arr, (1, 2))
        self.assertTrue(np.all((normalized_arr >= 0) & (normalized_arr <= 1)),
                        "Normalized array values should be between 0 and 1.")

        # Test invalid axis input
        with self.assertRaises(ValueError):
            scaler.fit_transform(arr, (4,))

        with self.assertRaises(ValueError):
            scaler.fit_transform(arr, (-1,))

    def test_normalize_fit_transform_01(self):
        """
        use case specific unit test for 4d array of shape (batches, frequencies, 2, nodes)
        :return:
        """
        v = np.random.rand(2, 3, 2, 4)  # Random 4D array: 2 batches, 3 frequencies, real & imaginary, 4 nodes

        # Call the function with the generated 4D array
        scaler = DimMinMaxScaler()
        v = knp.array(v)
        normalized_v = scaler.fit_transform(v, (0, 3))
        normalization_params = scaler.get_params()
        # Check if the returned arrays have the correct shapes
        self.assertEqual(normalized_v.shape, v.shape)
        self.assertEqual(normalization_params['min_vals'].shape, (1, 3, 2, 1))
        self.assertEqual(normalization_params['max_vals'].shape, (1, 3, 2, 1))

        # Check if the values are normalized per frequency and per imaginary/real
        for freq in range(v.shape[1]):
            for ri in range(v.shape[2]):
                values = normalized_v[:, freq, ri, :]
                min_val = amin(values)
                max_val = amax(values)

                self.assertTrue(0 <= min_val <= 1,
                                f"Minimum value {min_val} is out of range [0, 1] for freq={freq}, ri={ri}")
                self.assertTrue(0 <= max_val <= 1,
                                f"Maximum value {max_val} is out of range [0, 1] for freq={freq}, ri={ri}")

                # Check if the values are correctly normalized
                min_v = normalization_params['min_vals'][0, freq, ri, 0]
                max_v = normalization_params['max_vals'][0, freq, ri, 0]
                for batch in range(v.shape[0]):
                    for node in range(v.shape[3]):
                        original_value = v[batch, freq, ri, node]
                        normalized_value = (original_value - min_v) / (max_v - min_v)
                        self.assertAlmostEqual(normalized_value, normalized_v[batch, freq, ri, node], delta=1e-6)

    def test_normalize_fit_transform_02(self):
        # Test various input array dimensions and normalization axes
        test_cases = [
            (np.random.rand(5, 6), (0,)),
            (np.random.rand(5, 6), (1,)),
            (np.random.rand(3, 4, 5), (0, 1)),
            (np.random.rand(3, 4, 5), (2,)),
            (np.random.rand(2, 3, 4, 5), (0, 2)),
            (np.random.rand(2, 3, 4, 5), (1, 3)),
        ]
        scaler = DimMinMaxScaler()
        for arr, axes in test_cases:
            arr = knp.array(arr)
            normalized_arr = scaler.fit_transform(arr, axes)
            self.assertTrue(np.all((normalized_arr >= 0) & (normalized_arr <= 1)),
                            "Normalized array values should be between 0 and 1.")

            # Check if the original and reverted arrays are close
            original_arr = scaler.revert(normalized_arr)
            self.assertTrue(np.allclose(arr, original_arr), "Reverted array should be close to the original array.")

        # Test invalid axis input
        with self.assertRaises(ValueError):
            scaler.fit_transform(test_cases[0][0], (2,))

        with self.assertRaises(ValueError):
            scaler.fit_transform(test_cases[0][0], (-1,))

    def test_normalize_fit_transform_no_axes(self):
        scaler = DimMinMaxScaler()
        data = np.array([[1, 2], [3, 4]])
        data = knp.array(data, dtype=float)
        # Expected behavior when no axis is specified
        expected = np.array([[0., 0.33333333], [0.66666667, 1.]])

        result = scaler.fit_transform(data)
        self.assertTrue(np.allclose(result, expected))

    def test_revert_normalization_00(self):
        # Create a 4-dimensional numpy array
        arr = np.random.rand(3, 4, 5, 6)
        arr = knp.array(arr)
        scaler = DimMinMaxScaler()
        # Normalize the array along axis 1 and 2
        normalized_arr = scaler.fit_transform(arr, (1, 2))

        # Revert the normalization to get back the original array
        original_arr = scaler.revert(normalized_arr)

        # Check if the original and reverted arrays are close
        self.assertTrue(np.allclose(arr, original_arr), "Reverted array should be close to the original array.")

    def test_revert_normalization_01(self):
        # Test various input array dimensions and normalization axes
        test_cases = [
            (np.random.rand(5, 6), (0,)),
            (np.random.rand(5, 6), (1,)),
            (np.random.rand(3, 4, 5), (0, 1)),
            (np.random.rand(3, 4, 5), (2,)),
            (np.random.rand(2, 3, 4, 5), (0, 2)),
            (np.random.rand(2, 3, 4, 5), (1, 3)),
        ]
        scaler = DimMinMaxScaler()
        for arr, axes in test_cases:
            arr = knp.array(arr)
            normalized_arr = scaler.fit_transform(arr, axes)

            # Revert the normalization to get back the original array
            original_arr = scaler.revert(normalized_arr)

            # Check if the original and reverted arrays are close
            self.assertTrue(np.allclose(arr, original_arr), "Reverted array should be close to the original array.")

    def test_z_score_normalization(self):
        arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        arr = knp.array(arr, dtype=float)
        scaler = DimStandardScaler()
        normalized_arr = scaler.fit_transform(arr, (0,))

        # Check the means are close to 0 and standard deviations close to 1 for each column (axis=0)
        self.assertTrue(np.allclose(np.mean(normalized_arr, axis=0), 0))
        self.assertTrue(np.allclose(np.std(normalized_arr, axis=0), 1))
        params = scaler.get_params()
        # Check the normalization parameters are correct
        self.assertTrue(np.allclose(params['mean_vals'], np.mean(arr, axis=0)))
        self.assertTrue(np.allclose(params['std_vals'], np.std(arr, axis=0)))

    def test_z_score_normalization_revert(self):
        arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        arr = knp.array(arr, dtype=float)
        scaler = DimStandardScaler()
        normalized_arr = scaler.fit_transform(arr, (0,))
        reverted_arr = scaler.revert(normalized_arr)

        # Check the reverted array is close to the original array
        np.testing.assert_array_almost_equal(reverted_arr, arr)

    def test_error_conditions(self):
        # Test error conditions of the scaler.fit_transform function

        arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        arr = knp.array(arr, dtype=float)
        scaler = DimStandardScaler()

        # Test error when an invalid axis is specified
        with self.assertRaises(ValueError):
            scaler.fit_transform(arr, (2,))

    def test_3d_z_score_normalization(self):
        arr = np.random.rand(5, 5, 5)
        arr = knp.array(arr, dtype="float64")
        scaler = DimStandardScaler()
        normalized_arr = scaler.fit_transform(arr, (2,))
        params = scaler.get_params()
        # Check the means are close to 0 and standard deviations close to 1 for each column (axis=2)
        self.assertTrue(np.allclose(np.mean(normalized_arr, axis=2), 0))
        self.assertTrue(np.allclose(np.std(normalized_arr, axis=2), 1))

        # Check the normalization parameters are correct (squeeze is required to remove the extra dimension)
        self.assertTrue(np.allclose(np.squeeze(params['mean_vals']), np.mean(arr, axis=2)))
        self.assertTrue(np.allclose(np.squeeze(params['std_vals']), np.std(arr, axis=2)))

    def test_4d_z_score_normalization_over_two_axes(self):
        arr = np.random.rand(5, 5, 5, 5)
        arr = knp.array(arr, dtype="float64")
        scaler = DimStandardScaler()
        normalized_arr = scaler.fit_transform(arr, (2, 3))
        params = scaler.get_params()
        # Check the means are close to 0 and standard deviations close to 1 for each column (axis=2 and 3)
        self.assertTrue(np.allclose(np.mean(normalized_arr, axis=(2, 3)), 0))
        self.assertTrue(np.allclose(np.std(normalized_arr, axis=(2, 3)), 1))

        # Check the normalization parameters are correct
        self.assertTrue(np.allclose(np.squeeze(params['mean_vals']), np.mean(arr, axis=(2, 3))))
        self.assertTrue(np.allclose(np.squeeze(params['std_vals']), np.std(arr, axis=(2, 3))))


if __name__ == '__main__':
    unittest.main()
