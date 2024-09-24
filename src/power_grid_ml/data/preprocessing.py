from power_grid_ml.data.container import DataContainer


class Preprocessor:
    def scale(self, data_container:DataContainer):
        pass

    def unify_voltage_level(self, data_container:DataContainer, to_pu:bool, target_kv:float):
        # voltages, currents and admittance matrix
        pass

    def revert_voltage_level(self, data_container:DataContainer):
        pass

    def rotate_angles_to_reference(self, data_container:DataContainer):
        pass

    def remove_outliers(self, data_container:DataContainer):
        pass

    def remove_correlated_features(self, data_container:DataContainer):
        pass

    def remove_low_variance_features(self, data_container:DataContainer):
        pass

    def remove_irrelevant_features(self, data_container:DataContainer):
        """
        Remove irrelevant features from the data container
        e.g. harmonics with no magnitude
        :param data_container:
        :return:
        """
        pass

    # split complex conversion