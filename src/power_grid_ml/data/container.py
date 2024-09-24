import keras.src.ops.numpy as np
import pandas as pd


class DataContainer:
    def __init__(self):
        self.edge_features = None
        # line length
        # ! admittance / impedance
        # label (switch, transformer, line, switch status)
        # cable type
        # ! line currents
        self.node_features = None
        # consumer structure (residential, commercial, industrial)
        # consumer load (EV, heating, ...)
        # producer structure (solar, wind, ...)
        # ! nodal voltages
        # ! injected currents
        self.graph_features = None
        # structure (residential, commercial, industrial, mixed)
        # expected irradiance - possibly on a per-node basis
        # expected wind speed - possibly on a per-node basis
        self.admittance_matrix = None
        self.base_kvs = None
        pass

    def add_data_pandas(self, train: pd.DataFrame, test: pd.DataFrame, validation: pd.DataFrame):
        """
        pandas dataframe is a multi-index dataframe with the following indices:
        - step
        - freq
        - bus
        columns may include any of the following
        ['v1', 'v2', 'v3', 'i1', 'i2', 'i3', 'vangle1', 'vangle2', 'vangle3', 'iangle1', 'iangle2', 'iangle3']
        where currents are injected currents at the given node and voltage represent nodal voltages
        :param train:
        :param test:
        :param validation:
        :return:
        """
        pass

    def add_data_tensors(self, train: np.array, test: np.array, validation: np.array, phase=1,
                         datatype='nodal_voltage'):
        """
        :param train:
        :param test:
        :param validation:
        :param phase: 1, 2, 3
        :param datatype: nodal_voltage or injected_current
        :return:
        """
        axes_description = {
            'step': 0,
            'harmonic': 1,
            'complex': 2,
            'node': 3
        }
        pass
