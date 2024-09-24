import os

from power_grid_ml.data.container import DataContainer
from power_grid_ml.data.sample_data import create_tree_graph, get_sample_line_features, get_sample_node_features
os.environ["KERAS_BACKEND"] = "torch"
import keras
from power_grid_ml.data import DataLoaderSQL

from power_grid_ml.config import ConfigManager, config_path

experiment = "experiment_cigrelv.yaml"
# Load configuration
config_manager = ConfigManager(os.path.join(config_path, experiment))

# get test data
# Number of nodes in the graph
NUM_NODES = 20
# Create the tree graph and adjacency list
adjacency_list = create_tree_graph(NUM_NODES)
# Get sample line features with the correct number of lines
line_features_df = get_sample_line_features(adjacency_list)
# Get sample node features
node_features_df = get_sample_node_features(NUM_NODES)

# Create data container
data_container = DataContainer()
data_container.add_data_pandas(train_df, test_df, validation_df)
pass