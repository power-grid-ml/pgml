import os

from power_grid_ml.data.container import DataContainer

os.environ["KERAS_BACKEND"] = "torch"
import keras
from power_grid_ml.data import DataLoaderSQL

from power_grid_ml.config import ConfigManager, config_path

experiment = "experiment_cigrelv.yaml"
# Load configuration
config_manager = ConfigManager(os.path.join(config_path, experiment))

# Load data
url = config_manager.get_db_url()
data_loader = DataLoaderSQL(url)
datasets_config = config_manager.get_config_value('datasets')
train_df = data_loader.load(datasets_config['training']['table'], columns=datasets_config['columns'])
test_df = data_loader.load(datasets_config['test']['table'], columns=datasets_config['columns'])
validation_df = data_loader.load(datasets_config['validation']['table'], columns=datasets_config['columns'])

# Create data container
data_container = DataContainer()
data_container.add_data_pandas(train_df, test_df, validation_df)
pass