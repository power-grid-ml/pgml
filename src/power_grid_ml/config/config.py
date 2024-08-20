import yaml
class ConfigManager:
    def __init__(self, config_file: str):
        self.config = self.read_config(config_file)

    def read_config(self, config_file: str):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def get_config(self):
        return self.config

    def get_config_value(self, key: str):
        return self.config[key]