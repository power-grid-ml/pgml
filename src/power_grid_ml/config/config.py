import os

import yaml

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
config_path = os.path.join(base_path, "config")


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

    def get_db_url(self):
        if 'database' not in self.config['general']:
            raise KeyError("Database configuration not found in the config file")
        db_config = self.config['general']['database']
        if 'port' in db_config:
            return f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        else:
            return f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['dbname']}"
