import os
from pathlib import Path
# get environment variable for config path
resource_dir = os.environ.get('PGML_RESOURCE_DIR')
# if not set, use the default path
if not resource_dir or not os.path.exists(resource_dir):
    resource_dir = Path(__file__).parent.parent.parent.parent
config_dir = Path(resource_dir, "config")
