import json
from project_root import PROJECT_ROOT
from cachetools.func import ttl_cache


@ttl_cache(maxsize=1, ttl=30 * 60)
def get_config():
    config_path = PROJECT_ROOT / "data" / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    return config
