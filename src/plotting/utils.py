from pathlib import Path

from networks.ic_configs import CONFIGS


def decode_path(path: Path):
    seed = int(path.parent.name.split("seed")[1])
    method = path.parent.parent.name
    setting = path.parent.parent.parent.name

    ee_config = None
    for config_name in CONFIGS.keys():
        if config_name in method:
            ee_config = config_name
            method = method.replace("_" + config_name, "")
    if ee_config is None:
        ee_config = "no_ee"
    return setting, method, ee_config, seed
