import yaml
import h5py


def load_yaml(filename) -> dict:
    with open(filename) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    return settings


def save_yaml(filename, settings):
    with open(filename, "w") as file:
        yaml.dump(settings, file, sort_keys=False)


def save_yaml_to_hdf(hdf_file_path: str, key: str, yaml_dict: dict):
    with h5py.File(hdf_file_path, "a") as f:
        if key in f:
            del f[key]
        yaml_str = yaml.dump(yaml_dict, sort_keys=False)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset(key, data=yaml_str, dtype=dt)


def load_yaml_to_hdf(hdf_file_path: str, key: str) -> dict:
    with h5py.File(hdf_file_path, "r") as f:
        yaml_str = f[key][()].decode() if isinstance(f[key][()], bytes) else f[key][()]
    return yaml.safe_load(yaml_str)
