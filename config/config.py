from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal
import yaml
import os

# Enum for the dataset format


class DatasetFormat:
    # Both the CPP code and the data of the functions are loaded from PICKLE files
    PICKLE = "PICKLE"
    # We look for informations in the pickle files, if something is missing we get it from cpp files in a dynamic way
    HYBRID = "HYBRID"

    @staticmethod
    def from_string(s):
        if s == "PICKLE":
            return DatasetFormat.PICKLE
        elif s == "HYBRID":
            return DatasetFormat.HYBRID
        else:
            raise ValueError("Unknown dataset format")


@dataclass
class TiramisuConfig:
    tiramisu_path: str = ""
    workspace: str = "./workspace/"
    experiment_dir: str = ""



@dataclass
class DatasetConfig:
    dataset_format: DatasetFormat = DatasetFormat.HYBRID
    cpps_path: str = ""
    dataset_path: str = ""
    save_path: str = ""
    shuffle: bool = False
    seed: int = None
    saving_frequency: int = 10000

    def __init__(self, dataset_config_dict: Dict):
        self.dataset_format = DatasetFormat.from_string(
            dataset_config_dict["dataset_format"])
        self.cpps_path = dataset_config_dict["cpps_path"]
        self.dataset_path = dataset_config_dict["dataset_path"]
        self.save_path = dataset_config_dict["save_path"]
        self.shuffle = dataset_config_dict["shuffle"]
        self.seed = dataset_config_dict["seed"]
        self.saving_frequency = dataset_config_dict["saving_frequency"]

eckpoint: str = ""


@dataclass
class Experiment:
    legality_speedup: float = 1.0
    beam_search_order: bool = False
    max_time_in_minutes: int = 5 
    max_slowdown: int = 80

@dataclass
class Test:
    skip_execute_schedules: bool = False


@dataclass
class EnvVars:
    TIRAMISU_ROOT: str = ""
    CONDA_ENV: str = ""
    LD_LIBRARY_PATH: str = "${CONDA_ENV}/lib:${TIRAMISU_ROOT}/3rdParty/Halide/build/src:${TIRAMISU_ROOT}/3rdParty/llvm/build/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib:"



@dataclass
class AutoSchedulerConfig:

    tiramisu: TiramisuConfig
    dataset: DatasetConfig
    experiment: Experiment
    env_vars: EnvVars
    test: Test


    def __post_init__(self):
        if isinstance(self.tiramisu, dict):
            self.tiramisu = TiramisuConfig(**self.tiramisu)
        if isinstance(self.dataset, dict):
            self.dataset = DatasetConfig(self.dataset)
        if isinstance(self.experiment, dict):
            self.experiment = Experiment(**self.experiment)
        if isinstance(self.env_vars, dict):
            self.env_vars = EnvVars(**self.env_vars)
        if isinstance(self.test, dict):
            self.test = Test(**self.test)
    


def read_yaml_file(path):
    with open(path) as yaml_file:
        return yaml_file.read()


def parse_yaml_file(yaml_string: str) -> Dict[Any, Any]:
    return yaml.safe_load(yaml_string)


def dict_to_config(parsed_yaml: Dict[Any, Any]) -> AutoSchedulerConfig:
    tiramisu = TiramisuConfig(**parsed_yaml["tiramisu"])
    dataset = DatasetConfig(parsed_yaml["dataset"])
    experiment = Experiment(**parsed_yaml["experiment"])
    env_vars = EnvVars(**parsed_yaml["env_vars"])
    test = Test(**parsed_yaml["test"])
    return AutoSchedulerConfig(tiramisu, dataset,  experiment, env_vars,test)


class Config(object):
    config = None

    @classmethod
    def init(self, config_yaml="./config/config.yaml"):
        parsed_yaml_dict = parse_yaml_file(read_yaml_file(config_yaml))
        Config.config = dict_to_config(parsed_yaml_dict)
