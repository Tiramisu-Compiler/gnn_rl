from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
import random
from typing import Any

import numpy as np
import config.config as cfg
import ray


@dataclass
class TiramisuProgramCache:
    execution_times: dict[str, dict[str, float]]
    program_annotation: dict[str, Any]
    schedules_legality: dict[str, bool]
    schedules_solver: dict[str, tuple[int, int]]
    tags: set[str]
    isl_ast: dict[str, str]

    def add(
        self,
        schedule_str: str,
        is_legal: bool,
        isl_ast_str: str,
        skewing_factors: tuple[int, int] | None = None,
    ):
        self.schedules_legality[schedule_str] = is_legal
        self.isl_ast[schedule_str] = isl_ast_str
        if skewing_factors is not None:
            self.schedules_solver[schedule_str] = skewing_factors

    def add_execution_time(
        self, machine: str, schedule_str: str, execution_time: float
    ):
        if machine not in self.execution_times:
            self.execution_times[machine] = {}
        self.execution_times[machine][schedule_str] = execution_time

    def execution_time(self, machine: str, schedule_str: str) -> float | None:
        return self.execution_times.get(machine, {}).get(schedule_str, None)

    def to_dict(self):
        return asdict(self)


class DatasetActor:
    """
    DatasetActor is a class that is used to read the dataset and update it.
    It is used to read the dataset from disk and update it with the new functions.
    It is also used to save the dataset to disk.

    """

    def __init__(
        self,
        config: cfg.DatasetConfig,
    ):
        self.dataset_path = config.dataset_path
        self.path_to_save_dataset = config.save_path
        self.shuffle = config.shuffle
        self.seed = config.seed
        self.saving_frequency = config.saving_frequency

        self.dataset = {}
        self.function_names = []
        self.current_function_index = 0
        self.nbr_updates = 0
        self.dataset_name = config.dataset_path.split("/")[-1].split(".")[0]

        self.cpps_path = config.cpps_path
        self.cpps = {}
        self.tags = config.tags

        print(
            f"reading dataset in full pkl format: dataset pkl from {self.dataset_path} and cpps pkl from {self.cpps_path}"
        )

        with open(self.dataset_path, "rb") as f:
            self.dataset: dict[str, dict] = pickle.load(f)
            self.function_names: list[str] = list(self.dataset.keys())
            if self.tags:
                print(f"Filtering dataset by tags: {self.tags}")
                filtered_function_names: list[str] = []
                for program in self.function_names:
                    if any(tag in self.dataset[program]["tags"] for tag in self.tags):
                        filtered_function_names.append(program)
                self.function_names = filtered_function_names

        with open(self.cpps_path, "rb") as f:
            self.cpps: dict[str, str] = pickle.load(f)

        # Shuffle the dataset (can be used with random sampling turned off to get a random order)
        if self.shuffle:
            # Set the seed if specified (for reproducibility)
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self.function_names)

    @property
    def dataset_size(self):
        return len(self.function_names)

    def get_next_function(self, random=False):
        if random:
            function_name: str = np.random.choice(self.function_names)
        # Choose the next function sequentially
        else:
            function_name = self.function_names[
                self.current_function_index % self.dataset_size
            ]
            self.current_function_index += 1

        return (
            function_name,
            TiramisuProgramCache(**self.dataset[function_name]),
            self.cpps[function_name],
        )

    # Update the dataset with the new function
    def update_dataset(self, function_name: str, function_dict: dict) -> bool:
        """
        Update the dataset with the new function

        Arguments:
        function_name (str): name of the function
        function_dict (dict): dictionary containing the function schedules

        Returns:
        bool: True if the dataset was saved successfully
        """
        for key in function_dict.keys():
            self.dataset[function_name][key] = function_dict[key]

        self.nbr_updates += 1
        # print(f"# updates: {self.nbr_updates}")
        if self.nbr_updates % self.saving_frequency == 0:
            if self.nbr_updates % (2 * self.saving_frequency):
                return self.save_dataset_to_disk(version=2)
            else:
                return self.save_dataset_to_disk(version=1)
        return False

    def get_function_by_name(self, function_name: str):
        return (
            function_name,
            TiramisuProgramCache(**self.dataset[function_name]),
            self.cpps[function_name],
        )

    def save_dataset_to_disk(self, version=1) -> bool:
        """
        Save the dataset to disk
        :param version: version of the dataset to save (1 or 2)
        :return: True if the dataset was saved successfully
        """
        print("[Start] Save the legality_annotations_dict to disk")
        updated_dataset_path = (
            Path(self.path_to_save_dataset)
            / f"{self.dataset_name}_updated_{version}.pkl"
        )

        with updated_dataset_path.open("wb") as f:
            pickle.dump(self.dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("[Done] Save the legality_annotations_dict to disk")
        return True


@ray.remote
class DatasetActorRemote(DatasetActor):
    def __init__(
        self,
        config: cfg.DatasetConfig,
    ):
        super().__init__(config)  # pragma: no cover
