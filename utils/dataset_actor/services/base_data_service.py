from abc import abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
from typing import Any


@dataclass
class TiramisuProgramCache:
    execution_times: dict[str, dict[str, float]]
    program_annotation: dict[str, Any]
    schedules_legality: dict[str, bool]
    schedules_solver: dict[str, tuple[int, int]]
    tags: set[str]
    isl_ast: dict[str, str]

    def machine_execution_times(self, machine: str):
        return self["execution_times"].get(machine, {})

    def add(
        self,
        schedule_str: str,
        is_legal: bool | None = None,
        isl_ast_str: str | None = None,
        skewing_factors: tuple[int, int] | None = None,
    ):
        if is_legal is not None:
            self.schedules_legality[schedule_str] = is_legal
        if isl_ast_str is not None:
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

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class BaseDataService:
    def __init__(
        self,
        dataset_path: str,
        path_to_save_dataset: str,
        shuffle: bool = False,
        seed: int = None,
        saving_frequency: int = 10000,
    ) -> None:
        self.dataset_path = dataset_path
        self.path_to_save_dataset = path_to_save_dataset
        self.shuffle = shuffle
        self.seed = seed
        self.saving_frequency = saving_frequency

        self.dataset = {}
        self.function_names = []
        self.dataset_size = 0
        self.current_function_index = 0
        self.nbr_updates = 0
        self.dataset_name = dataset_path.split("/")[-1].split(".")[0]

    @abstractmethod
    def get_next_function(self, random=False) -> tuple[str, TiramisuProgramCache, str]:
        pass

    # Update the dataset with the new function
    def update_dataset(self, function_name: str, function_dict: dict) -> bool:
        """
        Update the dataset with the new function
        :param function_name: name of the function
        :param function_dict: dictionary containing the function schedules
        :return: True if the dataset was saved successfully
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

    # Save the dataset to disk
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
