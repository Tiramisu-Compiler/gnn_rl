from collections import namedtuple
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from time import time

import ray
import numpy as np
import torch
import torch.nn as nn
import math
from torch_geometric.data import Data

from agent.tiramisu_interface import TiramisuInterface, program_compatible_with_model
from config.config import AutoSchedulerConfig, Config
from utils.dataset_actor import DatasetActor

logger = logging.getLogger(__name__)

Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "value", "log_prob", "entropy", "actions_mask"),
)


@dataclass
class TimePerStage:
    graph: float = 0.0
    get_next_function: float = 0.0
    tiramisu_interface_init: float = 0.0
    model_eval: float = 0.0
    get_mask: list[float] = field(default_factory=list)
    get_action: list[float] = field(default_factory=list)
    apply_action: list[float] = field(default_factory=list)
    reward_process: list[float] = field(default_factory=list)
    get_next_state: list[float] = field(default_factory=list)
    rollout: float = 0.0
    update_dataset: float = 0.0

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4)

    def __str__(self):
        return json.dumps(self.__dict__)


class RolloutWorker:
    def __init__(
        self,
        dataset_worker: DatasetActor,
        config: AutoSchedulerConfig,
        worker_id: int = 0,
        function_name: str = None,
    ):
        Config.config = config
        self.dataset_worker = dataset_worker
        self.tiramisu_interface: TiramisuInterface = None
        self.tiralib_config_path = config.tiralib_config_path

        # Variables related to workers and the environment
        self.worker_id = worker_id
        self.current_program = None

        # Variables related to the RL+Tiramisu train cycle
        self.state = None
        self.previous_speedup = None
        self.steps = None
        self.times_per_stages = TimePerStage()

        # Initializing values and the episode
        self.reset(function_name)

    def reset(self, function_name: str = None):
        self.times_per_stages = TimePerStage()
        start_time = time()
        is_program_model_compatible = False
        while not is_program_model_compatible:
            logger.info("Getting next function")
            if function_name:
                function_name, function_data, cpp_code = (
                    self.dataset_worker.get_function_by_name(function_name)
                )
            else:
                function_name, function_data, cpp_code = ray.get(
                    self.dataset_worker.get_next_function.remote()
                )

            annotations = function_data.program_annotation
            is_program_model_compatible = program_compatible_with_model(annotations)
        end_time = time()
        self.times_per_stages.get_next_function = end_time - start_time
        start_time = end_time

        self.current_program = function_name
        self.tiramisu_interface = TiramisuInterface(
            cpp_code,
            self.tiralib_config_path,
            cache=function_data,
            machine=Config.config.machine,
        )
        end_time = time()
        self.times_per_stages.tiramisu_interface_init = end_time - start_time
        start_time = end_time

        node_feats, edge_index, it_index, comp_index = self.tiramisu_interface.graph

        self.previous_speedup = 1
        self.steps = 0
        self.state = (node_feats, edge_index, it_index)
        self.times_per_stages.graph = time() - start_time

    def rollout(self, model: nn.Module, device: str):
        rollout_start_time = time()
        start_time = rollout_start_time
        model.to(device)
        model.eval()
        trajectory = []
        done = False
        log_trajectory = "#" * 50
        log_trajectory += f"\nFunction  : {self.current_program}"
        print("\n")
        print("#" * 50)
        print(f"Function : {self.current_program}")
        print(f"trajectory : {trajectory}")
        end_time = time()
        self.times_per_stages.model_eval = end_time - start_time
        start_time = end_time
        while not done:
            prev_actions_mask = self.tiramisu_interface.get_mask()
            end_time = time()
            self.times_per_stages.get_mask.append(end_time - start_time)
            start_time = end_time
            self.steps += 1
            (node_feats, edge_index, it_index) = self.state
            data = Data(
                x=torch.tensor(node_feats, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.int)
                .transpose(0, 1)
                .contiguous(),
            ).to(device)

            with torch.no_grad():
                action, action_log_prob, entropy, value = model(
                    data, torch.tensor(prev_actions_mask).to(device)
                )
                action = action.item()
                action_log_prob = action_log_prob.item()
                value = value.item()
            end_time = time()
            self.times_per_stages.get_action.append(end_time - start_time)
            start_time = end_time

            print(f"Running Action : {action}")

            result = self.tiramisu_interface.apply_action(action)
            end_time = time()
            self.times_per_stages.apply_action.append(end_time - start_time)
            start_time = end_time

            done = result.done
            if result.crashed:
                logger.info("Crashed applying the action. Skipping this action")
                continue

            reward = self.reward_process(action, result.is_legal, result.speedup)
            end_time = time()
            self.times_per_stages.reward_process.append(end_time - start_time)
            start_time = end_time

            trajectory.append(
                (
                    (np.copy(node_feats), np.copy(edge_index)),
                    action,
                    reward,
                    value,
                    action_log_prob,
                    entropy,
                    prev_actions_mask,
                )
            )
            if self.steps == 40:
                done = True

            if not done:
                new_node_feats, new_edge_index, it_index, _ = (
                    self.tiramisu_interface.graph
                )

                self.state = (new_node_feats, new_edge_index, it_index)
                end_time = time()
                self.times_per_stages.get_next_state.append(end_time - start_time)
            start_time = time()

            current_log = (
                f"\nStep : {self.steps}"
                + f"\nAction ID : {action}"
                + f"\nLegality : {result.is_legal}"
                + f"\nActions Sequence So far : {self.tiramisu_interface.action_indices}"
                + "\n"
            )
            print(current_log)
            log_trajectory += current_log

        else:
            start_time = time()
            self.times_per_stages.rollout = start_time - rollout_start_time
            if self.tiramisu_interface.cache:
                if type(self.dataset_worker) is DatasetActor:
                    self.dataset_worker.update_dataset(
                        self.current_program, self.tiramisu_interface.cache.to_dict()
                    )
                else:
                    ray.get(
                        self.dataset_worker.update_dataset.remote(
                            self.current_program,
                            self.tiramisu_interface.cache.to_dict(),
                        )
                    )
                self.times_per_stages.update_dataset = time() - start_time

        # clean up created files
        # delete files with the filename in workspace
        if Config.config.clean_files:
            for filename in Path(Config.config.tiramisu.workspace).glob("*"):
                if self.current_program in filename.name:
                    filename.unlink()
        print(f"End of episode : {self.current_program}")
        print(f"Schedule : {self.tiramisu_interface.schedule}")
        print(f"actions : {self.tiramisu_interface.action_indices}")
        print(f"Speedup : {self.previous_speedup}")
        print(f"Time per stage for {self.current_program} : {self.times_per_stages}")
        return RolloutResult(
            self.current_program,
            trajectory,
            self.previous_speedup,
            str(self.tiramisu_interface.schedule),
            log_trajectory,
        )

    def reward_process(self, action, legality, total_speedup):
        switching_branch_penality = 1
        illegal_action_penality = 1
        max_speedup = np.inf
        log_base = 4

        if legality:
            if action != 55:
                # If the action is not Next
                instant_speedup = total_speedup / self.previous_speedup
                self.previous_speedup = total_speedup
            else:
                instant_speedup = switching_branch_penality
        else:
            instant_speedup = illegal_action_penality

        instant_speedup = np.clip(instant_speedup, 0, max_speedup)

        reward = math.log(instant_speedup, log_base)
        print(
            f"Reward : {reward}, Total Speedup : {total_speedup}, instant_speedup : {instant_speedup}"
        )

        return reward


@ray.remote
class RolloutWorkerRemote(RolloutWorker):
    def __init__(
        self,
        dataset_worker: DatasetActor,
        config: AutoSchedulerConfig,
        worker_id: int = 0,
    ):
        super().__init__(dataset_worker, config, worker_id)  # pragma: no cover


@dataclass
class RolloutResult:
    function_name: str
    trajectory: list[Transition]
    speedup: float
    schedule: str
    log_trajectory: str
