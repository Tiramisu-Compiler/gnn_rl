from collections import namedtuple
import logging
from pathlib import Path

import ray
import numpy as np
import torch
import torch.nn as nn
import math
from torch_geometric.data import Data

from agent.tiramisu_interface import TiramisuInterface, program_compatible_with_model
from config.config import AutoSchedulerConfig, Config
from utils.dataset_actor.dataset_actor import DatasetActor

logger = logging.getLogger(__name__)

Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "value", "log_prob", "entropy", "actions_mask"),
)


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

        # Initializing values and the episode
        self.reset(function_name)

    def reset(self, function_name: str = None):
        model_compatible_program = False
        while not model_compatible_program:
            logger.info("Getting next function")
            if function_name:
                function_name, function_data, cpp_code = (
                    self.dataset_worker.get_function_by_name(function_name)
                )
            else:
                function_name, function_data, cpp_code = ray.get(
                    self.dataset_worker.get_next_function.remote()
                )

            annotations = function_data["program_annotation"]
            model_compatible_program = program_compatible_with_model(annotations)

        self.current_program = function_name
        self.tiramisu_interface = TiramisuInterface(cpp_code, self.tiralib_config_path)

        node_feats, edge_index, it_index, comp_index = self.tiramisu_interface.graph

        self.previous_speedup = 1
        self.steps = 0
        self.state = (node_feats, edge_index, it_index)
        self.previous_action = None

    def rollout(self, model: nn.Module, device: str):
        model.to(device)
        model.eval()
        trajectory = []
        done = False
        log_trajectory = "#" * 50
        log_trajectory += f"\nFunction  : {self.current_program}"
        print(f"Function : {self.current_program}")
        print(f"trajectory : {trajectory}")

        while not done:
            prev_actions_mask = self.tiramisu_interface.get_mask()
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
                    data, torch.tensor(self.tiramisu_interface.get_mask()).to(device)
                )
                action = action.item()
                action_log_prob = action_log_prob.item()
                value = value.item()

            result = self.tiramisu_interface.apply_action(action)

            done = result.done
            if result.crashed:
                logger.info("Crashed applying the action. Skipping this action")
                continue

            reward = self.reward_process(action, result.is_legal, result.speedup)

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

            new_node_feats, new_edge_index, it_index, _ = self.tiramisu_interface.graph

            self.state = (new_node_feats, new_edge_index, it_index)

            current_log = (
                f"\nStep : {self.steps}"
                + f"\nAction ID : {action}"
                + f"\nLegality : {result.is_legal}"
                + f"\nActions Sequence So far : {self.tiramisu_interface.action_indices}"
                + "\n"
            )
            print(current_log)
            log_trajectory += current_log

            if self.steps == 40:
                done = True

        # else:
        #     schedule_object = self.tiramisu_api.scheduler_service.schedule_object

        #     tiramisu_program_dict = (
        #         self.tiramisu_api.get_current_tiramisu_program_dict()
        #     )
        #     ray.get(
        #         self.dataset_worker.update_dataset.remote(
        #             self.current_program, tiramisu_program_dict
        #         )
        #     )

        # clean up created files
        # delete files with the filename in workspace
        for filename in Path(Config.config.tiramisu.workspace).glob("*"):
            if self.current_program in filename.name:
                filename.unlink()

        print(f"Schedule : {self.tiramisu_interface.schedule}")
        print(f"actions : {self.tiramisu_interface.action_indices}")
        return {
            "trajectory": trajectory,
            "speedup": self.previous_speedup,
            "schedule": str(self.tiramisu_interface.schedule),
            "log_trajectory": log_trajectory,
        }

    def reward_process(self, action, legality, total_speedup):
        switching_branch_penality = 1
        illegal_action_penality = 1
        max_speedup = np.inf
        log_base = 4

        if legality:
            if action != 55:
                # If the action is not Next
                self.previous_action = action
                instant_speedup = total_speedup / self.previous_speedup
                self.previous_speedup = total_speedup
            else:
                instant_speedup = switching_branch_penality
        else:
            instant_speedup = illegal_action_penality

        instant_speedup = np.clip(instant_speedup, 0, max_speedup)

        reward = math.log(instant_speedup, log_base)

        return reward


@ray.remote
class RolloutWorkerRemote(RolloutWorker):
    def __init__(
        self,
        dataset_worker: DatasetActor,
        config: AutoSchedulerConfig,
        worker_id: int = 0,
    ):
        super().__init__(dataset_worker, config, worker_id)
