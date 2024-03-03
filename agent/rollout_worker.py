from collections import namedtuple

import ray
import torch
import torch.nn as nn
import math
from torch_geometric.data import Data, Batch
import agent.graph_utils as graph_utils
from config.config import Config
from env_api.scheduler.models.actions_mask import ActionsMask
from env_api.tiramisu_api import TiramisuEnvAPI
import numpy as np


def apply_action(
    tiramisu_api: TiramisuEnvAPI,
    action,
    node_feats,
    edge_index,
    it_index,
    worker_id="0",
):
    done = False
    match action[0]:
        case 0:
            loop_level = action[1]
            speedup, legality, actions_mask = tiramisu_api.skew(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                worker_id=worker_id,
            )
            if legality:
                skewing_params = (
                    tiramisu_api.scheduler_service.schedule_object.schedule_list[
                        -1
                    ].params[-2:]
                )
                branch = tiramisu_api.scheduler_service.current_branch
                its = tiramisu_api.scheduler_service.branches[branch].common_it
                graph_utils.apply_skewing(
                    [its[loop_level], its[loop_level + 1]],
                    skewing_params,
                    node_feats,
                    it_index,
                )
        case 1:
            loop_level = action[1]
            speedup, legality, actions_mask = tiramisu_api.reverse(
                loop_level=loop_level, worker_id=worker_id
            )
            if legality:
                branch = tiramisu_api.scheduler_service.current_branch
                its = tiramisu_api.scheduler_service.branches[branch].common_it
                graph_utils.apply_reversal(its[loop_level], node_feats, it_index)
        case 2:
            loop_level = action[1]
            speedup, legality, actions_mask = tiramisu_api.parallelize(
                loop_level=loop_level, worker_id=worker_id
            )
            if legality:
                branch = tiramisu_api.scheduler_service.current_branch
                its = tiramisu_api.scheduler_service.branches[branch].common_it
                graph_utils.apply_parallelization(its[loop_level], node_feats, it_index)
        case 3:
            loop_level1, loop_level2 =[(i,j) for i in range(4) for j in range(i+1,5)][action[1]]
             
            speedup, legality, actions_mask = tiramisu_api.interchange(
                loop_level1=loop_level1,
                loop_level2=loop_level2 ,
                worker_id=worker_id,
            )
            if legality:
                branch = tiramisu_api.scheduler_service.current_branch
                its = tiramisu_api.scheduler_service.branches[branch].common_it
                graph_utils.apply_interchange(
                    [its[loop_level1], its[loop_level2]], edge_index, it_index
                )

        case 4:
            factor = 4 + action[1]
            speedup, legality, actions_mask = tiramisu_api.unroll(
                unrolling_factor=2**factor,
                worker_id=worker_id,
            )
            if legality:
                branch = tiramisu_api.scheduler_service.current_branch
                its = tiramisu_api.scheduler_service.branches[branch].common_it
                graph_utils.apply_unrolling(its[-1], 2**factor, node_feats, it_index)

        case 5:
            loop_level = action[1]
            x, y = action[2] // 3, action[2] % 3
            size_x, size_y = [32, 64, 128][x], [32, 64, 128][y]
            speedup, legality, actions_mask = tiramisu_api.tile2D(
                loop_level1=loop_level,
                loop_level2=loop_level + 1,
                size_x=size_x,
                size_y=size_y,
                worker_id=worker_id,
            )
            if legality:
                branch = tiramisu_api.scheduler_service.current_branch
                its = tiramisu_api.scheduler_service.branches[branch].common_it
                graph_utils.apply_tiling(
                    [its[loop_level], its[loop_level + 1]],
                    [size_x, size_y],
                    node_feats,
                    it_index,
                )
        case 6:
            next_branch_mask = tiramisu_api.scheduler_service.next_branch()
            if next_branch_mask == None:
                speedup, legality, actions_mask = (
                    1,
                    True,
                    ActionsMask(),
                )
                done = True
            else:
                speedup, legality, actions_mask = (
                    1,
                    True,
                    next_branch_mask,
                )
                branch = tiramisu_api.scheduler_service.current_branch
                node_feats = graph_utils.focus_on_iterators(
                    tiramisu_api.scheduler_service.branches[branch].common_it,
                    node_feats,
                    it_index,
                )

    return speedup, node_feats, edge_index, legality, actions_mask, done


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "value", "log_prob", "entropy")
)


@ray.remote
class RolloutWorker:
    def __init__(self, dataset_worker, config, worker_id=0):
        Config.config = config
        self.tiramisu_api = TiramisuEnvAPI(local_dataset=False)
        self.dataset_worker = dataset_worker

        # Variables related to workers and the environment
        self.worker_id = worker_id
        self.current_program = None

        # Variables related to the RL+Tiramisu train cycle
        self.state = None
        self.actions_mask = None
        self.previous_speedup = None
        self.steps = None

        # Initializing values and the episode
        self.reset()

    def reset(self):
        actions_mask = None
        while actions_mask == None:
            prog_infos = ray.get(self.dataset_worker.get_next_function.remote())
            actions_mask = self.tiramisu_api.set_program(*prog_infos)

        self.current_program = prog_infos[0]

        self.actions_mask = actions_mask
        annotations = (
            self.tiramisu_api.scheduler_service.schedule_object.prog.annotations
        )
        node_feats, edge_index, it_index, comp_index = graph_utils.build_graph(
            annotations
        )

        node_feats = graph_utils.focus_on_iterators(
            self.tiramisu_api.scheduler_service.branches[0].common_it,
            node_feats,
            it_index,
        )

        self.previous_speedup = 1
        self.steps = 0
        self.state = (node_feats, edge_index, it_index)

    def rollout(self, model: nn.Module, device: str):
        model.to(device)
        model.eval()
        trajectory = []
        done = False
        log_trajectory = "#" * 50
        log_trajectory += f"\nFunction  : {self.current_program}"

        while not done:
            self.steps += 1
            (node_feats, edge_index, it_index) = self.state
            data = Data(
                x=torch.tensor(node_feats, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.int)
                .transpose(0, 1)
                .contiguous(),
            ).to(device)

            with torch.no_grad():
                action, raw_action, action_log_prob, entropy, value = model(
                    data=Batch.from_data_list([data]).to(device),
                    action_mask=self.actions_mask.to(device),
                )
                action_log_prob = action_log_prob.item()
                value = value.item()
            (
                total_speedup,
                new_node_feats,
                new_edge_index,
                legality,
                actions_mask,
                done,
            ) = apply_action(
                self.tiramisu_api,
                action,
                np.copy(node_feats),
                np.copy(edge_index),
                it_index,
                worker_id=str(self.worker_id),
            )
            self.actions_mask = actions_mask

            reward = self.reward_process(action, legality, total_speedup)

            trajectory.append(
                (
                    (np.copy(node_feats), np.copy(edge_index)),
                    raw_action,
                    reward,
                    value,
                    action_log_prob,
                    entropy,
                )
            )
            self.state = (new_node_feats, new_edge_index, it_index)

            log_trajectory += (
                f"\nStep : {self.steps}"
                + f"\nAction ID : {action}"
                + f"\nLegality : {legality}"
                + f"\nActions Sequence So far : {self.tiramisu_api.scheduler_service.schedule_object.schedule_str}"
                + "\n"
            )

            done = done or (self.steps == 25)

        else:
            schedule_object = self.tiramisu_api.scheduler_service.schedule_object

            tiramisu_program_dict = (
                self.tiramisu_api.get_current_tiramisu_program_dict()
            )
            self.dataset_worker.update_dataset.remote(
                self.current_program, tiramisu_program_dict
            )

        return {
            "trajectory": trajectory,
            "speedup": self.previous_speedup,
            "schedule_object": schedule_object,
            "log_trajectory": log_trajectory,
        }

    def reward_process(self, action, legality, total_speedup):
        switching_branch_penality = 0.5
        illegal_action_penality = 0.9
        max_speedup = np.inf
        log_base = 4

        if legality:
            if action[0] != 6:
                # If the action is not Next
                instant_speedup = total_speedup / self.previous_speedup
                self.previous_speedup = total_speedup
            else:
                instant_speedup = switching_branch_penality
        else:
            instant_speedup = illegal_action_penality

        instant_speedup = np.clip(instant_speedup, 0, max_speedup)

        reward = math.log(instant_speedup, log_base)

        return reward
