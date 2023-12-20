import ray
import torch
import random
from config.config import Config
from env_api.tiramisu_api import TiramisuEnvAPI
from rl.utils.actions import *
from gnn.utils.graph_builder import build_graph
import torch.nn as nn
from torch_geometric.data import Data
import math

BIG_NUMBER = 1e20


@ray.remote
class RolloutWorker:
    def __init__(self, dataset_worker, exploration_worker, config, worker_id=0):
        Config.config = config
        self.tiramisu_api = TiramisuEnvAPI(local_dataset=False)
        self.dataset_worker = dataset_worker
        self.exploration_worker = exploration_worker

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
        valid_program = None
        while not valid_program:
            prog_infos = ray.get(self.dataset_worker.get_next_function.remote())
            valid_program, actions_mask = self.tiramisu_api.set_program(*prog_infos)
            self.current_program = prog_infos[0]

        self.actions_mask = torch.tensor(actions_mask)
        annotations = (
            self.tiramisu_api.scheduler_service.schedule_object.prog.annotations
        )
        node_feats, edge_index, it_index, comp_index = build_graph(annotations)

        node_feats = focus_on_iterators(
            self.tiramisu_api.scheduler_service.branches[0].common_it,
            node_feats,
            it_index,
        )

        self.previous_speedup = 1
        self.steps = 0
        self.state = (node_feats, edge_index, it_index)

    def rollout(self, model: nn.Module, device: str):
        model.to(device)
        trajectory = []
        done = False
        while not done:
            self.steps += 1
            (node_feats, edge_index, it_index) = self.state
            eps_decay = ray.get(self.exploration_worker.get_value.remote())
            if torch.rand(1) > eps_decay:
                data = Data(
                    x=torch.tensor(node_feats, dtype=torch.float32),
                    edge_index=torch.tensor(edge_index, dtype=torch.int)
                    .transpose(0, 1)
                    .contiguous(),
                ).to(device)
                with torch.no_grad():
                    actions = model(data).squeeze(dim=0).cpu()
                    masked_actions = actions - self.actions_mask * BIG_NUMBER
                    action = masked_actions.argmax(dim=-1).item()
            else:
                non_masked_actions = torch.nonzero(self.actions_mask - 1).view(-1)
                action = random.Random().sample(
                    population=non_masked_actions.tolist(), k=1
                )[0]

            (
                speedup,
                new_node_feats,
                new_edge_index,
                legality,
                actions_mask,
                done,
            ) = apply_flattened_action(
                self.tiramisu_api, action, node_feats, edge_index, it_index
            )
            self.actions_mask = torch.tensor(actions_mask)

            instant_speedup = 1
            if legality and not action == 31 and not done:
                if speedup <= 0:
                    speedup = 0.01
                instant_speedup = speedup / self.previous_speedup
                self.previous_speedup = speedup

            reward = math.log(instant_speedup, 2)

            if done:
                trajectory.append(((node_feats, edge_index), action, None, reward))
            else:
                trajectory.append(
                    (
                        (node_feats, edge_index),
                        action,
                        (new_node_feats, new_edge_index),
                        reward,
                    )
                )

            self.state = (new_node_feats, new_edge_index, it_index)

        return {"trajectory": trajectory, "speedup": self.previous_speedup}
