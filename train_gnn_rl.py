import torch
import torch.nn as nn
import ray
import numpy as np
from rl.algorithms.dqn.q_update import update_weights
from rl.algorithms.dqn.epsilon_decay import EpsilonDecay
from gnn.models.gat import GAT
from rl.utils.replay_buffer import ReplayBuffer, Transition
from rl.algorithms.dqn.rollout_worker import RolloutWorker
import mlflow

from config.config import Config
from rllib_ray_utils.dataset_actor.dataset_actor import DatasetActor


NUM_EPISODES = 20_000
EPS_START = 0.9
EPS_END = 0.05
EPS_THRES = 1000
LAMBDA = 0.95
TAU = 0.2
BATCH_SIZE = 128
BIG_NUMBER = 1e20
NUM_ROLLOUT_WORKERS = 10

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    Config.init()
    exploration_worker = EpsilonDecay.remote(EPS_END, EPS_START, EPS_THRES)
    dataset_worker = DatasetActor.remote(Config.config.dataset)

    train_dqn = GAT(input_size=718, num_heads=4, hidden_size=172, num_outputs=32).to(
        device
    )
    target_dqn = GAT(input_size=718, num_heads=4, hidden_size=172, num_outputs=32).to(
        device
    )
    target_dqn.load_state_dict(train_dqn.state_dict())

    replay_buffer = ReplayBuffer(capacity=10000)

    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(train_dqn.parameters(), lr=0.001, weight_decay=0.0001)

    rollout_workers = [
        RolloutWorker.options(
            num_cpus=1, num_gpus=1 / (NUM_ROLLOUT_WORKERS + 1)
        ).remote(dataset_worker, exploration_worker, Config.config, worker_id=i)
        for i in range(NUM_ROLLOUT_WORKERS)
    ]
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "NUM_EPISODES": 20_000,
                "EPS_START": 0.9,
                "EPS_END": 0.05,
                "EPS_THRES": 1000,
                "LAMBDA": 0.95,
                "TAU": 0.2,
                "BATCH_SIZE": 128,
            }
        )
        for episode in range(NUM_EPISODES):
            results = ray.get(
                [
                    rollout_workers[i].rollout.remote(train_dqn, device)
                    for i in range(NUM_ROLLOUT_WORKERS)
                ]
            )
            speedups = []
            for result in results:
                transitions = [Transition(*x) for x in result["trajectory"]]
                replay_buffer.memory.extend(transitions)
                speedups.append(result["speedup"])

            speedups = np.array(speedups)
            loss = update_weights(
                replay_buffer,
                target_dqn,
                train_dqn,
                criterion,
                optimizer,
                device,
                BATCH_SIZE,
                LAMBDA,
                TAU,
            )
            ray.get(
                [rollout_workers[i].reset.remote() for i in range(NUM_ROLLOUT_WORKERS)]
            )
            if loss != None:
                mlflow.log_metrics(
                    {
                        "Q Loss - Temporal Difference": loss,
                        "Speedup average": speedups.mean(),
                        "Speedup min": speedups.min(),
                        "Speedup max": speedups.max(),
                        "Epsilon": ray.get(exploration_worker.get_value.remote()),
                    },
                    step=episode,
                )
    mlflow.end_run()
