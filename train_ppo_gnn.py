import mlflow
import torch
import torch.nn as nn
from agent.policy_value_nn import GAT
from agent.rollout_worker import RolloutWorker, Transition
from config.config import Config
from utils.dataset_actor.dataset_actor import DatasetActor
import numpy as np
import ray
import math
from torch_geometric.data import Batch, Data
import argparse as arg

import os 

def create_dir(run_name):
    parent_dir = "/scratch/dl5133/Dev/RL-Agent/new_agent/experiment_dir/models"

    path = os.path.join(parent_dir, run_name) 

    try: 
        os.makedirs(path, exist_ok = True) 
    except OSError as error: 
        pass

    return path


num_updates = 2000
clip_epsilon = 0.3
gamma = 0.99
lambdaa = 0.95
value_coeff = 4
entropy_coeff_start = 0.001
entropy_coeff_finish = 0.0001
max_grad_norm = 10
batch_size = 4096
num_epochs = 5
mini_batch_size = 128
lr = 5e-4
weight_decay = 0.0001
total_steps = num_updates * batch_size



if "__main__" == __name__:
    parser = arg.ArgumentParser() 

    parser.add_argument("--num-nodes", default=1, type=int)

    parser.add_argument("--name", type=str, default="experiment_101")

    args = parser.parse_args()


    NUM_ROLLOUT_WORKERS = args.num_nodes

    if NUM_ROLLOUT_WORKERS > 1 :
        ray.init("auto")
    else : 
        ray.init()
    # Init global config to run the Tiramisu env
    Config.init()

    dataset_worker = DatasetActor.remote(Config.config.dataset)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ppo_agent = GAT(input_size=718, num_heads=4, hidden_size=128, num_outputs=56).to(
        device
    )
    optimizer = torch.optim.Adam(
        ppo_agent.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8
    )
    value_loss = nn.MSELoss()

    ppo_agent.load_state_dict(
        torch.load(
            "/scratch/dl5133/Dev/RL-Agent/new_agent/experiment_dir/models/experiment_2k_101/last_model_168.pt",
            map_location=torch.device(device)
        ),
    )

    rollout_workers = [
        RolloutWorker.options(
            num_cpus=24, num_gpus=0, scheduling_strategy="SPREAD"
        ).remote(dataset_worker, Config.config, worker_id=i)
        for i in range(NUM_ROLLOUT_WORKERS)
    ]

    run_name = args.name

    model_save_path = create_dir(run_name)

    with mlflow.start_run(
        run_name=run_name,
        run_id="3f83032459e946d5b5a6397e266e5b29"
    ) as run:
        mlflow.log_params(
            {
                # "num_updates": num_updates,
                # "clip_epsilon": clip_epsilon,
                # "gamma": gamma,
                # "lambdaa": lambdaa,
                # "value_coeff": value_coeff,
                # "entropy_coeff_start": entropy_coeff_start,
                # "entropy_coeff_finish": entropy_coeff_finish,
                # "max_grad_norm": max_grad_norm,
                # "batch_size": batch_size,
                # "num_epochs": num_epochs,
                # "mini_batch_size": mini_batch_size,
                # "lr": lr,
                # "weight_decay": weight_decay,
                # "NUM_ROLLOUT_WORKERS": NUM_ROLLOUT_WORKERS,
            }
        )
        best_performance = - np.inf
        global_steps = 686651
        for u in range(168,num_updates+1):
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] - (lr/(num_updates+100))

            # entropy_coeff = entropy_coeff_finish
            entropy_coeff = entropy_coeff_finish - (
                entropy_coeff_finish - entropy_coeff_start
            ) * np.exp(-200*(global_steps / total_steps))

            num_steps = 0
            b_actions = torch.Tensor([]).to(device)
            b_log_probs = torch.Tensor([]).to(device)
            b_rewards = torch.Tensor([]).to(device)
            b_values = torch.Tensor([]).to(device)
            b_advantages = torch.Tensor([]).to(device)
            b_returns = torch.Tensor([]).to(device)
            b_entropy = torch.Tensor([]).to(device)
            b_actions_mask = torch.Tensor([]).to(device)
            b_states = []
            b_speedups = []
            avg_episode_length = 0
            m = 0

            while num_steps < batch_size:
                results = ray.get(
                    [
                        rollout_workers[i].rollout.remote(ppo_agent.to("cpu"), "cpu")
                        for i in range(NUM_ROLLOUT_WORKERS)
                    ]
                )

                for result in results:
                    b_speedups.append(math.log(result["speedup"], 4))
                    trajectory_len = len(result["trajectory"])
                    full_trajectory = Transition(*zip(*result["trajectory"]))
                    avg_episode_length = (m * avg_episode_length) / (
                        m + 1
                    ) + trajectory_len / (m + 1)
                    m += 1
                    num_steps += trajectory_len

                    actions = torch.Tensor(full_trajectory.action).to(device)
                    log_probs = torch.Tensor(full_trajectory.log_prob).to(device)
                    rewards = torch.Tensor(full_trajectory.reward).to(device)
                    values = torch.Tensor(full_trajectory.value).to(device)
                    entropies = torch.Tensor(full_trajectory.entropy).to(device)
                    # actions_mask = torch.Tensor(full_trajectory.actions_mask).to(device)
                    # Calculating advantages and lambda returns
                    advantages = torch.zeros(trajectory_len).to(device)
                    returns = torch.zeros(trajectory_len).to(device)

                    states = [None] * trajectory_len

                    states[-1] = Data(
                        x=torch.tensor(
                            full_trajectory.state[-1][0], dtype=torch.float32
                        ),
                        edge_index=torch.tensor(
                            full_trajectory.state[-1][1], dtype=torch.int
                        )
                        .transpose(0, 1)
                        .contiguous(),
                    )

                    advantages[-1] = rewards[-1] - values[-1]

                    for t in reversed(range(trajectory_len - 1)):
                        td = rewards[t] + gamma * values[t + 1] - values[t]
                        advantages[t] = td + gamma * lambdaa * advantages[t + 1]
                        states[trajectory_len - 2 - t] = Data(
                            x=torch.tensor(
                                full_trajectory.state[trajectory_len - 2 - t][0],
                                dtype=torch.float32,
                            ),
                            edge_index=torch.tensor(
                                full_trajectory.state[trajectory_len - 2 - t][1],
                                dtype=torch.int,
                            )
                            .transpose(0, 1)
                            .contiguous(),
                        )

                    returns = advantages + values

                    b_actions = torch.cat([b_actions, actions]).to(device)
                    b_log_probs = torch.cat([b_log_probs, log_probs]).to(device)
                    b_advantages = torch.cat([b_advantages, advantages]).to(device)
                    b_returns = torch.cat([b_returns, returns]).to(device)
                    b_entropy = torch.cat([b_entropy, entropies]).to(device)
                    # b_actions_mask = torch.cat([b_actions_mask, actions_mask]).to(device)
                    b_states.extend(states)

                ray.get(
                    [
                        rollout_workers[i].reset.remote()
                        for i in range(NUM_ROLLOUT_WORKERS)
                    ]
                )

            b_speedups = torch.Tensor(b_speedups)
            b_states = Batch.from_data_list(b_states).to(device)
            batch_indices = torch.arange(num_steps).to(device)

            ppo_agent.to(device)
            ppo_agent.train()

            v_loss_mean = 0
            policy_loss_mean = 0
            total_loss_mean = 0

            s = 0

            for e in range(num_epochs):
                np.random.shuffle(batch_indices)
                for b in range(0, batch_size, mini_batch_size):
                    start, end = b, b + mini_batch_size
                    rand_ind = batch_indices[start:end]
                    _, new_log_prob, new_entropy, new_value = ppo_agent(
                        Batch.from_data_list(b_states[rand_ind]).to(device),
                        actions_mask=None,
                        action=b_actions[rand_ind],
                    )
                    ratio = new_log_prob - b_log_probs[rand_ind]
                    ratio.exp()

                    clipped_ratio = torch.clamp(
                        ratio, 1 - clip_epsilon, 1 + clip_epsilon
                    )
                    clipped_loss = torch.min(
                        ratio * b_advantages[rand_ind],
                        clipped_ratio * b_advantages[rand_ind],
                    )
                    clip_loss = -clipped_loss.mean()

                    v_loss = value_loss(new_value.reshape(-1), b_returns[rand_ind])

                    ent_loss = new_entropy.mean()
                    loss = clip_loss + value_coeff * v_loss - entropy_coeff * ent_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(ppo_agent.parameters(), max_grad_norm)
                    optimizer.step()

                    v_loss_mean = (v_loss_mean * s) / (s + 1) + v_loss.item() / (s + 1)
                    policy_loss_mean = (policy_loss_mean * s) / (
                        s + 1
                    ) + clip_loss.item() / (s + 1)
                    total_loss_mean = (total_loss_mean * s) / (s + 1) + loss.item() / (
                        s + 1
                    )
                    s += 1

            global_steps += num_steps

            speedups_mean = b_speedups.mean().item()

            torch.save(ppo_agent.state_dict(), f"{model_save_path}/last_model_{u}.pt")
            if (u - 168) >= 10 : 
                os.remove(f"{model_save_path}/last_model_{u-10}.pt")
            if best_performance < speedups_mean:
                torch.save(ppo_agent.state_dict(), f"{model_save_path}/best_model_{u}.pt")
                best_performance = speedups_mean

            infos = {
                "Total Loss": total_loss_mean,
                "Value Loss": v_loss_mean,
                "Policy Loss": policy_loss_mean,
                "Entropy ": b_entropy.mean().item(),
                "Reward average": speedups_mean,
                "Reward min": b_speedups.min().item(),
                "Reward max": b_speedups.max().item(),
                "Episode length mean": avg_episode_length,
                "Learning rate" : optimizer.param_groups[0]["lr"],
            }
            print(infos)
            mlflow.log_metrics(
                infos,
                step=global_steps,
            )
        mlflow.end_run()

    ray.shutdown()


