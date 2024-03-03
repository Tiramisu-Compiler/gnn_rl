import mlflow
import torch
import torch.nn as nn
from agent.policy_value_nn import GATMultiDiscrete
from agent.rollout_worker import RolloutWorker, Transition
from config.config import Config
from utils.dataset_actor.dataset_actor import DatasetActor
import numpy as np
import ray
import math
from torch_geometric.data import Batch, Data


num_updates = 1000
clip_epsilon = 0.3
gamma = 0.99
lambdaa = 0.95
value_coeff = 1
entropy_coeff_start = 0.1
entropy_coeff_finish = 0
max_grad_norm = 1
batch_size = 512
num_epochs = 4
mini_batch_size = 64
start_lr = 1e-4
final_lr = 1e-4
weight_decay = 0
total_steps = num_updates * batch_size
NUM_ROLLOUT_WORKERS = 8
CPUS_PER_WORKER = 12


if "__main__" == __name__:
    ray.init("auto")
    # Init global config to run the Tiramisu env
    Config.init()

    dataset_worker = DatasetActor.remote(Config.config.dataset)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ppo_agent = GATMultiDiscrete(
        input_size=718,
        num_heads=4,
        hidden_size=32,
        num_types=7,
        num_loops=4,
        i_actions=10,
        u_actons=3,
        t_actions=9,
    ).to(device)
    optimizer = torch.optim.Adam(
        ppo_agent.parameters(), lr=start_lr, weight_decay=weight_decay, eps=1e-5
    )
    value_loss = nn.MSELoss()

    # ppo_agent.load_state_dict(
    #     torch.load(
    #         "/scratch/dl5133/Dev/RL-Agent/new_agent/models/model_exec_training_with_init_103_62.pt",
    #         map_location=torch.device(device)
    #     ),
    # )

    rollout_workers = [
        RolloutWorker.options(
            num_cpus=CPUS_PER_WORKER, num_gpus=0, scheduling_strategy="SPREAD"
        ).remote(dataset_worker, Config.config, worker_id=i)
        for i in range(NUM_ROLLOUT_WORKERS)
    ]

    run_name = "exec_training_bench_md_9"

    with mlflow.start_run(
        run_name=run_name,
        # run_id="8f80a3b96ea04676928053f7fd90aa4d"
    ) as run:
        mlflow.log_params(
            {
                "num_updates": num_updates,
                "clip_epsilon": clip_epsilon,
                "gamma": gamma,
                "lambdaa": lambdaa,
                "value_coeff": value_coeff,
                "entropy_coeff_start": entropy_coeff_start,
                "entropy_coeff_finish": entropy_coeff_finish,
                "max_grad_norm": max_grad_norm,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "mini_batch_size": mini_batch_size,
                "start_lr": start_lr,
                "final_lr": final_lr,
                "weight_decay": weight_decay,
                "NUM_ROLLOUT_WORKERS": NUM_ROLLOUT_WORKERS,
            }
        )
        best_performance = 0.3
        global_steps = 0
        for u in range(num_updates):
            # optimizer.param_groups[0]["lr"] = final_lr - (final_lr - start_lr) * np.exp(
            #     -2 * u / num_updates
            # )

            # entropy_coeff = entropy_coeff_finish
            entropy_coeff = entropy_coeff_finish - (
                entropy_coeff_finish - entropy_coeff_start
            ) * np.exp(-50 * global_steps / total_steps)

            num_steps = 0
            b_actions = torch.Tensor([]).to(device)
            b_log_probs = torch.Tensor([]).to(device)
            b_rewards = torch.Tensor([]).to(device)
            b_values = torch.Tensor([]).to(device)
            b_advantages = torch.Tensor([]).to(device)
            b_returns = torch.Tensor([]).to(device)
            b_entropy = torch.Tensor([]).to(device)
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

                    actions = torch.cat(full_trajectory.action, dim=1).to(device)
                    log_probs = torch.Tensor(full_trajectory.log_prob).to(device)
                    rewards = torch.Tensor(full_trajectory.reward).to(device)
                    values = torch.Tensor(full_trajectory.value).to(device)
                    entropies = torch.Tensor(full_trajectory.entropy).to(device)
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

                    b_actions = torch.cat([b_actions, actions], dim=1).to(device)
                    b_log_probs = torch.cat([b_log_probs, log_probs]).to(device)
                    b_advantages = torch.cat([b_advantages, advantages]).to(device)
                    b_returns = torch.cat([b_returns, returns]).to(device)
                    b_entropy = torch.cat([b_entropy, entropies]).to(device)
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
                    _, _ , new_log_prob, new_entropy, new_value = ppo_agent(
                        data=Batch.from_data_list(b_states[rand_ind]).to(device),
                        input_actions=b_actions.T[rand_ind].T.type(torch.long).to(device),
                        action_mask=None,
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

            if best_performance < speedups_mean:
                torch.save(
                    ppo_agent.state_dict(),
                    f"./experiment_dir/models/model_{run_name}_{u}.pt",
                )
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
            }
            print(infos)
            mlflow.log_metrics(
                infos,
                step=global_steps,
            )
        mlflow.end_run()

    ray.shutdown()
