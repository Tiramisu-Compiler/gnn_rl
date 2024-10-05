import math
import mlflow
from numpy import mean
from torch.distributions import Categorical 
import torch.nn as nn 
import argparse as arg
import ray
import torch 
from config.config import Config
from agent.rollout_worker import RolloutWorker, Transition
from utils.dataset_actor.dataset_actor import DatasetActor

class RandomAgent(nn.Module): 
    def __init__(self) -> None:
        super(RandomAgent,self).__init__()

    def forward(self, data=None, actions_mask=None, action=None): 
        logits = - actions_mask * 1e8
        probs = Categorical(logits=logits)
        return probs.sample(), torch.tensor([0]),torch.tensor([0]),torch.tensor([0])
    

total_steps = 10_000_000
batch_size = 1000
    

if "__main__" == __name__ : 

    parser = arg.ArgumentParser() 
    parser.add_argument("--num-nodes", default=1, type=int)
    parser.add_argument("--name", type=str, default="experiment_101")
    args = parser.parse_args()
    run_name = args.name
    NUM_ROLLOUT_WORKERS = args.num_nodes


    if NUM_ROLLOUT_WORKERS > 1 :
        ray.init("auto")
    else : 
        ray.init()
    

    Config.init()
    dataset_worker = DatasetActor.remote(Config.config.dataset)
    
    
    device = "cpu"
    random_agent = RandomAgent().to(device)

    rollout_workers = [
        RolloutWorker.options(
            num_cpus=24, num_gpus=0, scheduling_strategy="SPREAD"
        ).remote(dataset_worker, Config.config, worker_id=200+i)
        for i in range(NUM_ROLLOUT_WORKERS)
    ]

    with mlflow.start_run(
        run_name=run_name,
    ) as run:
        mlflow.log_params(
            {
                "Total Steps" : total_steps,
                "NUM_ROLLOUT_WORKERS": NUM_ROLLOUT_WORKERS,
            }
        )
        global_steps = 0
        while global_steps < total_steps :

            num_steps = 0
            b_speedups = []
            avg_episode_length = 0
            m = 0

            while num_steps < batch_size:
                results = ray.get(
                    [
                        rollout_workers[i].rollout.remote(random_agent.to("cpu"), "cpu")
                        for i in range(NUM_ROLLOUT_WORKERS)
                    ]
                )

                for result in results:
                    b_speedups.append(math.log(result["speedup"], 4))
                    trajectory_len = len(result["trajectory"])
                    avg_episode_length = (m * avg_episode_length) / (
                        m + 1
                    ) + trajectory_len / (m + 1)
                    m += 1
                    num_steps += trajectory_len


                ray.get(
                    [
                        rollout_workers[i].reset.remote()
                        for i in range(NUM_ROLLOUT_WORKERS)
                    ]
                )
            global_steps += num_steps

            speedups_mean = mean(b_speedups)


            infos = {
                "Reward average": speedups_mean,
                "Reward min": min(b_speedups),
                "Reward max": max(b_speedups),
                "Episode length mean": avg_episode_length,
            }
            print(infos)
            mlflow.log_metrics(
                infos,
                step=global_steps,
            )
        mlflow.end_run()

    ray.shutdown()