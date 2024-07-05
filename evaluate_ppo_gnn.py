import torch
from agent.policy_value_nn import GAT
from agent.rollout_worker import RolloutWorker
from config.config import Config
from env_api.core.services.compiling_service import CompilingService
from utils.dataset_actor.dataset_actor import DatasetActor
import ray
import json
import os
import argparse as arg


def write_cpp_file(schedule_object):
    tiramisu_prog = schedule_object.prog
    optim_list = schedule_object.schedule_list
    cpp_code = CompilingService.get_schedule_code(
        tiramisu_program=tiramisu_prog, optims_list=optim_list
    )
    CompilingService.write_cpp_code(
        cpp_code,
        os.path.join(
            Config.config.tiramisu.experiment_dir,
            "evaluation",
            schedule_object.prog.name,
        ),
    )


if "__main__" == __name__:

    parser = arg.ArgumentParser() 

    parser.add_argument("--num-nodes", default=1, type=int)

    args = parser.parse_args()

    NUM_ROLLOUT_WORKERS = args.num_nodes

    full_log = ""
    ray.init()
    # ray.init("auto")
    # ray.init()
    # Init global config to run the Tiramisu env
    Config.init()

    dataset_worker = DatasetActor.remote(Config.config.dataset)
    device = "cpu"

    ppo_agent = GAT(input_size=718, num_heads=4, hidden_size=198, num_outputs=56).to(
        device
    )

    ppo_agent.load_state_dict(
        torch.load(
            "/scratch/dl5133/Dev/RL-Agent/new_agent/experiment_dir/models/experiment_2.5k_101/model_experiment_2.5k_101_962.pt",
            map_location=torch.device(device),
        )
    )

    rollout_workers = [
        RolloutWorker.options(
            num_cpus=10, num_gpus=0, scheduling_strategy="SPREAD"
        ).remote(dataset_worker, Config.config, worker_id=i)
        for i in range(NUM_ROLLOUT_WORKERS)
    ]

    num_functions = ray.get(dataset_worker.get_dataset_size.remote())

    res = {}

    for _ in range(num_functions // NUM_ROLLOUT_WORKERS):
        results = ray.get(
            [
                rollout_workers[i].rollout.remote(ppo_agent, "cpu")
                for i in range(NUM_ROLLOUT_WORKERS)
            ]
        )
        for result in results:
            full_log += result["log_trajectory"]
            res[result["schedule_object"].prog.name] = {}
            res[result["schedule_object"].prog.name]["schedule"] = result[
                "schedule_object"
            ].schedule_str
            res[result["schedule_object"].prog.name]["speedup"] = result["speedup"]
            write_cpp_file(result["schedule_object"])

        ray.get([rollout_workers[i].reset.remote() for i in range(NUM_ROLLOUT_WORKERS)])

    results = ray.get(
        [
            rollout_workers[i].rollout.remote(ppo_agent, "cpu")
            for i in range(num_functions % NUM_ROLLOUT_WORKERS)
        ]
    )
    for result in results:
        full_log += result["log_trajectory"]
        res[result["schedule_object"].prog.name] = {}
        res[result["schedule_object"].prog.name]["schedule"] = result[
            "schedule_object"
        ].schedule_str
        res[result["schedule_object"].prog.name]["speedup"] = result["speedup"]
        write_cpp_file(result["schedule_object"])

    ray.get(
        [
            rollout_workers[i].reset.remote()
            for i in range(num_functions % NUM_ROLLOUT_WORKERS)
        ]
    )

    with open("./results.json", "w") as file:
        json.dump(res, file)

    with open("./log.txt", "w") as file:
        file.write(full_log)

    ray.shutdown()
