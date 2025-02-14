import json
from pathlib import Path
import torch
from agent.policy_value_nn import GAT
from agent.rollout_worker import RolloutWorker
from config.config import Config
from utils.dataset_actor.dataset_actor import DatasetActor
import argparse as arg


if "__main__" == __name__:
    parser = arg.ArgumentParser()

    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    MODEL_PATH = args.model

    full_log = ""

    # Init global config to run the Tiramisu env
    Config.init()

    dataset_worker = DatasetActor(Config.config.dataset)
    device = "cpu"

    ppo_agent = GAT(input_size=720, num_heads=4, hidden_size=128, num_outputs=56).to(
        device
    )

    ppo_agent.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device(device),
        )
    )

    results = {}
    logs = ""

    csv_file = Path("./results.csv").open("w")
    csv_file.write("function;schedule;speedup\n")

    for function in dataset_worker.dataset_service.dataset:
        print(f"Evaluting function {function}")
        # _, function_data, function_cpp = dataset_worker.get_function_by_name(function)
        rollout_worker = RolloutWorker(
            dataset_worker, Config.config, function_name=function
        )
        result = rollout_worker.rollout(ppo_agent, device)
        results[function] = {
            "schedule": result.schedule,
            "speedup": result.speedup,
        }
        logs += result.log_trajectory
        csv_file.write(f"{function};{result.schedule};{result.speedup}\n")

    csv_file.close()
    Path("./results.json").write_text(json.dumps(results, indent=4))
    Path("./log.txt").write_text(logs)
