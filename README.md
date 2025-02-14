# Tiramisu GNN Autoscheduler

This is a repository for the Tiramisu GNN Autoscheduler, a tool for training graph neural networks based reinforcement learning agents. The tool is built on top of the [Tiramisu](https://github.com/Tiramisu-Compiler/tiramisu), a polyhedral compiler for deep learning and scientific computing.

The project uses [TiraLib](https://github.com/Tiramisu-Compiler/TiraLib) as the backend for communications with the Tiramisu compiler. TiraLib is a Python library that provides an interface to write and run Tiramisu programs. It uses [TiralibCpp](https://github.com/skourta/TiraLibCpp) as a backend to compile and run the Tiramisu programs.

## Installation

To install the Tiramisu GNN Autoscheduler, you need to install the following dependencies:
- [Tiramisu](https://github.com/Tiramisu-Compiler/tiramisu)
- [TiralibCpp](https://github.com/skourta/TiraLibCpp)
- [Poetry](https://python-poetry.org/)
After installing the dependencies, you can run the following commands to train an agent:

- Clone the repository:
```bash
git clone https://github.com/Tiramisu-Compiler/gnn_rl
```
- Install the requirements :
```bash
poetry install
```

- Create a config for training by copying the example config and make sure both the directory with the headers and library files for Tiramisu and TiraLibCpp are correctly added to the config:
```bash
cp config/config.yaml.temp config/config.yaml
```


- Create a TraLib config just like in the [example template](https://github.com/Tiramisu-Compiler/TiraLib/blob/main/config.yaml.example) and save it in the same directory as the config.yaml
```bash
wget https://raw.githubusercontent.com/Tiramisu-Compiler/TiraLib/main/config.yaml.example -O config/tiralib_config.yaml
```

- Use the `rl_exec_job.sh.temp` script as a template to create a script that will run the training job. Make sure to update the paths to the config files and the Tiramisu GNN Autoscheduler repository in the script.

- You can also run the training on the current machine by running the following command:
```bash
python train_ppo_gnn.py --num-nodes=$NBR_NODES --name$NAME_OF_TRAINING
```