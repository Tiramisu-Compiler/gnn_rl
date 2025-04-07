from unittest.mock import patch
from pytest import fixture
from agent.policy_value_nn import GAT
from agent.tiramisu_interface import TiramisuInterface
from config.config import Config
from utils.dataset_actor import DatasetActor


@fixture
def conf_init():
    Config.init("tests/config.yaml")


@fixture
def dataset_actor(conf_init):
    return DatasetActor(Config.config.dataset)


@fixture
@patch("tiralib.tiramisu.schedule.Schedule.execute", return_value=[2.0])
def ti_cvt(_, dataset_actor):
    function_name, cache, cpp = dataset_actor.get_function_by_name(
        "function_cvtcolor_MEDIUM"
    )
    return TiramisuInterface(
        cpp,
        tiralib_config_path=Config.config.tiralib_config_path,
        cache=cache,
        machine=Config.config.machine,
    )


@fixture
@patch("tiralib.tiramisu.schedule.Schedule.execute", return_value=[2.0])
def ti_mvt(_, dataset_actor):
    function_name, cache, cpp = dataset_actor.get_function_by_name(
        "function_mvt_MEDIUM"
    )
    return TiramisuInterface(
        cpp,
        tiralib_config_path=Config.config.tiralib_config_path,
        cache=cache,
        machine=Config.config.machine,
    )


@fixture
def test_dataset_actor(conf_init):
    config = Config.config.dataset
    config.dataset_path = "tests/benchmarks/test_programs.pkl"
    config.cpps_path = "tests/benchmarks/test_programs_cpps.pkl"
    return DatasetActor(config)


@fixture
@patch("tiralib.tiramisu.schedule.Schedule.execute", return_value=[2.0])
def _1_node_branch_ti(_, test_dataset_actor):
    function_name, cache, cpp = test_dataset_actor.get_function_by_name(
        "function014733"
    )
    return TiramisuInterface(
        cpp,
        tiralib_config_path=Config.config.tiralib_config_path,
        cache=cache,
        machine=Config.config.machine,
    )


@fixture
def gnn_model():
    return GAT(input_size=720, num_heads=4, hidden_size=128, num_outputs=56).to("cpu")
