from pytest import fixture
from agent.tiramisu_interface import TiramisuInterface
from config.config import Config
from utils.dataset_actor.dataset_actor import DatasetActor


@fixture
def conf_init():
    Config.init("tests/config.yaml")


@fixture
def dataset_actor(conf_init):
    return DatasetActor(Config.config.dataset)


def test_cache(dataset_actor):
    function_name, cache, cpp = dataset_actor.get_function_by_name(
        "function_cvtcolor_MEDIUM"
    )
    assert function_name == "function_cvtcolor_MEDIUM"
    assert cache is not None
    assert cpp is not None

    ti = TiramisuInterface(
        cpp,
        tiralib_config_path=Config.config.tiralib_config_path,
        cache=cache,
        machine=Config.config.machine,
    )
    assert ti.tiramisu_program.name == "function_cvtcolor_MEDIUM"
    schedule_str = "I(L0,L1,comps=['comp02'])"
    assert schedule_str not in ti.cache.schedules_legality
    assert schedule_str not in ti.cache.isl_ast
    assert Config.config.machine not in ti.cache.execution_times
    # INTERCHANGE 0,1
    ti.apply_action(0)
    assert schedule_str in ti.cache.schedules_legality
    assert ti.cache.schedules_legality[schedule_str] is True
    assert schedule_str in ti.cache.isl_ast
    assert schedule_str in ti.cache.execution_times[Config.config.machine]
