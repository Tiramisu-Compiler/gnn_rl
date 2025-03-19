from unittest.mock import patch

import pytest
from agent.tiramisu_interface import TiramisuInterface
from config.config import Config


@patch("agent.tiramisu_interface.median_execution_time", return_value=1.5)
def test_cache(_, dataset_actor):
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


def test_init_without_cache(dataset_actor):
    function_name, cache, cpp = dataset_actor.get_function_by_name(
        "function_cvtcolor_MEDIUM"
    )
    assert function_name == "function_cvtcolor_MEDIUM"
    assert cache is not None
    assert cpp is not None

    ti = TiramisuInterface(
        cpp,
        tiralib_config_path=Config.config.tiralib_config_path,
        machine=Config.config.machine,
    )
    assert ti.tiramisu_program.name == "function_cvtcolor_MEDIUM"
    assert ti.cache is None
    assert ti.tiramisu_program.server is not None
    assert ti.machine == Config.config.machine
    assert ti._initial_execution_time is None
    with patch(
        "tiralib.tiramisu.schedule.Schedule.execute", return_value=[3.0, 2.0, 5.0]
    ):
        assert ti.initial_execution_time == 3.0


def test_init_with_cache(dataset_actor):
    function_name, cache, cpp = dataset_actor.get_function_by_name(
        "function_cvtcolor_MEDIUM"
    )
    assert function_name == "function_cvtcolor_MEDIUM"

    ti = TiramisuInterface(
        cpp,
        tiralib_config_path=Config.config.tiralib_config_path,
        machine=Config.config.machine,
        cache=cache,
    )
    assert ti.tiramisu_program.name == "function_cvtcolor_MEDIUM"
    assert ti.cache is not None
    assert ti.tiramisu_program.server is None
    assert ti.machine == Config.config.machine
    assert ti._initial_execution_time is None
    assert ti.cache.execution_time(Config.config.machine, "empty") is None
    with patch(
        "tiralib.tiramisu.schedule.Schedule.execute", return_value=[3.0, 2.0, 5.0]
    ) as mock_execute:
        assert ti.initial_execution_time == 3.0
        mock_execute.assert_called_once()
        assert ti.initial_execution_time == 3.0
        mock_execute.assert_called_once()
    assert ti.cache.execution_time(Config.config.machine, "empty") == 3.0

    ti = TiramisuInterface(
        cpp,
        tiralib_config_path=Config.config.tiralib_config_path,
        machine=Config.config.machine,
        cache=cache,
    )
    assert ti.initial_execution_time == 3.0


def test_branches(ti_cvt, dataset_actor):
    print(ti_cvt.tiramisu_program.tree)
    assert len(ti_cvt.branches) == 1
    assert ti_cvt.current_branch_index == 0
    assert ti_cvt.current_branch == [("comp02", 0), ("comp02", 1), ("comp02", 2)]

    function_name, cache, cpp = dataset_actor.get_function_by_name(
        "function_blur_MEDIUM"
    )
    ti = TiramisuInterface(
        cpp,
        tiralib_config_path=Config.config.tiralib_config_path,
        cache=cache,
        machine=Config.config.machine,
    )
    assert len(ti.branches) == 1
    assert ti.current_branch_index == 0


def test_init_with_cache_no_server(dataset_actor):
    function_name, cache, cpp = dataset_actor.get_function_by_name(
        "function_cvtcolor_MEDIUM"
    )
    assert function_name == "function_cvtcolor_MEDIUM"

    ti = TiramisuInterface(
        cpp,
        tiralib_config_path=Config.config.tiralib_config_path,
        machine=Config.config.machine,
        cache=cache,
        use_server=False,
    )
    assert ti.tiramisu_program.name == "function_cvtcolor_MEDIUM"
    assert ti.cache is not None
    assert ti.tiramisu_program.server is None
    assert ti.machine == Config.config.machine
    assert ti._initial_execution_time is None
    assert ti.cache.execution_time(Config.config.machine, "empty") is None
    with patch(
        "tiralib.tiramisu.schedule.Schedule.execute", return_value=[3.0, 2.0, 5.0]
    ) as mock_execute:
        assert ti.initial_execution_time == 3.0
        mock_execute.assert_called_once()
        assert ti.initial_execution_time == 3.0
        mock_execute.assert_called_once()
        assert ti.tiramisu_program.server is None


def test_init_no_cache_no_server(dataset_actor):
    function_name, cache, cpp = dataset_actor.get_function_by_name(
        "function_cvtcolor_MEDIUM"
    )
    assert function_name == "function_cvtcolor_MEDIUM"

    with pytest.raises(ValueError):
        TiramisuInterface(
            cpp,
            tiralib_config_path=Config.config.tiralib_config_path,
            machine=Config.config.machine,
            use_server=False,
        )
