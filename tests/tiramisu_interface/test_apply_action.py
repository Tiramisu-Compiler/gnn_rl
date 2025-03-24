from unittest.mock import MagicMock, PropertyMock, patch
import pytest

from agent.tiramisu_interface import ActionSlices, ApplyActionResult, TiramisuInterface
from tiralib.tiramisu import Schedule

from config.config import Config


def tilings_2d():
    tilings = []
    for i in range(14, 50, 4):
        tilings.extend(
            [
                (
                    i + j,
                    f"T2(L{j},L{j+1},{ActionSlices.tiling_size(i)[0]},{ActionSlices.tiling_size(i)[1]},comps=['comp02'])",
                )
                if j < 2
                else (i + j, "raise")
                for j in range(0, 4)
            ]
        )

    return tilings


@pytest.mark.parametrize(
    "action, expected_action_str",
    [
        (0, "I(L0,L1,comps=['comp02'])"),
        (1, "I(L1,L2,comps=['comp02'])"),
        (2, "raise"),
        (3, "raise"),
        (4, "R(L0,comps=['comp02'])"),
        (5, "R(L1,comps=['comp02'])"),
        (6, "R(L2,comps=['comp02'])"),
        (7, "raise"),
        (8, "raise"),
        (9, "S(L0,L1,0,0,comps=['comp02'])"),
        (10, "S(L1,L2,0,0,comps=['comp02'])"),
        (11, "raise"),
        (12, "P(L0,comps=['comp02'])"),
        (13, "P(L1,comps=['comp02'])"),
    ]
    + tilings_2d()
    + [
        (50, "U(L2,1,comps=['comp02'])"),
        (51, "U(L2,2,comps=['comp02'])"),
        (52, "U(L2,4,comps=['comp02'])"),
        (53, "U(L2,8,comps=['comp02'])"),
        (54, "U(L2,16,comps=['comp02'])"),
        (55, "raise"),
    ],
)
def test_action_index_to_tiralib_action(ti_cvt, action, expected_action_str):
    if expected_action_str == "raise":
        with pytest.raises(ValueError):
            action_obj = ti_cvt.action_index_to_tiralib_action(action)
    else:
        action_obj = ti_cvt.action_index_to_tiralib_action(action)
        action_obj.initialize_action_for_tree(ti_cvt.tree)
        assert str(action_obj) == expected_action_str


def test_action_index_to_tiralib_action_invalid_unrolling_and_parallelization(
    _1_node_branch_ti,
):
    with pytest.raises(ValueError):
        _1_node_branch_ti.action_index_to_tiralib_action(50)

    _1_node_branch_ti.current_branch_index = 1

    with pytest.raises(ValueError):
        _1_node_branch_ti.action_index_to_tiralib_action(13)


@patch("agent.tiramisu_interface.TiramisuInterface.get_mask", return_value=[1] * 56)
def test_apply_action_all_actions_masked(get_mask, ti_cvt):
    result = ti_cvt.apply_action(0)
    assert result == ApplyActionResult(
        is_legal=True, speedup=1, done=True, crashed=False
    )


def test_apply_action_next_branch(_1_node_branch_ti):
    ti = _1_node_branch_ti
    result = ti.apply_action(55)
    assert result == ApplyActionResult(
        is_legal=True, speedup=1, done=False, crashed=False
    )
    assert ti.current_branch_index == 1

    result = ti.apply_action(55)
    assert result == ApplyActionResult(
        is_legal=True, speedup=1, done=False, crashed=False
    )
    assert ti.current_branch_index == 2

    result = ti.apply_action(55)
    assert result == ApplyActionResult(
        is_legal=True, speedup=1, done=True, crashed=False
    )
    assert ti.current_branch_index == 2


@patch(
    "agent.tiramisu_interface.TiramisuInterface.schedule_is_legal", return_value=False
)
def test_apply_action_illegal_action(schedule_is_legal, _1_node_branch_ti):
    ti = _1_node_branch_ti
    result = ti.apply_action(0)
    assert result == ApplyActionResult(
        is_legal=False, speedup=1, done=False, crashed=False
    )


@patch(
    "agent.tiramisu_interface.TiramisuInterface.schedule_is_legal", return_value=True
)
def test_apply_action_legal_action_with_cache(schedule_is_legal, _1_node_branch_ti):
    ti = _1_node_branch_ti
    ti.cache = MagicMock()
    ti.cache.execution_time = lambda _, y: 2.0 if y == "empty" else 1.0
    result = ti.apply_action(0)
    assert result == ApplyActionResult(
        is_legal=True, speedup=2, done=False, crashed=False
    )


@patch(
    "agent.tiramisu_interface.TiramisuInterface.schedule_is_legal", return_value=True
)
@patch("agent.tiramisu_interface.median_execution_time", lambda x: 1.0 if x else 2.0)
def test_apply_action_legal_action_without_cache(schedule_is_legal, _1_node_branch_ti):
    ti = _1_node_branch_ti
    ti.schedule
    ti.cache = None
    result = ti.apply_action(0)
    assert result == ApplyActionResult(
        is_legal=True, speedup=2, done=False, crashed=False
    )


@patch(
    "agent.tiramisu_interface.TiramisuInterface.schedule_is_legal",
    side_effect=ValueError("Execution crashed"),
)
def test_apply_action_execution_crashed(schedule_is_legal, _1_node_branch_ti):
    ti = _1_node_branch_ti
    ti.schedule
    ti.cache = None
    result = ti.apply_action(0)
    assert result == ApplyActionResult(
        is_legal=False, speedup=1, done=False, crashed=True
    )


@patch(
    "agent.tiramisu_interface.TiramisuInterface.schedule_is_legal", return_value=True
)
@patch("agent.tiramisu_interface.median_execution_time", lambda x: 1.0 if x else 2.0)
@patch(
    "tiralib.tiramisu.tiramisu_tree.TiramisuTree.depth",
    new_callable=PropertyMock(return_value=6),
)
def test_apply_action_new_tree_exceeds_depth(
    depth, schedule_is_legal, _1_node_branch_ti
):
    ti = _1_node_branch_ti
    ti.schedule
    result = ti.apply_action(0)
    assert result == ApplyActionResult(
        is_legal=True, speedup=2, done=True, crashed=False
    )


@patch("tiralib.tiramisu.tiramisu_tree.TiramisuTree.from_isl_ast_string_list")
def test_schedule_is_legal_cache_hit(isl_ast, ti_cvt):
    ti = ti_cvt
    ti.cache = MagicMock()
    ti.cache.schedules_legality = {
        "I(L0,L1,comps=['comp02'])": True,
    }
    ti.schedule = Schedule.from_sched_str(
        "I(L0,L1,comps=['comp02'])", ti.tiramisu_program
    )
    assert ti.schedule_is_legal(ti.schedule)


@patch("tiralib.tiramisu.tiramisu_tree.TiramisuTree.from_isl_ast_string_list")
@patch(
    "tiralib.tiramisu.function_server.FunctionServer.run",
    return_value=MagicMock(
        name="cvt", legality=True, isl_ast="isl_ast", exec_times=[2.0], success=True
    ),
)
def test_schedule_is_legal_no_cache(run, isl_ast, ti_cvt):
    ti = ti_cvt
    ti.cache = None
    ti.schedule = Schedule.from_sched_str(
        "I(L0,L1,comps=['comp02'])", ti.tiramisu_program
    )
    assert ti.schedule_is_legal(ti.schedule)


@patch("tiralib.tiramisu.tiramisu_tree.TiramisuTree.from_isl_ast_string_list")
@patch(
    "tiralib.tiramisu.function_server.FunctionServer.run",
    return_value=MagicMock(
        name="cvt",
        legality=True,
        isl_ast="isl_ast",
        exec_times=[2.0],
        success=True,
        additional_info="skewing_factors:1,1",
    ),
)
def test_schedule_is_legal_skewing(run, isl_ast, ti_cvt):
    ti = ti_cvt
    ti.cache = None
    ti.schedule = Schedule.from_sched_str(
        "S(L0,L1,0,0,comps=['comp02'])", ti.tiramisu_program
    )
    assert ti.schedule_is_legal(ti.schedule)
    assert ti.schedule.optims_list[0].factors == [1, 1]


@patch("tiralib.tiramisu.tiramisu_tree.TiramisuTree.from_isl_ast_string_list")
@patch(
    "tiralib.tiramisu.function_server.FunctionServer.run",
    return_value=MagicMock(
        name="cvt",
        legality=True,
        isl_ast="isl_ast",
        exec_times=[2.0],
        success=True,
        additional_info="skewing_factors:1,1",
    ),
)
def test_schedule_is_legal_skewing_factors_set_by_user(run, isl_ast, ti_cvt):
    ti = ti_cvt
    ti.cache = None
    ti.schedule = Schedule.from_sched_str(
        "S(L0,L1,2,2,comps=['comp02'])", ti.tiramisu_program
    )
    assert ti.schedule_is_legal(ti.schedule)
    assert ti.schedule.optims_list[0].factors == [2, 2]


@patch("tiralib.tiramisu.tiramisu_tree.TiramisuTree.from_isl_ast_string_list")
def test_schedule_is_legal_no_server(isl_ast, dataset_actor):
    function_name, cache, cpp = dataset_actor.get_function_by_name(
        "function_mvt_MEDIUM"
    )
    ti = TiramisuInterface(
        cpp,
        tiralib_config_path=Config.config.tiralib_config_path,
        cache=cache,
        machine=Config.config.machine,
        use_server=False,
    )
    ti.cache = MagicMock()
    ti.cache.schedules_legality = {
        "S(L0,L1,1,1,comps=['comp03', 'comp04'])": None,
    }
    ti.schedule = Schedule.from_sched_str(
        "S(L0,L1,0,0,comps=['comp03', 'comp04'])", ti.tiramisu_program
    )
    with (
        patch(
            "tiralib.tiramisu.function_server.FunctionServer.run",
        ) as run,
        patch(
            "tiralib.tiramisu.compiling_service.CompilingService.compile_legality",
            return_value=(True, MagicMock()),
        ),
        patch(
            "tiralib.tiramisu.compiling_service.CompilingService.call_skewing_solver",
            return_value=(1, 1),
        ) as skewing_solver,
    ):
        assert ti.schedule_is_legal(ti.schedule)
        run.assert_not_called()
        skewing_solver.assert_called_once()
        assert ti.schedule.optims_list[0].factors == [1, 1]


@patch("tiralib.tiramisu.tiramisu_tree.TiramisuTree.from_isl_ast_string_list")
def test_schedule_is_legal_no_server_skewing_set_by_user(isl_ast, dataset_actor):
    function_name, cache, cpp = dataset_actor.get_function_by_name(
        "function_mvt_MEDIUM"
    )
    ti = TiramisuInterface(
        cpp,
        tiralib_config_path=Config.config.tiralib_config_path,
        cache=cache,
        machine=Config.config.machine,
        use_server=False,
    )
    ti.cache = MagicMock()
    ti.cache.schedules_legality = {
        "S(L0,L1,1,1,comps=['comp03', 'comp04'])": None,
    }
    ti.schedule = Schedule.from_sched_str(
        "S(L0,L1,2,2,comps=['comp03', 'comp04'])", ti.tiramisu_program
    )
    with (
        patch(
            "tiralib.tiramisu.function_server.FunctionServer.run",
        ) as run,
        patch(
            "tiralib.tiramisu.compiling_service.CompilingService.compile_legality",
            return_value=(True, MagicMock()),
        ),
        patch(
            "tiralib.tiramisu.compiling_service.CompilingService.call_skewing_solver",
        ) as skewing_solver,
    ):
        assert ti.schedule_is_legal(ti.schedule)
        run.assert_not_called()
        skewing_solver.assert_not_called()
        assert ti.schedule.optims_list[0].factors == [2, 2]
