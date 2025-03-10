from unittest.mock import patch
import pytest
from agent.tiramisu_interface import (
    NEXT_ACTION_INDEX,
    ActionSlices,
    _get_level_action_indices_tuple_actions,
)
from tiralib.tiramisu import Schedule


@pytest.mark.parametrize(
    "level, start_index, expected",
    [
        (0, ActionSlices.SKEWING.start, [ActionSlices.SKEWING.start]),
        (0, ActionSlices.INTERCHANGE.start, [ActionSlices.INTERCHANGE.start]),
        (0, ActionSlices.TILING2D.start, [ActionSlices.TILING2D.start]),
        (0, ActionSlices.TILING2D.start + 12, [ActionSlices.TILING2D.start + 12]),
        (
            1,
            ActionSlices.SKEWING.start,
            [ActionSlices.SKEWING.start, ActionSlices.SKEWING.start + 1],
        ),
        (
            1,
            ActionSlices.INTERCHANGE.start,
            [ActionSlices.INTERCHANGE.start, ActionSlices.INTERCHANGE.start + 1],
        ),
        (
            1,
            ActionSlices.TILING2D.start,
            [ActionSlices.TILING2D.start, ActionSlices.TILING2D.start + 1],
        ),
        (
            1,
            ActionSlices.TILING2D.start + 12,
            [ActionSlices.TILING2D.start + 12, ActionSlices.TILING2D.start + 13],
        ),
        (
            2,
            ActionSlices.SKEWING.start,
            [ActionSlices.SKEWING.start + 1, ActionSlices.SKEWING.start + 2],
        ),
        (
            2,
            ActionSlices.INTERCHANGE.start,
            [ActionSlices.INTERCHANGE.start + 1, ActionSlices.INTERCHANGE.start + 2],
        ),
        (
            2,
            ActionSlices.TILING2D.start,
            [ActionSlices.TILING2D.start + 1, ActionSlices.TILING2D.start + 2],
        ),
        (
            2,
            ActionSlices.TILING2D.start + 12,
            [ActionSlices.TILING2D.start + 13, ActionSlices.TILING2D.start + 14],
        ),
        (3, ActionSlices.SKEWING.start, [ActionSlices.SKEWING.stop - 1]),
        (
            3,
            ActionSlices.INTERCHANGE.start,
            [ActionSlices.INTERCHANGE.start + 2, ActionSlices.INTERCHANGE.start + 3],
        ),
        (
            3,
            ActionSlices.TILING2D.start,
            [ActionSlices.TILING2D.start + 2, ActionSlices.TILING2D.start + 3],
        ),
        (
            3,
            ActionSlices.TILING2D.start + 12,
            [ActionSlices.TILING2D.start + 14, ActionSlices.TILING2D.start + 15],
        ),
        (4, ActionSlices.SKEWING.start, []),
        (4, ActionSlices.INTERCHANGE.start, [ActionSlices.INTERCHANGE.start + 3]),
        (4, ActionSlices.TILING2D.start, [ActionSlices.TILING2D.start + 3]),
        (4, ActionSlices.TILING2D.start + 12, [ActionSlices.TILING2D.start + 15]),
        (5, ActionSlices.SKEWING.start, []),
        (5, ActionSlices.INTERCHANGE.start, []),
        (5, ActionSlices.TILING2D.start, []),
        (5, ActionSlices.TILING2D.start + 12, []),
    ],
)
def test_get_level_action_indices_tuple_actions(level, start_index, expected):
    assert _get_level_action_indices_tuple_actions(level, start_index) == expected


def test_mask_empty_schedule(ti_cvt):
    mask = ti_cvt.get_mask()

    interchange_part = [0, 0, 1, 1]
    reversal_part = [0, 0, 0, 1, 1]
    skewing_part = [0, 0, 1]
    parallelize_part = [0, 0]
    tiling2d_part = [0, 0, 1, 1] * 9
    unrolling_part = [0, 0, 0, 0, 0]
    next_action = [0]

    assert (
        mask[ActionSlices.INTERCHANGE] == interchange_part
    ).all(), "Interchange mask is wrong"
    assert (
        mask[ActionSlices.REVERSAL] == reversal_part
    ).all(), "Reversal mask is wrong"
    assert (mask[ActionSlices.SKEWING] == skewing_part).all(), "Skewing mask is wrong"
    assert (
        mask[ActionSlices.PARALLELIZATION] == parallelize_part
    ).all(), "Parallelize mask is wrong"
    assert (
        mask[ActionSlices.TILING2D] == tiling2d_part
    ).all(), "Tiling2D mask is wrong"
    assert (
        mask[ActionSlices.UNROLLING] == unrolling_part
    ).all(), "Unrolling mask is wrong"
    assert (mask[55] == next_action).all(), "Next action mask is wrong"


def test_mask_after_skewing_illegal(ti_cvt):
    result = ti_cvt.apply_action(9)
    assert result.is_legal is False
    mask = ti_cvt.get_mask()

    interchange_part = [0, 0, 1, 1]
    reversal_part = [0, 0, 0, 1, 1]
    skewing_part = [1, 0, 1]
    parallelize_part = [0, 0]
    tiling2d_part = [0, 0, 1, 1] * 9
    unrolling_part = [0, 0, 0, 0, 0]
    next_action = [0]

    assert (
        mask[ActionSlices.INTERCHANGE] == interchange_part
    ).all(), "Interchange mask is wrong"
    assert (
        mask[ActionSlices.REVERSAL] == reversal_part
    ).all(), "Reversal mask is wrong"
    assert (mask[ActionSlices.SKEWING] == skewing_part).all(), "Skewing mask is wrong"
    assert (
        mask[ActionSlices.PARALLELIZATION] == parallelize_part
    ).all(), "Parallelize mask is wrong"
    assert (
        mask[ActionSlices.TILING2D] == tiling2d_part
    ).all(), "Tiling2D mask is wrong"
    assert (
        mask[ActionSlices.UNROLLING] == unrolling_part
    ).all(), "Unrolling mask is wrong"
    assert (mask[55] == next_action).all(), "Next action mask is wrong"


@patch("agent.tiramisu_interface.median_execution_time", return_value=1.5)
def test_mask_after_skewing_legal(_, ti_mvt):
    result = ti_mvt.apply_action(9)
    assert result.is_legal is True
    mask = ti_mvt.get_mask()

    interchange_part = [1, 1, 1, 1]
    reversal_part = [1, 1, 1, 1, 1]
    skewing_part = [1, 1, 1]
    parallelize_part = [0, 0]
    tiling2d_part = [1, 1, 1, 1] * 9
    unrolling_part = [0, 0, 0, 0, 0]
    next_action = [0]

    assert (
        mask[ActionSlices.INTERCHANGE] == interchange_part
    ).all(), "Interchange mask is wrong"
    assert (
        mask[ActionSlices.REVERSAL] == reversal_part
    ).all(), "Reversal mask is wrong"
    assert (mask[ActionSlices.SKEWING] == skewing_part).all(), "Skewing mask is wrong"
    assert (
        mask[ActionSlices.PARALLELIZATION] == parallelize_part
    ).all(), "Parallelize mask is wrong"
    assert (
        mask[ActionSlices.TILING2D] == tiling2d_part
    ).all(), "Tiling2D mask is wrong"
    assert (
        mask[ActionSlices.UNROLLING] == unrolling_part
    ).all(), "Unrolling mask is wrong"
    assert (mask[55] == next_action).all(), "Next action mask is wrong"


@patch("agent.tiramisu_interface.median_execution_time", return_value=1.5)
def test_mask_after_parallelisation(_, ti_mvt):
    result = ti_mvt.apply_action(12)
    assert result.is_legal is True
    mask = ti_mvt.get_mask()

    interchange_part = [1, 1, 1, 1]
    reversal_part = [1, 1, 1, 1, 1]
    skewing_part = [1, 1, 1]
    parallelize_part = [1, 0]
    tiling2d_part = [0, 1, 1, 1] * 9
    unrolling_part = [0] * 5
    next_action = [0]

    assert (
        mask[ActionSlices.INTERCHANGE] == interchange_part
    ).all(), "Interchange mask is wrong"
    assert (
        mask[ActionSlices.REVERSAL] == reversal_part
    ).all(), "Reversal mask is wrong"
    assert (mask[ActionSlices.SKEWING] == skewing_part).all(), "Skewing mask is wrong"
    assert (
        mask[ActionSlices.PARALLELIZATION] == parallelize_part
    ).all(), "Parallelize mask is wrong"
    assert (
        mask[ActionSlices.TILING2D] == tiling2d_part
    ).all(), "Tiling2D mask is wrong"
    assert (
        mask[ActionSlices.UNROLLING] == unrolling_part
    ).all(), "Unrolling mask is wrong"
    assert (mask[55] == next_action).all(), "Next action mask is wrong"


@patch("agent.tiramisu_interface.median_execution_time", return_value=1.5)
def test_mask_after_parallelisation_and_tiling(_, ti_mvt):
    result = ti_mvt.apply_action(12)
    assert result.is_legal is True
    result = ti_mvt.apply_action(14)
    mask = ti_mvt.get_mask()

    interchange_part = [1, 1, 1, 1]
    reversal_part = [1, 1, 1, 1, 1]
    skewing_part = [1, 1, 1]
    parallelize_part = [1, 1]
    tiling2d_part = [1, 1, 1, 1] * 9
    unrolling_part = [0] * 5
    next_action = [0]

    assert (
        mask[ActionSlices.INTERCHANGE] == interchange_part
    ).all(), "Interchange mask is wrong"
    assert (
        mask[ActionSlices.REVERSAL] == reversal_part
    ).all(), "Reversal mask is wrong"
    assert (mask[ActionSlices.SKEWING] == skewing_part).all(), "Skewing mask is wrong"
    assert (
        mask[ActionSlices.PARALLELIZATION] == parallelize_part
    ).all(), "Parallelize mask is wrong"
    assert (
        mask[ActionSlices.TILING2D] == tiling2d_part
    ).all(), "Tiling2D mask is wrong"
    assert (
        mask[ActionSlices.UNROLLING] == unrolling_part
    ).all(), "Unrolling mask is wrong"
    assert (mask[55] == next_action).all(), "Next action mask is wrong"


@patch("agent.tiramisu_interface.median_execution_time", return_value=1.5)
@patch(
    "agent.tiramisu_interface.TiramisuInterface.schedule_is_legal", return_value=True
)
def test_mask_after_unrolling(_, schedule_is_legal, ti_mvt):
    result = ti_mvt.apply_action(51)
    assert result.is_legal is True
    mask = ti_mvt.get_mask()

    interchange_part = [1, 1, 1, 1]
    reversal_part = [1, 1, 1, 1, 1]
    skewing_part = [1, 1, 1]
    parallelize_part = [1, 1]
    tiling2d_part = [1, 1, 1, 1] * 9
    unrolling_part = [1, 1, 1, 1, 1]
    next_action = [0]

    assert (
        mask[ActionSlices.INTERCHANGE] == interchange_part
    ).all(), "Interchange mask is wrong"
    assert (
        mask[ActionSlices.REVERSAL] == reversal_part
    ).all(), "Reversal mask is wrong"
    assert (mask[ActionSlices.SKEWING] == skewing_part).all(), "Skewing mask is wrong"
    assert (
        mask[ActionSlices.PARALLELIZATION] == parallelize_part
    ).all(), "Parallelize mask is wrong"
    assert (
        mask[ActionSlices.TILING2D] == tiling2d_part
    ).all(), "Tiling2D mask is wrong"
    assert (
        mask[ActionSlices.UNROLLING] == unrolling_part
    ).all(), "Unrolling mask is wrong"
    assert (mask[55] == next_action).all(), "Next action mask is wrong"


@patch("agent.tiramisu_interface.median_execution_time", return_value=1.5)
def test_mask_1_node_branch(_, _1_node_branch_ti):
    ti = _1_node_branch_ti
    ti.apply_action(NEXT_ACTION_INDEX)
    assert len(ti.current_branch) == 1, "Branch is not of length 1"
    mask = ti.get_mask()

    interchange_part = [1, 1, 1, 1]
    reversal_part = [1, 1, 0, 1, 1]
    skewing_part = [1, 1, 1]
    parallelize_part = [1, 1]
    tiling2d_part = [1, 1, 1, 1] * 9
    unrolling_part = [0, 0, 0, 0, 0]
    next_action = [0]

    assert (
        mask[ActionSlices.INTERCHANGE] == interchange_part
    ).all(), "Interchange mask is wrong"
    assert (
        mask[ActionSlices.REVERSAL] == reversal_part
    ).all(), "Reversal mask is wrong"
    assert (mask[ActionSlices.SKEWING] == skewing_part).all(), "Skewing mask is wrong"
    assert (
        mask[ActionSlices.PARALLELIZATION] == parallelize_part
    ).all(), "Parallelize mask is wrong"
    assert (
        mask[ActionSlices.TILING2D] == tiling2d_part
    ).all(), "Tiling2D mask is wrong"
    assert (
        mask[ActionSlices.UNROLLING] == unrolling_part
    ).all(), "Unrolling mask is wrong"
    assert (mask[55] == next_action).all(), "Next action mask is wrong"


@patch("agent.tiramisu_interface.median_execution_time", return_value=1.5)
@patch("tiralib.tiramisu.schedule.Schedule.update_tree_from_isl_ast")
def test_mask_unsupported_action(_, update, ti_cvt):
    ti = ti_cvt
    ti.schedule = Schedule.from_sched_str(
        "T3(L0,L1,L2,32,32,32,comps=['comp02'])", ti.tiramisu_program
    )
    with pytest.raises(ValueError):
        ti.apply_action(50)


@patch("agent.tiramisu_interface.median_execution_time", return_value=1.5)
@patch(
    "agent.tiramisu_interface.TiramisuInterface.schedule_is_legal", return_value=True
)
def test_mask_after_interchange(_, schedule_is_legal, ti_cvt):
    result = ti_cvt.apply_action(0)
    assert result.is_legal is True
    mask = ti_cvt.get_mask()

    interchange_part = [1, 0, 1, 1]
    reversal_part = [0, 0, 0, 1, 1]
    skewing_part = [0, 0, 1]
    parallelize_part = [0, 0]
    tiling2d_part = [0, 0, 1, 1] * 9
    unrolling_part = [0, 0, 0, 0, 0]
    next_action = [0]

    assert (
        mask[ActionSlices.INTERCHANGE] == interchange_part
    ).all(), "Interchange mask is wrong"
    assert (
        mask[ActionSlices.REVERSAL] == reversal_part
    ).all(), "Reversal mask is wrong"
    assert (mask[ActionSlices.SKEWING] == skewing_part).all(), "Skewing mask is wrong"
    assert (
        mask[ActionSlices.PARALLELIZATION] == parallelize_part
    ).all(), "Parallelize mask is wrong"
    assert (
        mask[ActionSlices.TILING2D] == tiling2d_part
    ).all(), "Tiling2D mask is wrong"
    assert (
        mask[ActionSlices.UNROLLING] == unrolling_part
    ).all(), "Unrolling mask is wrong"
    assert (mask[55] == next_action).all(), "Next action mask is wrong"
