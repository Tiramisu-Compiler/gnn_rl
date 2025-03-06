import pytest

from agent.tiramisu_interface import ActionSlices


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
