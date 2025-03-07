import pytest

from agent.tiramisu_interface import ActionSlices, program_compatible_with_model


@pytest.mark.parametrize(
    "index, params",
    [
        (14, (32, 32)),
        (18, (64, 64)),
        (22, (128, 128)),
        (26, (32, 64)),
        (30, (32, 128)),
        (34, (64, 32)),
        (38, (64, 128)),
        (42, (128, 32)),
        (46, (128, 64)),
        (50, "raise"),
    ],
)
def test_ActionSlices(index, params):
    if params == "raise":
        with pytest.raises(ValueError):
            ActionSlices.tiling_size(index)
    else:
        assert ActionSlices.tiling_size(index) == params


# def program_compatible_with_model(annotations):
#     max_accesses = 15
#     min_accesses = 0
#     max_iterators = MAX_ITERATOR_DEPTH
#     computations_dict = annotations["computations"]

#     # Making sure every computation doesn't exceed the limit of the cost model , if the model is updated change the conditions
#     for comp_name in computations_dict:
#         comp_dict = computations_dict[comp_name]
#         if (
#             len(comp_dict["accesses"]) > max_accesses
#             or len(comp_dict["accesses"]) < min_accesses
#         ):
#             return False
#         if len(comp_dict["iterators"]) > max_iterators:
#             return False

#     return True


@pytest.mark.parametrize(
    "annotations, expected",
    [
        (
            {
                "computations": {
                    "comp01": {"accesses": [1, 2, 3], "iterators": [1, 2, 3]},
                    "comp02": {"accesses": [1, 2, 3], "iterators": [1, 2, 3]},
                }
            },
            True,
        ),
        (
            {
                "computations": {
                    "comp01": {
                        "accesses": [
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15,
                            16,
                        ],
                        "iterators": [1, 2, 3],
                    },
                    "comp02": {"accesses": [1, 2, 3], "iterators": [1, 2, 3]},
                }
            },
            False,
        ),
        (
            {
                "computations": {
                    "comp01": {"accesses": [1, 2, 3], "iterators": [1, 2, 3]},
                    "comp02": {"accesses": [1, 2, 3], "iterators": [1, 2, 3, 4, 5, 6]},
                }
            },
            False,
        ),
    ],
)
def test_program_compatible_with_model(annotations, expected):
    assert program_compatible_with_model(annotations) == expected
