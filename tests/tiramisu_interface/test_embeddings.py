from unittest.mock import patch
import numpy as np
import pytest

from agent.graph_utils import (
    encode_data_type,
    isl_map_to_write_access_matrix,
    pad_access_matrix,
)
from agent.tiramisu_interface import (
    BUFFER_ACCESS_EMBEDDING_START,
    MAX_ITERATOR_DEPTH,
    VECTOR_SIZE,
)


# TYPE_TAG = 0
# FOCUS_TAG = -11
# LOWER_BOUND_IS_INT_TAG = -10
# LOWER_BOUND_VALUE_TAG = -9
# UPPER_BOUND_IS_INT_TAG = -8
# UPPER_BOUND_VALUE_TAG = -7
# PARALLELIZATION_TAG = -6
# REVERSAL_TAG = -5
# UNROLLING_FACTOR_TAG = -4
# TILE_SIZE_TAG = -3
# SKEWING_FACTOR_1_TAG = -2
# SKEWING_FACTOR_2_TAG = -1


def test_tree_to_iterator_vectors(ti_cvt):
    iterator_vectors = ti_cvt._tree_to_iterator_vectors()
    assert len(iterator_vectors) == 3

    i00 = -np.ones(VECTOR_SIZE)
    # Type of embedding 0 for vector 1 for computation
    i00[0] = 0
    # 11 valid tags all at the end of the embedding
    i00[-11:] = [0, 1, 0, 1, 322, 0, 0, 0, 0, 0, 0]
    assert np.array_equal(iterator_vectors[("comp02", 0)], i00)

    i01 = -np.ones(VECTOR_SIZE)
    i01[0] = 0
    i01[-11:] = [0, 1, 0, 1, 130, 0, 0, 0, 0, 0, 0]
    assert np.array_equal(iterator_vectors[("comp02", 1)], i01)

    i02 = -np.ones(VECTOR_SIZE)
    i02[0] = 0
    i02[-11:] = [0, 1, 0, 1, 5, 0, 0, 0, 0, 0, 0]
    assert np.array_equal(iterator_vectors[("comp02", 2)], i02)


@patch("agent.tiramisu_interface.median_execution_time", return_value=1.5)
def test_get_comp_annotations(_, ti_cvt):
    comp02 = ti_cvt._get_comp_annotations("comp02")
    assert comp02.get("absolute_order") == 1
    assert comp02.get("comp_is_reduction")
    ti_cvt.apply_action(51)
    updated_comp02 = ti_cvt._get_comp_annotations("_comp02_update_0")
    assert updated_comp02 == comp02

    # inexistent comp
    with pytest.raises(ValueError):
        ti_cvt._get_comp_annotations("comp03")


def test_encode_data_type():
    assert np.array_equal([1, 0, 0], encode_data_type("int32"))
    assert np.array_equal([0, 1, 0], encode_data_type("float32"))
    assert np.array_equal([0, 0, 1], encode_data_type("float64"))
    assert encode_data_type("whatever") is None


@pytest.mark.parametrize(
    "access_matrix, padded_access_matrix, max_depth",
    [
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[1, 2, -1, -1], [3, 4, -1, -1], [-1, -1, -1, -1]]),
            2,
        ),
        (
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            np.array(
                [
                    [1, 2, 3, -1, -1],
                    [4, 5, 6, -1, -1],
                    [7, 8, 9, -1, -1],
                    [-1, -1, -1, -1, -1],
                ]
            ),
            3,
        ),
        (
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                ]
            ),
            2,
        ),
    ],
)
def test_pad_access_matrix(access_matrix, padded_access_matrix, max_depth):
    access_matrix = np.array(access_matrix)
    padded_access_matrix = np.array(padded_access_matrix)
    assert np.array_equal(
        pad_access_matrix(access_matrix, max_depth), padded_access_matrix
    )


def test_isl_map_to_write_access_matrix():
    isl_map = "{ comp02[i00, i01, i02] -> buf02[i00, i01] }"
    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    assert np.array_equal(isl_map_to_write_access_matrix(isl_map), matrix)

    isl_map = "{ comp02[i00, i01, i02] -> buf02[i00, i01, i02] }"
    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    assert np.array_equal(isl_map_to_write_access_matrix(isl_map), matrix)

    isl_map = "{ comp02[i00] -> buf02[i00, i01, i02, i03] }"
    matrix = np.array([[1, 0], [0, 0], [0, 0], [0, 0]])
    assert np.array_equal(isl_map_to_write_access_matrix(isl_map), matrix)

    isl_map = "{ comp02[i00] -> buf02[i00 + 2, i01, i02, i03] }"
    matrix = np.array([[1, 2], [0, 0], [0, 0], [0, 0]])
    assert np.array_equal(isl_map_to_write_access_matrix(isl_map), matrix)

    isl_map = "{ comp02[i00, i01] -> buf02[3*i00, i01 + 2, i02, i03] }"
    matrix = np.array([[3, 0, 0], [0, 1, 2], [0, 0, 0], [0, 0, 0]])
    assert np.array_equal(isl_map_to_write_access_matrix(isl_map), matrix)

    with pytest.raises(ValueError):
        isl_map_to_write_access_matrix("whatever")


def test_annotations_to_comps_vectors(ti_cvt):
    comp_vectors = ti_cvt._annotations_to_comps_vectors()
    assert len(comp_vectors) == 1

    comp02 = -np.ones(VECTOR_SIZE)
    comp02[0] = 1
    comp02[1] = 1
    comp02[2:5] = [1, 0, 0]
    comp02[5] = 1
    comp02[6] = 2
    write_padded_matrix = pad_access_matrix(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]]), MAX_ITERATOR_DEPTH
    ).reshape(-1)
    comp02[7:BUFFER_ACCESS_EMBEDDING_START] = write_padded_matrix
    buf_2 = np.concatenate(
        [
            pad_access_matrix(
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]), MAX_ITERATOR_DEPTH
            ).reshape(-1),
            [1],
            [2 + 1],
        ]
    )
    buf_0 = np.concatenate(
        [
            pad_access_matrix(
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]), MAX_ITERATOR_DEPTH
            ).reshape(-1),
            [0],
            [0 + 1],
        ]
    )
    buf_1 = np.concatenate(
        [
            pad_access_matrix(np.array([[0, 0, 1, 0]]), MAX_ITERATOR_DEPTH).reshape(-1),
            [0],
            [1 + 1],
        ]
    )
    access_embedding = np.concatenate([buf_2, buf_0, buf_1])
    comp02[
        BUFFER_ACCESS_EMBEDDING_START : BUFFER_ACCESS_EMBEDDING_START
        + access_embedding.shape[0]
    ] = access_embedding

    assert np.array_equal(comp_vectors["comp02"], comp02)
