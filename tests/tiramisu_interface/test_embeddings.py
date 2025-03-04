from unittest.mock import patch
import numpy as np
import pytest

from agent.graph_utils import (
    encode_data_type,
    isl_map_to_write_access_matrix,
    pad_access_matrix,
)
from agent.tiramisu_interface import VECTOR_SIZE


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


def test_pad_access_matrix():
    access_matrix = np.array([[1, 2], [3, 4]])
    padded_access_matrix = np.array([[1, 2, -1, -1], [3, 4, -1, -1], [-1, -1, -1, -1]])
    assert np.array_equal(pad_access_matrix(access_matrix, 2), padded_access_matrix)


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


# def test_annotations_to_comps_vectors(ti_cvt):
#     comp_vectors = ti_cvt._annotations_to_comps_vectors()
#     assert len(comp_vectors) == 1

#     comp02 = -np.ones(VECTOR_SIZE)
#     comp02[0] = 1
#     comp02[1] = 1
#     comp02[2:5] = [1, 0, 0]
#     comp02[5] = 1
#     comp02[6] = 2
#     write_matrix = isl_map_to_write_access_matrix(
#         ti_cvt._get_comp_annotations("comp02")["write_access_relation"]
#     )
#     padded_matrix = pad_access_matrix(write_matrix, MAX_ITERATOR_DEPTH).reshape(-1)
#     comp02[7 : 7 + MAX_ITERATOR_DEPTH]
