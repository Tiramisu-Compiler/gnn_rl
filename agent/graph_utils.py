import numpy as np
import re


def isl_to_write_matrix(isl_map):
    comp_iterators_str = re.findall(r"\[(.*)\]\s*->", isl_map)[0]
    buffer_iterators_str = re.findall(r"->\s*\w*\[(.*)\]", isl_map)[0]
    buffer_iterators_str = re.sub(r"\w+'\s=", "", buffer_iterators_str)
    comp_iter_names = re.findall(r"(?:\s*(\w+))+", comp_iterators_str)
    buf_iter_names = re.findall(r"(?:\s*(\w+))+", buffer_iterators_str)
    matrix = np.zeros([len(buf_iter_names), len(comp_iter_names) + 1])
    for i, buf_iter in enumerate(buf_iter_names):
        for j, comp_iter in enumerate(comp_iter_names):
            if buf_iter == comp_iter:
                matrix[i, j] = 1
                break
    return matrix


def pad_access_matrix(access_matrix, max_depth):
    padded_access_matrix = np.ones((max_depth + 1, max_depth + 2)) * -1
    padded_access_matrix[: access_matrix.shape[0], : access_matrix.shape[1]] = (
        access_matrix
    )
    return padded_access_matrix


def encode_data_type(data_type):
    if data_type == "int32":
        return [1, 0, 0]
    elif data_type == "float32":
        return [0, 1, 0]
    elif data_type == "float64":
        return [0, 0, 1]
