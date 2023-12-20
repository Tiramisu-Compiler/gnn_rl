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


def iterators_to_vectors(annotations):
    it_dict = {}
    iter_vector_size = 718
    size_of_comp_vector = 709
    for it in annotations["iterators"]:
        single_iter_vector = -np.ones(iter_vector_size)
        single_iter_vector[0] = 0
        single_iter_vector[-9:] = 0
        # lower value
        single_iter_vector[size_of_comp_vector + 1] = annotations["iterators"][it][
            "lower_bound"
        ]
        # upper value
        single_iter_vector[size_of_comp_vector + 2] = annotations["iterators"][it][
            "upper_bound"
        ]
        it_dict[it] = single_iter_vector

    return it_dict


def pad_access_matrix(access_matrix, max_depth):
    padded_access_matrix = np.ones((max_depth + 1, max_depth + 2)) * -1
    padded_access_matrix[
        : access_matrix.shape[0], : access_matrix.shape[1]
    ] = access_matrix
    return padded_access_matrix


def encode_data_type(data_type):
    if data_type == "int32":
        return [1, 0, 0]
    elif data_type == "float32":
        return [0, 1, 0]
    elif data_type == "float64":
        return [0, 0, 1]


def comps_to_vectors(annotations):
    comp_vector_size = 718
    max_depth = 5
    dict_comp = {}
    for comp in annotations["computations"]:
        single_comp_vector = -np.ones(comp_vector_size)
        # This means that this vector has data related to a computation and not an iterator
        single_comp_vector[0] = 1
        comp_dict = annotations["computations"][comp]
        # This field represents the absolute order of execution of computations
        single_comp_vector[1] = comp_dict["absolute_order"]
        # a vector of one-hot encoding of possible 3 data-types
        single_comp_vector[2:5] = encode_data_type(comp_dict["data_type"])
        single_comp_vector[5] = +comp_dict["comp_is_reduction"]
        # The write-to buffer id
        single_comp_vector[6] = +comp_dict["write_buffer_id"]
        # We add a vector of write access
        write_matrix = isl_to_write_matrix(comp_dict["write_access_relation"])
        padded_matrix = pad_access_matrix(write_matrix, max_depth).reshape(-1)
        single_comp_vector[7 : 7 + padded_matrix.shape[0]] = padded_matrix
        # We add vector of read access
        for index, read_access_dict in enumerate(comp_dict["accesses"]):
            read_access_matrix = pad_access_matrix(
                np.array(read_access_dict["access_matrix"]), max_depth
            ).reshape(-1)
            read_access_matrix = np.append(
                read_access_matrix, +read_access_dict["access_is_reduction"]
            )
            read_access_matrix = np.append(
                read_access_matrix, read_access_dict["buffer_id"] + 1
            )
            read_access_size = read_access_matrix.shape[0]
            single_comp_vector[
                49 + index * read_access_size : 49 + (index + 1) * read_access_size
            ] = read_access_matrix
        dict_comp[comp] = single_comp_vector
    return dict_comp


def build_graph(annotations):
    it_vector_dict = iterators_to_vectors(annotations)
    comp_vector_dict = comps_to_vectors(annotations)
    it_index = {}
    comp_index = {}
    num_iterators = len(annotations["iterators"])
    for i, it in enumerate(it_vector_dict):
        it_index[it] = i
    for i, comp in enumerate(comp_vector_dict):
        comp_index[comp] = i

    edge_index = []
    node_feats = None

    for it in annotations["iterators"]:
        for child_it in annotations["iterators"][it]["child_iterators"]:
            edge_index.append([it_index[it], it_index[child_it]])

        for child_comp in annotations["iterators"][it]["computations_list"]:
            edge_index.append([it_index[it], num_iterators + comp_index[child_comp]])
    node_feats = np.stack(
        [
            *[arr for arr in it_vector_dict.values()],
            *[arr for arr in comp_vector_dict.values()],
        ],
    )
    return node_feats, np.array(edge_index), it_index, comp_index
