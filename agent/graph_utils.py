import numpy as np
import re


def parse_isl_map(isl_map: str):
    # Extract the computation and buffer parts
    match = re.match(r".*{\s*(\w+)\[([^\]]+)\]\s*->\s*(\w+)\[([^\]]+)\]\s*}", isl_map)
    if not match:
        raise ValueError("Invalid ISL map format")

    _ = match.group(1)
    comp_iterators = match.group(2).split(",")
    _ = match.group(3)
    buffer_accesses = match.group(4).split(",")

    # Strip any spaces around iterators or accesses
    comp_iterators = [it.strip() for it in comp_iterators]
    buffer_accesses = [access.strip() for access in buffer_accesses]

    return comp_iterators, buffer_accesses


def extract_affine_coefficients(access, comp_iterators):
    """
    Extract coefficients of iterators in the affine expression.
    Example: access = 'i1 + 2*j2', comp_iterators = ['i1', 'j2', 'k']
    Output: [1, 2, 0, 0] (coefficients for i1, j2, k, and scalar term)
    """
    # remove all spaces
    access = access.replace(" ", "")

    # Initialize coefficients (including one for the scalar term)
    coefficients = [0] * (len(comp_iterators) + 1)

    # Match terms like '2*i1' or 'i1' or '-i1', and extract the coefficients
    for i, it in enumerate(comp_iterators):
        # Modify the pattern to match iterator names with numbers or underscores
        term_pattern = re.compile(r"([+-]?\d*)\s*\*?\s*(" + re.escape(it) + r")(\b|$)")
        match = term_pattern.search(access)
        if match:
            coeff = match.group(1)
            coefficients[i] = (
                int(coeff)
                if coeff and coeff != "+" and coeff != "-"
                else 1
                if coeff == "" or coeff == "+"
                else -1
            )

    # Handle the scalar part, which is just a constant not attached to any iterator
    scalar_pattern = re.compile(
        r"([+-]?\b\d+\b)(?![\*\w])"
    )  # Match isolated constants (scalars) not tied to iterators
    scalar_match = scalar_pattern.search(access)

    if scalar_match:
        coefficients[-1] = int(scalar_match.group(1))

    return coefficients


def isl_map_to_write_access_matrix(isl_map: str):
    # Parse the ISL map
    comp_iterators, buffer_accesses = parse_isl_map(isl_map)

    # Initialize the access matrix (rows = buffer dimensions, columns = iterators + scalar)
    access_matrix = []

    # Process each buffer access and extract affine coefficients
    for access in buffer_accesses:
        row = extract_affine_coefficients(access, comp_iterators)
        access_matrix.append(row)

    return np.array(access_matrix).tolist()


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
