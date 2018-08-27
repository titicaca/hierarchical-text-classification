"""
Util functions for HTC
"""
import numpy as np
from scipy.sparse import issparse, csr_matrix
from collections import defaultdict
import pickle
from queue import Queue
from hierarchical_text_classification.htc.constant import ROOT
import logging


def get_level_label_index(graph, labels, level, root=ROOT):
    """
    Get labels for a given level of the graph
    :param graph: graph instance
    :param labels: array of labels
    :param level: int level of the labels in the graph
    :param root: Graph root node
    :return: array of label index for the given level in the graph
    """
    if level < 1:
        raise ValueError("Param level should be an integer greater than 0")
    i = 0
    q_cur = Queue()
    q_cur.put(root)
    while level > i:
        i += 1
        logging.debug("Scanning level %s of the tree" % str(i))
        q_prev = q_cur
        q_cur = Queue()
        while not q_prev.empty():
            n = q_prev.get()
            for child in graph.successors(n):
                q_cur.put(child)
        logging.debug("Level %s of the tree: %s" % (str(i), list(q_cur.queue)))
    level_labels = list(q_cur.queue)
    res = []
    for label in level_labels:
        idx = find_array_index(labels, label)
        if idx == -1:
            raise ValueError("Cannot find label %s index" %label)
        res.append(idx)
    return res


def make_flat_hierarchy(targets, root):
    """
    Create a trivial (flat) hiearchy, linking all given targets to given root node.

    """
    adjacency = defaultdict(list)
    for target in targets:
        adjacency[root].append(target)
    return adjacency


def is_estimator(obj):
    if hasattr(obj.__class__, "fit"):
        return True


def apply_along_rows(func, X):
    """
    Apply function row-wise to input matrix X.
    This will work for dense matrices (eg np.ndarray)
    as well as for CSR sparse matrices.

    """
    if issparse(X):
        return np.array([
            func(X.getrow(i))
            for i in range(X.shape[0])
        ])
    else:
        # XXX might break vis-a-vis this issue merging: https://github.com/numpy/numpy/pull/8511
        # See discussion over issue with truncated string when using np.apply_along_axis here:
        #   https://github.com/numpy/numpy/issues/8352
        return np.array([
            func(X[i:i+1, :])
            for i in range(X.shape[0])
        ])


def convert_label_matrix_to_vector(matrix, labels):
    res = []
    for arr in matrix:
        if sum(arr) != 1:
            raise ValueError("Multi-labels are detected, only single label output can be converted to vector")
        for i in range(len(arr)):
            if arr[i] == 1:
                res.append(labels[i])
                break
    if len(res) != len(matrix):
        raise ValueError("Internal error: inconsistent length of converted result and matrix")
    return np.array(res)


def convert_label_vector_to_csr_matrix(vector, labels, sep=None):
    data = []
    row_idx = []
    col_idx = []
    for i in range(len(vector)):
        if sep is not None:
            label_arr = vector[i].split(sep)
        else:
            label_arr = [vector[i]]
        for label in label_arr:
            icol = find_array_index(labels, label)
            if icol == -1:
                logging.warning("cannot find index for label: " + label)
                continue
            row_idx.append(i)
            col_idx.append(icol)
            data.append(1)
    return csr_matrix((data, (row_idx, col_idx)), shape=(len(vector), len(labels)), dtype=int)


def find_array_index(array, element):
    for i in range(len(array)):
        if array[i] == element:
            return i
    return -1


def extract_rows_csr(matrix, rows):
    """
    Parameters
    ----------
    matrix : (sparse) csr_matrix

    rows : list of row ids

    Returns
    -------
    matrix_: (sparse) csr_matrix
        Transformed by extracting the desired rows from `matrix`

    """
    if not isinstance(matrix, csr_matrix):
        matrix = csr_matrix(matrix)

    # Short circuit if we want a blank matrix
    if len(rows) == 0:
        return csr_matrix(matrix.shape)

    # Keep a record of the desired rows
    indptr = np.zeros(matrix.indptr.shape, dtype=np.int32)
    indices = []
    data = []

    # Keep track of the current index pointer
    indices_count = 0

    for i in range(matrix.shape[0]):
        indptr[i] = indices_count

        if i in rows:
            indices.append(matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]])
            data.append(matrix.data[matrix.indptr[i]:matrix.indptr[i+1]])
            indices_count += len(matrix.data[matrix.indptr[i]:matrix.indptr[i+1]])

    indptr[-1] = indices_count

    indices = np.concatenate(indices)
    data = np.concatenate(data)

    return csr_matrix((data, indices, indptr), shape=matrix.shape)


def nnz_rows_ix(X):
    """Return row indices which have at least one non-zero column value."""
    return np.unique(X.nonzero()[0])


def get_nonzero_rows(X):
    return X[nnz_rows_ix(X), :]


def build_binary_training_data(pos_features, neg_features):
    targets = np.zeros(pos_features.shape[0])
    pos_idx = nnz_rows_ix(pos_features)
    targets[pos_idx] = 1
    neg_idx = nnz_rows_ix(neg_features)
    targets[neg_idx] = -1

    features = pos_features + neg_features
    features = get_nonzero_rows(features)
    targets = targets[nnz_rows_ix(targets)]
    return features, targets


def save_model(model_object, model_path, model_name=None):
    # TODO fix for Pickle TypeError: can't pickle _thread.RLock objects
    if model_path[-1] != '/':
        model_path = model_path + '/'
    if model_name is None:
        model_name = "hierarchical_classifier.pkl"
    file_path = model_path + model_name
    with open(file_path, 'wb') as fid:
        pickle.dump(model_object, fid)


def load_model(model_path, model_name=None):
    if model_path[-1] != '/':
        model_path = model_path + '/'
    if model_name is None:
        model_name = "hierarchical_classifier.pkl"
    file_path = model_path + model_name
    with open(file_path, 'rb') as fid:
        return pickle.load(fid)
