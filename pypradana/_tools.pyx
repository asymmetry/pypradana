# cython: infer_types=True

import numpy as np
cimport cython

DTYPE = np.float


cdef int _find_modules_check(
    float x,
    float y,
    float module_x,
    float module_y,
    float module_size_x,
    float module_size_y,
):
    if ((abs(x - module_x) < module_size_x / 2)
            and (abs(y - module_y) < module_size_y / 2)):
        return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def _find_modules(
    float[:] x,
    float[:] y,
    float[:] module_x,
    float[:] module_y,
    float[:] module_size_x,
    float[:] module_size_y,
):
    cdef Py_ssize_t data_length = x.shape[0]
    cdef Py_ssize_t module_length = module_x.shape[0]

    result = np.full(data_length, -1, dtype=np.intc)
    cdef int[:] result_view = result

    cdef int is_inside
    cdef Py_ssize_t i, j

    for i in range(data_length):
        for j in range(module_length):
            is_inside = _find_modules_check(
                x[i],
                y[i],
                module_x[j],
                module_y[j],
                module_size_x[j],
                module_size_y[j],
            )
            if is_inside == 1:
                result_view[i] = j
                break

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def _get_module_e_correction(
    float[:] x,
    float[:] y,
    int[:] id_,
    int[:] type_,
    float[:, :, :, :] histogram,
    float[:, :, :] edge_x,
    float[:, :, :] edge_y,
):
    cdef Py_ssize_t data_length = x.shape[0]
    cdef Py_ssize_t bins = edge_x.shape[2]

    result = np.full(data_length, -1, dtype=np.float32)
    cdef float[:] result_view = result

    cdef Py_ssize_t i, j
    cdef int x_bin, y_bin

    for i in range(data_length):
        x_bin, y_bin = 0, 0
        for j in range(1, bins):
            if x[i] < edge_x[type_[i]][id_[i]][j]:
                x_bin = j - 1
                break
        for j in range(1, bins):
            if y[i] < edge_y[type_[i]][id_[i]][j]:
                y_bin = j - 1
                break
        result_view[i] = histogram[type_[i]][id_[i]][x_bin][y_bin]

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def _not_at_dead_module(
    float[:] x,
    float[:] y,
    float[:] module_x,
    float[:] module_y,
    float[:] cut_size,
):
    cdef Py_ssize_t data_length = x.shape[0]
    cdef Py_ssize_t module_length = module_x.shape[0]

    result = np.ones(data_length, dtype=np.intc)
    cdef int[:] result_view = result

    cdef Py_ssize_t i, j
    cdef int temp

    for i in range(data_length):
        temp = 1
        for j in range(module_length):
            distance2 = (x[i] - module_x[j])**2 + (y[i] - module_y[j])**2
            if distance2 < cut_size[j]**2:
                temp = 0
                break
        result_view[i] = temp

    return result
