# cython: infer_types=True

import numpy as np
cimport cython

ctypedef fused int_type:
    int
    long

ctypedef fused float_type:
    float
    double

cdef double _m_e = 0.5109989461


cdef long _find_modules_check(
    double x,
    double y,
    double module_x,
    double module_y,
    double module_size_x,
    double module_size_y,
):
    if ((abs(x - module_x) < module_size_x / 2)
            and (abs(y - module_y) < module_size_y / 2)):
        return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def _find_modules(
    float_type[:] x,
    float_type[:] y,
    float[:] module_x,
    float[:] module_y,
    float[:] module_size_x,
    float[:] module_size_y,
):
    cdef Py_ssize_t data_length = x.shape[0]
    cdef Py_ssize_t module_length = module_x.shape[0]

    result = np.full(data_length, -1, dtype=np.int64)
    cdef long[:] result_view = result

    cdef Py_ssize_t i, j
    cdef long is_inside

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
    float_type[:] x,
    float_type[:] y,
    int_type[:] id_,
    int_type[:] type_,
    float[:, :, :, :] histogram,
    float[:, :, :] edge_x,
    float[:, :, :] edge_y,
):
    cdef Py_ssize_t data_length = x.shape[0]
    cdef Py_ssize_t bins = edge_x.shape[2]

    result = np.full(data_length, -1, dtype=np.double)
    cdef double[:] result_view = result

    cdef Py_ssize_t i, j
    cdef long x_bin, y_bin

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
    float_type[:] x,
    float_type[:] y,
    float[:] module_x,
    float[:] module_y,
    float[:] cut_size,
):
    cdef Py_ssize_t data_length = x.shape[0]
    cdef Py_ssize_t module_length = module_x.shape[0]

    result = np.ones(data_length, dtype=np.int64)
    cdef long[:] result_view = result

    cdef Py_ssize_t i, j
    cdef long temp
    cdef double distance2

    for i in range(data_length):
        temp = 1
        for j in range(module_length):
            distance2 = (x[i] - module_x[j])**2 + (y[i] - module_y[j])**2
            if distance2 < cut_size[j]**2:
                temp = 0
                break
        result_view[i] = temp

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def _get_double_arm_cut(
    int_type[:] event_number,
    float_type[:] e_beam,
    float_type[:] e,
    float_type[:] e_cut_min,
    float_type[:] e_cut_max,
    float_type[:] phi,
    float_type[:] r,
    float coplanerity_cut,
    float hycal_z,
    float vertex_z_cut,
):
    cdef Py_ssize_t data_length = event_number.shape[0]

    result = np.zeros(data_length, dtype=np.int64)
    cdef long[:] result_view = result

    cdef Py_ssize_t i, j
    cdef long found_i, found_j
    cdef double elasticity, elasticity_cut_min, elasticity_cut_max
    cdef double d_phi
    cdef double vertex_z
    cdef double elasticity_save

    elasticity_save = 1e+6
    for i in range(data_length):
        found_i, found_j = -1, -1
        for j in range(i + 1, data_length):
            if event_number[i] != event_number[j]:
                elasticity_save = 1e+6
                break

            elasticity = e_beam[i] - (e[i] + e[j])
            elasticity_cut_min = -np.sqrt(e_cut_min[i]**2 + e_cut_min[j]**2)
            elasticity_cut_max = np.sqrt(e_cut_max[i]**2 + e_cut_max[j]**2)
            if (elasticity < elasticity_cut_min
                    or elasticity > elasticity_cut_max):
                continue

            d_phi = phi[i] + np.pi - phi[j]
            if d_phi > np.pi:
                d_phi -= 2 * np.pi
            if abs(d_phi) > coplanerity_cut:
                continue

            vertex_z = np.sqrt(
                (_m_e * r[i] * r[j] + e_beam[i] * r[i] * r[j]) / 2 / _m_e)
            if abs(vertex_z - hycal_z) > vertex_z_cut:
                continue

            if abs(elasticity / elasticity_cut_max) < elasticity_save:
                found_i, found_j = i, j
                elasticity_save = abs(elasticity)

        if found_i != -1 and found_j != -1:
            result_view[found_i] = 1
            result_view[found_j] = 1

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def _get_gem_efficiency(
    float_type[:] theta,
    float[:] eff,
    float[:] edge,
):
    cdef Py_ssize_t data_length = theta.shape[0]
    cdef Py_ssize_t edge_length = edge.shape[0]

    result = np.ones(data_length, dtype=np.double)
    cdef double[:] result_view = result

    cdef Py_ssize_t i, j
    cdef double temp

    for i in range(data_length):
        for j in range(1, edge_length):
            if theta[i] < edge[j]:
                result_view[i] = eff[j - 1]
                break

    return result


cdef long _is_inside_gem_spacers_check(double x, double y):
    if (abs(x - 161.55) < 1.5 or abs(x - 344.45) < 1.5 or abs(y + 409.3) < 1.5
            or abs(y + 204) < 1.5 or abs (y) < 1.5 or abs(y - 204) < 1.5
            or abs(y - 409.3) < 1.5):
        return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def _is_inside_gem_spacers(
    float_type[:] gem_x,
    float_type[:] gem_y,
):
    cdef Py_ssize_t data_length = gem_x.shape[0]

    result = np.zeros(data_length, dtype=np.int64)
    cdef long[:] result_view = result

    cdef Py_ssize_t i
    cdef double x, y

    for i in range(data_length):
        if gem_x[i] > 0:
            x = 0.918772774 * gem_x[i]
            y = 0.918772774 * gem_y[i]

            result_view[i] = _is_inside_gem_spacers_check(x, y)
        else:
            x = 0.925798253 * gem_x[i]
            y = 0.925798253 * gem_y[i]

            result_view[i] = _is_inside_gem_spacers_check(-x, y)

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def _match_hits(
    int_type[:] hycal_event_number,
    float_type[:] hycal_x,
    float_type[:] hycal_y,
    int_type[:] gem_event_number,
    float_type[:] gem_x,
    float_type[:] gem_y,
    float_type[:] cut,
):
    cdef Py_ssize_t hycal_data_length = hycal_event_number.shape[0]
    cdef Py_ssize_t gem_data_length = gem_event_number.shape[0]

    result = np.zeros(hycal_data_length, dtype=np.int64)
    used = np.zeros(gem_data_length, dtype=np.int64)
    cdef long[:] result_view = result
    cdef long[:] used_view = used

    cdef Py_ssize_t i, j
    cdef Py_ssize_t j_start, j_start_new, i_prev
    cdef double distance2, min_distance2
    cdef long found

    j_start, j_start_new, i_prev = 0, 0, 0
    for i in range(hycal_data_length):
        if hycal_event_number[i] != hycal_event_number[i_prev]:
            j_start = j_start_new
            i_prev = i

        min_distance2 = 1e12
        found = -1
        for j in range(j_start, gem_data_length):
            if hycal_event_number[i] < gem_event_number[j]:
                j_start_new = j
                break

            if (hycal_event_number[i] > gem_event_number[j]
                    or used_view[j] == 1):
                continue

            distance2 = (hycal_x[i] - gem_x[j])**2 + (hycal_y[i] - gem_y[j])**2
            if distance2 < cut[i]**2 and distance2 < min_distance2:
                min_distance2 = distance2
                found = j

        result_view[i] = found
        if found >= 0:
            used_view[found] = 1

    return result
