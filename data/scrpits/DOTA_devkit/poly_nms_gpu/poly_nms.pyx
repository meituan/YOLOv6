import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "poly_nms.hpp":
    void _poly_nms(np.int32_t*, int*, np.float32_t*, int, int, float, int)

def poly_gpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh,
            np.int32_t device_id=0):
    cdef int boxes_num = dets.shape[0]
    cdef int boxes_dim = dets.shape[1]
    cdef int num_out
    cdef np.ndarray[np.int32_t, ndim=1] \
        keep = np.zeros(boxes_num, dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] \
        scores = dets[:, 8]
    cdef np.ndarray[np.int_t, ndim=1] \
        order = scores.argsort()[::-1]
    cdef np.ndarray[np.float32_t, ndim=2] \
        sorted_dets = dets[order, :]
    _poly_nms(&keep[0], &num_out, &sorted_dets[0, 0], boxes_num, boxes_dim, thresh, device_id)
    keep = keep[:num_out]
    return list(order[keep])
