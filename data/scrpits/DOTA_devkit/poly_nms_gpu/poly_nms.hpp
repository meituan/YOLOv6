//
// Created by dingjian on 18-5-24.
//

#ifndef DOTA_DEVKIT_POLY_NMS_HPP
#define DOTA_DEVKIT_POLY_NMS_HPP


void _poly_nms(int* keep_out, int* num_out, const float* polys_host, int polys_num,
            int polys_dim, float nms_overlap_thresh, int device_id);

#endif //DOTA_DEVKIT_POLY_NMS_HPP
