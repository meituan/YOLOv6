/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"
#include "trt_utils.h"

static const int NUM_CLASSES_YOLO = 80;
static const float NMS_THRESHOLD = 0.65;
static const float CLS_THRESHOLD = 0.5;

extern "C" bool NvDsInferParseCustomYoloV6(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

/* This is a sample bounding box parsing function for the sample YoloV3 detector model */
static NvDsInferParseObjectInfo convertBBox(const float& bx, const float& by, const float& bw,
                                     const float& bh, const uint& netW, const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution
    float xCenter = bx ;
    float yCenter = by ;
    float x0 = xCenter - bw / 2;
    float y0 = yCenter - bh / 2;
    float x1 = x0 + bw;
    float y1 = y0 + bh;

    x0 = clamp(x0, 0, netW);
    y0 = clamp(y0, 0, netH);
    x1 = clamp(x1, 0, netW);
    y1 = clamp(y1, 0, netH);

    b.left = x0;
    b.width = clamp(x1 - x0, 0, netW);
    b.top = y0;
    b.height = clamp(y1 - y0, 0, netH);

    return b;
}

static void addBBoxProposal(const float bx, const float by, const float bw, const float bh,
                     const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBox(bx, by, bw, bh, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

/*
static inline std::vector<const NvDsInferLayerInfo*>
SortLayers(const std::vector<NvDsInferLayerInfo> & outputLayersInfo)
{
    std::vector<const NvDsInferLayerInfo*> outLayers;
    for (auto const &layer : outputLayersInfo) 
    {
        outLayers.push_back (&layer);
    }
    std::sort(outLayers.begin(), outLayers.end(),
        [](const NvDsInferLayerInfo* a, const NvDsInferLayerInfo* b) {
            return a->inferDims.d[1] < b->inferDims.d[1];
        });
    return outLayers;
}
*/
/*
static std::vector<NvDsInferParseObjectInfo>
decodeYoloV6Tensor(
    const float* detections, 
    const uint gridSizeW, const uint gridSizeH, const uint stride, const uint numBBoxes,
    const uint numOutputClasses, const uint& netW,
    const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;
    for (uint y = 0; y < gridSizeH; ++y) {
        for (uint x = 0; x < gridSizeW; ++x) {
            for (uint b = 0; b < numBBoxes; ++b)
            {
                const int numGridCells = gridSizeH * gridSizeW;
                const int bbindex = y * gridSizeW + x;
                const float bx
                    = x + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 0)];
                const float by
                    = y + detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 1)];
                const float bw
                    = stride * exp(detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 2)]);
                const float bh
                    = stride * exp(detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 3)]);

                const float objectness
                    = detections[bbindex + numGridCells * (b * (5 + numOutputClasses) + 4)];

                float maxProb = 0.0f;
                int maxIndex = -1;

                for (uint i = 0; i < numOutputClasses; ++i)
                {
                    float prob
                        = (detections[bbindex
                                      + numGridCells * (b * (5 + numOutputClasses) + (5 + i))]);

                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                maxProb = objectness * maxProb;

                addBBoxProposal(bx, by, bw, bh, stride, netW, netH, maxIndex, maxProb, binfo);
            }
        }
    }
    return binfo;
}
*/

static std::vector<NvDsInferParseObjectInfo>
nonMaximumSuppression(const float nmsThresh, std::vector<NvDsInferParseObjectInfo> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };

    auto computeIoU
        = [&overlap1D](NvDsInferParseObjectInfo& bbox1, NvDsInferParseObjectInfo& bbox2) -> float {
        float overlapX
            = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
        float overlapY
            = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
        float area1 = (bbox1.width) * (bbox1.height);
        float area2 = (bbox2.width) * (bbox2.height);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
                     [](const NvDsInferParseObjectInfo& b1, const NvDsInferParseObjectInfo& b2) {
                         return b1.detectionConfidence > b2.detectionConfidence;
                     });

    std::vector<NvDsInferParseObjectInfo> out;
    for (auto i : binfo)
    {
        bool keep = true;
        for (auto j : out)
        {
            if (keep)
            {
                float overlap = computeIoU(i, j);
                keep = overlap <= nmsThresh;
            }
            else
                break;
        }
        if (keep) out.push_back(i);
    }
    return out;
}


static std::vector<NvDsInferParseObjectInfo>
parseYoloV6BBox(const NvDsInferLayerInfo& feat, const uint numOutputClasses, const uint& netW,
    const uint& netH)
{
    std::vector<std::vector<NvDsInferParseObjectInfo>> binfo;   
    binfo.resize(NUM_CLASSES_YOLO);

    const float* detections = (const float*)feat.buffer;
    auto numBBoxes = feat.inferDims.d[0];
    const int numBBoxCells = feat.inferDims.d[1];

    for (uint b = 0; b < numBBoxes; ++b)
    {
        const float bx
                    = detections[b * numBBoxCells + 0];
        const float by
            = detections[b * numBBoxCells + 1];
        const float bw
            = detections[b * numBBoxCells + 2];
        const float bh
            = detections[b * numBBoxCells + 3];

        const float objectness
            = detections[b * numBBoxCells + 4];

        float maxProb = 0.0f;
        int maxIndex = -1;

        for (uint i = 0; i < numOutputClasses; ++i)
        {
            float prob
                = (detections[b * numBBoxCells + (5 + i)]);

            if (prob > maxProb)
            {
                maxProb = prob;
                maxIndex = i;
            }
        }
        maxProb = objectness * maxProb;
        if(maxProb > CLS_THRESHOLD)
        {
            // std::vector<NvDsInferParseObjectInfo> bboxInfo;
            addBBoxProposal(bx, by, bw, bh, netW, netH, maxIndex, maxProb, binfo[maxIndex]);
            // binfo[maxIndex].push_back(bboxInfo);
        }
    }
    // NMS
    std::vector<NvDsInferParseObjectInfo> objects = {};
    for(int cls_id = 0; cls_id < NUM_CLASSES_YOLO; ++cls_id)
    {
        std::vector<NvDsInferParseObjectInfo> outObjs = nonMaximumSuppression(NMS_THRESHOLD, binfo[cls_id]);
        objects.insert(objects.end(), outObjs.begin(), outObjs.end());
    }
    
    return objects;
}

static bool NvDsInferParseYoloV6(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // const uint kNUM_BBOXES = 1;
    // const uint kLAYER_NUM = 1;

    // const std::vector<const NvDsInferLayerInfo*> sortedLayers =
    //     SortLayers (outputLayersInfo);

    // if (sortedLayers.size() != kLAYER_NUM) {
    //     std::cerr << "ERROR: yoloV6 output layer.size: " << sortedLayers.size()
    //               << " does not match mask.size: " << kLAYER_NUM << std::endl;
    //     return false;
    // }

    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    

    // for (uint idx = 0; idx < sortedLayers.size(); ++idx) 
    // {
    // dimensions: batch, 8400, 85
    // bbox order: center_x, center_y, width, height
    const NvDsInferLayerInfo &layer = outputLayersInfo[0]; 

    assert(layer.inferDims.numDims == 2);



    /*
        const uint gridSizeH = layer.inferDims.d[1];
        const uint gridSizeW = layer.inferDims.d[2];
        const uint stride = DIVUP(networkInfo.width, gridSizeW);
        assert(stride == DIVUP(networkInfo.height, gridSizeH));

        std::vector<NvDsInferParseObjectInfo> outObjs =
            decodeYoloV6Tensor((const float*)(layer.buffer), gridSizeW, gridSizeH, stride, kNUM_BBOXES,
                        NUM_CLASSES_YOLO, networkInfo.width, networkInfo.height);
    */
    // objects.insert(objects.end(), outObjs.begin(), outObjs.end());
    // }

    std::vector<NvDsInferParseObjectInfo> objects = parseYoloV6BBox(
        layer, NUM_CLASSES_YOLO, networkInfo.width, networkInfo.height );

    objectList = objects;           // 赋值运算符被调用

    return true;
}

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomYoloV6(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseYoloV6 (
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV6);