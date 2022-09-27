/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "trt_utils.h"

#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <functional>
#include <algorithm>
#include <math.h>

#include "NvInferPlugin.h"

static void leftTrim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

static void rightTrim(std::string& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

std::string trim(std::string s)
{
    leftTrim(s);
    rightTrim(s);
    return s;
}

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

bool fileExists(const std::string fileName, bool verbose)
{
    if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName)))
    {
        if (verbose) std::cout << "File does not exist : " << fileName << std::endl;
        return false;
    }
    return true;
}

std::vector<float> loadWeights(const std::string weightsFilePath, const std::string& networkType)
{
    assert(fileExists(weightsFilePath));
    std::cout << "Loading pre-trained weights..." << std::endl;
    std::ifstream file(weightsFilePath, std::ios_base::binary);
    assert(file.good());
    std::string line;

    if (networkType == "yolov2")
    {
        // Remove 4 int32 bytes of data from the stream belonging to the header
        file.ignore(4 * 4);
    }
    else if ((networkType == "yolov3") || (networkType == "yolov3-tiny")
             || (networkType == "yolov2-tiny"))
    {
        // Remove 5 int32 bytes of data from the stream belonging to the header
        file.ignore(4 * 5);
    }
    else
    {
        std::cout << "Invalid network type" << std::endl;
        assert(0);
    }

    std::vector<float> weights;
    char floatWeight[4];
    while (!file.eof())
    {
        file.read(floatWeight, 4);
        assert(file.gcount() == 4);
        weights.push_back(*reinterpret_cast<float*>(floatWeight));
        if (file.peek() == std::istream::traits_type::eof()) break;
    }
    std::cout << "Loading weights of " << networkType << " complete!"
              << std::endl;
    std::cout << "Total Number of weights read : " << weights.size() << std::endl;
    return weights;
}

std::string dimsToString(const nvinfer1::Dims d)
{
    std::stringstream s;
    assert(d.nbDims >= 1);
    for (int i = 0; i < d.nbDims - 1; ++i)
    {
        s << std::setw(4) << d.d[i] << " x";
    }
    s << std::setw(4) << d.d[d.nbDims - 1];

    return s.str();
}

int getNumChannels(nvinfer1::ITensor* t)
{
    nvinfer1::Dims d = t->getDimensions();
    assert(d.nbDims == 3);

    return d.d[0];
}

uint64_t get3DTensorVolume(nvinfer1::Dims inputDims)
{
    assert(inputDims.nbDims == 3);
    return inputDims.d[0] * inputDims.d[1] * inputDims.d[2];
}

nvinfer1::ILayer* netAddMaxpool(int layerIdx, std::map<std::string, std::string>& block,
                                nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "maxpool");
    assert(block.find("size") != block.end());
    assert(block.find("stride") != block.end());

    int size = std::stoi(block.at("size"));
    int stride = std::stoi(block.at("stride"));

    nvinfer1::IPoolingLayer* pool
        = network->addPooling(*input, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{size, size});
    assert(pool);
    std::string maxpoolLayerName = "maxpool_" + std::to_string(layerIdx);
    pool->setStride(nvinfer1::DimsHW{stride, stride});
    pool->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    pool->setName(maxpoolLayerName.c_str());

    return pool;
}

nvinfer1::ILayer* netAddConvLinear(int layerIdx, std::map<std::string, std::string>& block,
                                   std::vector<float>& weights,
                                   std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
                                   int& inputChannels, nvinfer1::ITensor* input,
                                   nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "convolutional");
    assert(block.find("batch_normalize") == block.end());
    assert(block.at("activation") == "linear");
    assert(block.find("filters") != block.end());
    assert(block.find("pad") != block.end());
    assert(block.find("size") != block.end());
    assert(block.find("stride") != block.end());

    int filters = std::stoi(block.at("filters"));
    int padding = std::stoi(block.at("pad"));
    int kernelSize = std::stoi(block.at("size"));
    int stride = std::stoi(block.at("stride"));
    int pad;
    if (padding)
        pad = (kernelSize - 1) / 2;
    else
        pad = 0;
    // load the convolution layer bias
    nvinfer1::Weights convBias{nvinfer1::DataType::kFLOAT, nullptr, filters};
    float* val = new float[filters];
    for (int i = 0; i < filters; ++i)
    {
        val[i] = weights[weightPtr];
        weightPtr++;
    }
    convBias.values = val;
    trtWeights.push_back(convBias);
    // load the convolutional layer weights
    int size = filters * inputChannels * kernelSize * kernelSize;
    nvinfer1::Weights convWt{nvinfer1::DataType::kFLOAT, nullptr, size};
    val = new float[size];
    for (int i = 0; i < size; ++i)
    {
        val[i] = weights[weightPtr];
        weightPtr++;
    }
    convWt.values = val;
    trtWeights.push_back(convWt);
    nvinfer1::IConvolutionLayer* conv = network->addConvolution(
        *input, filters, nvinfer1::DimsHW{kernelSize, kernelSize}, convWt, convBias);
    assert(conv != nullptr);
    std::string convLayerName = "conv_" + std::to_string(layerIdx);
    conv->setName(convLayerName.c_str());
    conv->setStride(nvinfer1::DimsHW{stride, stride});
    conv->setPadding(nvinfer1::DimsHW{pad, pad});

    return conv;
}

nvinfer1::ILayer* netAddConvBNLeaky(int layerIdx, std::map<std::string, std::string>& block,
                                    std::vector<float>& weights,
                                    std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
                                    int& inputChannels, nvinfer1::ITensor* input,
                                    nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "convolutional");
    assert(block.find("batch_normalize") != block.end());
    assert(block.at("batch_normalize") == "1");
    assert(block.at("activation") == "leaky");
    assert(block.find("filters") != block.end());
    assert(block.find("pad") != block.end());
    assert(block.find("size") != block.end());
    assert(block.find("stride") != block.end());

    bool batchNormalize, bias;
    if (block.find("batch_normalize") != block.end())
    {
        batchNormalize = (block.at("batch_normalize") == "1");
        bias = false;
    }
    else
    {
        batchNormalize = false;
        bias = true;
    }
    // all conv_bn_leaky layers assume bias is false
    assert(batchNormalize == true && bias == false);
    UNUSED(batchNormalize);
    UNUSED(bias);

    int filters = std::stoi(block.at("filters"));
    int padding = std::stoi(block.at("pad"));
    int kernelSize = std::stoi(block.at("size"));
    int stride = std::stoi(block.at("stride"));
    int pad;
    if (padding)
        pad = (kernelSize - 1) / 2;
    else
        pad = 0;

    /***** CONVOLUTION LAYER *****/
    /*****************************/
    // batch norm weights are before the conv layer
    // load BN biases (bn_biases)
    std::vector<float> bnBiases;
    for (int i = 0; i < filters; ++i)
    {
        bnBiases.push_back(weights[weightPtr]);
        weightPtr++;
    }
    // load BN weights
    std::vector<float> bnWeights;
    for (int i = 0; i < filters; ++i)
    {
        bnWeights.push_back(weights[weightPtr]);
        weightPtr++;
    }
    // load BN running_mean
    std::vector<float> bnRunningMean;
    for (int i = 0; i < filters; ++i)
    {
        bnRunningMean.push_back(weights[weightPtr]);
        weightPtr++;
    }
    // load BN running_var
    std::vector<float> bnRunningVar;
    for (int i = 0; i < filters; ++i)
    {
        // 1e-05 for numerical stability
        bnRunningVar.push_back(sqrt(weights[weightPtr] + 1.0e-5));
        weightPtr++;
    }
    // load Conv layer weights (GKCRS)
    int size = filters * inputChannels * kernelSize * kernelSize;
    nvinfer1::Weights convWt{nvinfer1::DataType::kFLOAT, nullptr, size};
    float* val = new float[size];
    for (int i = 0; i < size; ++i)
    {
        val[i] = weights[weightPtr];
        weightPtr++;
    }
    convWt.values = val;
    trtWeights.push_back(convWt);
    nvinfer1::Weights convBias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    trtWeights.push_back(convBias);
    nvinfer1::IConvolutionLayer* conv = network->addConvolution(
        *input, filters, nvinfer1::DimsHW{kernelSize, kernelSize}, convWt, convBias);
    assert(conv != nullptr);
    std::string convLayerName = "conv_" + std::to_string(layerIdx);
    conv->setName(convLayerName.c_str());
    conv->setStride(nvinfer1::DimsHW{stride, stride});
    conv->setPadding(nvinfer1::DimsHW{pad, pad});

    /***** BATCHNORM LAYER *****/
    /***************************/
    size = filters;
    // create the weights
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, nullptr, size};
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, nullptr, size};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, size};
    float* shiftWt = new float[size];
    for (int i = 0; i < size; ++i)
    {
        shiftWt[i]
            = bnBiases.at(i) - ((bnRunningMean.at(i) * bnWeights.at(i)) / bnRunningVar.at(i));
    }
    shift.values = shiftWt;
    float* scaleWt = new float[size];
    for (int i = 0; i < size; ++i)
    {
        scaleWt[i] = bnWeights.at(i) / bnRunningVar[i];
    }
    scale.values = scaleWt;
    float* powerWt = new float[size];
    for (int i = 0; i < size; ++i)
    {
        powerWt[i] = 1.0;
    }
    power.values = powerWt;
    trtWeights.push_back(shift);
    trtWeights.push_back(scale);
    trtWeights.push_back(power);
    // Add the batch norm layers
    nvinfer1::IScaleLayer* bn = network->addScale(
        *conv->getOutput(0), nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(bn != nullptr);
    std::string bnLayerName = "batch_norm_" + std::to_string(layerIdx);
    bn->setName(bnLayerName.c_str());
    /***** ACTIVATION LAYER *****/
    /****************************/
    nvinfer1::ITensor* bnOutput = bn->getOutput(0);
    nvinfer1::IActivationLayer* leaky = network->addActivation(
        *bnOutput, nvinfer1::ActivationType::kLEAKY_RELU);
    leaky->setAlpha(0.1);
    assert(leaky != nullptr);
    std::string leakyLayerName = "leaky_" + std::to_string(layerIdx);
    leaky->setName(leakyLayerName.c_str());

    return leaky;
}

nvinfer1::ILayer* netAddUpsample(int layerIdx, std::map<std::string, std::string>& block,
                                 std::vector<float>& weights,
                                 std::vector<nvinfer1::Weights>& trtWeights, int& inputChannels,
                                 nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "upsample");
    nvinfer1::Dims inpDims = input->getDimensions();
    assert(inpDims.nbDims == 3);
    assert(inpDims.d[1] == inpDims.d[2]);
    int h = inpDims.d[1];
    int w = inpDims.d[2];
    int stride = std::stoi(block.at("stride"));
    // add pre multiply matrix as a constant
    nvinfer1::Dims preDims{3,
                           {1, stride * h, w}};

    int size = stride * h * w;
    nvinfer1::Weights preMul{nvinfer1::DataType::kFLOAT, nullptr, size};
    float* preWt = new float[size];
    /* (2*h * w)
    [ [1, 0, ..., 0],
      [1, 0, ..., 0],
      [0, 1, ..., 0],
      [0, 1, ..., 0],
      ...,
      ...,
      [0, 0, ..., 1],
      [0, 0, ..., 1] ]
    */
    for (int i = 0, idx = 0; i < h; ++i)
    {
        for (int s = 0; s < stride; ++s)
        {
            for (int j = 0; j < w; ++j, ++idx)
            {
                preWt[idx] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
    preMul.values = preWt;
    trtWeights.push_back(preMul);
    nvinfer1::IConstantLayer* preM = network->addConstant(preDims, preMul);
    assert(preM != nullptr);
    std::string preLayerName = "preMul_" + std::to_string(layerIdx);
    preM->setName(preLayerName.c_str());
    // add post multiply matrix as a constant
    nvinfer1::Dims postDims{3,
                            {1, h, stride * w}};

    size = stride * h * w;
    nvinfer1::Weights postMul{nvinfer1::DataType::kFLOAT, nullptr, size};
    float* postWt = new float[size];
    /* (h * 2*w)
    [ [1, 1, 0, 0, ..., 0, 0],
      [0, 0, 1, 1, ..., 0, 0],
      ...,
      ...,
      [0, 0, 0, 0, ..., 1, 1] ]
    */
    for (int i = 0, idx = 0; i < h; ++i)
    {
        for (int j = 0; j < stride * w; ++j, ++idx)
        {
            postWt[idx] = (j / stride == i) ? 1.0 : 0.0;
        }
    }
    postMul.values = postWt;
    trtWeights.push_back(postMul);
    nvinfer1::IConstantLayer* post_m = network->addConstant(postDims, postMul);
    assert(post_m != nullptr);
    std::string postLayerName = "postMul_" + std::to_string(layerIdx);
    post_m->setName(postLayerName.c_str());
    // add matrix multiply layers for upsampling
    nvinfer1::IMatrixMultiplyLayer* mm1
        = network->addMatrixMultiply(*preM->getOutput(0), nvinfer1::MatrixOperation::kNONE, *input,
                                     nvinfer1::MatrixOperation::kNONE);
    assert(mm1 != nullptr);
    std::string mm1LayerName = "mm1_" + std::to_string(layerIdx);
    mm1->setName(mm1LayerName.c_str());
    nvinfer1::IMatrixMultiplyLayer* mm2
        = network->addMatrixMultiply(*mm1->getOutput(0), nvinfer1::MatrixOperation::kNONE,
                                     *post_m->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    assert(mm2 != nullptr);
    std::string mm2LayerName = "mm2_" + std::to_string(layerIdx);
    mm2->setName(mm2LayerName.c_str());
    return mm2;
}

void printLayerInfo(std::string layerIndex, std::string layerName, std::string layerInput,
                    std::string layerOutput, std::string weightPtr)
{
    std::cout << std::setw(6) << std::left << layerIndex << std::setw(15) << std::left << layerName;
    std::cout << std::setw(20) << std::left << layerInput << std::setw(20) << std::left
              << layerOutput;
    std::cout << std::setw(6) << std::left << weightPtr << std::endl;
}
