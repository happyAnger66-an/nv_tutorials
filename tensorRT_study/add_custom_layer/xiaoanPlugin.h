/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_XIAOAN_PLUGIN_H
#define TRT_CLIXIAOANP_PLUGIN_H

#include "NvInferPluginBase.h"
#include "NvInferRuntime.h"

#include <cstdlib>
#include <string>
#include <vector>
#include <assert.h>

using namespace nvinfer1;

REGISTER_TENSORRT_PLUGIN(XiaoanPluginCreator);

class XiaoanPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
{

    IPluginCapability *getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        // All plugin interface methods are noexcept and care should be
        // taken not to throw exceptions across the API boundary. It is
        // recommended to catch any exceptions and return a value that
        // appropriately represents the error status.
        try
        {
            if (type == PluginCapabilityType::kBUILD)
            {
                return static_cast<IPluginV3OneBuild *>(this);
            }
            if (type == PluginCapabilityType::kRUNTIME)
            {
                return static_cast<IPluginV3OneRuntime *>(this);
            }

            ASSERT(type == PluginCapabilityType::kCORE);
            return static_cast<IPluginV3OneCore *>(this);
        }
        catch (...)
        {
            // log error
        }
        return nullptr;
    }

    int32_t getNbOutputs() const noexcept override
    {
        return 2;
    }

    int32_t getOutputDataTypes(
        DataType *outputTypes, int32_t nbOutputs, DataType const *inputTypes, int32_t nbInputs) const noexcept override
    {
        outputTypes[0] = DataType::kINT32;
        outputTypes[1] = DataType::kINT32;
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        bool typeOk{false};
        if (pos == 0)
        {
            typeOk = inOut[0].desc.type == DataType::kFLOAT || inOut[0].desc.type == DataType::kHALF;
        }
        else if (pos == 1)
        {
            typeOk = inOut[1].desc.type == DataType::kINT32;
        }
        else // pos == 2
        {
            // size tensor outputs must be NCHW INT32
            typeOk = inOut[2].desc.type == DataType::kINT32;
        }

        return inOut[pos].desc.format == PluginFormat::kLINEAR && typeOk;
    }

    int32_t getOutputShapes(DimsExprs const *inputs, int32_t nbInputs, DimsExprs const *shapeInputs,
                            int32_t nbShapeInputs, DimsExprs *outputs, int32_t nbOutputs, IExprBuilder &exprBuilder) noexcept override
    {
        // The input tensor must be 2-D
        if (inputs[0].nbDims != 2)
        {
            return -1;
        }

        outputs[0].nbDims = 2;

        auto upperBound = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[0], *inputs[0].d[1]);

        // On average, we can assume that half of all elements will be non-zero
        auto optValue = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *upperBound, *exprBuilder.constant(2));
        auto numNonZeroSizeTensor = exprBuilder.declareSizeTensor(1, *optValue, *upperBound);

        if (!mRowOrder)
        {
            outputs[0].d[0] = exprBuilder.constant(2);
            outputs[0].d[1] = numNonZeroSizeTensor;
        }
        else
        {
            outputs[0].d[0] = numNonZeroSizeTensor;
            outputs[0].d[1] = exprBuilder.constant(2);
        }

        // output at index 1 is a size tensor
        outputs[1].nbDims = 0; // size tensors must be declared as 0-D

        return 0;
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out,
                            int32_t nbOutputs) noexcept override
    {
        return 0;
    }

    int32_t onShapeChange(
        PluginTensorDesc const *in, int32_t nbInputs, PluginTensorDesc const *out, int32_t nbOutputs) noexcept override
    {
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs,
                    void *const *outputs, void *workspace, cudaStream_t stream) noexcept override
    {

        int32_t const R = inputDesc[0].dims.d[0];
        int32_t const C = inputDesc[0].dims.d[1];

        auto type = inputDesc[0].type;

        if (!(type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kFLOAT))
        {
            sample::gLogError << "Unsupported: Sample only supports DataType::kHALF and DataType::FLOAT" << std::endl;
            return -1;
        }

        cudaMemsetAsync(outputs[1], 0, sizeof(int32_t), stream);

        if (workspace == nullptr)
        {
            sample::gLogError << "Unsupported: workspace is null" << std::endl;
            return -1;
        }

        if (!mRowOrder)
        {
            // When constructing a column major output, the kernel needs to be aware of the total number of non-zero
            // elements so as to write the non-zero indices at the correct places. Therefore, we will launch the kernel
            // twice: first, only to calculate the total non-zero count, which will be stored in workspace; and
            // then to actually write the non-zero indices to the outputs[0] buffer.
            cudaMemsetAsync(workspace, 0, sizeof(int32_t), stream);
            nonZeroIndicesHelper(type, inputs[0], nullptr, workspace, 0, R, C, mRowOrder, stream);
            nonZeroIndicesHelper(type, inputs[0], outputs[0], outputs[1], workspace, R, C, mRowOrder, stream);
        }
        else
        {
            nonZeroIndicesHelper(type, inputs[0], outputs[0], outputs[1], 0, R, C, mRowOrder, stream);
        }

        return 0;
    }
};

class XiaoanPluginCreator : public IPluginCreatorV3One
{
public:
    XiaoanPluginCreator()
    {
        mPluginAttributes.clear();
        mPluginAttributes.emplace_back(PluginField("rowOrder", nullptr, PluginFieldType::kINT32, 1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    char const *getPluginName() const noexcept override
    {
        return "XiaoanPlugin";
    }

    char const *getPluginVersion() const noexcept override
    {
        return "0";
    }

    PluginFieldCollection const *getFieldNames() noexcept override
    {
        return &mFC;
    }

    IPluginV3 *createPlugin(char const *name, PluginFieldCollection const *fc, TensorRTPhase phase) noexcept override
    {
        try
        {
            bool rowOrder{true};
            for (int32_t i = 0; i < fc->nbFields; ++i)
            {
                auto const fieldName(fc->fields[i].name);
                if (std::strcmp(fieldName, "rowOrder") == 0)
                {
                    rowOrder = *static_cast<bool const *>(fc->fields[i].data);
                }
            }
            return new XiaoanPlugin(rowOrder);
        }
        catch (std::exception const &e)
        {
            sample::gLogError << e.what() << std::endl;
        }
        return nullptr;
    }

    char const *getPluginNamespace() const noexcept override
    {
        return "";
    }

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
};

#endif