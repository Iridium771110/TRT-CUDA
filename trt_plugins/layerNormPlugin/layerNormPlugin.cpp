/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "layerNormPlugin.h"
#include "layerNormKernel.h"

using namespace nvinfer1;
using namespace plugin;

using nvinfer1::plugin::LayerNormalizationPlugin;
using nvinfer1::plugin::LayerNormalizationPluginCreator;

namespace
{
char const* kLAYER_NORM_PLUGIN_NAME{"LayerNormalization_test"};
char const* kLAYER_NORM_PLUGIN_VERSION{"1"};
size_t constexpr kSERIALIZATION_SIZE{sizeof(float)};
} // namespace

LayerNormalizationPlugin::LayerNormalizationPlugin(std::string const& name, float mEpsilon)
    : mName(name)
    , mEpsilon(mEpsilon)
{
}

LayerNormalizationPlugin::LayerNormalizationPlugin(std::string const& name, void const* buffer, size_t length)
    : mName(name)
{
    PLUGIN_VALIDATE(buffer != nullptr);
    PLUGIN_VALIDATE(length == kSERIALIZATION_SIZE);

    char const* d = static_cast<char const*>(buffer);
    char const* a = d;

    mEpsilon = read<float>(d);

    PLUGIN_VALIDATE(d == a + length);
}

IPluginV2DynamicExt* LayerNormalizationPlugin::clone() const noexcept
{
    try
    {
        auto plugin = new LayerNormalizationPlugin(*this);
        plugin->setPluginNamespace(mNameSpace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t LayerNormalizationPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DataType LayerNormalizationPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

DimsExprs LayerNormalizationPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    return inputs[0];
}

bool LayerNormalizationPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{

    switch (pos)
    {
    case 0:
        return ((inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF)
            && (inOut[0].format == TensorFormat::kLINEAR))
            || ((inOut[0].type == DataType::kINT8)
            && (inOut[0].format == TensorFormat::kCHW4 || inOut[0].format == TensorFormat::kCHW32));
    case 1:
    case 2:
        return (inOut[pos].type == inOut[0].type)
            || ((inOut[0].type == DataType::kINT8) && (inOut[pos].type == DataType::kHALF));
    case 3: return (inOut[pos].type == inOut[0].type) && (inOut[pos].format == inOut[0].format);
    default: // should NOT be here!
        return false;
    }
    return false;
}

void LayerNormalizationPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
}

size_t LayerNormalizationPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t LayerNormalizationPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{

    int32_t gridSize = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    int32_t nHiddenSize = 1;
    for (int32_t i = 2; i < inputDesc[0].dims.nbDims; ++i)
    {
        nHiddenSize *= inputDesc[0].dims.d[i];
    }
    int32_t status = -1;

    switch (inputDesc[0].type)
    {
    case DataType::kFLOAT:
    {
        auto const input = static_cast<float const*>(inputs[0]);
        auto const gamma = static_cast<float const*>(inputs[1]);
        auto const beta = static_cast<float const*>(inputs[2]);
        auto output = static_cast<float*>(outputs[0]);

        status = computeLayerNorm<float>(gridSize, nHiddenSize, input, gamma, beta, output, mEpsilon, stream);
        break;
    }
    case DataType::kHALF:
    {

        auto const input = static_cast<half const*>(inputs[0]);
        auto const gamma = static_cast<half const*>(inputs[1]);
        auto const beta = static_cast<half const*>(inputs[2]);
        auto output = static_cast<half*>(outputs[0]);

        status = computeLayerNorm<half>(gridSize, nHiddenSize, input, gamma, beta, output, mEpsilon, stream);
        break;
    }
    case DataType::kINT8:
    {
        float const dqScaleIn = inputDesc[0].scale;
        float const qScale = 1.f / outputDesc[0].scale;
        auto const input = static_cast<int8_t const*>(inputs[0]);
        auto output = static_cast<int8_t*>(outputs[0]);
        auto const gamma = static_cast<half const*>(inputs[1]);
        auto const beta = static_cast<half const*>(inputs[2]);

        status = computeLayerNormQDQ(
            gridSize, nHiddenSize, input, gamma, beta, output, dqScaleIn, qScale, mEpsilon, stream);
        break;
    }
    default:
    {
        PLUGIN_FAIL("DataType not implemented yet");
        break;
    }
    }
    return status;
}

void LayerNormalizationPlugin::destroy() noexcept
{
    delete this;
}

int32_t LayerNormalizationPlugin::initialize() noexcept
{
    return 0;
}

void LayerNormalizationPlugin::terminate() noexcept {}

size_t LayerNormalizationPlugin::getSerializationSize() const noexcept
{
    return kSERIALIZATION_SIZE;
}

void LayerNormalizationPlugin::serialize(void* buffer) const noexcept
{
    PLUGIN_ASSERT(buffer != nullptr);
    char* d = static_cast<char*>(buffer);
    char* a = d;
    write(d, mEpsilon); // float
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void LayerNormalizationPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNameSpace = pluginNamespace;
}

char const* LayerNormalizationPlugin::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

char const* LayerNormalizationPlugin::getPluginType() const noexcept
{
    return kLAYER_NORM_PLUGIN_NAME;
}

char const* LayerNormalizationPlugin::getPluginVersion() const noexcept
{
    return kLAYER_NORM_PLUGIN_VERSION;
}

PluginFieldCollection LayerNormalizationPluginCreator::mFC{};
std::vector<PluginField> LayerNormalizationPluginCreator::mPluginAttributes;

LayerNormalizationPluginCreator::LayerNormalizationPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

LayerNormalizationPluginCreator::~LayerNormalizationPluginCreator() {}

IPluginV2* LayerNormalizationPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const* fields = fc->fields;

        // default values
        float mEpsilon = 1e-5F;

        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "epsilon"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                mEpsilon = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
            }
        }
        return new LayerNormalizationPlugin(name, mEpsilon);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* LayerNormalizationPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        PLUGIN_VALIDATE(serialData != nullptr);
        return new LayerNormalizationPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* LayerNormalizationPluginCreator::getPluginName() const noexcept
{
    return kLAYER_NORM_PLUGIN_NAME;
}

char const* LayerNormalizationPluginCreator::getPluginVersion() const noexcept
{
    return kLAYER_NORM_PLUGIN_VERSION;
}

PluginFieldCollection const* LayerNormalizationPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

REGISTER_TENSORRT_PLUGIN(LayerNormalizationPluginCreator);