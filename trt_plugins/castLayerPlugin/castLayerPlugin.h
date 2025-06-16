#ifndef TRT_CASTLAYER_PLUGIN_H
#define TRT_CASTLAYER_PLUGIN_H

#include "plugin.h"
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <vector>
#include <cuda.h>
#include "cuda_fp16.h"

#include "checkMacrosPlugin.h"

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include <unordered_map>

//this layer may not verified yet

namespace custom_plugin_kernel{

    template<typename InType, typename OutType>
    int customCastKernelExector(const void* input_p, void* output_p, int element_num, cudaStream_t stream);

    typedef int (*customCastKernelExectorFuncPtr)(const void*, void*, int, cudaStream_t);

}

namespace nvinfer1
{
namespace plugin
{

class CastLayerPlugin : public IPluginV2DynamicExt
{
public:
    CastLayerPlugin() = delete;
    CastLayerPlugin(std::string const& name, DataType input_type, DataType output_type);
    CastLayerPlugin(std::string const& name, void const* buffer, size_t length);
    ~CastLayerPlugin() override = default;

    // Method inherited from IPluginV2
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // Method inherited from IPluginV2Ext
    DataType getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    // Method inherited from IPluginV2DynamicExt
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
    const std::string mName_;
    std::string mNameSpace_;
    nvinfer1::DataType input_type_;
    DataType output_type_;
    int input_type_size_;
    int output_type_size_;
    static std::unordered_map<int32_t, custom_plugin_kernel::customCastKernelExectorFuncPtr> executor_map_;
    static int32_t convertNvDType2Key(nvinfer1::DataType type1, nvinfer1::DataType type2);
    static std::unordered_map<DataType, int> nv_type_size_table_;
};

class CastLayerPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    CastLayerPluginCreator();
    ~CastLayerPluginCreator();
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;
    IPluginV2* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
    IPluginV2* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    //这两似乎没啥卵用
    static PluginFieldCollection mFC_;
    static std::vector<PluginField> mPluginAttributes_;
    static std::unordered_map<int32_t, DataType> nv_type_table_;
};

}}

#endif