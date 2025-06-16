#ifndef TRT_AXESNORMALIZATION_PLUGIN_H
#define TRT_AXESNORMALIZATION_PLUGIN_H

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
//当前先仅支持axes均在尾部且连续，其它情况以后再说，通用情况复杂，再议
//部分情况可能没有验证
namespace custom_plugin_kernel{

    int customTailNormalizationKernelExector(const void* input_p, void* output_p, const float* scale_p, const float* bias_p, float eps,
                                            std::vector<int> &shape, std::string &data_type, int scale_dim_id, cudaStream_t stream);
}

namespace nvinfer1
{
namespace plugin
{

class AxesNormalizationPlugin : public IPluginV2DynamicExt
{
public:
    AxesNormalizationPlugin() = delete;
    AxesNormalizationPlugin(std::string const& name, DataType data_type, uint32_t axes_mask, Weights scales, Weights bias, float eps);//正常malloc使用
    AxesNormalizationPlugin(DataType data_type, uint32_t axes_mask, Weights scales, Weights bias, Dims axes,
                            void* scales_p, void* bias_p, std::vector<int> run_shape_, float eps, bool initialed); // clone使用，应当避免新的malloc
    AxesNormalizationPlugin(std::string const& name, void const* buffer, size_t length);//正常malloc使用
    ~AxesNormalizationPlugin() override;

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

    //self defined
    bool setEpsFactor(float eps);
    bool setName(std::string name);
private:
    std::string name_;
    std::string nameSpace_;
    DataType data_type_;
    uint32_t axes_mask_;
    Weights scales_;
    Weights bias_;
    Dims axes_;
    void* scales_p_ = nullptr;
    void* bias_p_ = nullptr;
    std::vector<int> run_shape_ = {1, 1, 1};
    float eps_ = 1e-5f;
    bool initialed_ = false;

    std::vector<float> bias_data_zone_;
    std::vector<float> scales_data_zone_;

    static std::unordered_map<DataType, std::string> trt_type_string_table_;
};

class AxesNormalizationPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    AxesNormalizationPluginCreator();
    ~AxesNormalizationPluginCreator();
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