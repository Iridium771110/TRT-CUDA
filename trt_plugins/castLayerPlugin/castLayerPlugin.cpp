#include "castLayerPlugin.h"

using namespace nvinfer1;
using namespace plugin;

using nvinfer1::plugin::CastLayerPlugin;
using nvinfer1::plugin::CastLayerPluginCreator;

namespace
{
char const* kCAST_LAYER_PLUGIN_NAME{"CastLayerPlugin_custom"};
char const* kCAST_LAYER_PLUGIN_VERSION{"1"};
size_t constexpr kCAST_LAYER_SERIALIZATION_SIZE{2*sizeof(DataType)};
} // namespace

std::unordered_map<DataType, int> CastLayerPlugin::nv_type_size_table_ = {{DataType::kFLOAT, 4},
                                                                {DataType::kHALF, 2},
                                                                {DataType::kINT32, 4},
                                                                {DataType::kINT8, 1}};

int32_t CastLayerPlugin::convertNvDType2Key(nvinfer1::DataType type1, nvinfer1::DataType type2){
    return static_cast<int32_t>(type1) * 100 + static_cast<int32_t>(type2);
}

CastLayerPlugin::CastLayerPlugin(std::string const& name, DataType input_type, DataType output_type)
    : mName_(name)
    , input_type_(input_type)
    , output_type_(output_type)
{
    input_type_size_ = nv_type_size_table_.at(input_type_);
    output_type_size_ = nv_type_size_table_.at(output_type_);
}

CastLayerPlugin::CastLayerPlugin(std::string const& name, void const* buffer, size_t length)
    : mName_(name)
{
    PLUGIN_VALIDATE(buffer != nullptr);
    PLUGIN_VALIDATE(length == kCAST_LAYER_SERIALIZATION_SIZE);

    char const* d = static_cast<char const*>(buffer);
    char const* a = d;

    input_type_ = read<DataType>(d);
    output_type_ = read<DataType>(d);
    input_type_size_ = nv_type_size_table_.at(input_type_);
    output_type_size_ = nv_type_size_table_.at(output_type_);

    PLUGIN_VALIDATE(d == a + length);
}

IPluginV2DynamicExt* CastLayerPlugin::clone() const noexcept
{
    //将这个plugin对象克隆一份给TensorRT的builder、network或者engine，注意如果涉及指针空间可能需要新的构造函数等
    try
    {
        auto plugin = new CastLayerPlugin(this->mName_, this->input_type_, this->output_type_);
        plugin->setPluginNamespace(mNameSpace_.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t CastLayerPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DataType CastLayerPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return output_type_;
}

DimsExprs CastLayerPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    return inputs[0];
}

size_t CastLayerPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

void CastLayerPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (nbInputs != 1 || nbOutputs != 1){
        std::cout << "CastLayerPlugin acquire 1 input & 1 output, but get " << nbInputs << "input and " << nbOutputs << " output" << std::endl;
    }
}

void CastLayerPlugin::destroy() noexcept
{
    delete this;
}

int32_t CastLayerPlugin::initialize() noexcept
{
    //初始化函数，在这个插件准备开始run之前执行。
    // 主要初始化一些提前开辟空间的参数，一般是一些cuda操作需要的参数(例如conv操作需要执行卷积操作，我们就需要提前开辟weight和bias的显存)
    return 0;
}

void CastLayerPlugin::terminate() noexcept {
    // 析构函数则需要执行terminate，terminate函数就是释放这个op之前开辟的一些显存空间
}

size_t CastLayerPlugin::getSerializationSize() const noexcept
{
    return kCAST_LAYER_SERIALIZATION_SIZE;
}

void CastLayerPlugin::serialize(void* buffer) const noexcept
{
    PLUGIN_ASSERT(buffer != nullptr);
    char* d = static_cast<char*>(buffer);
    char* a = d;
    write<DataType>(d, input_type_);
    write<DataType>(d, output_type_);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void CastLayerPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNameSpace_ = pluginNamespace;
}

char const* CastLayerPlugin::getPluginNamespace() const noexcept
{
    return mNameSpace_.c_str();
}

char const* CastLayerPlugin::getPluginType() const noexcept
{
    return kCAST_LAYER_PLUGIN_NAME;
}

char const* CastLayerPlugin::getPluginVersion() const noexcept
{
    return kCAST_LAYER_PLUGIN_VERSION;
}

bool CastLayerPlugin::supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept{
    //TensorRT调用此方法以判断pos索引的输入/输出是否支持inOut[pos].format和inOut[pos].type指定的格式/数据类型。
    //本处仅有1in-1out
    if (pos < 0 || pos >= 2){
        std::cout << "CastLayerPlugin acquire 1 input & 1 output, but get index " << pos << ", which is out of inout range" << std::endl;
        return false;
    }
    const PluginTensorDesc* in = inOut;
    const PluginTensorDesc* out = inOut + nbInputs;
    switch (pos){
        case 0:
        return in[0].type == DataType::kFLOAT || in[0].type == DataType::kHALF || in[0].type == DataType::kINT32 || in[0].type == DataType::kINT8;
        case 1:
        return out[0].format == in[0].format &&
            (out[0].type == DataType::kFLOAT || out[0].type == DataType::kHALF || out[0].type == DataType::kINT32 || out[0].type == DataType::kINT8);
    }
    return false;
}

int32_t CastLayerPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    //可以不用workspace，在initial里面开辟，在teminate里面释放，析构调用terminate
    Dims input_dim = inputDesc[0].dims;
    int element_num = 1;
    for (int i = 0; i < input_dim.nbDims; i++) element_num *= input_dim.d[i];
    const void* input_p = inputs[0];
    void* output_p = outputs[0];
    int32_t func_key = convertNvDType2Key(DataType(input_type_), DataType(output_type_));
    if (executor_map_.count(func_key) == 0){
        std::cout << "unsupported datatype, either input or output, support are kFLOAT, kHALF, kINT32, kINT8" << std::endl;
        return -1;
    }
    custom_plugin_kernel::customCastKernelExectorFuncPtr executor_p = executor_map_.at(func_key);
    int32_t status = executor_p(input_p, output_p, element_num, stream);
    return status;
}

PluginFieldCollection CastLayerPluginCreator::mFC_{};
std::vector<PluginField> CastLayerPluginCreator::mPluginAttributes_;
std::unordered_map<int32_t, DataType> CastLayerPluginCreator::nv_type_table_ = {
                                                                        {static_cast<int32_t>(DataType::kFLOAT), DataType::kFLOAT}, //{0, DataType::kFLOAT},
                                                                        {static_cast<int32_t>(DataType::kHALF), DataType::kHALF}, //{1, DataType::kHALF},
                                                                        {static_cast<int32_t>(DataType::kINT8), DataType::kINT8}, //{2, DataType::kINT8},
                                                                        {static_cast<int32_t>(DataType::kINT32), DataType::kINT32}, //{3, DataType::kINT32}
};

CastLayerPluginCreator::CastLayerPluginCreator()
{
    mPluginAttributes_.clear();
    mPluginAttributes_.emplace_back(PluginField("input_type", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes_.emplace_back(PluginField("output_type", nullptr, PluginFieldType::kINT32, 1));
    mFC_.nbFields = mPluginAttributes_.size();
    mFC_.fields = mPluginAttributes_.data();
}

CastLayerPluginCreator::~CastLayerPluginCreator() {}

IPluginV2* CastLayerPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const* fields = fc->fields;
        DataType input_type, output_type;
        int complete_check = 0;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "input_type") || !strcmp(attrName, "output_type"))
            {
                complete_check ++;
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                int32_t type_id = static_cast<int32_t>(*(static_cast<int32_t const*>(fields[i].data)));
                if (nv_type_table_.count(type_id) == 0){
                    std::cout << "unsupported type id " << type_id << std::endl;
                    std::cout << "use 0 for DataType::kFLOAT or 1 for DataType::kHALF or 2 for DataType::kINT8 or 3 for DataType::kINT32" << std::endl;
                    return nullptr;
                }
                if (!strcmp(attrName, "input_type")) input_type = nv_type_table_.at(type_id);
                else output_type = nv_type_table_.at(type_id);
            }
        }
        if (complete_check != 2){
            std::cout << "uncompatible castlayer plugin creation, need 2 attributes but get " << complete_check << std::endl;
            return nullptr;
        }
        return new CastLayerPlugin(name, input_type, output_type);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* CastLayerPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        PLUGIN_VALIDATE(serialData != nullptr);
        return new CastLayerPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* CastLayerPluginCreator::getPluginName() const noexcept
{
    return kCAST_LAYER_PLUGIN_NAME;
}

char const* CastLayerPluginCreator::getPluginVersion() const noexcept
{
    return kCAST_LAYER_PLUGIN_VERSION;
}

PluginFieldCollection const* CastLayerPluginCreator::getFieldNames() noexcept
{
    return &mFC_;
}

REGISTER_TENSORRT_PLUGIN(CastLayerPluginCreator);