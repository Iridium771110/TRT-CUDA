#include "axesNormalizationPlugin.h"

using namespace nvinfer1;
using namespace plugin;

using nvinfer1::plugin::AxesNormalizationPlugin;
using nvinfer1::plugin::AxesNormalizationPluginCreator;

namespace
{
char const* kAXES_NORMALIZATION_PLUGIN_NAME{"AxesNormalizationPlugin_custom"};
char const* kAXES_NORMALIZATION_PLUGIN_VERSION{"1"};
size_t constexpr kAXES_NORMALiZATION_SERIALIZATION_SIZE{sizeof(DataType) + sizeof(uint32_t) + 2*sizeof(Weights) + sizeof(float)};
} // namespace

std::unordered_map<DataType, std::string> AxesNormalizationPlugin::trt_type_string_table_ = {
                                                                        {DataType::kFLOAT, "float"},
                                                                        {DataType::kHALF, "half"},
};

AxesNormalizationPlugin::AxesNormalizationPlugin(std::string const& name, DataType data_type, uint32_t axes_mask, Weights scales, Weights bias, float eps)
    : name_(name)
    , data_type_(data_type)
    , axes_mask_(axes_mask)
    , scales_(scales)
    , bias_(bias)
    , eps_(eps)
{
    std::vector<int> axes(0);
    for (int i = 0; i < 32; i++){
        if ((axes_mask_ >> i) & 0x1) axes.emplace_back(i);
    }
    axes_.nbDims = axes.size();
    if (axes_.nbDims > axes_.MAX_DIMS){
        std::cout << "in AxesLayernorm " << name << " axes number out of range " << axes_.MAX_DIMS << std::endl;
        axes_.nbDims = axes_.MAX_DIMS;
    }
    for (int i = 0; i < axes_.nbDims; i++) axes_.d[i] = axes[i];
    scales_data_zone_.resize(scales_.count);
    bias_data_zone_.resize(bias_.count);
    for (int i = 0; i < scales_.count; i++) scales_data_zone_[i] = reinterpret_cast<const float*>(scales_.values)[i];
    for (int i = 0; i < bias_.count; i++) bias_data_zone_[i] = reinterpret_cast<const float*>(bias_.values)[i];
    scales_.values = reinterpret_cast<const void*>(scales_data_zone_.data());
    bias_.values = reinterpret_cast<const void*>(bias_data_zone_.data());
}
AxesNormalizationPlugin::AxesNormalizationPlugin(DataType data_type, uint32_t axes_mask, Weights scales, Weights bias, Dims axes,
                            void* scales_p, void* bias_p, std::vector<int> run_shape, float eps, bool initialed)
    : data_type_(data_type)
    , axes_mask_(axes_mask)
    , scales_(scales)
    , bias_(bias)
    , axes_(axes)
    , scales_p_(scales_p)
    , bias_p_(bias_p)
    , run_shape_(run_shape)
    , eps_(eps)
    , initialed_(initialed)
{
}
AxesNormalizationPlugin::AxesNormalizationPlugin(std::string const& name, void const* buffer, size_t length)
    : name_(name)
{
    PLUGIN_VALIDATE(buffer != nullptr);
    
    char const* d = static_cast<char const*>(buffer);
    char const* a = d;
    data_type_ = read<DataType>(d);
    axes_mask_ = read<uint32_t>(d);
    scales_ = read<Weights>(d);
    bias_ = read<Weights>(d);
    eps_ = read<float>(d);
    scales_data_zone_.resize(scales_.count);
    bias_data_zone_.resize(bias_.count);
    for (int i = 0; i < scales_.count; i++) scales_data_zone_[i] = read<float>(d);
    for (int i = 0; i < bias_.count; i++) bias_data_zone_[i] = read<float>(d);
    scales_.values = reinterpret_cast<const void*>(scales_data_zone_.data());
    bias_.values = reinterpret_cast<const void*>(bias_data_zone_.data());
    std::vector<int> axes(0);
    for (int i = 0; i < 32; i++){
        if ((axes_mask_ >> i) & 0x1) axes.emplace_back(i);
    }
    axes_.nbDims = axes.size();
    if (axes_.nbDims > axes_.MAX_DIMS){
        std::cout << "in AxesLayernorm " << name << " axes number out of range " << axes_.MAX_DIMS << std::endl;
        axes_.nbDims = axes_.MAX_DIMS;
    }
    for (int i = 0; i < axes_.nbDims; i++) axes_.d[i] = axes[i];
    PLUGIN_VALIDATE(length == kAXES_NORMALiZATION_SERIALIZATION_SIZE + sizeof(float)*(scales_.count + bias_.count));
    PLUGIN_VALIDATE(d == a + length);
}
AxesNormalizationPlugin::~AxesNormalizationPlugin(){
    terminate(); // 此处似乎不必有这个，可能内部和inital函数有调用关联
}

IPluginV2DynamicExt* AxesNormalizationPlugin::clone() const noexcept{
    //将这个plugin对象克隆一份给TensorRT的builder、network或者engine，注意如果涉及指针空间可能需要新的构造函数等
    try{
        cudaDeviceSynchronize();
        auto plugin = new AxesNormalizationPlugin(this->data_type_, this->axes_mask_, this->scales_, this->bias_, 
                                                this->axes_, this->scales_p_, this->bias_p_, this->run_shape_, this->eps_, false);
        plugin->setPluginNamespace(nameSpace_.c_str());
        return plugin;
    }
    catch (std::exception const& e){
        caughtError(e);
    }
    return nullptr;
}
void AxesNormalizationPlugin::serialize(void* buffer) const noexcept{
    PLUGIN_ASSERT(buffer != nullptr);
    char* d = static_cast<char*>(buffer);
    char* a = d;
    write<DataType>(d, data_type_);
    write<uint32_t>(d, axes_mask_);
    write<Weights>(d, scales_);
    write<Weights>(d, bias_);
    write<float>(d, eps_);
    for (int i = 0; i < scales_.count; i++) write<float>(d, reinterpret_cast<const float*>(scales_.values)[i]);
    for (int i = 0; i < bias_.count; i++) write<float>(d, reinterpret_cast<const float*>(bias_.values)[i]);

    PLUGIN_ASSERT(d == a + getSerializationSize());
}
void AxesNormalizationPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept{
        //本处配置可以部分用于运行时检查
    if (nbInputs != 1 || nbOutputs != 1){
        std::cout << "AxesNoramlizationPlugin aquire 1 input & 1 output, but get " << nbInputs << "input and " << nbOutputs << " output" << std::endl;
    }
    if (in->desc.format != nvinfer1::TensorFormat::kLINEAR){
        std::cout<<"invalid input format "<<std::endl;
    }
}
int32_t AxesNormalizationPlugin::initialize() noexcept{
    //初始化函数，在这个插件准备开始run之前执行。
    // 主要初始化一些提前开辟空间的参数，一般是一些cuda操作需要的参数(例如conv操作需要执行卷积操作，我们就需要提前开辟weight和bias的显存)
    if (scales_.count != bias_.count){
        std::cout << "AxesNormalization " << name_ << " in compatible scale/bias length " << scales_.count << ' ' << bias_.count << std::endl;
    }
    if (scales_.type != DataType::kFLOAT || bias_.type != DataType::kFLOAT){
        std::cout << "AxesNormalization " << name_ << " scale/bias type should be float" << std::endl;
    }
    if (trt_type_string_table_.count(data_type_) == 0){
        std::cout << "AxesNormalization " << name_ << " in&out data type should be float or half" << std::endl;
    }
    cudaMalloc(reinterpret_cast<void**>(&scales_p_), sizeof(float) * scales_.count);
    cudaMalloc(reinterpret_cast<void**>(&bias_p_), sizeof(float) * bias_.count);
    cudaMemcpy(scales_p_, scales_.values, sizeof(float) * scales_.count, cudaMemcpyHostToDevice);
    cudaMemcpy(bias_p_, bias_.values, sizeof(float) * bias_.count, cudaMemcpyHostToDevice);
    return 0;
}
void AxesNormalizationPlugin::terminate() noexcept {
    // 析构函数则需要执行terminate，terminate函数就是释放这个op之前开辟的一些显存空间
    if (!initialed_){
        scales_p_ = nullptr;
        bias_p_ = nullptr;
        return;
    }
    if (scales_p_) cudaFree(scales_p_);
    if (bias_p_) cudaFree(bias_p_);
    scales_p_ = nullptr;
    bias_p_ = nullptr;
}
void AxesNormalizationPlugin::destroy() noexcept{
    delete this;
}
bool AxesNormalizationPlugin::supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept{
    //TensorRT调用此方法以判断pos索引的输入/输出是否支持inOut[pos].format和inOut[pos].type指定的格式/数据类型。
    //在build engine时的核选择等处中，依此来获得可用的那些进行比较选择
    //也以此可以指定中间内部变量可以选用的格式类型等
    //本处仅有1in-1out
    if (pos < 0 || pos >= 2){
        std::cout << "AxesNoramlizationPlugin acquire 1 input & 1 output, but get index " << pos << ", which is out of inout range" << std::endl;
        return false;
    }
    const PluginTensorDesc* in = inOut;
    const PluginTensorDesc* out = inOut + nbInputs;
    if (in[0].format != nvinfer1::TensorFormat::kLINEAR) return false;
    switch (pos){
        case 0:
        return in[0].type == DataType::kFLOAT || in[0].type == DataType::kHALF;
        case 1:
        return out[0].format == in[0].format && (out[0].type == in[0].type);
    }
    return false;
}

int32_t AxesNormalizationPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept{
    const void* input_p = inputs[0];
    void* output_p = outputs[0];
    Dims input_dim = inputDesc[0].dims;
    uint32_t check_dim = 0U;
    int tail_dim = 1;
    for (int i = 2; i < input_dim.nbDims; i++){
        check_dim |= 1U << i;
        tail_dim *= input_dim.d[i];
    }
    for (int i = 0; i < axes_.nbDims; i++){
        check_dim ^= 1U << axes_.d[i];
    }
    
    if (axes_.nbDims + 2 != input_dim.nbDims || check_dim != 0){
        std::cout << "AxesNormalization " << name_ << "invalid dimensions of input" << std::endl;
        for (int i = 0; i < input_dim.nbDims; i++) std::cout << input_dim.d[i] << ' ';
        std::cout << std::endl;
        std::cout << check_dim <<' '<< axes_.nbDims << ' '<< run_shape_[1]<<std::endl;
        return -2;
    }
    data_type_ = inputDesc[0].type; 
    run_shape_[0] = input_dim.d[0];
    run_shape_[1] = input_dim.d[1];
    run_shape_[2] = tail_dim;
    if (run_shape_[1] == scales_.count) custom_plugin_kernel::customTailNormalizationKernelExector(
                                                        input_p, output_p, reinterpret_cast<float*>(scales_p_), reinterpret_cast<float*>(bias_p_),
                                                        eps_, run_shape_, trt_type_string_table_.at(data_type_), 1, stream);
    else if (run_shape_[2] == scales_.count) custom_plugin_kernel::customTailNormalizationKernelExector(
                                                        input_p, output_p, reinterpret_cast<float*>(scales_p_), reinterpret_cast<float*>(bias_p_),
                                                        eps_, run_shape_, trt_type_string_table_.at(data_type_), 2, stream);
    else{
        std::cout << "AxesNormalization " << name_ << "invalid dimensions of input" << std::endl;
        std::cout << "for instance norm support channel as scale dim, for layer norm support tail as scale dim yet" << std::endl;
        return -2;
    }
    return 0;
}

char const* AxesNormalizationPlugin::getPluginType() const noexcept{
    return kAXES_NORMALIZATION_PLUGIN_NAME;
}
char const* AxesNormalizationPlugin::getPluginVersion() const noexcept{
    return kAXES_NORMALIZATION_PLUGIN_VERSION;
}
int32_t AxesNormalizationPlugin::getNbOutputs() const noexcept{
    return 1;
}
size_t AxesNormalizationPlugin::getSerializationSize() const noexcept{
    return kAXES_NORMALiZATION_SERIALIZATION_SIZE + sizeof(float)*scales_.count + sizeof(float)*bias_.count;
}
char const* AxesNormalizationPlugin::getPluginNamespace() const noexcept{
    return nameSpace_.c_str();
}
DataType AxesNormalizationPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept{
    return data_type_;
}
DimsExprs AxesNormalizationPlugin::getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept{
    return inputs[0];
}
size_t AxesNormalizationPlugin::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept{
    return 0;
}

void AxesNormalizationPlugin::setPluginNamespace(char const* pluginNamespace) noexcept{
    nameSpace_ = pluginNamespace;
}
bool AxesNormalizationPlugin::setEpsFactor(float eps){
    eps_ = eps;
    return true;
}
bool AxesNormalizationPlugin::setName(std::string name){
    name_ = name;
    return true;
}

PluginFieldCollection AxesNormalizationPluginCreator::mFC_{};
std::vector<PluginField> AxesNormalizationPluginCreator::mPluginAttributes_;
std::unordered_map<int32_t, DataType> AxesNormalizationPluginCreator::nv_type_table_ = {
                                                                        {static_cast<int32_t>(DataType::kFLOAT), DataType::kFLOAT}, //{0, DataType::kFLOAT},
                                                                        {static_cast<int32_t>(DataType::kHALF), DataType::kHALF}, //{1, DataType::kHALF},
};

AxesNormalizationPluginCreator::AxesNormalizationPluginCreator(){
    mPluginAttributes_.clear();
    mPluginAttributes_.emplace_back(PluginField("data_type", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes_.emplace_back(PluginField("axes_mask", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes_.emplace_back(PluginField("scales", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes_.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes_.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC_.nbFields = mPluginAttributes_.size();
    mFC_.fields = mPluginAttributes_.data();
}
AxesNormalizationPluginCreator::~AxesNormalizationPluginCreator() {}

IPluginV2* AxesNormalizationPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept{
    try{
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const* fields = fc->fields;
        DataType data_type;
        uint32_t axes_mask;
        Weights scales, bias;
        float eps;
        int complete_check = 0;
        for (int32_t i = 0; i < fc->nbFields; ++i){
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "data_type")){
                complete_check ++;
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                int32_t type_id = static_cast<int32_t>(*(static_cast<int32_t const*>(fields[i].data)));
                if (nv_type_table_.count(type_id) == 0){
                    std::cout << "unsupported type id " << type_id << "use 0 for DataType::kFLOAT or 1 for DataType::kHALF" << std::endl;
                    return nullptr;
                }
                data_type = nv_type_table_.at(type_id);
            }
            else if (!strcmp(attrName, "axes_mask")){
                complete_check ++;
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                axes_mask = static_cast<uint32_t>(*(static_cast<int32_t const*>(fields[i].data)));
                if (axes_mask == 0 || axes_mask >= 256){
                    std::cout << "invalid axes mask " << axes_mask << " should > 0 and < 256" << std::endl;
                    return nullptr;
                }
            }
            else if (!strcmp(attrName, "scales")){
                complete_check ++;
                scales.type = DataType::kFLOAT;
                scales.count = fields[i].length;
                scales.values = fields[i].data;
            }
            else if (!strcmp(attrName, "bias")){
                complete_check ++;
                bias.type = DataType::kFLOAT;
                bias.count = fields[i].length;
                bias.values = fields[i].data;
            }
            else if (!strcmp(attrName, "eps")){
                complete_check ++;
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                eps = reinterpret_cast<const float*>(fields[i].data)[0];
            }
        }

        if (complete_check != 5){
            std::cout << "uncompatible AxesNoramlization plugin creation, need 5 attributes but get " << complete_check << std::endl;
            return nullptr;
        }
        if (scales.count != bias.count){
            std::cout << "uncompatible AxesNoramlization plugin creation for scales number not equal to bias number " << std::endl;
            return nullptr;
        }
        return new AxesNormalizationPlugin(name, data_type, axes_mask, scales, bias, eps);
    }
    catch (std::exception const& e){
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* AxesNormalizationPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept{
    try{
        PLUGIN_VALIDATE(serialData != nullptr);
        return new AxesNormalizationPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e){
        caughtError(e);
    }
    return nullptr;
}

char const* AxesNormalizationPluginCreator::getPluginName() const noexcept{
    return kAXES_NORMALIZATION_PLUGIN_NAME;
}
char const* AxesNormalizationPluginCreator::getPluginVersion() const noexcept{
    return kAXES_NORMALIZATION_PLUGIN_VERSION;
}
PluginFieldCollection const* AxesNormalizationPluginCreator::getFieldNames() noexcept{
    return &mFC_;
}

REGISTER_TENSORRT_PLUGIN(AxesNormalizationPluginCreator);