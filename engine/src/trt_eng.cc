#include "trt_eng.h"

namespace engine{

    TRTEngine::~TRTEngine(){
#ifdef JETSON_ORIN
        if (on_dla_){
            general_buffer_pool_.clearBufferPool(dev_handle_);
            general_sync_obj_pool_.clearSyncObjPool(dev_handle_);
        }
#endif
        std::cout<<"deconstruct trt eng"<<std::endl;
    }

    void TRTEngine::destroy(){
        //似乎智能指针之后调用的话会发生 double free
        delete this;
    }
    bool TRTEngine::enginePrepare(std::string net_file_path, std::string eng_file_path){
        return true;
    }
    bool TRTEngine::loadEngineModel(const std::string& model_file_path){
        std::ifstream eng_file(model_file_path, std::ios::binary);
        if (!eng_file.good()){
            std::cout<< "eng: " << engine_name_ <<" engine file path wrong: "<<model_file_path<<" no such file"<<std::endl;
            return false;
        }
        eng_file.seekg(0, eng_file.end);
        int64_t end_pos = eng_file.tellg();
        eng_file.seekg(0, eng_file.beg);
        int64_t beg_pos = eng_file.tellg();
        int64_t file_length = end_pos - beg_pos;
        engine_file_data_.resize(file_length);
        eng_file.read(engine_file_data_.data(), file_length);
        eng_file.close();
        return true;
    }
    bool TRTEngine::initEngineModel(){
        if (on_dla_){
#ifdef JETSON_ORIN
            dla_ret_ = cudlaCreateDevice(device_id_, &dev_handle_, CUDLA_STANDALONE);
            if (dla_ret_ != cudlaSuccess){
                std::cout<<"failed create dla device"<<std::endl;
                return false;
            }
            dla_ret_ = cudlaModuleLoadFromMemory(dev_handle_, reinterpret_cast<uint8_t*>(engine_file_data_.data()), engine_file_data_.size(), &module_handle_, 0);
            if (dla_ret_ != cudlaSuccess){
                std::cout<<"failed load dla into device"<<std::endl;
                return false;
            }
            dla_ret_ = cudlaModuleGetAttributes(module_handle_, CUDLA_NUM_INPUT_TENSORS, &dla_model_attr_);
            dla_ret_ = cudlaModuleGetAttributes(module_handle_, CUDLA_NUM_OUTPUT_TENSORS, &dla_model_attr_);
            if (inputs_num_ != dla_model_attr_.numInputTensors || outputs_num_ != dla_model_attr_.numOutputTensors){
                std::cout<<"dla input num or output num not consistent"<<std::endl;
                return false;
            }
            std::vector<cudlaModuleTensorDescriptor> dla_input_tensor_desc(inputs_num_);
            std::vector<cudlaModuleTensorDescriptor> dla_output_tensor_desc(outputs_num_);
            dla_model_attr_.inputTensorDesc = dla_input_tensor_desc.data();
            dla_ret_ = cudlaModuleGetAttributes(module_handle_, CUDLA_INPUT_TENSOR_DESCRIPTORS, &dla_model_attr_); 
            dla_model_attr_.outputTensorDesc = dla_output_tensor_desc.data();
            dla_ret_ = cudlaModuleGetAttributes(module_handle_, CUDLA_OUTPUT_TENSOR_DESCRIPTORS, &dla_model_attr_);
            dla_inputs_p_vec_.resize(inputs_num_);
            dla_outputs_p_vec_.resize(outputs_num_);
            dla_inputs_name_vec_.resize(inputs_num_);
            dla_outputs_name_vec_.resize(outputs_num_);
            for (int i = 0; i < inputs_num_; i++){
                std::string name = dla_input_tensor_desc[i].name;
                uint64_t size = dla_input_tensor_desc[i].size;
                general_buffer_pool_.createBuffer(dev_handle_, name, size);
                nvscibuffer::GeneralBufferPtr tmp_buffer_p = general_buffer_pool_.getBufferPtr(name);
                dla_mem_map_.emplace(name, tmp_buffer_p->getDlaPtr());
                gpu_mem_map_.emplace(name, tmp_buffer_p->getGpuPtr());
                cpu_mem_map_.emplace(name, tmp_buffer_p->getCpuPtr());
                dla_mem_size_map_.emplace(name, size);
                dla_inputs_p_vec_[i] = tmp_buffer_p->getDlaPtr();
                dla_inputs_name_vec_[i] = name;
                std::cout<<name<<std::endl;
            }
            for (int i = 0; i < outputs_num_; i++){
                std::string name = dla_output_tensor_desc[i].name;
                uint64_t size = dla_output_tensor_desc[i].size;
                general_buffer_pool_.createBuffer(dev_handle_, name, size);
                nvscibuffer::GeneralBufferPtr tmp_buffer_p = general_buffer_pool_.getBufferPtr(name);
                dla_mem_map_.emplace(name, tmp_buffer_p->getDlaPtr());
                gpu_mem_map_.emplace(name, tmp_buffer_p->getGpuPtr());
                cpu_mem_map_.emplace(name, tmp_buffer_p->getCpuPtr());
                dla_mem_size_map_.emplace(name, size);
                dla_outputs_p_vec_[i] = tmp_buffer_p->getDlaPtr();
                dla_outputs_name_vec_[i] = name;
                std::cout<<name<<std::endl;
            }
            general_sync_obj_pool_.createSyncObj(dev_handle_, engine_name_ + "sync_obj");
            nvscisync::GeneralSyncObjPtr tmp_sync_obj_p = general_sync_obj_pool_.getSyncObjPtr(engine_name_ + "sync_obj");
            dla_wait_cpu_events_p_ = tmp_sync_obj_p->getDlaWaitCpuEventPtr();
            dla_signal_cpu_events_p_ = tmp_sync_obj_p->getDlaSignalCpuEventPtr();
            dla_wait_cuda_events_p_ = tmp_sync_obj_p->getDlaWaitGpuEventPtr();
            dla_signal_cuda_events_p_ = tmp_sync_obj_p->getDlaSignalGpuEventPtr();
            dla_wait_cpu_event_obj_p_ = tmp_sync_obj_p->getDlaWaitCpuEventObjPtr();
            dla_signal_cpu_event_context_p_ = tmp_sync_obj_p->getDlaSignalCpuEventContextPtr();
            cuda_signal_dla_sema_p_ = tmp_sync_obj_p->getCudaSignalDlaSemaPtr();
            cuda_wait_dla_sema_p_ = tmp_sync_obj_p->getCudaWaitDlaSemaPtr();
            cuda_signal_dla_sema_param_p_ = tmp_sync_obj_p->getCudaSignalDlaSemaParamPtr();
            cuda_wait_dla_sema_param_p_ = tmp_sync_obj_p->getCudaWaitDlaSemaParamPtr();

            dla_sync_cpu_task_.moduleHandle = module_handle_;
            dla_sync_cpu_task_.outputTensor = dla_outputs_p_vec_.data(); //should a ** -- theoretically is an array of buffer
            dla_sync_cpu_task_.numOutputTensors = outputs_num_;
            dla_sync_cpu_task_.numInputTensors = inputs_num_;
            dla_sync_cpu_task_.inputTensor = dla_inputs_p_vec_.data(); //should a ** -- theoretically is an array of buffer
            dla_sync_cpu_task_.waitEvents = dla_wait_cpu_events_p_;
            dla_sync_cpu_task_.signalEvents = dla_signal_cpu_events_p_;

            dla_sync_gpu_task_.moduleHandle = module_handle_;
            dla_sync_gpu_task_.outputTensor = dla_outputs_p_vec_.data(); //should a ** -- theoretically is an array of buffer
            dla_sync_gpu_task_.numOutputTensors = outputs_num_;
            dla_sync_gpu_task_.numInputTensors = inputs_num_;
            dla_sync_gpu_task_.inputTensor = dla_inputs_p_vec_.data(); //should a ** -- theoretically is an array of buffer
            dla_sync_gpu_task_.waitEvents = dla_wait_cuda_events_p_;
            dla_sync_gpu_task_.signalEvents = dla_signal_cuda_events_p_;
#endif
        }
        else{
            runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
            engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engine_file_data_.data(), engine_file_data_.size()));
            if (engine_ == nullptr){
                std::cout<< "eng: " << engine_name_ <<" failed to deserialize nv tensorRT engine"<<std::endl;
                return false;
            }
            context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
            if (context_ == nullptr){
                std::cout<< "eng: " << engine_name_ <<" failed to create nv TensorRT context"<<std::endl;
                return false;
            }
            engine_file_data_.resize(0);

            if (inputs_num_ != input_buffers_.size()){
                std::cout<< "eng: " << engine_name_ <<" has "<<inputs_num_<<" input tensors but get "<<input_buffers_.size()<<std::endl;
                return false;
            }
            if (outputs_num_ != output_buffers_.size()){
                std::cout<< "eng: " << engine_name_ <<" has "<<outputs_num_<<" output tensors but get "<<output_buffers_.size()<<std::endl;
                return false;
            }
            if (inputs_num_ + outputs_num_ != engine_->getNbIOTensors()){
                std::cout<< "eng: " << engine_name_ <<" inconsist for IO tensor number "<<std::endl;
                return false;
            }
            for (int i = 0; i < inputs_num_; i++) context_->setInputTensorAddress(engine_->getIOTensorName(i),
                                                                getBufferPtr(engine_->getIOTensorName(i))->getDataPtr<void>());
#ifdef x86_64
            for (int i = 0; i < outputs_num_; i++) context_->setOutputTensorAddress(engine_->getIOTensorName(i + inputs_num_), 
                                                                getBufferPtr(engine_->getIOTensorName(i + inputs_num_))->getDataPtr<void>());
#elif defined(JETSON_ORIN)
            for (int i = 0; i < outputs_num_; i++) context_->setTensorAddress(engine_->getIOTensorName(i + inputs_num_), 
                                                                getBufferPtr(engine_->getIOTensorName(i + inputs_num_))->getDataPtr<void>());
#else
            std::cout<<"unsupported platform for current"<<std::endl;
            return -2;
#endif        
        }

        return true;
    }
    bool TRTEngine::inferEngineModel(){
        if (on_dla_){
#ifdef JETSON_ORIN
            if (sync_type_ == 1){
                dla_ret_ = cudlaSubmitTask(dev_handle_, &dla_sync_cpu_task_, 1, NULL, 0);
                sci_ret_ = NvSciSyncObjSignal(*dla_wait_cpu_event_obj_p_);
                sci_ret_ = NvSciSyncFenceWait(reinterpret_cast<NvSciSyncFence*>(dla_signal_cpu_events_p_->eofFences[0].fence), *dla_signal_cpu_event_context_p_, -1);
            }
            else{
                cuda_ret_ = cudaSignalExternalSemaphoresAsync(cuda_signal_dla_sema_p_, cuda_signal_dla_sema_param_p_, 1, stream_);
                dla_ret_ = cudlaSubmitTask(dev_handle_, &dla_sync_gpu_task_, 1, NULL, 0);
                cuda_ret_ = cudaWaitExternalSemaphoresAsync(cuda_wait_dla_sema_p_, cuda_wait_dla_sema_param_p_, 1, stream_);
            }
#endif
        }
        else{
            context_->enqueueV3(stream_);
            cudaStreamSynchronize(stream_);
        }
        return true;
    }
    bool TRTEngine::testModelInferTime(int repeat_times){
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        for (int i = 0; i < repeat_times; i++){
            cudaEventRecord(start, stream_);
            context_->enqueueV3(stream_);
            cudaEventRecord(end, stream_);
            cudaEventSynchronize(end);
            float cost;
            cudaEventElapsedTime(&cost, start, end);
            std::cout<<"trt model test time: "<<cost<<"ms"<<std::endl;
        }
        cudaStreamSynchronize(stream_);
        return true;
    }
    bool TRTEngine::saveEngineModel(const std::string& model_file_path){
        nvinfer1::IHostMemory* serializedModel = engine_->serialize();
        std::ofstream out_eng_file(model_file_path, std::ios::binary);
        if (!out_eng_file.good()){
            std::cout<<"failed to save engine file in path: "<<std::endl;
            return -2;
        }
        out_eng_file.write(reinterpret_cast<char*>(serializedModel->data()), serializedModel->size());
        out_eng_file.close();
        delete serializedModel;
        return true;
    }
    int64_t TRTEngine::buildEngineModel(const char* const data_p, int64_t start_byte, std::string engine_save_path,
                                    std::string build_type, std::string fallback_type, int dla_id, bool rebuild){
        if (common::fileExist(engine_save_path)){
            std::cout<<"engine file "<<engine_save_path<<" exist"<<std::endl;
            if (!rebuild){
                std::cout<<"no rebuild required, skip the building phase"<<std::endl;
                return 1;
            }
        }
#ifndef JETSON_ORIN
        if (dla_id >= 0){
            std::cout<<"unsupport dla in this platform"<<std::endl;
            return -3;
        }
#endif
        const char* cur_data_p = data_p + start_byte;
        int eng_header_len;
        std::memcpy(&eng_header_len, cur_data_p, 4);
        cur_data_p += 4;
        std::string eng_header_str;
        eng_header_str.resize(eng_header_len);
        std::memcpy(eng_header_str.data(), cur_data_p, eng_header_len);
        cur_data_p += eng_header_len;
        nlohmann::json eng_header = nlohmann::json::parse(eng_header_str);
        int nodes_cfg_len = eng_header.at("node_cfg_len").get<int>();
        int tensors_cfg_len = eng_header.at("tensor_cfg_len").get<int>();
        int constant_data_len = eng_header.at("constant_data_len").get<int>();
std::cout<<"ready engine header"<<std::endl;
std::cout<<eng_header_str<<std::endl;
        std::string nodes_cfg_str;
        std::string tensors_cfg_str;
        std::vector<char> constant_data(constant_data_len);
        nodes_cfg_str.resize(nodes_cfg_len);
        tensors_cfg_str.resize(constant_data_len);
        std::memcpy(nodes_cfg_str.data(), cur_data_p, nodes_cfg_len);
        cur_data_p += nodes_cfg_len;
        std::memcpy(tensors_cfg_str.data(), cur_data_p, tensors_cfg_len);
        cur_data_p += tensors_cfg_len;
        std::vector<nlohmann::json> nodes_cfg = nlohmann::json::parse(nodes_cfg_str);

        std::memcpy(constant_data.data(), cur_data_p, constant_data_len);
        std::vector<nlohmann::json> tensors_cfg = nlohmann::json::parse(tensors_cfg_str);
std::cout<<"ready cfg header"<<std::endl;
std::cout<<cur_data_p + constant_data_len - data_p<<std::endl;
std::cout<<constant_data.size()<<' '<<tensors_cfg.size()<<std::endl;
        char* constant_data_p = constant_data.data();
        std::cout<<reinterpret_cast<float*>(constant_data_p)[0]<<std::endl;
        for (int idx = 0; idx < tensors_cfg.size(); idx++){
            std::string name = tensors_cfg[idx].at("name").get<std::string>();
            std::string data_type = tensors_cfg[idx].at("dtype").get<std::string>();
            std::vector<int> shape = tensors_cfg[idx].at("shape").get<std::vector<int>>();
            int64_t data_length = tensors_cfg[idx].at("byte_length").get<int64_t>();
            memory::CPUBufferPtr tensor_data_buffer_cpu = std::make_shared<memory::CPUBuffer>(name, data_length);
            tensor_data_buffer_cpu->copyFromCPU(constant_data_p, data_length);
            initializer_cpu_buffers_table_.emplace(name, tensor_data_buffer_cpu);
            initializer_tensors_table_.emplace(name, std::make_shared<memory::Tensor>(name, data_type, data_length, shape));
            constant_data_p += data_length;
        }
std::cout<<"ready initializer tensor"<<std::endl;
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger_);
        nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
        nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U);
        if (build_type == "int8"){
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            if (fallback_type == "float16"){
                config->setFlag(nvinfer1::BuilderFlag::kFP16);
            }
            else if (fallback_type != "float32"){
                std::cout << "trt engine fallback type must be float16 or float32, but get "<<fallback_type<<std::endl;
                return -3;
            }
        }
        else if (build_type == "float16"){
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        else if (build_type != "float32"){
            std::cout << "trt engine build type must be int8, float16 or float32, but get "<<build_type<<std::endl;
            return -3;
        }
        if (dla_id == 0 || dla_id == 1){
            config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            config->setDLACore(dla_id);
            config->setEngineCapability(nvinfer1::EngineCapability::kDLA_STANDALONE);
            config->setFlag(nvinfer1::BuilderFlag::kDIRECT_IO);
        }
        tensors_cfg = eng_header.at("input").get<std::vector<nlohmann::json>>();
        std::vector<nvinfer1::ITensor*> inputs(tensors_cfg.size());
        //让输入输出强行是NCHW，图像标准形式，注意检查
        for (int idx = 0; idx < inputs.size(); idx++){
            nlohmann::json tensor_cfg = tensors_cfg[idx];
            std::string name = tensor_cfg.at("name").get<std::string>();
            std::string dtype = tensor_cfg.at("dtype").get<std::string>();
            std::vector<int> shape = tensors_cfg[idx].at("shape").get<std::vector<int>>();
            nvinfer1::DataType data_type = nv_type_table_.at(dtype);
            nvinfer1::Dims4 dims(shape[0], shape[1], shape[2], shape[3]);
            nvinfer1::ITensor* input = network->addInput(name.c_str(), data_type, dims);
            nv_tensor_table_.emplace(name, input);
            input->setDynamicRange(-1.0f, 1.0f);
            if (dla_id == 0 || dla_id == 1){
                input->setAllowedFormats(1U << int(nvinfer1::TensorFormat::kDLA_LINEAR));
                input->setType(nvinfer1::DataType::kHALF);
            }
            else{
                input->setAllowedFormats(1U << int(nvinfer1::TensorFormat::kLINEAR));
            }
        }
std::cout<<"ready input tensor"<<std::endl;
        for (int idx = 0; idx < nodes_cfg.size(); idx++){
            nlohmann::json node_cfg = nodes_cfg[idx];
            addModule(network, node_cfg);
            // if (node_cfg.at("name") == "/upsampling_layer/conv/conv.1/InstanceNormalization") break;
        }
std::cout<<"ready add layer"<<std::endl;
        tensors_cfg = eng_header.at("output").get<std::vector<nlohmann::json>>();
        for (int idx = 0; idx < tensors_cfg.size(); idx++){
            nvinfer1::ITensor* output = nv_tensor_table_.at(tensors_cfg[idx].at("name").get<std::string>());
            if (dla_id == 0 || dla_id == 1){
                output->setAllowedFormats(1U << int(nvinfer1::TensorFormat::kDLA_LINEAR));
                output->setType(nvinfer1::DataType::kHALF);
            }
            else{
                output->setAllowedFormats(1U << int(nvinfer1::TensorFormat::kLINEAR));
            }
            network->markOutput(*output);
        }
        // nvinfer1::ITensor* output = nv_tensor_table_.at("/upsampling_layer/conv/conv.1/InstanceNormalization_output_0");
        // network->markOutput(*output);
std::cout<<"ready output tensor"<<std::endl;

        nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
std::cout<<"ready build serialized"<<std::endl;
        std::ofstream out_eng_file(engine_save_path, std::ios::binary);
        if (!out_eng_file.good()){
            std::cout<<"failed to save engine file in path: "<<std::endl;
            return -2;
        }
        out_eng_file.write(reinterpret_cast<char*>(serializedModel->data()), serializedModel->size());
        out_eng_file.close();

        delete serializedModel;
        delete network;
        delete config;
        delete builder;
        return 0;
    }
    bool TRTEngine::addConstantTensor(nvinfer1::INetworkDefinition* network, std::string tensor_name){
        std::string dtype = initializer_tensors_table_.at(tensor_name)->getElementType();
        std::vector<int> shape = initializer_tensors_table_.at(tensor_name)->getShape();
        nvinfer1::DataType data_type = nv_type_table_.at(dtype);
        int element_num = initializer_tensors_table_.at(tensor_name)->getElementNum();
        nvinfer1::Dims dim;
        dim.nbDims = shape.size();
        for (int i = 0; i < dim.nbDims; i++) dim.d[i] = shape[i];
        nvinfer1::Weights tensor_data{data_type, nullptr, 0};
        tensor_data.count = element_num;
        tensor_data.values = initializer_cpu_buffers_table_.at(tensor_name)->getDataPtr<void>();
        nvinfer1::IConstantLayer* const_layer_p = network->addConstant(dim, tensor_data);
        nvinfer1::ITensor* const_tensor_p = const_layer_p->getOutput(0);
        const_tensor_p->setName(tensor_name.c_str());
        nv_tensor_table_.emplace(tensor_name, const_tensor_p);
        const_tensor_p->setDynamicRange(-1.0f, 1.0f);
        // std::cout<<tensor_name<<' '<<tensor_data.values<<' '<<tensor_data.count<<' '<<dtype
        //         <<' '<<initializer_cpu_buffers_table_.at(tensor_name)->getDataByteSize()<<std::endl;
        return true;
    }
    nvinfer1::ITensor* TRTEngine::internBoardcastShuffle(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* tensor_p, int aim_rank, int start_dim){
        //add dim use 1, start_dim -> first dim in res for input
        nvinfer1::Dims cur_shape = tensor_p->getDimensions();
        nvinfer1::IShuffleLayer* reshape_layer_p = network->addShuffle(*tensor_p);
        nvinfer1::Dims aim_shape;
        aim_shape.nbDims = aim_rank;
        for (int i = 0; i < aim_rank; i++) aim_shape.d[i] = 1;
        for (int i = 0; i < cur_shape.nbDims; i++) aim_shape.d[i + start_dim] = cur_shape.d[i];
        reshape_layer_p->setReshapeDimensions(aim_shape);
        nvinfer1::ITensor* output_tensor_p = reshape_layer_p->getOutput(0);
        output_tensor_p->setDynamicRange(-1.0f, 1.0f);
        return output_tensor_p;
    }
    int64_t TRTEngine::addModule(nvinfer1::INetworkDefinition* network, nlohmann::json node_cfg){
        std::string node_name = node_cfg.at("name").get<std::string>();
        // std::cout<<"add layer "<<node_name<<std::endl;
        std::string node_type = node_cfg.at("type").get<std::string>();
        std::vector<std::string> node_inputs_name = node_cfg.at("input").get<std::vector<std::string>>();
        std::vector<std::string> node_outputs_name = node_cfg.at("output").get<std::vector<std::string>>();
        nlohmann::json node_attr = node_cfg.at("attr").get<nlohmann::json>();
        std::vector<nvinfer1::ITensor*> node_inputs(node_inputs_name.size());
        std::vector<nvinfer1::ITensor*> node_outputs(node_outputs_name.size());
        for (int i = 0; i < node_inputs_name.size(); i++){
            if ((initializer_tensors_table_.count(node_inputs_name[i]) != 0) && (nv_tensor_table_.count(node_inputs_name[i]) == 0)){
                addConstantTensor(network, node_inputs_name[i]);
            }
            node_inputs[i] = nv_tensor_table_.at(node_inputs_name[i]);
            // nvinfer1::Dims dims = node_inputs[i]->getDimensions();
            // for (int j = 0; j < dims.nbDims; j++) std::cout<<dims.d[j]<<' ';
            // std::cout<<std::endl;
        }
        if ((node_type == "Add") ||
                (node_type == "Sub") ||
                (node_type == "Mul") ||
                (node_type == "Div")){
            int dim0 = node_inputs[0]->getDimensions().nbDims;
            int dim1 = node_inputs[1]->getDimensions().nbDims;
            if (dim0 < dim1) node_inputs[0] = internBoardcastShuffle(network, node_inputs[0], dim1, dim1 - dim0);
            else if (dim0 > dim1) node_inputs[1] = internBoardcastShuffle(network, node_inputs[1], dim0, dim0 - dim1);
            else{
                nvinfer1::Dims dims0 = node_inputs[0]->getDimensions();
                nvinfer1::Dims dims1 = node_inputs[1]->getDimensions();
                for (int i = 0; i < dim0; i++){
                    if (dims0.d[i] != dims1.d[i] && dims0.d[i] != 1 && dims1.d[i] != 1){
                        std::cout<<"invalid input shapes for elementwise layer"<<std::endl;
                        return -1;
                    }
                }
            }
            nvinfer1::IElementWiseLayer* element_layer_p = network->addElementWise(*node_inputs[0], *node_inputs[1], nv_elem_op_table_.at(node_type));
            node_outputs[0] = element_layer_p->getOutput(0);
        }
        else if (node_type == "Sqrt"){
            nvinfer1::IUnaryLayer* sqrt_layer_p = network->addUnary(*node_inputs[0], nvinfer1::UnaryOperation::kSQRT);
            node_outputs[0] = sqrt_layer_p->getOutput(0);
        }
        else if (node_type == "Cast"){
#ifdef x86_64
            nvinfer1::ICastLayer* cast_layer_p = network->addCast(*node_inputs[0], nv_type_table_.at(node_attr.at("output_type").get<std::string>()));
            node_outputs[0] = cast_layer_p->getOutput(0);
#elif defined(JETSON_ORIN)
            nvinfer1::IPluginCreator* cast_plugin_creator = getPluginRegistry()->getPluginCreator("CastLayerPlugin_custom", "1");
            std::vector<nvinfer1::PluginField> cast_plugin_field;
            int32_t input_type_id = static_cast<int32_t>(node_inputs[0]->getType());
            std::string output_type_str = node_attr.at("output_type").get<std::string>();
            if (nv_type_table_.count(output_type_str) == 0){
                std::cout << "unsupported output type for cast layer " << output_type_str<<std::endl;
                return -1;
            }
            int32_t output_type_id = static_cast<int32_t>(nv_type_table_.at(output_type_str));
            cast_plugin_field.emplace_back(nvinfer1::PluginField("input_type", &input_type_id, nvinfer1::PluginFieldType::kINT32, 1));
            cast_plugin_field.emplace_back(nvinfer1::PluginField("output_type", &output_type_id, nvinfer1::PluginFieldType::kINT32, 1));
            nvinfer1::PluginFieldCollection cast_plugin_data;
            cast_plugin_data.nbFields = cast_plugin_field.size();
            cast_plugin_data.fields = cast_plugin_field.data();
            nvinfer1::IPluginV2* cast_plugin_obj_p = cast_plugin_creator->createPlugin(node_name.c_str(), &cast_plugin_data);
            nvinfer1::IPluginV2Layer* cast_plugin_layer_p = network->addPluginV2(&node_inputs[0], 1, *cast_plugin_obj_p);
            node_outputs[0] = cast_plugin_layer_p->getOutput(0);
#else
            std::cout<<"unsupported platform for cast layer"<<std::endl;
            return -2;
#endif
        }
        else if (node_type == "Concat"){
            nvinfer1::IConcatenationLayer* concat_layer_p = network->addConcatenation(node_inputs.data(), node_inputs.size());
            concat_layer_p->setAxis(node_attr.at("axis").get<int>());
            node_outputs[0] = concat_layer_p->getOutput(0);
        }
        else if (node_type == "Conv"){
            std::string weight_name = node_attr.at("weight_name").get<std::string>();
            std::string bias_name = node_attr.at("bias_name").get<std::string>();
            std::string weight_type = initializer_tensors_table_.at(weight_name)->getElementType();
            std::string bias_type = initializer_tensors_table_.at(bias_name)->getElementType();
            int out_channel = initializer_tensors_table_.at(weight_name)->getShape()[0];
            int in_channel = initializer_tensors_table_.at(weight_name)->getShape()[1];
            std::vector<int> kernel_size = node_attr.at("kernel_shape").get<std::vector<int>>();
            nvinfer1::Weights weight = {nv_type_table_.at(weight_type), nullptr, 0};
            weight.count = initializer_tensors_table_.at(weight_name)->getElementNum();
            weight.values = initializer_cpu_buffers_table_.at(weight_name)->getDataPtr<void>();

            // std::vector<float> tmp = common::loadDataFromFile<float, float>("/home/dong/WS/FoundationPose/test_conv_weight.bin");
            // for (int i = 0; i < 8; i++) std::cout<<reinterpret_cast<const float*>(weight.values)[i]<<' ';
            // std::cout <<std::endl;
            // for (int i = 0; i < 8; i++) std::cout<<tmp[i]<<' ';
            // std::cout <<std::endl;

            nvinfer1::Weights bias = {nv_type_table_.at(bias_type), nullptr, 0};
            bias.count = initializer_tensors_table_.at(bias_name)->getElementNum();
            bias.values = initializer_cpu_buffers_table_.at(bias_name)->getDataPtr<void>();

            // std::vector<float> tmp_b = common::loadDataFromFile<float, float>("/home/dong/WS/FoundationPose/test_conv_bias.bin");
            // for (int i = 0; i < 8; i++) std::cout<<reinterpret_cast<const float*>(bias.values)[i]<<' ';
            // std::cout <<std::endl;
            // for (int i = 0; i < 8; i++) std::cout<<tmp_b[i]<<' ';
            // std::cout <<std::endl;

            nvinfer1::IConvolutionLayer* conv_layer_p = network->addConvolutionNd(*node_inputs[0], out_channel,
                                                                                    nvinfer1::DimsHW{kernel_size[0], kernel_size[1]},
                                                                                    weight, bias);
            std::vector<int> pads = node_attr.at("pads").get<std::vector<int>>();
            std::vector<int> dilations = node_attr.at("dilations").get<std::vector<int>>();
            std::vector<int> strides = node_attr.at("strides").get<std::vector<int>>();
            conv_layer_p->setDilationNd(nvinfer1::Dims2{dilations[0], dilations[1]});
            conv_layer_p->setStrideNd(nvinfer1::Dims2{strides[0], strides[1]});
            conv_layer_p->setPrePadding(nvinfer1::Dims2{pads[0], pads[2]});
            conv_layer_p->setPostPadding(nvinfer1::Dims2{pads[1], pads[3]});
            conv_layer_p->setNbGroups(node_attr.at("group").get<int>());
            node_outputs[0] = conv_layer_p->getOutput(0);
        }
        else if (node_type == "Relu"){
            nvinfer1::IActivationLayer* relu_layer_p = network->addActivation(*node_inputs[0], nvinfer1::ActivationType::kRELU);
            node_outputs[0] = relu_layer_p->getOutput(0);
        }
        else if (node_type == "Slice"){
#if !defined x86_64 && !defined JETSON_ORIN
                std::cout<<"supporting problem in Slice layer for current trt version & paltform setting"<<std::endl;
                return -1;
#endif
            int axis = node_attr.at("axis").get<int>();
            int start = node_attr.at("start").get<int>();
            int stride = node_attr.at("step").get<int>();
            int size = node_attr.at("end").get<int>() - start;
#ifdef x86_64
            nvinfer1::ISliceLayer* slice_layer_p = network->addSlice(*node_inputs[0], nvinfer1::Dims{1,{start}}, nvinfer1::Dims{1,{size}},
                                                                    nvinfer1::Dims{1,{stride}});
            slice_layer_p->setAxes(nvinfer1::Dims{1,{axis}});
            node_outputs[0] = slice_layer_p->getOutput(0);
#endif
#ifdef JETSON_ORIN
            nvinfer1::Dims input_dim = node_inputs[0]->getDimensions();
            nvinfer1::Dims starts = input_dim;
            nvinfer1::Dims strides = input_dim;
            nvinfer1::Dims sizes = input_dim;
            for (int i = 0; i < input_dim.nbDims; i++){
                starts.d[i] = 0;
                strides.d[i] = 1;
            }
            starts.d[axis] = start;
            strides.d[axis] = stride;
            sizes.d[axis] = size;
            nvinfer1::ISliceLayer* slice_layer_p = network->addSlice(*node_inputs[0], starts, sizes, strides);
            node_outputs[0] = slice_layer_p->getOutput(0);
#endif
        }
        else if (node_type == "Reshape"){
            nvinfer1::IShuffleLayer* reshape_layer_p = network->addShuffle(*node_inputs[0]);
            std::vector<int> attr_shape = node_attr.at("shape").get<std::vector<int>>();
            nvinfer1::Dims shape;
            shape.nbDims = attr_shape.size();
            for (int i = 0; i < shape.nbDims; i++) shape.d[i] = attr_shape[i];
            reshape_layer_p->setReshapeDimensions(shape);
            node_outputs[0] = reshape_layer_p->getOutput(0);
        }
        else if (node_type == "Transpose"){
            nvinfer1::IShuffleLayer* tranpose_layer_p = network->addShuffle(*node_inputs[0]);
            std::vector<int> sequences = node_attr.at("sequences").get<std::vector<int>>();
            nvinfer1::Permutation seq;
            for (int i = 0; i < sequences.size(); i++) seq.order[i] = sequences[i];
            tranpose_layer_p->setFirstTranspose(seq);
            node_outputs[0] = tranpose_layer_p->getOutput(0);
        }
        else if (node_type == "MatMul"){
            int dim0 = node_inputs[0]->getDimensions().nbDims;
            int dim1 = node_inputs[1]->getDimensions().nbDims;
            if (dim0 < dim1) node_inputs[0] = internBoardcastShuffle(network, node_inputs[0], dim1, dim1 - dim0);
            else if (dim0 > dim1) node_inputs[1] = internBoardcastShuffle(network, node_inputs[1], dim0, dim0 - dim1);
            nvinfer1::IMatrixMultiplyLayer* matmul_layer_p = network->addMatrixMultiply(*node_inputs[0], nvinfer1::MatrixOperation::kNONE,
                                                                                        *node_inputs[1], nvinfer1::MatrixOperation::kNONE);
            node_outputs[0] = matmul_layer_p->getOutput(0);
        }
        else if (node_type == "Unsqueeze"){
            nvinfer1::IShuffleLayer* reshape_layer_p = network->addShuffle(*node_inputs[0]);
            int axis = node_attr.at("axis").get<int>();
            nvinfer1::Dims shape = node_inputs[0]->getDimensions();
            shape.nbDims += 1;
            for (int i = shape.nbDims - 1; i > axis; i--) shape.d[i] = shape.d[i - 1];
            shape.d[axis] = 1;
            reshape_layer_p->setReshapeDimensions(shape);
            node_outputs[0] = reshape_layer_p->getOutput(0);
        }
        else if (node_type == "Squeeze"){
            nvinfer1::IShuffleLayer* reshape_layer_p = network->addShuffle(*node_inputs[0]);
            std::vector<int> axes = node_attr.at("axes").get<std::vector<int>>();
            nvinfer1::Dims in_shape = node_inputs[0]->getDimensions();
            nvinfer1::Dims out_shape;
            out_shape.nbDims = in_shape.nbDims - axes.size();
            int out_shape_idx = 0;
            for (int i = 0; i < axes.size(); i++) in_shape.d[axes[i]] = -1;
            for (int i = 0; i < in_shape.nbDims; i++){
                if (in_shape.d[i] != -1){
                    out_shape.d[out_shape_idx] = in_shape.d[i];
                    out_shape_idx ++;
                }
            }
            reshape_layer_p->setReshapeDimensions(out_shape);
            node_outputs[0] = reshape_layer_p->getOutput(0);
        }
        else if (node_type == "Gather"){
            std::vector<int> indices = node_attr.at("indice").get<std::vector<int>>();
            int axis = node_attr.at("axis").get<int>();
            std::string name = node_name + "indices";
            std::string data_type = "int32";
            std::vector<int> shape = {int(indices.size())};
            int64_t data_length = sizeof(int) * indices.size();
            memory::CPUBufferPtr tensor_data_buffer_cpu = std::make_shared<memory::CPUBuffer>(name, data_length);
            tensor_data_buffer_cpu->copyFromCPU(indices.data(), data_length);
            initializer_cpu_buffers_table_.emplace(name, tensor_data_buffer_cpu);
            initializer_tensors_table_.emplace(name, std::make_shared<memory::Tensor>(name, data_type, data_length, shape));
            addConstantTensor(network, name);
            nvinfer1::IGatherLayer* gather_layer_p = network->addGather(*node_inputs[0], *nv_tensor_table_.at(name), axis);
            node_outputs[0] = gather_layer_p->getOutput(0);
        }
        else if (node_type == "Softmax"){
            int axis = node_attr.at("axis").get<int>();
            nvinfer1::ISoftMaxLayer* softmax_layer_p = network->addSoftMax(*node_inputs[0]);
            softmax_layer_p->setAxes(1U << axis);
            node_outputs[0] = softmax_layer_p->getOutput(0);
        }
        else if (node_type == "Gemm"){
            float alpha = node_attr.at("alpha").get<float>();
            float beta = node_attr.at("beta").get<float>();
            if (alpha > 1.0f + 1e-5 || alpha < 1.0f - 1e-5 || beta > 1.0f + 1e-5 || beta < 1.0f - 1e-5){
                std::cout<<"Gemm will be treated as conv1x1 layer, request alpha, beta = 1 in onnx file"<<std::endl;
                return -1;
            }
            std::string weight_tensor_name = node_attr.at("weight_name").get<std::string>();
            std::string bias_tensor_name = node_attr.at("bias_name").get<std::string>();
            std::string weight_type = initializer_tensors_table_.at(weight_tensor_name)->getElementType();
            int out_channel = initializer_tensors_table_.at(weight_tensor_name)->getShape()[0];
            nvinfer1::Weights weight{nv_type_table_.at(weight_type), nullptr, 0};
            weight.count = initializer_tensors_table_.at(weight_tensor_name)->getElementNum();
            weight.values = initializer_cpu_buffers_table_.at(weight_tensor_name)->getDataPtr<void>();
            std::string bias_type = initializer_tensors_table_.at(bias_tensor_name)->getElementType();
            nvinfer1::Weights bias{nv_type_table_.at(bias_type), nullptr, 0};
            bias.count = initializer_tensors_table_.at(bias_tensor_name)->getElementNum();
            bias.values = initializer_cpu_buffers_table_.at(bias_tensor_name)->getDataPtr<void>();
            nvinfer1::Dims input_shape = node_inputs[0]->getDimensions();
            if (input_shape.nbDims != 2){
                std::cout<<"Gemm input should be M*K and treated as NC*1*1, but get dims is "<<input_shape.nbDims<<std::endl;
                return -1;
            }
            nvinfer1::IShuffleLayer* reshape_input_layer_p = network->addShuffle(*node_inputs[0]);
            reshape_input_layer_p->setReshapeDimensions(nvinfer1::Dims4{input_shape.d[0], input_shape.d[1], 1, 1});
            nvinfer1::ITensor* reshaped_input_p = reshape_input_layer_p->getOutput(0);
            nvinfer1::IConvolutionLayer* conv1x1_layer_p = network->addConvolutionNd(*reshaped_input_p, out_channel, nvinfer1::DimsHW{1,1}, weight, bias);
            nvinfer1::ITensor* conv_output_p = conv1x1_layer_p->getOutput(0);
            nvinfer1::IShuffleLayer* reshape_output_layer_p = network->addShuffle(*conv_output_p);
            reshape_output_layer_p->setReshapeDimensions(nvinfer1::Dims2{input_shape.d[0], out_channel});
            node_outputs[0] = reshape_output_layer_p->getOutput(0);
        }
        else if (node_type == "LayerNormalization"){
            std::string scale_name = node_attr.at("weight_name").get<std::string>();
            std::string bias_name = node_attr.at("bias_name").get<std::string>();
            float eps = node_attr.at("eps").get<float>();
            int axis = node_attr.at("axis").get<int>();
            uint32_t axes_mask = 1U << axis;
#ifdef x86_64            
            addConstantTensor(network, scale_name);
            addConstantTensor(network, bias_name);
            if (node_inputs[0]->getDimensions().nbDims > nv_tensor_table_.at(scale_name)->getDimensions().nbDims){
                nv_tensor_table_.at(scale_name) = internBoardcastShuffle(network, nv_tensor_table_.at(scale_name), node_inputs[0]->getDimensions().nbDims,
                                                        node_inputs[0]->getDimensions().nbDims - nv_tensor_table_.at(scale_name)->getDimensions().nbDims);
            }
            if (node_inputs[0]->getDimensions().nbDims > nv_tensor_table_.at(bias_name)->getDimensions().nbDims){
                nv_tensor_table_.at(bias_name) = internBoardcastShuffle(network, nv_tensor_table_.at(bias_name), node_inputs[0]->getDimensions().nbDims,
                                                        node_inputs[0]->getDimensions().nbDims - nv_tensor_table_.at(bias_name)->getDimensions().nbDims);
            }
            nvinfer1::INormalizationLayer* layernorm_layer_p = network->addNormalization(*node_inputs[0],
                                                                                        *nv_tensor_table_.at(scale_name),
                                                                                        *nv_tensor_table_.at(bias_name),
                                                                                        axes_mask);
            layernorm_layer_p->setEpsilon(eps);
            node_outputs[0] = layernorm_layer_p->getOutput(0);
#elif defined(JETSON_ORIN)
            nvinfer1::IPluginCreator* axes_norm_plugin_creator = getPluginRegistry()->getPluginCreator("AxesNormalizationPlugin_custom", "1");
            std::vector<nvinfer1::PluginField> axes_norm_plugin_field;
            int32_t input_type_id = static_cast<int32_t>(node_inputs[0]->getType());
            axes_norm_plugin_field.emplace_back(nvinfer1::PluginField("data_type", &input_type_id, nvinfer1::PluginFieldType::kINT32, 1));
            axes_norm_plugin_field.emplace_back(nvinfer1::PluginField("axes_mask", &axes_mask, nvinfer1::PluginFieldType::kINT32, 1));
            axes_norm_plugin_field.emplace_back(nvinfer1::PluginField("eps", &eps, nvinfer1::PluginFieldType::kFLOAT32, 1));
            int scale_ele_num = initializer_tensors_table_.at(scale_name)->getElementNum();
            int bias_ele_num = initializer_tensors_table_.at(bias_name)->getElementNum();
            void* scale_data_p = initializer_cpu_buffers_table_.at(scale_name)->getDataPtr<void>();
            void* bias_data_p = initializer_cpu_buffers_table_.at(bias_name)->getDataPtr<void>();
            axes_norm_plugin_field.emplace_back(nvinfer1::PluginField("scales", scale_data_p, nvinfer1::PluginFieldType::kFLOAT32, scale_ele_num));
            axes_norm_plugin_field.emplace_back(nvinfer1::PluginField("bias", bias_data_p, nvinfer1::PluginFieldType::kFLOAT32, bias_ele_num));
            nvinfer1::PluginFieldCollection axes_norm_plugin_data;
            axes_norm_plugin_data.nbFields = axes_norm_plugin_field.size();
            axes_norm_plugin_data.fields = axes_norm_plugin_field.data();
            nvinfer1::IPluginV2* axes_norm_plugin_obj_p = axes_norm_plugin_creator->createPlugin(node_name.c_str(), &axes_norm_plugin_data);
            nvinfer1::IPluginV2Layer* instance_norm_plugin_layer_p = network->addPluginV2(&node_inputs[0], 1, *axes_norm_plugin_obj_p);
            node_outputs[0] = instance_norm_plugin_layer_p->getOutput(0);
#else
            std::cout<<"unsupported platform for cast layer"<<std::endl;
            return -2;
#endif
        }
        else if (node_type == "ReduceMean"){
            std::vector<int> axes = node_attr.at("axes").get<std::vector<int>>();
            bool keep_dims = static_cast<bool>(node_attr.at("keepdims").get<int>());
            uint32_t axes_mask = 0x00000000;
            for (int i = 0; i < axes.size(); i++) axes_mask |= (1U << axes[i]);
            nvinfer1::IReduceLayer* reduce_mean_layer_p = network->addReduce(*node_inputs[0], nvinfer1::ReduceOperation::kAVG, axes_mask, keep_dims);
            node_outputs[0] = reduce_mean_layer_p->getOutput(0);
        }
        else if (node_type == "MaxPool"){
            int ceil_mode = node_attr.at("ceil_mode").get<int>();
            std::vector<int> kernel_shape = node_attr.at("kernel_shape").get<std::vector<int>>();
            std::vector<int> pads = node_attr.at("pads").get<std::vector<int>>();
            std::vector<int> strides = node_attr.at("strides").get<std::vector<int>>();
            int dim = kernel_shape.size();
            nvinfer1::Dims kernel_shape_dim = {dim, {}};
            nvinfer1::Dims pre_pads_dim = {dim, {}};
            nvinfer1::Dims post_pads_dim = {dim, {}};
            nvinfer1::Dims strides_dim = {dim, {}};
            for (int i = 0; i < dim; i++){
                kernel_shape_dim.d[i] = kernel_shape[i];
                pre_pads_dim.d[i] = pads[i*2];
                post_pads_dim.d[i] = pads[i*2 + 1];
                strides_dim.d[i] = strides[i];
            }
            nvinfer1::PaddingMode nv_pad_mod;
            if (ceil_mode == 1) nv_pad_mod = nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP;
            else if (ceil_mode == 0) nv_pad_mod = nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN;
            else{
                std::cout<<"unknown ceil mode in pooling layer"<<std::endl;
                return -1;
            }
            nvinfer1::IPoolingLayer* max_pool_layer_p = network->addPoolingNd(*node_inputs[0], nvinfer1::PoolingType::kMAX, kernel_shape_dim);
            max_pool_layer_p->setPaddingMode(nv_pad_mod);
            max_pool_layer_p->setPrePadding(pre_pads_dim);
            max_pool_layer_p->setPostPadding(post_pads_dim);
            max_pool_layer_p->setStrideNd(strides_dim);
            node_outputs[0] = max_pool_layer_p->getOutput(0);
        }
        else if (node_type == "Resize"){
            std::string coord_trans_mode = node_attr.at("coordinate_transformation_mode").get<std::string>();
            std::string interpolation_mode = node_attr.at("mode").get<std::string>();
            std::string nearest_mode = node_attr.at("nearest_mode").get<std::string>();
            float cubic_coeff = node_attr.at("cubic_coeff_a").get<float>();
            std::vector<float> scales = node_attr.at("scales").get<std::vector<float>>();
            nvinfer1::IResizeLayer* resize_layer_p = network->addResize(*node_inputs[0]);
            nvinfer1::ResizeCoordinateTransformation nv_coord_trans_mode;
            if (coord_trans_mode == "half_pixel") nv_coord_trans_mode = nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL;
            else{
                std::cout<<"unknown coordinate transformation mode for resize layer yet"<<std::endl;
                return -1;
            }
            nvinfer1::ResizeRoundMode nv_round_mode;
            if (nearest_mode == "floor") nv_round_mode = nvinfer1::ResizeRoundMode::kFLOOR;
            else{
                std::cout<<"unknown round mode for resize layer yet"<<std::endl;
                return -1;
            }
            nvinfer1::InterpolationMode nv_interpolation_mode;
            if (interpolation_mode == "linear") nv_interpolation_mode = nvinfer1::InterpolationMode::kLINEAR;
            else{
                std::cout<<"unknown interpolation mode for resize layer yet"<<std::endl;
                return -1;
            }
            resize_layer_p->setCoordinateTransformation(nv_coord_trans_mode);
            resize_layer_p->setResizeMode(nv_interpolation_mode);
            resize_layer_p->setNearestRounding(nv_round_mode);
            resize_layer_p->setCubicCoeff(cubic_coeff);
            resize_layer_p->setScales(scales.data(), scales.size());
            node_outputs[0] = resize_layer_p->getOutput(0);
        }
        else if (node_type == "InstanceNormalization"){
            std::string scale_name = node_attr.at("weight_name").get<std::string>();
            std::string bias_name = node_attr.at("bias_name").get<std::string>();
            float eps = node_attr.at("eps").get<float>();
            uint32_t axes_mask = 0;
            //suppose is BC... 2-last dims will do norm
            int input_shape = node_inputs[0]->getDimensions().nbDims;
            for (int i = 2; i < input_shape; i++) axes_mask |= 1U << i;
#ifdef x86_64
            addConstantTensor(network, scale_name);
            addConstantTensor(network, bias_name);
            if (node_inputs[0]->getDimensions().nbDims > nv_tensor_table_.at(scale_name)->getDimensions().nbDims){
                nv_tensor_table_.at(scale_name) = internBoardcastShuffle(network, nv_tensor_table_.at(scale_name), node_inputs[0]->getDimensions().nbDims, 1);
            }
            if (node_inputs[0]->getDimensions().nbDims > nv_tensor_table_.at(bias_name)->getDimensions().nbDims){
                nv_tensor_table_.at(bias_name) = internBoardcastShuffle(network, nv_tensor_table_.at(bias_name), node_inputs[0]->getDimensions().nbDims, 1);
            }
            nvinfer1::INormalizationLayer* layernorm_layer_p = network->addNormalization(*node_inputs[0],
                                                                                        *nv_tensor_table_.at(scale_name),
                                                                                        *nv_tensor_table_.at(bias_name),
                                                                                        axes_mask);
            layernorm_layer_p->setEpsilon(eps);
            node_outputs[0] = layernorm_layer_p->getOutput(0);
#elif defined(JETSON_ORIN)
            nvinfer1::IPluginCreator* axes_norm_plugin_creator = getPluginRegistry()->getPluginCreator("AxesNormalizationPlugin_custom", "1");
            std::vector<nvinfer1::PluginField> axes_norm_plugin_field;
            int32_t input_type_id = static_cast<int32_t>(node_inputs[0]->getType());
            axes_norm_plugin_field.emplace_back(nvinfer1::PluginField("data_type", &input_type_id, nvinfer1::PluginFieldType::kINT32, 1));
            axes_norm_plugin_field.emplace_back(nvinfer1::PluginField("axes_mask", &axes_mask, nvinfer1::PluginFieldType::kINT32, 1));
            axes_norm_plugin_field.emplace_back(nvinfer1::PluginField("eps", &eps, nvinfer1::PluginFieldType::kFLOAT32, 1));
            int scale_ele_num = initializer_tensors_table_.at(scale_name)->getElementNum();
            int bias_ele_num = initializer_tensors_table_.at(bias_name)->getElementNum();
            void* scale_data_p = initializer_cpu_buffers_table_.at(scale_name)->getDataPtr<void>();
            void* bias_data_p = initializer_cpu_buffers_table_.at(bias_name)->getDataPtr<void>();
            axes_norm_plugin_field.emplace_back(nvinfer1::PluginField("scales", scale_data_p, nvinfer1::PluginFieldType::kFLOAT32, scale_ele_num));
            axes_norm_plugin_field.emplace_back(nvinfer1::PluginField("bias", bias_data_p, nvinfer1::PluginFieldType::kFLOAT32, bias_ele_num));
            nvinfer1::PluginFieldCollection axes_norm_plugin_data;
            axes_norm_plugin_data.nbFields = axes_norm_plugin_field.size();
            axes_norm_plugin_data.fields = axes_norm_plugin_field.data();
            nvinfer1::IPluginV2* axes_norm_plugin_obj_p = axes_norm_plugin_creator->createPlugin(node_name.c_str(), &axes_norm_plugin_data);
            nvinfer1::IPluginV2Layer* instance_norm_plugin_layer_p = network->addPluginV2(&node_inputs[0], 1, *axes_norm_plugin_obj_p);
            node_outputs[0] = instance_norm_plugin_layer_p->getOutput(0);
#else
            std::cout<<"unsupported platform for cast layer"<<std::endl;
            return -2;
#endif
        }
        else{
            std::cout<<"unsupported node type yet for "<<node_type<<std::endl;
            return -1;
        }
        for (int i = 0; i < node_outputs_name.size(); i++){
            node_outputs[i]->setDynamicRange(-1.0f, 1.0f);
            node_outputs[i]->setName(node_outputs_name[i].c_str());
            nv_tensor_table_.emplace(node_outputs_name[i], node_outputs[i]);
            // nvinfer1::Dims dim = node_outputs[i]->getDimensions();
            // std::cout<<node_name<<", "<<node_outputs_name[i]<<": "<<dim.nbDims<<std::endl;
            // for (int j = 0; j < dim.nbDims; j++) std::cout<<dim.d[j]<<' ';
            // std::cout<<std::endl;
        }
        return 0;
    }

    bool TRTEngine::setInputsShape(std::vector<std::vector<int64_t>> &inputs_shape){
        inputs_shape_ = inputs_shape;
        return true;
    }
    bool TRTEngine::setOutputsShape(std::vector<std::vector<int64_t>> &outputs_shape){
        outputs_shape_ = outputs_shape;
        return true;
    }

    bool TRTEngine::loadEngineModel(const std::string &model_file_path, 
                                    const bool &on_dla, const char &sync_type, const int &device_id){
        on_dla_ = on_dla;
        sync_type_ = sync_type;
        device_id_ = device_id;
#ifndef JETSON_ORIN
        if (on_dla_){
            std::cout<<"unsupport dla in this platform"<<std::endl;
            return -3;
        }
#endif
        if (on_dla_ && sync_type_ == 0) std::cout<<"engine on gpu only support sync type 0(gpu-cpu)"<<std::endl;
        if (!on_dla_ && (sync_type_ == 1 || sync_type_ == 2)) std::cout<<"engine on dla only support sync type 1(dla-cpu) or 2(dla-gpu)"<<std::endl;
        std::ifstream eng_file(model_file_path, std::ios::binary);
        if (!eng_file.good()){
            std::cout<< "eng: " << engine_name_ <<" engine file path wrong: "<<model_file_path<<" no such file"<<std::endl;
            return false;
        }
        eng_file.seekg(0, eng_file.end);
        int64_t end_pos = eng_file.tellg();
        eng_file.seekg(0, eng_file.beg);
        int64_t beg_pos = eng_file.tellg();
        int64_t file_length = end_pos - beg_pos;
        engine_file_data_.resize(file_length);
        eng_file.read(engine_file_data_.data(), file_length);
        eng_file.close();
        return true;
    }
    bool TRTEngine::waitSignalAndLaunchEngine(){
        if (on_dla_){
#ifdef JETSON_ORIN
            if (sync_type_ == 1){
                dla_ret_ = cudlaSubmitTask(dev_handle_, &dla_sync_cpu_task_, 1, NULL, 0);
                sci_ret_ = NvSciSyncObjSignal(*dla_wait_cpu_event_obj_p_);
            }
            else{
                cuda_ret_ = cudaSignalExternalSemaphoresAsync(cuda_signal_dla_sema_p_, cuda_signal_dla_sema_param_p_, 1, stream_);
                dla_ret_ = cudlaSubmitTask(dev_handle_, &dla_sync_gpu_task_, 1, NULL, 0);
            }
#endif
        }
        else{
            context_->enqueueV3(stream_);
        }
        return true;
    }
    bool TRTEngine::syncAfterLaunchEngine(){
        if (on_dla_){
#ifdef JETSON_ORIN
            if (sync_type_ == 1){
                sci_ret_ = NvSciSyncFenceWait(reinterpret_cast<NvSciSyncFence*>(dla_signal_cpu_events_p_->eofFences[0].fence), 
                                            *dla_signal_cpu_event_context_p_, -1);
            }
            else{
                cuda_ret_ = cudaWaitExternalSemaphoresAsync(cuda_wait_dla_sema_p_, cuda_wait_dla_sema_param_p_, 1, stream_);
            }
#endif
        }
        else{
            cudaStreamSynchronize(stream_);
        }
        return true;
    }
    bool TRTEngine::updateIOMemPtr(const std::string &name, void* mem_p){
        if (on_dla_){
#ifdef JETSON_ORIN
            //谨慎使用，仅会改变dla-task执行部分的dla mem p，其它的相关p均未改变，需要注意对应关系
            for (int i = 0; i < inputs_num_; i++){
                if (dla_inputs_name_vec_[i] == name){
                    dla_inputs_p_vec_[i] = reinterpret_cast<uint64_t*>(mem_p);
                    return true;
                }
            }
            for (int i = 0; i < outputs_num_; i++){
                if (dla_outputs_name_vec_[i] == name){
                    dla_outputs_p_vec_[i] = reinterpret_cast<uint64_t*>(mem_p);
                    return true;
                }
            }
#endif
        }
        else{
            context_->setTensorAddress(name.c_str(), mem_p);
        }
        return true;
    }

    bool TRTEngine::registed_ = [](){
        std::string type_name = "TRTEngine";
        EngineFactory::registEngineCreator(type_name, 
                                            [](){return BaseEnginePtr(std::make_shared<TRTEngine>());});
        return true;
    }();

}