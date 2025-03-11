#include <iostream>
#include <cuda_runtime_api.h>
#include "cudla_context_hybrid.h"
#include "cudla_context_standalone.h"
#include <NvInfer.h>
#include <string>
#include <cstring>
#include <string.h>
#include <fstream>
#include <vector>
#include "nlohmann/json.hpp"
#include <unordered_map>
#include <cuda_fp16.h>

    bool loadFile(const std::string& file_path, std::vector<char> &file_data){
        std::ifstream file(file_path, std::ios::binary);
        if (!file.good()){
            std::cout <<" engine file path wrong: "<<file_path<<" no such file"<<std::endl;
            return false;
        }
        file.seekg(0, file.end);
        int64_t end_pos = file.tellg();
        file.seekg(0, file.beg);
        int64_t beg_pos = file.tellg();
        int64_t file_length = end_pos - beg_pos;
        file_data.resize(file_length);
        file.read(file_data.data(), file_length);
        file.close();
        return true;
    }

    class Logger : public nvinfer1::ILogger{
    void log(Severity severity, const char* msg) noexcept override{
        if (severity <= Severity::kINFO) std::cout<<msg<<std::endl;
    }
};

template <typename SrcT, typename DstT>
void castData(SrcT* src_p, DstT* dst_p, int length){
    for (int i = 0; i < length; i++) dst_p[i] = static_cast<DstT>(src_p[i]);
}

int main(){
    std::unordered_map<std::string, std::vector<char>> initializer_cpu_buffers_table;
    std::unordered_map<std::string, std::vector<int>> initializer_tensor_shape;
    const std::unordered_map<std::string, nvinfer1::DataType> nv_type_table = {
                                                            {"float", nvinfer1::DataType::kFLOAT},
                                                            {"float16", nvinfer1::DataType::kHALF},
                                                            {"int8", nvinfer1::DataType::kINT8},
                                                            {"int32", nvinfer1::DataType::kINT32}};
    std::string net_path = "../dla_test.net";
    std::string eng_path = "../dla_test.eng";
    // std::string cfg_path = "../dla_test.json";
    std::string input_data_path = "";
    std::string output_ref_path = "";
    std::string build_type = "float16";
    std::string fallback_type = "float16";
    std::vector<char> net_data;
    std::unordered_map<std::string, int> inputs_byte_length_map;
    std::unordered_map<std::string, int> outputs_byte_length_map;
    std::vector<std::string> inputs_name;
    std::vector<std::string> outputs_name;
    loadFile(net_path, net_data);
    const char* cur_data_p = net_data.data();
    // std::vector<char> cfg_data;
    // loadFile(cfg_path, cfg_data);
    // nlohmann::json 
    int start_byte = 0;
    int eng_header_len;
    std::memcpy(&eng_header_len, cur_data_p, 4);
    cur_data_p += 4;
    std::string eng_header_str;
        eng_header_str.resize(eng_header_len);
        std::memcpy(eng_header_str.data(), cur_data_p, eng_header_len);
        cur_data_p += eng_header_len;
        nlohmann::json eng_header = nlohmann::json::parse(eng_header_str);
    std::cout<<eng_header_str<<std::endl;
    int nodes_cfg_len = eng_header.at("node_cfg_len").get<int>();
        int tensors_cfg_len = eng_header.at("tensor_cfg_len").get<int>();
        int constant_data_len = eng_header.at("constant_data_len").get<int>();
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
        std::vector<nlohmann::json> tensors_cfg = nlohmann::json::parse(tensors_cfg_str);
        std::memcpy(constant_data.data(), cur_data_p, constant_data_len);
        char* constant_data_p = constant_data.data();
        // std::cout<<reinterpret_cast<float*>(constant_data_p)[0]<<std::endl;

        for (int idx = 0; idx < tensors_cfg.size(); idx++){
            std::string name = tensors_cfg[idx].at("name").get<std::string>();
            std::string data_type = tensors_cfg[idx].at("dtype").get<std::string>();
            std::vector<int> shape = tensors_cfg[idx].at("shape").get<std::vector<int>>();
            int64_t data_length = tensors_cfg[idx].at("byte_length").get<int64_t>();
            std::vector<char> tensor_data_buffer_cpu(data_length);
            std::memcpy(tensor_data_buffer_cpu.data(), constant_data_p, data_length);
            // memory::CPUBufferPtr tensor_data_buffer_cpu = std::make_shared<memory::CPUBuffer>(name, data_length);
            // tensor_data_buffer_cpu->copyFromCPU(constant_data_p, data_length);
            initializer_cpu_buffers_table.emplace(name, tensor_data_buffer_cpu);
            // initializer_tensors_table_.emplace(name, std::make_shared<memory::Tensor>(name, data_type, data_length, shape));
            initializer_tensor_shape.emplace(name, shape);
            constant_data_p += data_length;
        }


    Logger logger;
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
        nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
        nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U);
        if (build_type == "int8"){
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            if (fallback_type == "float16"){
                config->setFlag(nvinfer1::BuilderFlag::kFP16); 
            }
            else if (fallback_type != "float32"){
                std::cout<<"trt engine fallback type must be float16 or float32, but get "<<fallback_type<<std::endl;
                return -3;
            }
        }
        else if (build_type == "float16"){
            config->setFlag(nvinfer1::BuilderFlag::kFP16); 
        }
        else if (build_type != "float32"){
            std::cout<<"trt engine fallback type must be int8, float16 or float32, but get "<<fallback_type<<std::endl;
            return -3;
        }

        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(0);
        // config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        config->setEngineCapability(nvinfer1::EngineCapability::kDLA_STANDALONE);
        config->setFlag(nvinfer1::BuilderFlag::kDIRECT_IO);

        tensors_cfg = eng_header.at("input").get<std::vector<nlohmann::json>>();
        std::vector<nvinfer1::ITensor*> inputs(tensors_cfg.size());
        //让输入输出强行是NCHW，图像标准形式，注意检查
        std::unordered_map<std::string, nvinfer1::ITensor*> nv_tensor_table;
        for (int idx = 0; idx < inputs.size(); idx++){
            nlohmann::json tensor_cfg = tensors_cfg[idx];
            std::string name = tensor_cfg.at("name").get<std::string>();
            std::string dtype = tensor_cfg.at("dtype").get<std::string>();
            std::vector<int> shape = tensors_cfg[idx].at("shape").get<std::vector<int>>();
            nvinfer1::DataType data_type = nv_type_table.at(dtype);
            nvinfer1::Dims4 dims(shape[0], shape[1], shape[2], shape[3]);
            nvinfer1::ITensor* input = network->addInput(name.c_str(), data_type, dims);
            nv_tensor_table.emplace(name, input);
            input->setDynamicRange(-1.0f, 1.0f);
            input->setAllowedFormats(1U << int(nvinfer1::TensorFormat::kDLA_LINEAR));//kDLA_HWC4
            //should has a extern switch from config.json
            input->setType(nvinfer1::DataType::kHALF);
            inputs_byte_length_map.emplace(name, tensor_cfg.at("byte_length").get<int>());
            inputs_name.emplace_back(name);
        }

        for (int idx = 0; idx < nodes_cfg.size(); idx++){
            nlohmann::json node_cfg = nodes_cfg[idx];
            // if (node_cfg.at("name") == "/upsampling_layer/conv/conv.1/InstanceNormalization") break;
            std::string node_name = node_cfg.at("name").get<std::string>();
            // std::cout<<"add layer "<<node_name<<std::endl;
            std::string node_type = node_cfg.at("type").get<std::string>();
            std::vector<std::string> node_inputs_name = node_cfg.at("input").get<std::vector<std::string>>();
            std::vector<std::string> node_outputs_name = node_cfg.at("output").get<std::vector<std::string>>();
            nlohmann::json node_attr = node_cfg.at("attr").get<nlohmann::json>();
            std::vector<nvinfer1::ITensor*> node_inputs(node_inputs_name.size());
            std::vector<nvinfer1::ITensor*> node_outputs(node_outputs_name.size());

            for (int i = 0; i < node_inputs.size(); i++){
                node_inputs[i] = nv_tensor_table.at(node_inputs_name[i]);
            }

            std::string weight_name = node_attr.at("weight_name").get<std::string>();
            std::string bias_name = node_attr.at("bias_name").get<std::string>();
            // std::string weight_type = initializer_tensors_table_.at(weight_name)->getElementType();
            // std::string bias_type = initializer_tensors_table_.at(bias_name)->getElementType();
            // int out_channel = initializer_tensors_table_.at(weight_name)->getShape()[0];
            // int in_channel = initializer_tensors_table_.at(weight_name)->getShape()[1];
            std::string weight_type = "float";
            std::string bias_type = "float";
            std::vector<int> weight_shape = initializer_tensor_shape.at(weight_name);
            int weight_num = 1;
            for (int i = 0; i < weight_shape.size(); i++) weight_num *= weight_shape[i];
            int out_channel = weight_shape[0];
            int in_channel = weight_shape[1];
            std::vector<int> kernel_size = node_attr.at("kernel_shape").get<std::vector<int>>();
            nvinfer1::Weights weight = {nv_type_table.at(weight_type), nullptr, 0};
            weight.count = weight_num;
            weight.values = initializer_cpu_buffers_table.at(weight_name).data();

            std::vector<int> bias_shape = initializer_tensor_shape.at(bias_name);
            int bias_num = 1;
            for (int i = 0; i < bias_shape.size(); i++) bias_num *= bias_shape[i];
            nvinfer1::Weights bias = {nv_type_table.at(bias_type), nullptr, 0};
            bias.count = bias_num;
            bias.values = initializer_cpu_buffers_table.at(bias_name).data();

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
            
            for (int i = 0; i < node_outputs.size(); i++){
                node_outputs[i]->setDynamicRange(-1.0f, 1.0f);
                node_outputs[i]->setName(node_outputs_name[i].c_str());
                nv_tensor_table.emplace(node_outputs_name[i], node_outputs[i]);
            }
        }

        tensors_cfg = eng_header.at("output").get<std::vector<nlohmann::json>>();
        for (int idx = 0; idx < tensors_cfg.size(); idx++){
            nvinfer1::ITensor* output = nv_tensor_table.at(tensors_cfg[idx].at("name").get<std::string>());
            // output->setAllowedFormats(1U << int(nvinfer1::TensorFormat::kLINEAR));
            output->setAllowedFormats(1U << int(nvinfer1::TensorFormat::kDLA_LINEAR));
            output->setType(nvinfer1::DataType::kHALF);
            network->markOutput(*output);
            std::string name = tensors_cfg[idx].at("name").get<std::string>();
            outputs_byte_length_map.emplace(name, tensors_cfg[idx].at("byte_length").get<int>());
            outputs_name.emplace_back(name);
            nvinfer1::Dims output_shape = output->getDimensions();
            for (int j = 0; j < output_shape.nbDims; j++) std::cout<<output_shape.d[j]<<' ';std::cout<<std::endl;
        }

        nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);

        std::ofstream out_eng_file(eng_path, std::ios::binary);
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


// cuDLAContextStandalone test("../dla_test.eng");
    cudaError cuda_ret;
    cudlaDevHandle dev_handle;
    cudlaStatus dla_ret;
    dla_ret = cudlaCreateDevice(0, &dev_handle, CUDLA_STANDALONE);
    std::vector<char> model_data;
    loadFile(eng_path, model_data);
    cudlaModule module_handle;
    dla_ret = cudlaModuleLoadFromMemory(dev_handle, reinterpret_cast<uint8_t*>(model_data.data()), model_data.size(), &module_handle, 0);
    if (dla_ret != cudlaSuccess){
        std::cout<<"failed load dla model to dla device"<<std::endl;
    }

    cudlaModuleAttribute dla_model_attr;
    dla_ret = cudlaModuleGetAttributes(module_handle, CUDLA_NUM_INPUT_TENSORS, &dla_model_attr);
    dla_ret = cudlaModuleGetAttributes(module_handle, CUDLA_NUM_OUTPUT_TENSORS, &dla_model_attr);
    int num_dla_input = dla_model_attr.numInputTensors;
    int num_dla_output = dla_model_attr.numOutputTensors;
    std::cout<<num_dla_input<<' '<<num_dla_output<<std::endl;
    std::vector<cudlaModuleTensorDescriptor> dla_input_tensor_desc(num_dla_input);
    std::vector<cudlaModuleTensorDescriptor> dla_output_tensor_desc(num_dla_output);
    dla_model_attr.inputTensorDesc = dla_input_tensor_desc.data();
    dla_ret = cudlaModuleGetAttributes(module_handle, CUDLA_INPUT_TENSOR_DESCRIPTORS, &dla_model_attr); 
    dla_model_attr.outputTensorDesc = dla_output_tensor_desc.data();
    dla_ret = cudlaModuleGetAttributes(module_handle, CUDLA_OUTPUT_TENSOR_DESCRIPTORS, &dla_model_attr);
    if (dla_ret != cudlaSuccess){
        std::cout<<"failed to get dla model attr info"<<std::endl;
    } std::cout<<"finish to get dla model attr info"<<std::endl;
    
    NvSciBufModule sci_buf_module = nullptr;
    NvSciError sci_ret;
    sci_ret = NvSciBufModuleOpen(&sci_buf_module);
    NvSciBufObj input_buf;//buf allocated natively for dla?
    uint64_t input_byte_size = inputs_byte_length_map.at(inputs_name[0]);
    NvSciBufAttrList input_unreconciled_attr_list;
    NvSciBufAttrList input_reconciled_attr_list;
    NvSciBufAttrList input_conflict_attr_list;
    NvSciBufType              input_bufType         = NvSciBufType_RawBuffer;
    uint64_t input_bufSize = input_byte_size;
    uint64_t input_align = 128;//align to 128 byte
    bool                      input_cpu_access_flag = true;
    NvSciBufAttrValAccessPerm input_perm            = NvSciBufAccessPerm_ReadWrite;
    CUuuid   input_uuid;
    CUresult input_cuda_ret = cuDeviceGetUuid(&input_uuid, 0);
    int input_num_attr = 6;
    NvSciBufAttrKeyValuePair input_raw_buf_attrs[input_num_attr] = {
        {NvSciBufGeneralAttrKey_Types, &input_bufType, sizeof(input_bufType)},
        {NvSciBufRawBufferAttrKey_Size, &input_bufSize, sizeof(input_bufSize)},
        {NvSciBufRawBufferAttrKey_Align, &input_align, sizeof(input_align)},
        {NvSciBufGeneralAttrKey_NeedCpuAccess, &input_cpu_access_flag, sizeof(input_cpu_access_flag)},//allow cpu access
        {NvSciBufGeneralAttrKey_RequiredPerm, &input_perm, sizeof(input_perm)},
        {NvSciBufGeneralAttrKey_GpuId, &input_uuid, sizeof(input_uuid)},//for gpu must have
    };
    if (sci_ret != NvSciError_Success){
        std::cout<<"failed to set input attr info"<<std::endl;
    } std::cout<<"finish to set input attr info"<<std::endl;

    sci_ret = NvSciBufAttrListCreate(sci_buf_module, &input_unreconciled_attr_list);
    sci_ret = NvSciBufAttrListSetAttrs(input_unreconciled_attr_list, input_raw_buf_attrs, input_num_attr);
    sci_ret = NvSciBufAttrListReconcile(&input_unreconciled_attr_list, 1, &input_reconciled_attr_list, &input_conflict_attr_list);
    sci_ret = NvSciBufObjAlloc(input_reconciled_attr_list, &input_buf);
    void* input_buf_cpu_p;
    sci_ret = NvSciBufObjGetCpuPtr(input_buf, &input_buf_cpu_p);
    if (sci_ret != NvSciError_Success){
        std::cout<<"failed to alloc input buf"<<std::endl;
    } std::cout<<"finish to alloc input buf"<<std::endl;

    cudlaExternalMemoryHandleDesc cudla_input_ext_mem_desc;
    std::memset(&cudla_input_ext_mem_desc, 0, sizeof(cudlaExternalMemoryHandleDesc));
    uint64_t* dla_input_buf_p; //this can be accessed by cpu
    cudla_input_ext_mem_desc.extBufObject = (void*)input_buf;
    cudla_input_ext_mem_desc.size         = dla_input_tensor_desc[0].size;
    dla_ret = cudlaImportExternalMemory(dev_handle, &cudla_input_ext_mem_desc, &dla_input_buf_p, 0);
    if (dla_ret != cudlaSuccess){
        std::cout<<" failed to import dla input mem buf"<<std::endl;
    }
    std::cout<<" finish import dla input mem buf "<<cudla_input_ext_mem_desc.size<<std::endl;
    cudaExternalMemoryHandleDesc cuda_input_ext_mem_handle_desc;
    memset(&cuda_input_ext_mem_handle_desc, 0, sizeof(cuda_input_ext_mem_handle_desc));
    cuda_input_ext_mem_handle_desc.type                  = cudaExternalMemoryHandleTypeNvSciBuf;
    cuda_input_ext_mem_handle_desc.handle.nvSciBufObject = input_buf;
    cuda_input_ext_mem_handle_desc.size                  = cudla_input_ext_mem_desc.size;
    cudaExternalMemory_t cuda_input_ext_mem;
    cudaImportExternalMemory(&cuda_input_ext_mem, &cuda_input_ext_mem_handle_desc);
    cudaExternalMemoryBufferDesc cuda_input_ext_mem_buf_desc;
    memset(&cuda_input_ext_mem_buf_desc, 0, sizeof(cuda_input_ext_mem_buf_desc));
    cuda_input_ext_mem_buf_desc.offset = 0;
    cuda_input_ext_mem_buf_desc.size   = cudla_input_ext_mem_desc.size;
    void* input_buf_gpu_p;
    cudaExternalMemoryGetMappedBuffer(&input_buf_gpu_p, cuda_input_ext_mem, &cuda_input_ext_mem_buf_desc);

    NvSciBufObj output_buf;//buf allocated natively for dla?
    uint64_t output_byte_size = outputs_byte_length_map.at(outputs_name[0]);
    NvSciBufAttrList output_unreconciled_attr_list;
    NvSciBufAttrList output_reconciled_attr_list;
    NvSciBufAttrList output_conflict_attr_list;
    NvSciBufType              output_bufType         = NvSciBufType_RawBuffer;
    uint64_t output_bufSize = output_byte_size;
    uint64_t output_align = 128;//align to 128 byte
    bool                      output_cpu_access_flag = true;
    NvSciBufAttrValAccessPerm output_perm            = NvSciBufAccessPerm_ReadWrite;
    CUuuid   output_uuid;
    CUresult output_cuda_ret = cuDeviceGetUuid(&output_uuid, 0);
    int output_num_attr = 6;
    NvSciBufAttrKeyValuePair output_raw_buf_attrs[output_num_attr] = {
        {NvSciBufGeneralAttrKey_Types, &output_bufType, sizeof(output_bufType)},
        {NvSciBufRawBufferAttrKey_Size, &output_bufSize, sizeof(output_bufSize)},
        {NvSciBufRawBufferAttrKey_Align, &output_align, sizeof(output_align)},
        {NvSciBufGeneralAttrKey_NeedCpuAccess, &output_cpu_access_flag, sizeof(output_cpu_access_flag)},//allow cpu access
        {NvSciBufGeneralAttrKey_RequiredPerm, &output_perm, sizeof(output_perm)},
        {NvSciBufGeneralAttrKey_GpuId, &output_uuid, sizeof(output_uuid)},//for gpu must have
    };
    if (sci_ret != NvSciError_Success){
        std::cout<<"failed to set output attr info"<<std::endl;
    } std::cout<<"finish to set output attr info"<<std::endl;

    sci_ret = NvSciBufAttrListCreate(sci_buf_module, &output_unreconciled_attr_list);
    sci_ret = NvSciBufAttrListSetAttrs(output_unreconciled_attr_list, output_raw_buf_attrs, output_num_attr);
    sci_ret = NvSciBufAttrListReconcile(&output_unreconciled_attr_list, 1, &output_reconciled_attr_list, &output_conflict_attr_list);
    sci_ret = NvSciBufObjAlloc(output_reconciled_attr_list, &output_buf);
    void* output_buf_cpu_p;
    sci_ret = NvSciBufObjGetCpuPtr(output_buf, &output_buf_cpu_p);
    if (sci_ret != NvSciError_Success){
        std::cout<<"failed to alloc output buf"<<std::endl;
    } std::cout<<"finish to alloc output buf"<<std::endl;

    cudlaExternalMemoryHandleDesc cudla_output_ext_mem_desc;
    std::memset(&cudla_output_ext_mem_desc, 0, sizeof(cudlaExternalMemoryHandleDesc));
    uint64_t* dla_output_buf_p; //this can be accessed by cpu
    cudla_output_ext_mem_desc.extBufObject = (void*)output_buf;
    cudla_output_ext_mem_desc.size         = dla_output_tensor_desc[0].size;
    dla_ret = cudlaImportExternalMemory(dev_handle, &cudla_output_ext_mem_desc, &dla_output_buf_p, 0);
    if (dla_ret != cudlaSuccess){
        std::cout<<" failed to import dla output mem buf"<<std::endl;
    } std::cout<<" finish import dla output mem buf "<<dla_output_tensor_desc[0].size<<std::endl;

    cudaExternalMemoryHandleDesc cuda_output_ext_mem_handle_desc;
    memset(&cuda_output_ext_mem_handle_desc, 0, sizeof(cuda_output_ext_mem_handle_desc));
    cuda_output_ext_mem_handle_desc.type                  = cudaExternalMemoryHandleTypeNvSciBuf;
    cuda_output_ext_mem_handle_desc.handle.nvSciBufObject = output_buf;
    cuda_output_ext_mem_handle_desc.size                  = cudla_output_ext_mem_desc.size;
    cudaExternalMemory_t cuda_output_ext_mem;
    cudaImportExternalMemory(&cuda_output_ext_mem, &cuda_output_ext_mem_handle_desc);
    cudaExternalMemoryBufferDesc cuda_output_ext_mem_buf_desc;
    memset(&cuda_output_ext_mem_buf_desc, 0, sizeof(cuda_output_ext_mem_buf_desc));
    cuda_output_ext_mem_buf_desc.offset = 0;
    cuda_output_ext_mem_buf_desc.size   = cudla_output_ext_mem_desc.size;
    void* output_buf_gpu_p;
    cudaExternalMemoryGetMappedBuffer(&output_buf_gpu_p, cuda_output_ext_mem, &cuda_output_ext_mem_buf_desc);

    NvSciSyncModule sci_sync_module;
    sci_ret = NvSciSyncModuleOpen(&sci_sync_module);
    NvSciSyncObj dla_wait_event_obj;
    NvSciSyncAttrList wait_event_waiter_attr_list;
    NvSciSyncAttrList wait_event_signaler_attr_list;
    NvSciSyncAttrList wait_event_reconciled_attr_list;
    NvSciSyncAttrList wait_event_conflict_attr_list;
    sci_ret = NvSciSyncAttrListCreate(sci_sync_module, &wait_event_waiter_attr_list);
    sci_ret = NvSciSyncAttrListCreate(sci_sync_module, &wait_event_signaler_attr_list);
    dla_ret = cudlaGetNvSciSyncAttributes(reinterpret_cast<uint64_t*>(wait_event_waiter_attr_list), CUDLA_NVSCISYNC_ATTR_WAIT);

    NvSciSyncObj dla_wait_cuda_event_obj;
    NvSciSyncAttrList wait_cuda_event_waiter_attr_list;
    NvSciSyncAttrList wait_cuda_event_signaler_attr_list;
    NvSciSyncAttrList wait_cuda_event_reconciled_attr_list;
    NvSciSyncAttrList wait_cuda_event_conflict_attr_list;
    sci_ret = NvSciSyncAttrListCreate(sci_sync_module, &wait_cuda_event_waiter_attr_list);
    sci_ret = NvSciSyncAttrListCreate(sci_sync_module, &wait_cuda_event_signaler_attr_list);
    dla_ret = cudlaGetNvSciSyncAttributes(
                reinterpret_cast<uint64_t*>(wait_cuda_event_waiter_attr_list), 
                CUDLA_NVSCISYNC_ATTR_SIGNAL);
    cuda_ret = cudaDeviceGetNvSciSyncAttributes(
                wait_cuda_event_signaler_attr_list, 
                0, 
                cudaNvSciSyncAttrWait); //??? like a bug, inverse setting but is ok
    NvSciSyncAttrList wait_cuda_event_attrs[2] = {wait_cuda_event_signaler_attr_list, wait_cuda_event_waiter_attr_list};
    sci_ret = NvSciSyncAttrListReconcile(wait_cuda_event_attrs, 2, &wait_cuda_event_reconciled_attr_list, &wait_cuda_event_conflict_attr_list);
    sci_ret = NvSciSyncObjAlloc(wait_cuda_event_reconciled_attr_list, &dla_wait_cuda_event_obj);
    if (sci_ret != NvSciError_Success){
        std::cout<<"failed to alloc wait gpu sync obj"<<std::endl;
    } else std::cout<<"finish to alloc wait gpu sync obj"<<std::endl;

    bool cpu_signaler = true;
    NvSciSyncAttrKeyValuePair cpu_signaler_keyValue[2];
    std::memset(cpu_signaler_keyValue, 0, sizeof(cpu_signaler_keyValue));
    NvSciSyncAccessPerm cpu_signaler_perm = NvSciSyncAccessPerm_SignalOnly;
    cpu_signaler_keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    cpu_signaler_keyValue[0].value = (void*)(&cpu_signaler);
    cpu_signaler_keyValue[0].len = sizeof(cpu_signaler);
    cpu_signaler_keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    cpu_signaler_keyValue[1].value = (void*)(&cpu_signaler_perm);
    cpu_signaler_keyValue[1].len = sizeof(cpu_signaler_perm);
    sci_ret = NvSciSyncAttrListSetAttrs(wait_event_signaler_attr_list, cpu_signaler_keyValue, 2);
    NvSciSyncAttrList wait_event_attrs[2] = {wait_event_signaler_attr_list, wait_event_waiter_attr_list};
    sci_ret = NvSciSyncAttrListReconcile(wait_event_attrs, 2, &wait_event_reconciled_attr_list, &wait_event_conflict_attr_list);
    sci_ret = NvSciSyncObjAlloc(wait_event_reconciled_attr_list, &dla_wait_event_obj);
    if (sci_ret != NvSciError_Success){
        std::cout<<"failed to alloc wait cpu sync obj"<<std::endl;
    } else std::cout<<"finish to alloc wait cpu sync obj"<<std::endl;

    NvSciSyncCpuWaitContext dla_signal_event_context;
    NvSciSyncObj dla_signal_event_obj;
    NvSciSyncAttrList signal_event_waiter_attr_list;
    NvSciSyncAttrList signal_event_signaler_attr_list;
    NvSciSyncAttrList signal_event_reconciled_attr_list;
    NvSciSyncAttrList signal_event_conflict_attr_list;
    sci_ret = NvSciSyncAttrListCreate(sci_sync_module, &signal_event_waiter_attr_list);
    sci_ret = NvSciSyncAttrListCreate(sci_sync_module, &signal_event_signaler_attr_list);
    dla_ret = cudlaGetNvSciSyncAttributes(reinterpret_cast<uint64_t*>(signal_event_signaler_attr_list), CUDLA_NVSCISYNC_ATTR_SIGNAL);

    NvSciSyncObj dla_signal_cuda_event_obj;
    NvSciSyncAttrList signal_cuda_event_waiter_attr_list;
    NvSciSyncAttrList signal_cuda_event_signaler_attr_list;
    NvSciSyncAttrList signal_cuda_event_reconciled_attr_list;
    NvSciSyncAttrList signal_cuda_event_conflict_attr_list;
    sci_ret = NvSciSyncAttrListCreate(sci_sync_module, &signal_cuda_event_waiter_attr_list);
    sci_ret = NvSciSyncAttrListCreate(sci_sync_module, &signal_cuda_event_signaler_attr_list);
    dla_ret = cudlaGetNvSciSyncAttributes(reinterpret_cast<uint64_t*>(signal_cuda_event_signaler_attr_list), CUDLA_NVSCISYNC_ATTR_SIGNAL);
    cuda_ret = cudaDeviceGetNvSciSyncAttributes(signal_cuda_event_waiter_attr_list, 0, cudaNvSciSyncAttrWait);
    NvSciSyncAttrList signal_cuda_event_attrs[2] = {signal_cuda_event_signaler_attr_list, signal_cuda_event_waiter_attr_list};
    sci_ret = NvSciSyncAttrListReconcile(signal_cuda_event_attrs, 2, &signal_cuda_event_reconciled_attr_list, &signal_cuda_event_conflict_attr_list);
    sci_ret = NvSciSyncObjAlloc(signal_cuda_event_reconciled_attr_list, &dla_signal_cuda_event_obj);
    if (sci_ret != NvSciError_Success){
        std::cout<<"failed to alloc signal gpu sync obj"<<std::endl;
    } else std::cout<<"finish to alloc signal gpu sync obj"<<std::endl;

    bool cpu_waiter = true;
    NvSciSyncAttrKeyValuePair cpu_waiter_keyValue[2];
    std::memset(cpu_waiter_keyValue, 0, sizeof(cpu_waiter_keyValue));
    NvSciSyncAccessPerm cpu_waiter_perm = NvSciSyncAccessPerm_WaitOnly;
    cpu_waiter_keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    cpu_waiter_keyValue[0].value = (void*)(&cpu_waiter);
    cpu_waiter_keyValue[0].len = sizeof(cpu_waiter);
    cpu_waiter_keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    cpu_waiter_keyValue[1].value = (void*)(&cpu_waiter_perm);
    cpu_waiter_keyValue[1].len = sizeof(cpu_waiter_perm);
    sci_ret = NvSciSyncAttrListSetAttrs(signal_event_waiter_attr_list, cpu_waiter_keyValue, 2);
    NvSciSyncAttrList signal_event_attrs[2] = {signal_event_signaler_attr_list, signal_event_waiter_attr_list};
    sci_ret = NvSciSyncAttrListReconcile(signal_event_attrs, 2, &signal_event_reconciled_attr_list, &signal_event_conflict_attr_list);
    sci_ret = NvSciSyncObjAlloc(signal_event_reconciled_attr_list, &dla_signal_event_obj);
    sci_ret = NvSciSyncCpuWaitContextAlloc(sci_sync_module, &dla_signal_event_context);
    if (sci_ret != NvSciError_Success){
        std::cout<<"failed to alloc signal cpu sync obj"<<std::endl;
    } else std::cout<<"finish to alloc signal cpu sync obj"<<std::endl;

    uint64_t* dla_wait_event_reg_p = nullptr;
    uint64_t* dla_signal_event_reg_p = nullptr;
    cudlaExternalSemaphoreHandleDesc sema_mem_desc = {0};
    std::memset(&sema_mem_desc, 0, sizeof(sema_mem_desc));
    sema_mem_desc.extSyncObject = dla_wait_event_obj;
    dla_ret = cudlaImportExternalSemaphore(dev_handle, &sema_mem_desc, &dla_wait_event_reg_p, 0);
    if (dla_ret != cudlaSuccess){
        std::cout<<" failed to import cpu sync semaphore into dla"<<std::endl;
    } else std::cout<<" finish import cpu sync semaphore into dla "<<std::endl;
    std::memset(&sema_mem_desc, 0, sizeof(sema_mem_desc));
    sema_mem_desc.extSyncObject = dla_signal_event_obj;
    dla_ret = cudlaImportExternalSemaphore(dev_handle, &sema_mem_desc, &dla_signal_event_reg_p, 0);
    if (dla_ret != cudlaSuccess){
        std::cout<<" failed to import cpu sync semaphore into dla"<<std::endl;
    } else std::cout<<" finish import cpu sync semaphore into dla "<<std::endl;
    uint64_t* dla_wait_cuda_event_reg_p = nullptr;
    uint64_t* dla_signal_cuda_event_reg_p = nullptr;
    std::memset(&sema_mem_desc, 0, sizeof(sema_mem_desc));
    sema_mem_desc.extSyncObject = dla_wait_cuda_event_obj;
    dla_ret = cudlaImportExternalSemaphore(dev_handle, &sema_mem_desc, &dla_wait_cuda_event_reg_p, 0);
    if (dla_ret != cudlaSuccess){
        std::cout<<dla_ret<<" failed to import gpu sync semaphore into dla"<<std::endl;
    } else std::cout<<" finish import gpu sync semaphore into dla "<<std::endl;
    std::memset(&sema_mem_desc, 0, sizeof(sema_mem_desc));
    sema_mem_desc.extSyncObject = dla_signal_cuda_event_obj;
    dla_ret = cudlaImportExternalSemaphore(dev_handle, &sema_mem_desc, &dla_signal_cuda_event_reg_p, 0);
    if (dla_ret != cudlaSuccess){
        std::cout<<" failed to import gpu sync semaphore into dla"<<std::endl;
    } else std::cout<<" finish import gpu sync semaphore into dla "<<std::endl;

    cudaExternalSemaphoreHandleDesc cuda_ext_sem_desc;
    memset(&cuda_ext_sem_desc, 0, sizeof(cuda_ext_sem_desc));
    cuda_ext_sem_desc.type                = cudaExternalSemaphoreHandleTypeNvSciSync;
    cuda_ext_sem_desc.handle.nvSciSyncObj = (void *)(dla_wait_cuda_event_obj);
    cudaExternalSemaphore_t cuda_signal_dla_sema;
    cuda_ret = cudaImportExternalSemaphore(&cuda_signal_dla_sema, &cuda_ext_sem_desc);
    if (cuda_ret != cudaSuccess){
        std::cout<<" failed to import sync semaphore into cuda"<<std::endl;
    } else std::cout<<" finish import sync semaphore into cuda "<<std::endl;

    memset(&cuda_ext_sem_desc, 0, sizeof(cuda_ext_sem_desc));
    cuda_ext_sem_desc.type                = cudaExternalSemaphoreHandleTypeNvSciSync;
    cuda_ext_sem_desc.handle.nvSciSyncObj = (void *)(dla_signal_cuda_event_obj);
    cudaExternalSemaphore_t cuda_wait_dla_sema;
    cuda_ret = cudaImportExternalSemaphore(&cuda_wait_dla_sema, &cuda_ext_sem_desc);
    if (cuda_ret != cudaSuccess){
        std::cout<<" failed to import sync semaphore into cuda"<<std::endl;
    } else std::cout<<" finish import sync semaphore into cuda "<<std::endl;

    NvSciSyncFence dla_wait_pre_fence = NvSciSyncFenceInitializer;
    NvSciSyncObjGenerateFence(dla_wait_event_obj, &dla_wait_pre_fence);
    cudlaWaitEvents* dla_wait_events_p;
    dla_wait_events_p = (cudlaWaitEvents*)malloc(sizeof(cudlaWaitEvents));
    dla_wait_events_p->numEvents = 1;
    CudlaFence* dla_wait_pre_fences_p = (CudlaFence*)malloc(dla_wait_events_p->numEvents * sizeof(CudlaFence));
    dla_wait_pre_fences_p[0].fence = &dla_wait_pre_fence;
    dla_wait_pre_fences_p[0].type = CUDLA_NVSCISYNC_FENCE;
    dla_wait_events_p->preFences = dla_wait_pre_fences_p;

    cudlaSignalEvents* dla_signal_events_p;
    dla_signal_events_p = (cudlaSignalEvents*)malloc(sizeof(cudlaSignalEvents));
    dla_signal_events_p->numEvents = 1;
    uint64_t** dla_cpu_devs_p = (uint64_t**)malloc(dla_signal_events_p->numEvents * sizeof(uint64_t*));
    dla_cpu_devs_p[0] = dla_signal_event_reg_p;
    dla_signal_events_p->devPtrs = dla_cpu_devs_p;
    NvSciSyncFence dla_signal_eof_fence = NvSciSyncFenceInitializer;
    dla_signal_events_p->eofFences = (CudlaFence*)malloc(dla_signal_events_p->numEvents * sizeof(CudlaFence));
    dla_signal_events_p->eofFences[0].fence = &dla_signal_eof_fence;
    dla_signal_events_p->eofFences[0].type = CUDLA_NVSCISYNC_FENCE;
    std::cout<<" finish set sync event for dla & cpu"<<std::endl;

    NvSciSyncFence dla_wait_cuda_pre_fence = NvSciSyncFenceInitializer;
    NvSciSyncObjGenerateFence(dla_wait_cuda_event_obj, &dla_wait_cuda_pre_fence);
    cudlaWaitEvents* dla_wait_cuda_events_p;
    dla_wait_cuda_events_p = (cudlaWaitEvents*)malloc(sizeof(cudlaWaitEvents));
    dla_wait_cuda_events_p->numEvents = 1;
    CudlaFence* dla_wait_cuda_pre_fences_p = (CudlaFence*)malloc(dla_wait_cuda_events_p->numEvents * sizeof(CudlaFence));
    dla_wait_cuda_pre_fences_p[0].fence = &dla_wait_cuda_pre_fence;
    dla_wait_cuda_pre_fences_p[0].type = CUDLA_NVSCISYNC_FENCE;
    dla_wait_cuda_events_p->preFences = dla_wait_cuda_pre_fences_p;
    cudaExternalSemaphoreSignalParams cuda_signal_dla_sema_param;
    memset(&cuda_signal_dla_sema_param, 0, sizeof(cuda_signal_dla_sema_param));
    cuda_signal_dla_sema_param.params.nvSciSync.fence = (void *)(&dla_wait_cuda_pre_fence);
    cuda_signal_dla_sema_param.flags                  = 0;
    
    cudlaSignalEvents* dla_signal_cuda_events_p;
    dla_signal_cuda_events_p = (cudlaSignalEvents*)malloc(sizeof(cudlaSignalEvents));
    dla_signal_cuda_events_p->numEvents = 1;
    uint64_t** dla_gpu_devs_p = (uint64_t**)malloc(dla_signal_cuda_events_p->numEvents * sizeof(uint64_t*));
    dla_gpu_devs_p[0] = dla_signal_cuda_event_reg_p;
    dla_signal_cuda_events_p->devPtrs = dla_gpu_devs_p;
    NvSciSyncFence dla_signal_cuda_eof_fence = NvSciSyncFenceInitializer;
    dla_signal_cuda_events_p->eofFences = (CudlaFence*)malloc(dla_signal_cuda_events_p->numEvents * sizeof(CudlaFence));
    dla_signal_cuda_events_p->eofFences[0].fence = &dla_signal_cuda_eof_fence;
    dla_signal_cuda_events_p->eofFences[0].type = CUDLA_NVSCISYNC_FENCE;
    cudaExternalSemaphoreWaitParams cuda_wait_dla_sema_param;
    memset(&cuda_wait_dla_sema_param, 0, sizeof(cuda_wait_dla_sema_param));
    cuda_wait_dla_sema_param.params.nvSciSync.fence = (void *)(&dla_signal_cuda_eof_fence);
    cuda_wait_dla_sema_param.flags                  = 0;
    std::cout<<" finish set sync event for dla & gpu"<<std::endl;


    input_data_path = "../dla_test_input.bin";
    output_ref_path = "../dla_test_output.bin";
    std::vector<char> input_data;
    std::vector<char> output_ref;
    loadFile(input_data_path, input_data);
    loadFile(output_ref_path, output_ref);
    std::vector<half> input_data_f16(4*512*512);
    for (int i = 0; i < 4*512*512; i++) input_data_f16[i] = static_cast<half>(reinterpret_cast<float*>(input_data.data())[i]);
    std::cout<<" finish input data transcript "<<std::endl;

    cudlaTask dla_task;
    dla_task.moduleHandle = module_handle;
    dla_task.outputTensor = &dla_output_buf_p; //should a ** -- theoretically is an array of buffer
    dla_task.numOutputTensors = 1;
    dla_task.numInputTensors = 1;
    dla_task.inputTensor = &dla_input_buf_p; //should a ** -- theoretically is an array of buffer
    dla_task.waitEvents = dla_wait_events_p;
    dla_task.signalEvents = dla_signal_events_p;  

    std::memset(input_buf_cpu_p, 0, sizeof(half)*input_data_f16.size());
    std::memset(output_buf_cpu_p, 0, sizeof(half)*32*16*32);
    std::memcpy(input_buf_cpu_p, input_data_f16.data(), sizeof(half)*input_data_f16.size());
    dla_ret = cudlaSubmitTask(dev_handle, &dla_task, 1, NULL, 0);
    if (dla_ret != cudlaSuccess){
        std::cout<<" failed to submit task to dla"<<std::endl;
    } else std::cout<<" finish submit task to dla "<<std::endl;
    sci_ret = NvSciSyncObjSignal(dla_wait_event_obj);
    sci_ret = NvSciSyncFenceWait(reinterpret_cast<NvSciSyncFence*>(dla_signal_events_p->eofFences[0].fence), dla_signal_event_context, -1);
    if (sci_ret != NvSciError_Success){
        std::cout<<" failed to wait dla signal"<<std::endl;
    } else std::cout<<" finish wait dla signal "<<std::endl;

    int last_dim = 16;
    std::vector<float> max_err_cpu(last_dim, 0.0f);
    std::vector<float> rec_ref_cpu(last_dim, 0.0f);
    std::vector<float> rec_res_cpu(last_dim, 0.0f);
    std::vector<float> ref_sum_cpu(last_dim, 0.0f);
    std::vector<float> err_sum_cpu(last_dim, 0.0f);
    for (int n = 0; n <32*16; n++){
        for (int c = 0; c < last_dim; c++){
            float ref = reinterpret_cast<float*>(output_ref.data())[n*16+c];
            float res = reinterpret_cast<half*>(output_buf_cpu_p)[n*32 + c];
            float err = std::abs(ref - res);
            if (err > max_err_cpu[c]){
                max_err_cpu[c] = err;
                rec_ref_cpu[c] = ref;
                rec_res_cpu[c] = res;
            }
            ref_sum_cpu[c] += std::abs(ref);
            err_sum_cpu[c] += err;
        }
    }
    for (int i = 0; i < 16; i++) std::cout<<i<<": res="<<rec_res_cpu[i]<<", ref="<<rec_ref_cpu[i]<<", max_err="<<max_err_cpu[i]
                                            <<", ref_sum="<<ref_sum_cpu[i]<<", err_sum="<<err_sum_cpu[i]<<std::endl;
    std::cout<< "finish cpu-dla test"<<std::endl;


    dla_task.moduleHandle = module_handle;
    dla_task.outputTensor = &dla_output_buf_p; //should a ** -- theoretically is an array of buffer
    dla_task.numOutputTensors = 1;
    dla_task.numInputTensors = 1;
    dla_task.inputTensor = &dla_input_buf_p; //should a ** -- theoretically is an array of buffer
    dla_task.waitEvents = dla_wait_cuda_events_p;
    dla_task.signalEvents = dla_signal_cuda_events_p;  
for (int loop = 0; loop <= 10; loop++){
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemsetAsync(input_buf_gpu_p, 0, 4*512*512*sizeof(half), stream);
    cudaMemsetAsync(output_buf_gpu_p, 0, 32*16*32*sizeof(half), stream);
    cudaMemcpyAsync(input_buf_gpu_p, input_data_f16.data(), 4*512*512*sizeof(half), cudaMemcpyHostToDevice, stream);
    // cudaStreamSynchronize(stream);
    cuda_ret = cudaSignalExternalSemaphoresAsync(&cuda_signal_dla_sema, &cuda_signal_dla_sema_param, 1, stream);
    if (cuda_ret != cudaSuccess){
        std::cout<<" failed to cuda signal for dla"<<std::endl;
    } else std::cout<<" finish cuda signal for dla "<<std::endl;
    dla_ret = cudlaSubmitTask(dev_handle, &dla_task, 1, NULL, 0);
    if (dla_ret != cudlaSuccess){
        std::cout<<" failed to submit task to dla"<<std::endl;
    } else std::cout<<" finish submit task to dla "<<std::endl;
    cuda_ret = cudaWaitExternalSemaphoresAsync(&cuda_wait_dla_sema, &cuda_wait_dla_sema_param, 1, stream);
    if (cuda_ret != cudaSuccess){
        std::cout<<" failed to cuda wait for dla"<<std::endl;
    } else std::cout<<" finish cuda wait for dla "<<std::endl;

    std::vector<half> output_res(32*16*32);
    cudaMemcpyAsync(output_res.data(), output_buf_gpu_p, sizeof(half)*32*16*32, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::vector<float> max_err(last_dim, 0.0f);
    std::vector<float> rec_ref(last_dim, 0.0f);
    std::vector<float> rec_res(last_dim, 0.0f);
    std::vector<float> ref_sum(last_dim, 0.0f);
    std::vector<float> err_sum(last_dim, 0.0f);
    for (int n = 0; n <32*16; n++){
        for (int c = 0; c < last_dim; c++){
            float ref = reinterpret_cast<float*>(output_ref.data())[n*16+c];
            float res = output_res[n*32 + c];
            float err = std::abs(ref - res);
            if (err > max_err[c]){
                max_err[c] = err;
                rec_ref[c] = ref;
                rec_res[c] = res;
            }
            ref_sum[c] += std::abs(ref);
            err_sum[c] += err;
        }
    }
    for (int i = 0; i < 16; i++) std::cout<<i<<": res="<<rec_res[i]<<", ref="<<rec_ref[i]<<", max_err="<<max_err[i]
                                            <<", ref_sum="<<ref_sum[i]<<", err_sum="<<err_sum[i]<<std::endl;
    std::cout<< "finish gpu-dla test"<<std::endl;
}
    // 
}
