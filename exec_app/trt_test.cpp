#include "common.h"
#include "base_eng.h"
// #include "trt_eng.h"
void getJsonFile(std::string path){
    std::ifstream file(path, std::ios::binary);
    if (!file.good()){
        std::cout<<"invalid file fath: "<<path<<std::endl;
        return;
    } 
    file.seekg(0, file.end);
    int64_t end_byte = file.tellg();
    file.seekg(0, file.beg);
    int64_t beg_byte = file.tellg();
    int64_t file_length = end_byte - beg_byte;
    std::cout<<file_length<<std::endl;
    int eng_header_len;
    file.read(reinterpret_cast<char*>(&eng_header_len), 4);
    std::cout<<eng_header_len<<std::endl;
    file.seekg(4, file.beg);
    int cur_pos = 4;
    std::string eng_header_str;
    eng_header_str.resize(eng_header_len);
    file.read(eng_header_str.data(), eng_header_len);
    std::cout<<eng_header_str<<std::endl;
    cur_pos += eng_header_len;
    file.seekg(cur_pos, file.beg);
    
    nlohmann::json eng_header = nlohmann::json::parse(eng_header_str.c_str());
    int nodes_cfg_len = eng_header.at("node_cfg_len").get<int>();
    int tensors_cfg_len = eng_header.at("tensor_cfg_len").get<int>();
    int serial_data_len = eng_header.at("constant_data_len").get<int>();

    std::string nodes_cfg_str;
    nodes_cfg_str.resize(nodes_cfg_len);
    file.read(nodes_cfg_str.data(), nodes_cfg_len);
    std::vector<nlohmann::json> nodes_cfg = nlohmann::json::parse(nodes_cfg_str.c_str());
    std::cout<<nodes_cfg.size()<<std::endl;
    cur_pos += nodes_cfg_len;
    file.seekg(cur_pos, file.beg);
    
    std::string tensors_cfg_str;
    tensors_cfg_str.resize(tensors_cfg_len);
    file.read(tensors_cfg_str.data(), tensors_cfg_len);
    std::vector<nlohmann::json> tensors_cfg = nlohmann::json::parse(tensors_cfg_str.c_str());
    std::cout<<tensors_cfg.size()<<std::endl;
    cur_pos += tensors_cfg_len;
    file.seekg(cur_pos, file.beg);

    std::vector<char> serial_data(serial_data_len);
    file.read(serial_data.data(), serial_data_len);

    std::cout<<eng_header.at("name").get<std::string>()<<std::endl;
    std::vector<nlohmann::json> inputs = eng_header.at("input").get<std::vector<nlohmann::json>>();
    std::cout<<inputs.size()<<std::endl;
    std::vector<int> inputs_len(inputs.size());
    
    for (int i = 0; i < inputs.size(); i++){
        inputs_len[i] = inputs[i].at("byte_length");
        
    }
    int sum = 0;
    for (int i = 0; i < tensors_cfg.size(); i++) sum += tensors_cfg[i].at("byte_length").get<int>();
    std::cout<<sum<<std::endl;
}

// void test_weight_cg(){
//     engine::Logger logger;
//     nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
//         nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
//         nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U);

//             std::string name_in = "test_input";
//             std::vector<int> shape_in = {1, 16, 40, 60};
//             nvinfer1::Dims4 dims_in(shape_in[0], shape_in[1], shape_in[2], shape_in[3]);
//             nvinfer1::ITensor* input = network->addInput(name_in.c_str(), nvinfer1::DataType::kFLOAT, dims_in);

//         std::vector<int> shape_const = {1, 16, 40, 60};
//         int element_num_const = 16 * 40 * 60;
//         nvinfer1::Dims4 dims_const(shape_const[0], shape_const[1], shape_const[2], shape_const[3]);
//         nvinfer1::Weights tensor_data_const{nvinfer1::DataType::kFLOAT, nullptr, 0};
//         std::vector<float> data_const(element_num_const, 1.0f);
//         void* data_const_gpu = nullptr;
//         cudaMalloc(reinterpret_cast<void**>(&data_const_gpu), sizeof(float)*element_num_const);
//         cudaMemcpy(data_const_gpu, data_const.data(), sizeof(float)*element_num_const, cudaMemcpyHostToDevice);
//         tensor_data_const.count = element_num_const;
//         tensor_data_const.values = data_const.data();
//         // tensor_data_const.values = data_const_gpu;
//         nvinfer1::IConstantLayer* const_layer_p = network->addConstant(dims_const, tensor_data_const);
//         nvinfer1::ITensor* const_tensor_p = const_layer_p->getOutput(0); 

//         nvinfer1::IElementWiseLayer* add_layer_p = network->addElementWise(*input, *const_tensor_p, nvinfer1::ElementWiseOperation::kSUM);
//         nvinfer1::ITensor* add_tensor_p = add_layer_p->getOutput(0);

//         int out_channel = 32;
//         int in_channel = 16;
//         int kernel_volume = 9;
//         int element_num_conv_weight = kernel_volume * in_channel * out_channel;
//         std::vector<float> weight_conv(element_num_conv_weight, 1.0f);
//         void* weight_conv_gpu = nullptr;
//         cudaMalloc(reinterpret_cast<void**>(&weight_conv_gpu), sizeof(float)*element_num_conv_weight);
//         cudaMemcpy(weight_conv_gpu, weight_conv.data(), sizeof(float)*element_num_conv_weight, cudaMemcpyHostToDevice);
//         nvinfer1::Weights weight = {nvinfer1::DataType::kFLOAT, nullptr, 0};
//         weight.count = element_num_conv_weight;
//         // weight.values = weight_conv.data();
//         weight.values = weight_conv_gpu;
//         int element_num_conv_bias = out_channel;
//         std::vector<float> bias_conv(element_num_conv_bias, 1.0f);
//         void* bias_conv_gpu = nullptr;
//         cudaMalloc(reinterpret_cast<void**>(&bias_conv_gpu), sizeof(float)*element_num_conv_bias);
//         cudaMemcpy(bias_conv_gpu, bias_conv.data(), sizeof(float)*element_num_conv_bias, cudaMemcpyHostToDevice);
//         nvinfer1::Weights bias = {nvinfer1::DataType::kFLOAT, nullptr, 0};
//         bias.count = element_num_conv_bias;
//         // bias.values = bias_conv.data();
//         bias.values = bias_conv_gpu;
//         nvinfer1::IConvolutionLayer* conv_layer_p = network->addConvolutionNd(*add_tensor_p, out_channel,
//                                                                             nvinfer1::DimsHW{3, 3},
//                                                                                 weight, bias);
//         conv_layer_p->setDilationNd(nvinfer1::Dims2{1, 1});
//         conv_layer_p->setStrideNd(nvinfer1::Dims2{1, 1});
//         conv_layer_p->setPrePadding(nvinfer1::Dims2{1, 1});
//         conv_layer_p->setPostPadding(nvinfer1::Dims2{1, 1});
//         conv_layer_p->setNbGroups(1);
//         nvinfer1::ITensor* conv_tensor_p = conv_layer_p->getOutput(0);

//         network->markOutput(*conv_tensor_p);

//         nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
            
//         std::ofstream out_eng_file("../eng_file/test_trt_eng.eng", std::ios::binary);
//         if (!out_eng_file.good()){
//             std::cout<<"failed to save engine file in path: "<<std::endl;
//         }
//         out_eng_file.write(reinterpret_cast<char*>(serializedModel->data()), serializedModel->size());
//         out_eng_file.close();

//         delete network;
//         delete config;
//         delete builder;
//         cudaFree(data_const_gpu);
//         cudaFree(weight_conv_gpu);
//         cudaFree(bias_conv_gpu);
// }

int main(int argc, char *argv[]){
    // test_weight_cg();
    // getJsonFile("/home/dong/WS/test/eng_test/trt/test_net.net");
    std::string cfg_path = argv[1];
    std::ifstream json_file(cfg_path);
    nlohmann::json config;
    json_file >> config;
    bool test_status = config.at("test_status").get<bool>();
    std::vector<nlohmann::json> engines = config.at("engines").get<std::vector<nlohmann::json>>();
    for (int engine_id = 0; engine_id < engines.size(); engine_id++){
        nlohmann::json engine_config = engines[engine_id];
        std::string engine_name = engine_config.at("eng_name").get<std::string>();
        std::string input_net_path = engine_config.at("net_path").get<std::string>();
        std::string output_eng_path = engine_config.at("eng_path").get<std::string>();
        bool rebuild_flag = engine_config.at("rebuild").get<bool>();
        std::string build_type = engine_config.at("build_type").get<std::string>();
        std::string fallback_type = engine_config.at("fallback_type").get<std::string>();
        int gpu_id = engine_config.at("gpu").get<int>();
        int dla_id = engine_config.at("dla").get<int>();
        std::vector<char> serial_data = common::loadDataFromFile<char, char>(input_net_path);
        std::string engine_type = "TRTEngine";
        engine::BaseEnginePtr model_p = engine::EngineFactory::createEngineInstance(engine_type);
        std::cout<<"start build"<<std::endl;
        model_p->buildEngineModel(serial_data.data(), 0, output_eng_path, build_type, fallback_type, dla_id, rebuild_flag);
        std::cout<<"end build"<<std::endl;

        std::vector<nlohmann::json> inputs_config = engine_config.at("inputs").get<std::vector<nlohmann::json>>();
        std::vector<nlohmann::json> outputs_config = engine_config.at("outputs").get<std::vector<nlohmann::json>>();
        int inputs_num = inputs_config.size();
        int outputs_num = outputs_config.size();
        std::vector<std::vector<char>> inputs_data(inputs_num);
        std::vector<std::vector<char>> outputs_ref_data(outputs_num);
        std::vector<memory::BaseBufferPtr> inputs_buffer_p(inputs_num, nullptr);
        std::vector<memory::BaseBufferPtr> outputs_buffer_p(outputs_num, nullptr);
        std::vector<std::vector<int64_t>> inputs_shape(inputs_num);
        std::vector<std::vector<int64_t>> outputs_shape(outputs_num);
        for (int i = 0; i < inputs_num; i++){
            std::string input_data_path = inputs_config[i].at("file_path").get<std::string>();
            inputs_shape[i] = inputs_config[i].at("shape").get<std::vector<int64_t>>();
            inputs_data[i] = common::loadDataFromFile<char, char>(input_data_path);
            std::string buffer_name = inputs_config[i].at("name").get<std::string>();
            inputs_buffer_p[i] = std::make_shared<memory::GPUBuffer>(buffer_name, inputs_data[i].size());
            inputs_buffer_p[i]->copyFromCPU(inputs_data[i].data(), inputs_data[i].size());
        }
        for (int i = 0; i < outputs_num; i++){
            std::string output_data_path = outputs_config[i].at("file_path").get<std::string>();
            outputs_shape[i] = outputs_config[i].at("shape").get<std::vector<int64_t>>();
            if (output_data_path == ""){
                int volume = 1;
                for (int j = 0; j < outputs_shape[i].size(); j++) volume *= outputs_shape[i][j];
                outputs_ref_data[i].resize(volume * 4);
            } 
            else outputs_ref_data[i] = common::loadDataFromFile<char, char>(output_data_path);
            std::string buffer_name = outputs_config[i].at("name").get<std::string>();
            outputs_buffer_p[i] = std::make_shared<memory::GPUBuffer>(buffer_name, outputs_ref_data[i].size());
            outputs_buffer_p[i]->setZeros();
        }
        cudaDeviceSynchronize();
        std::cout<<"start test infer"<<std::endl;
        model_p->setDeviceId(gpu_id);
        model_p->setEngineIONumber(inputs_num, outputs_num);
        model_p->setStreamInternal();
        model_p->setEngineName(engine_name);
        model_p->setInputsShape(inputs_shape);
        model_p->setOutputsShape(outputs_shape);
        model_p->setEngineIOBuffer(inputs_buffer_p, outputs_buffer_p);
        cudaDeviceSynchronize();
        std::cout<<"ready set"<<std::endl;
        model_p->loadEngineModel(output_eng_path);
        cudaDeviceSynchronize();
        std::cout<<"ready load"<<std::endl;
        model_p->initEngineModel();
        cudaDeviceSynchronize();
        std::cout<<"ready init"<<std::endl;
        // model_p->inferEngineModel();
        model_p->testModelInferTime(5);
        cudaDeviceSynchronize();
        std::cout<<"end test infer"<<std::endl;

        std::vector<char> res;
        int check_dim_idx = engine_config.at("test_check_dim").get<int>();
        for (int i = 0; i < outputs_num; i++){
            std::string output_data_path = outputs_config[i].at("file_path").get<std::string>();
            outputs_shape[i] = outputs_config[i].at("shape").get<std::vector<int64_t>>();
            if (output_data_path == ""){
                int volume = 1;
                for (int j = 0; j < outputs_shape[i].size(); j++) volume *= outputs_shape[i][j];
                std::string save_path = outputs_buffer_p[i]->getBufferName() + ".bin";
                for (int j = 0; j < save_path.size(); j++) if (save_path[j] == '/') save_path[j] = '-';
                save_path = "../test_data/" + save_path;
                model_p->getBufferPtr(outputs_buffer_p[i]->getBufferName())->saveToFile(save_path);
                continue;
            }
            res.resize(outputs_buffer_p[i]->getDataByteSize());
            model_p->getBufferPtr(outputs_buffer_p[i]->getBufferName())->copyToCPU(res.data(), res.size());
            if (check_dim_idx >= outputs_shape[i].size()){
                std::cout<<"invalid check dimension id, out of range"<<std::endl;
                return -1;
            }
            int head_dim = 1;
            int tail_dim = 1;
            int check_dim = 1;
            for (int j = 0; j < check_dim_idx; j++) head_dim *= outputs_shape[i][j];
            check_dim = outputs_shape[i][check_dim_idx];
            for (int j = check_dim_idx + 1; j < outputs_shape[i].size(); j++) tail_dim *= outputs_shape[i][j];
            std::cout<<outputs_buffer_p[i]->getBufferName()<<std::endl;
            common::maxErrCheck<float, float>(reinterpret_cast<float*>(res.data()), reinterpret_cast<float*>(outputs_ref_data[i].data()),
                                                head_dim, check_dim, tail_dim);
        } 
    }

    return 0;
}