#include "common.h"
#include "base_eng.h"

int main(int argc, char *argv[]){
    // test_weight_cg();
    // getJsonFile("/home/dong/WS/test/eng_test/trt/test_net.net");
    std::string cfg_path = argv[1];
    std::ifstream json_file(cfg_path);
    nlohmann::json config;
    json_file >> config;

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
        int sync_type = engine_config.at("sync_type").get<int>();
        if (dla_id == -1){
            std::cout << "this test only test dla engine"<<std::endl;
            return -2;
        }
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
        std::vector<std::vector<half>> inputs_data_f16(inputs_num);
        std::vector<std::vector<char>> outputs_ref_data(outputs_num);
        std::vector<std::vector<int64_t>> inputs_shape(inputs_num);
        std::vector<std::vector<int64_t>> outputs_shape(outputs_num);
        std::vector<std::string> inputs_name(inputs_num);
        std::vector<std::string> outputs_name(outputs_num);
        
        for (int i = 0; i < inputs_num; i++){
            std::string input_data_path = inputs_config[i].at("file_path").get<std::string>();
            inputs_shape[i] = inputs_config[i].at("shape").get<std::vector<int64_t>>();
            inputs_data[i] = common::loadDataFromFile<char, char>(input_data_path);
            std::string buffer_name = inputs_config[i].at("name").get<std::string>();
            inputs_name[i] = buffer_name;
        }
        for (int i = 0; i < outputs_num; i++){
            std::string output_data_path = outputs_config[i].at("file_path").get<std::string>();
            outputs_shape[i] = outputs_config[i].at("shape").get<std::vector<int64_t>>();
            outputs_ref_data[i] = common::loadDataFromFile<char, char>(output_data_path);
            std::string buffer_name = outputs_config[i].at("name").get<std::string>();
            outputs_name[i] = buffer_name;
        }

        model_p->setDeviceId(dla_id);
        model_p->setEngineIONumber(inputs_num, outputs_num);
        model_p->setStreamInternal();
        model_p->setEngineName(engine_name);
        model_p->setInputsShape(inputs_shape);
        model_p->setOutputsShape(outputs_shape);

        model_p->loadEngineModel(output_eng_path, true, sync_type, dla_id);
        model_p->initEngineModel();
        
        std::vector<void*> input_cpu_p_vec(inputs_num);
        std::vector<void*> input_gpu_p_vec(inputs_num);
        std::vector<uint64_t*> input_dla_p_vec(inputs_num);
        std::vector<int> input_size(inputs_num);
        std::vector<void*> output_cpu_p_vec(outputs_num);
        std::vector<void*> output_gpu_p_vec(outputs_num);
        std::vector<uint64_t*> output_dla_p_vec(outputs_num);
        std::vector<int> output_size(outputs_num);
        cudaStream_t stream = model_p->getStream();

        for (int i = 0; i < inputs_num; i++){std::cout<<inputs_name[i]<<std::endl;
            input_cpu_p_vec[i] = model_p->getCpuMemPtr(inputs_name[i]);
            input_gpu_p_vec[i] = model_p->getGpuMemPtr(inputs_name[i]);
            input_dla_p_vec[i] = model_p->getDlaMemPtr(inputs_name[i]);
            input_size[i] = model_p->getDlaMemSize(inputs_name[i]);
            inputs_data_f16[i].resize(inputs_data[i].size() / sizeof(float));
            for (int j = 0; j < inputs_data_f16[i].size(); j++) inputs_data_f16[i][j] = static_cast<half>(reinterpret_cast<float*>(inputs_data[i].data())[j]);
        }
        for (int i = 0; i < outputs_num; i++){std::cout<<outputs_name[i]<<std::endl;
            output_cpu_p_vec[i] = model_p->getCpuMemPtr(outputs_name[i]);
            output_gpu_p_vec[i] = model_p->getGpuMemPtr(outputs_name[i]);
            output_dla_p_vec[i] = model_p->getDlaMemPtr(outputs_name[i]);
            output_size[i] = model_p->getDlaMemSize(outputs_name[i]);
            
        }
        //拷贝中需要注意dla的内存排布和对齐规则，在零拷贝情况下直接考贝可能因为对齐规则不一造成错误
        if (sync_type == 1){
            for (int i = 0; i < inputs_num; i++) std::memset(input_cpu_p_vec[i], 0, input_size[i]);
            for (int i = 0; i < outputs_num; i++) std::memset(output_cpu_p_vec[i], 0, output_size[i]);
            for (int i = 0; i < inputs_num; i++) std::memcpy(input_cpu_p_vec[i], inputs_data_f16[i].data(), input_size[i]);
            // model_p->waitSignalAndLaunchEngine();
            // model_p->syncAfterLaunchEngine();
            model_p->inferEngineModel();
            for (int i = 0; i < outputs_num; i++){
                int ndim = outputs_shape[i].size();
                int check_dim = 1;
                for (int j = 0; j < ndim - 1; j++) check_dim *= outputs_shape[i][j];
                int align_block = 64 / 2;
                int last_dim = outputs_shape[i][ndim - 1];
                int aligned_last_dim = (last_dim + align_block - 1) / align_block * align_block;
                std::vector<float> max_err_cpu(last_dim, 0.0f);
                std::vector<float> rec_ref_cpu(last_dim, 0.0f);
                std::vector<float> rec_res_cpu(last_dim, 0.0f);
                std::vector<float> ref_sum_cpu(last_dim, 0.0f);
                std::vector<float> err_sum_cpu(last_dim, 0.0f);
                for (int n = 0; n <check_dim; n++){
                    for (int c = 0; c < last_dim; c++){
                        float ref = reinterpret_cast<float*>(outputs_ref_data[i].data())[n*last_dim + c];
                        float res = static_cast<float>(reinterpret_cast<half*>(output_cpu_p_vec[i])[n*aligned_last_dim + c]);
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
                std::cout<<"output: "<<outputs_name[i]<<std::endl;
                for (int j = 0; j < last_dim; j++) std::cout<<j<<": res="<<rec_res_cpu[j]<<", ref="<<rec_ref_cpu[j]<<", max_err="<<max_err_cpu[j]
                                                        <<", ref_sum="<<ref_sum_cpu[j]<<", err_sum="<<err_sum_cpu[j]<<std::endl;
            }
            std::cout<< "finish cpu-dla test"<<std::endl;
        }
        else{
            for (int i = 0; i < inputs_num; i++) cudaMemsetAsync(input_gpu_p_vec[i], 0, input_size[i], stream);
            for (int i = 0; i < outputs_num; i++) cudaMemsetAsync(output_gpu_p_vec[i], 0, output_size[i], stream);
            for (int i = 0; i < inputs_num; i++) cudaMemcpyAsync(input_gpu_p_vec[i], inputs_data_f16[i].data(), input_size[i], cudaMemcpyHostToDevice, stream);
            model_p->waitSignalAndLaunchEngine();
            model_p->syncAfterLaunchEngine();
            // model_p->inferEngineModel();
            for (int i = 0; i < outputs_num; i++){
                int ndim = outputs_shape[i].size();
                int check_dim = 1;
                for (int j = 0; j < ndim - 1; j++) check_dim *= outputs_shape[i][j];
                int align_block = 64 / 2;
                int last_dim = outputs_shape[i][ndim - 1];
                int aligned_last_dim = (last_dim + align_block - 1) / align_block * align_block;
                std::vector<half> output_res(check_dim*aligned_last_dim);
                cudaMemcpyAsync(output_res.data(), output_gpu_p_vec[i], output_size[i], cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                std::vector<float> max_err_gpu(last_dim, 0.0f);
                std::vector<float> rec_ref_gpu(last_dim, 0.0f);
                std::vector<float> rec_res_gpu(last_dim, 0.0f);
                std::vector<float> ref_sum_gpu(last_dim, 0.0f);
                std::vector<float> err_sum_gpu(last_dim, 0.0f);
                for (int n = 0; n <check_dim; n++){
                    for (int c = 0; c < last_dim; c++){
                        float ref = reinterpret_cast<float*>(outputs_ref_data[i].data())[n*last_dim + c];
                        float res = reinterpret_cast<half*>(output_res.data())[n*aligned_last_dim + c];
                        float err = std::abs(ref - res);
                        if (err > max_err_gpu[c]){
                            max_err_gpu[c] = err;
                            rec_ref_gpu[c] = ref;
                            rec_res_gpu[c] = res;
                        }
                        ref_sum_gpu[c] += std::abs(ref);
                        err_sum_gpu[c] += err;
                    }
                }
                std::cout<<"output: "<<outputs_name[i]<<std::endl;
                for (int j = 0; j < last_dim; j++) std::cout<<j<<": res="<<rec_res_gpu[j]<<", ref="<<rec_ref_gpu[j]<<", max_err="<<max_err_gpu[j]
                                                        <<", ref_sum="<<ref_sum_gpu[j]<<", err_sum="<<err_sum_gpu[j]<<std::endl;
            }
            std::cout<< "finish gpu-dla test"<<std::endl;
        }

        
    }

    return 0;
}