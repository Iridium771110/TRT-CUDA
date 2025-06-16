#ifndef TRT_ENG_H
#define TRT_ENG_H

#include "base_eng.h"
#include <NvInfer.h>
#ifdef JETSON_ORIN
#include "nvsci_buffer.h"
#include "nvsci_synchronize.h"
#endif
namespace engine{

class Logger : public nvinfer1::ILogger{
    void log(Severity severity, const char* msg) noexcept override{
        if (severity <= Severity::kWARNING) std::cout<<msg<<std::endl;
    }
};

class TRTEngine : public BaseEngine{
public:
    TRTEngine(){}
    TRTEngine(TRTEngine &&) = delete;
    TRTEngine(const TRTEngine &) = delete;
    TRTEngine & operator=(TRTEngine &&) = delete;
    TRTEngine & operator=(const TRTEngine &) = delete;
    ~TRTEngine();

    bool enginePrepare(std::string net_file_path, std::string eng_file_path) final;
    bool loadEngineModel(const std::string& model_file_path) final;
    bool initEngineModel() final;
    //注意dla情况下，内存长度有对齐需求
    bool inferEngineModel() final;
    bool saveEngineModel(const std::string& model_file_path) final;
    int64_t buildEngineModel(const char* const data_p, int64_t start_byte, std::string engine_save_path,
                                    std::string build_type, std::string fallback_type, int dla_id, bool rebuild = false) final;
    //build 和 load-execute 两个过程实际上是分离的
    //当前假定了多次推理之间，内存位置和大小都是固定的 -》实际上有可能是变化的，需要补接口
    bool testModelInferTime(int repeat_times) final;
    void destroy() final;
    
    bool setInputsShape(std::vector<std::vector<int64_t>> &inputs_shape) final;
    bool setOutputsShape(std::vector<std::vector<int64_t>> &outputs_shape) final;

    bool loadEngineModel(const std::string &model_file_path, const bool &on_dla, const char &sync_type, const int &device_id) final;
    bool waitSignalAndLaunchEngine() final;
    bool syncAfterLaunchEngine() final;
    bool updateIOMemPtr(const std::string &name, void* mem_p);
private:
    bool addConstantTensor(nvinfer1::INetworkDefinition* network, std::string tensor_name);
    int64_t addModule(nvinfer1::INetworkDefinition* network, nlohmann::json node_cfg);
    nvinfer1::ITensor* internBoardcastShuffle(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* tensor_p, int aim_rank, int start_dim);

    Logger logger_;
    std::vector<char> engine_file_data_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<std::vector<int64_t>> inputs_shape_;
    std::vector<std::vector<int64_t>> outputs_shape_;
    std::unordered_map<std::string, nvinfer1::ITensor*> nv_tensor_table_;

    char sync_type_ = 0; //0=gpu-cpu， 1=dla-cpu, 2=dla-gpu
    bool on_dla_ = false;
#ifdef JETSON_ORIN
    cudlaDevHandle dev_handle_;
    cudlaModule module_handle_;
    cudlaModuleAttribute dla_model_attr_;
    cudlaStatus dla_ret_;
    cudaError_t cuda_ret_;
    NvSciError sci_ret_;

    nvscibuffer::GeneralBufferPool general_buffer_pool_;
    nvscisync::GeneralSyncObjPool general_sync_obj_pool_;
    cudlaWaitEvents* dla_wait_cpu_events_p_;
    cudlaSignalEvents* dla_signal_cpu_events_p_;
    cudlaWaitEvents* dla_wait_cuda_events_p_;
    cudlaSignalEvents* dla_signal_cuda_events_p_;
    NvSciSyncObj* dla_wait_cpu_event_obj_p_;
    NvSciSyncCpuWaitContext* dla_signal_cpu_event_context_p_;
    cudaExternalSemaphore_t* cuda_signal_dla_sema_p_;
    cudaExternalSemaphore_t* cuda_wait_dla_sema_p_;
    cudaExternalSemaphoreSignalParams* cuda_signal_dla_sema_param_p_;
    cudaExternalSemaphoreWaitParams* cuda_wait_dla_sema_param_p_;

    std::vector<uint64_t*> dla_inputs_p_vec_;
    std::vector<uint64_t*> dla_outputs_p_vec_;
    cudlaTask dla_sync_cpu_task_;
    cudlaTask dla_sync_gpu_task_;
    std::vector<std::string> dla_inputs_name_vec_;
    std::vector<std::string> dla_outputs_name_vec_;
#endif
    const std::unordered_map<std::string, nvinfer1::DataType> nv_type_table_ = {
                                                            {"float", nvinfer1::DataType::kFLOAT},
                                                            {"float16", nvinfer1::DataType::kHALF},
                                                            {"int8", nvinfer1::DataType::kINT8},
                                                            {"int32", nvinfer1::DataType::kINT32}};
    const std::unordered_map<std::string, nvinfer1::ElementWiseOperation> nv_elem_op_table_ = {
                                                            {"Add", nvinfer1::ElementWiseOperation::kSUM},
                                                            {"Sub", nvinfer1::ElementWiseOperation::kSUB},
                                                            {"Mul", nvinfer1::ElementWiseOperation::kPROD},
                                                            {"Div", nvinfer1::ElementWiseOperation::kDIV}};
    static bool registed_;
};

}

#endif