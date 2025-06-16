#ifndef TORCH_ENG_H
#define TORCH_ENG_H

#include "base_eng.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

namespace engine{

class TorchEngine : public BaseEngine{
public:
    TorchEngine(){}
    TorchEngine(TorchEngine &&) = delete;
    TorchEngine(const TorchEngine &) = delete;
    TorchEngine & operator=(TorchEngine &&) = delete;
    TorchEngine & operator=(const TorchEngine &) = delete;

    bool enginePrepare(std::string net_file_path, std::string eng_file_path) final;
    bool loadEngineModel(const std::string& model_file_path) final;
    bool initEngineModel() final;
    bool inferEngineModel() final;
    bool saveEngineModel(const std::string& model_file_path) final;
    int64_t buildEngineModel(const char* const data_p, int64_t start_byte, std::string engine_save_path,
                                    std::string build_type, std::string fallback_type, int dla_id, bool rebuild = false) final;
    bool testModelInferTime(int repeat_times) final;
    void destroy() final;

    bool setInputsShape(std::vector<std::vector<int64_t>> &inputs_shape) final;
    bool setOutputsShape(std::vector<std::vector<int64_t>> &outputs_shape) final;
    
private:
    int64_t addModule(nlohmann::json node_cfg);

    std::shared_ptr<torch::jit::script::Module> model_ptr_;
    std::vector<torch::jit::IValue> inputs_;
    torch::jit::IValue output_;
    std::vector<at::IntArrayRef> inputs_shape_;
    std::vector<at::IntArrayRef> outputs_shape_;
    at::cuda::CUDAStream myStream_ = at::cuda::getStreamFromPool();
    static bool registed_;
};

typedef std::shared_ptr<TorchEngine> TorchEnginePtr;

}

#endif