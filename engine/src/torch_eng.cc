#include "torch_eng.h"

namespace engine{

    void TorchEngine::destroy(){
        delete this;
    }
    bool TorchEngine::enginePrepare(std::string net_file_path, std::string eng_file_path){
        return true;
    }
    bool TorchEngine::loadEngineModel(const std::string& model_file_path){
        if (torch::cuda::is_available()) std::cout<<"cuda is available for libtorch!"<<std::endl;
        else {
            std::cout<<"cuda is not available for libtorch!"<<std::endl;
            return false;
        }
        // at::NoGradGuard nograd;
        model_ptr_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_file_path));
        model_ptr_->to({torch::kCUDA, static_cast<c10::DeviceIndex>(device_id_)});
        cudaDeviceSynchronize();
        std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;
        return true;
    }
    bool TorchEngine::initEngineModel(){
        torch::TensorOptions option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        if (inputs_num_ != input_buffers_.size()){
            std::cout << "eng: " << engine_name_ << " inconsist for inputs_num and input_buffers. " << std::endl;
            return false;
        }
        if (outputs_num_ != output_buffers_.size()){
            std::cout << "eng: " << engine_name_ << " inconsist for outputs_num and output_buffers. " << std::endl;
            return false;
        }
        inputs_.resize(inputs_num_);
        for (int i = 0; i < inputs_num_; i++) inputs_.emplace_back(torch::from_blob(input_buffers_[i]->getDataPtr<void>(), inputs_shape_[i], option));
        return true;
    }
    bool TorchEngine::inferEngineModel(){
        myStream_ = at::cuda::getStreamFromExternal(stream_, device_id_);
        at::cuda::setCurrentCUDAStream(myStream_);
        torch::jit::IValue output = model_ptr_->forward(inputs_);
        auto out_dict = output.toGenericDict();
        for (int i = 0; i < outputs_num_; i++){
            cudaMemcpyAsync(output_buffers_[i]->getDataPtr<void>(), out_dict.at(output_buffers_[i]->getBufferName()).toTensor().data_ptr(),
                            output_buffers_[i]->getDataByteSize(), cudaMemcpyDeviceToDevice, stream_);
        }
        cudaStreamSynchronize(stream_);
        return true;
    }
    bool TorchEngine::testModelInferTime(int repeat_times){
        myStream_ = at::cuda::getStreamFromExternal(stream_, device_id_);
        at::cuda::setCurrentCUDAStream(myStream_);
        torch::jit::IValue output;
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        for (int i = 0; i < repeat_times; i++){
            cudaEventRecord(start, stream_);
            output = model_ptr_->forward(inputs_);
            cudaEventRecord(end, stream_);
            cudaEventSynchronize(end);
            float cost;
            cudaEventElapsedTime(&cost, start, end);
            std::cout<<"torch model test time: "<<cost<<"ms"<<std::endl;
        }
        auto out_dict = output.toGenericDict();
        for (int i = 0; i < outputs_num_; i++){
            cudaMemcpyAsync(output_buffers_[i]->getDataPtr<void>(), out_dict.at(output_buffers_[i]->getBufferName()).toTensor().data_ptr(),
                            output_buffers_[i]->getDataByteSize(), cudaMemcpyDeviceToDevice, stream_);
        }
        cudaStreamSynchronize(stream_);
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        return true;
    }
    bool TorchEngine::saveEngineModel(const std::string& model_file_path){
        return true;
    }
    int64_t TorchEngine::buildEngineModel(const char* const data_p, int64_t start_byte, std::string engine_save_path,
                                    std::string build_type, std::string fallback_type, int dla_id, bool rebuild){
        return 0;
    }
    int64_t TorchEngine::addModule(nlohmann::json node_cfg){
        return 0;
    }

    bool TorchEngine::setInputsShape(std::vector<std::vector<int64_t>> &inputs_shape){
        if (inputs_num_ != inputs_shape.size()){
            std::cout << "eng: " << engine_name_ << " inconsist for inputs_num and inputs_shape. " << std::endl;
            return false;
        }
        inputs_shape_.resize(inputs_num_);
        for (int i = 0; i < inputs_num_; i++){
            inputs_shape_[i] = at::IntArrayRef(inputs_shape[i].data(), inputs_shape_[i].size());
        }
        return true;
    }
    bool TorchEngine::setOutputsShape(std::vector<std::vector<int64_t>> &outputs_shape){
        if (outputs_num_ != outputs_shape.size()){
            std::cout << "eng: " << engine_name_ << " inconsist for outputs_num and outputs_shape. " << std::endl;
            return false;
        }
        outputs_shape_.resize(outputs_num_);
        for (int i = 0; i < outputs_num_; i++){
            outputs_shape_[i] = at::IntArrayRef(outputs_shape[i].data(), outputs_shape_[i].size());
        }
        return true;
    }
    
    bool TorchEngine::registed_ = [](){
        std::string type_name = "TorchEngine";
        EngineFactory::registEngineCreator(type_name, 
                                            [](){return BaseEnginePtr(std::make_shared<TorchEngine>());});
        return true;
    }();
    
}