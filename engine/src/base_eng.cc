#include "base_eng.h"

namespace engine{

    void BaseEngine::setEngineName(const std::string& name){
        engine_name_ = name;
    }
    void BaseEngine::setEngineIONumber(const int& input_num, const int& output_num){
        inputs_num_ = input_num;
        outputs_num_ = output_num;
    }
    void BaseEngine::setEngineIOBuffer(std::vector<memory::BaseBufferPtr> input_buffers, std::vector<memory::BaseBufferPtr> output_buffers){
        input_buffers_ = input_buffers;
        output_buffers_ = output_buffers;
        for (int i = 0; i < input_buffers.size(); i++){
            if (engine_io_buffer_table_.count(input_buffers[i]->getBufferName()) != 0){
                std::cout << "eng: " << engine_name_ << " buffer key " << input_buffers[i] << "already exist, buffer will not be replaced";
            }
            engine_io_buffer_table_.emplace(input_buffers[i]->getBufferName(), input_buffers[i]);
        }
        for (int i = 0; i < output_buffers.size(); i++){
            if (engine_io_buffer_table_.count(output_buffers[i]->getBufferName()) != 0){
                std::cout << "eng: " << engine_name_ << " buffer key " << output_buffers[i] << "already exist, buffer will not be replaced";
            }
            engine_io_buffer_table_.emplace(output_buffers[i]->getBufferName(), output_buffers[i]);
        }
    }
    bool BaseEngine::setStreamExternal(cudaStream_t &stream){
        stream_ = stream;
        return true;
    }
    bool BaseEngine::setStreamInternal(){
        cudaStreamCreate(&stream_);
        return true;
    }
    cudaStream_t BaseEngine::getStream(){
        return stream_;
    }
    std::string BaseEngine::getEngineName(){
        return engine_name_;
    }
    int BaseEngine::getEngineInputNum(){
        return inputs_num_;
    }
    int BaseEngine::getEngineOutputNum(){
        return outputs_num_;
    }
    memory::BaseBufferPtr BaseEngine::getBufferPtr(const std::string &buffer_name){
        if (engine_io_buffer_table_.count(buffer_name) != 0){
            return engine_io_buffer_table_.at(buffer_name);
        }
        else{
            std::cout << "eng: " << engine_name_ << " no such buffer name for in/output: " << buffer_name << std::endl;
            return nullptr;
        }
    }
    void BaseEngine::cudaCheck(cudaError_t ret, std::string cur_func_pos){
        if (ret != cudaSuccess) std::cout<<cur_func_pos<<": "<<cudaGetErrorString(ret)<<std::endl;
    }
    bool BaseEngine::setDeviceId(int id){
        device_id_ = id;
        return true;
    }

    void EngineFactory::registEngineCreator(const std::string &engine_type_name, std::function<BaseEnginePtr()> create_func){
        if (engine_type_table_.count(engine_type_name) != 0){
            std::cout << "engine type : " << engine_type_name << " already registed in factory, do nothing";
        }
        else {
            engine_type_table_.emplace(engine_type_name, create_func);
        }
    }
    BaseEnginePtr EngineFactory::createEngineInstance(std::string &engine_type_name){
        if (engine_type_table_.count(engine_type_name) == 0){
            return nullptr;
        }
        else{
            return engine_type_table_.at(engine_type_name)();
        }
    }
    std::unordered_map<std::string, std::function<BaseEnginePtr()>> EngineFactory::engine_type_table_;
}