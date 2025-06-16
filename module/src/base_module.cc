#include "base_module.h"

namespace module
{
    void BaseModule::setModuleName(const std::string& name){
        module_name_ = name;
    }
    void BaseModule::setModuleIONumber(const int& input_num, const int& output_num){
        inputs_num_ = input_num;
        outputs_num_ = output_num;
    }
    void BaseModule::setModuleIOBuffer(std::vector<memory::BaseBufferPtr> input_buffers, std::vector<memory::BaseBufferPtr> output_buffers){
        input_buffers_ = input_buffers;
        output_buffers_ = output_buffers;
        for (int i = 0; i < input_buffers.size(); i++){
            if (module_io_buffer_table_.count(input_buffers[i]->getBufferName()) != 0){
                std::cout << "module: " << module_name_ << " buffer key " << input_buffers[i] << "already exist, buffer will not be replaced";
            }
            module_io_buffer_table_.emplace(input_buffers[i]->getBufferName(), input_buffers[i]);
        }
        for (int i = 0; i < output_buffers.size(); i++){
            if (module_io_buffer_table_.count(output_buffers[i]->getBufferName()) != 0){
                std::cout << "module: " << module_name_ << " buffer key " << output_buffers[i] << "already exist, buffer will not be replaced";
            }
            module_io_buffer_table_.emplace(output_buffers[i]->getBufferName(), output_buffers[i]);
        }
    }
    bool BaseModule::setStreamExternal(cudaStream_t &stream){
        stream_ = stream;
        return true;
    }
    bool BaseModule::setStreamInternal(){
        cudaStreamCreate(&stream_);
        return true;
    }
    cudaStream_t BaseModule::getStream(){
        return stream_;
    }
    std::string BaseModule::getModuleName(){
        return module_name_;
    }
    int BaseModule::getModuleInputNum(){
        return inputs_num_;
    }
    int BaseModule::getModuleOutputNum(){
        return outputs_num_;
    }
    memory::BaseBufferPtr BaseModule::getBufferPtr(const std::string &buffer_name){
        if (module_io_buffer_table_.count(buffer_name) != 0){
            return module_io_buffer_table_.at(buffer_name);
        }
        else{
            std::cout << "module: " << module_name_ << " no such buffer name for in/output: " << buffer_name << std::endl;
            return nullptr;
        }
    }
    bool BaseModule::setDeviceId(int id){
        device_id_ = id;
        return true;
    }
    
    void ModuleFactory::registModuleCreator(const std::string &module_type_name, std::function<BaseModulePtr()> create_func){
        if (module_type_table_.count(module_type_name) != 0){
            std::cout << "module type : " << module_type_name << " already registed in factory, do nothing";
        }
        else {
            module_type_table_.emplace(module_type_name, create_func);
        }
    }
    BaseModulePtr ModuleFactory::createModuleInstance(std::string &module_type_name){
        if (module_type_table_.count(module_type_name) == 0){
            return nullptr;
        }
        else{
            return module_type_table_.at(module_type_name)();
        }
    }
    std::unordered_map<std::string, std::function<BaseModulePtr()>> ModuleFactory::module_type_table_;

} // namespace module
