#ifndef BASE_ENG_H
#define BASE_ENG_H

#include "common.h"
#include "mem_buffer.h"
#include "base_module.h"
#include <functional>

namespace engine{

class BaseEngine{
public:
    BaseEngine(){};
    BaseEngine(BaseEngine &&) = delete;
    BaseEngine(const BaseEngine &) = delete;
    BaseEngine & operator=(BaseEngine &&) = delete;
    BaseEngine & operator=(const BaseEngine &) = delete;
    virtual ~BaseEngine(){}

    void setEngineName(const std::string& name);
    void setEngineIONumber(const int& input_num, const int& output_num);
    void setEngineIOBuffer(std::vector<memory::BaseBufferPtr> input_buffers, std::vector<memory::BaseBufferPtr> output_buffers);
    bool setStreamExternal(cudaStream_t &stream);
    bool setStreamInternal();
    cudaStream_t getStream();
    std::string getEngineName();
    int getEngineInputNum();
    int getEngineOutputNum();
    memory::BaseBufferPtr getBufferPtr(const std::string &buffer_name);
    void cudaCheck(cudaError_t ret, std::string cur_func_pos);
    bool setDeviceId(int id);

    uint64_t* getDlaMemPtr(const std::string &name) {return dla_mem_map_.at(name);}
    void* getGpuMemPtr(const std::string &name) {return gpu_mem_map_.at(name);}
    void* getCpuMemPtr(const std::string &name) {return cpu_mem_map_.at(name);}
    uint64_t getDlaMemSize(const std::string &name) {return dla_mem_size_map_.at(name);}
    bool setDlaMemPtr(const std::string &name, uint64_t* dla_mem_p) {
        if (dla_mem_map_.count(name) == 0){
            std::cout<<"not a existed dla mem ptr key"<<std::endl;
            return false;
        } 
        else{
            dla_mem_map_[name] = dla_mem_p;
            return true;
        }
    }

    virtual bool enginePrepare(std::string net_file_path, std::string eng_file_path) = 0;
    virtual bool loadEngineModel(const std::string& model_file_path) = 0;
    virtual bool initEngineModel() = 0;
    virtual bool inferEngineModel() = 0;
    virtual bool saveEngineModel(const std::string& model_file_path) = 0;
    virtual int64_t buildEngineModel(const char* const data_p, int64_t start_byte, std::string engine_save_path,
                                    std::string build_type, std::string fallback_type, int dla_id, bool rebuild = false) = 0;
    virtual bool testModelInferTime(int repeat_times) = 0;
    virtual void destroy() = 0;

    virtual bool setInputsShape(std::vector<std::vector<int64_t>> &inputs_shape) = 0;
    virtual bool setOutputsShape(std::vector<std::vector<int64_t>> &outputs_shape) = 0;
    
    virtual bool loadEngineModel(const std::string &model_file_path, const bool &on_dla, const char &sync_type, const int &device_id) = 0;
    virtual bool waitSignalAndLaunchEngine() = 0;
    virtual bool syncAfterLaunchEngine() = 0;
    virtual bool updateIOMemPtr(const std::string &name, void* mem_p) = 0;
    
protected:
    cudaStream_t stream_;
    std::string engine_name_;
    int inputs_num_;
    int outputs_num_;
    std::vector<memory::BaseBufferPtr> input_buffers_;
    std::vector<memory::BaseBufferPtr> output_buffers_;
    int device_id_ = 0;
    std::unordered_map<std::string, module::BaseModulePtr> engine_modules_table_;
    std::unordered_map<std::string, memory::TensorPtr> initializer_tensors_table_;
    std::unordered_map<std::string, memory::GPUBufferPtr> initializer_gpu_buffers_table_;
    std::unordered_map<std::string, memory::CPUBufferPtr> initializer_cpu_buffers_table_;

    std::unordered_map<std::string, uint64_t*> dla_mem_map_;
    std::unordered_map<std::string, void*> gpu_mem_map_;
    std::unordered_map<std::string, void*> cpu_mem_map_;
    std::unordered_map<std::string, uint64_t> dla_mem_size_map_;

private:
    std::unordered_map<std::string, memory::BaseBufferPtr> engine_io_buffer_table_;
};
typedef std::shared_ptr<BaseEngine> BaseEnginePtr;

class EngineFactory{
public:
    static void registEngineCreator(const std::string &engine_type_name, std::function<BaseEnginePtr()> create_func);
    static BaseEnginePtr createEngineInstance(std::string &engine_type_name);
private:
    EngineFactory(){}
    ~EngineFactory(){}
    static std::unordered_map<std::string, std::function<BaseEnginePtr()>> engine_type_table_;
};

}

#endif