#ifndef MODULE_BASE_MODULE_H
#define MODULE_BASE_MODULE_H

#include "common.h"
#include <functional>
#include "mem_buffer.h"

namespace module{

    class BaseModule{
    public:
        BaseModule(){};
        BaseModule(BaseModule &&) = delete;
        BaseModule(const BaseModule &) = delete;
        BaseModule & operator=(BaseModule &&) = delete;
        BaseModule & operator=(const BaseModule &) = delete;
        virtual ~BaseModule(){}

        void setModuleName(const std::string& name);
        void setModuleIONumber(const int& input_num, const int& output_num);
        void setModuleIOBuffer(std::vector<memory::BaseBufferPtr> input_buffers, std::vector<memory::BaseBufferPtr> output_buffers);
        bool setStreamExternal(cudaStream_t &stream);
        bool setStreamInternal();
        cudaStream_t getStream();
        std::string getModuleName();
        int getModuleInputNum();
        int getModuleOutputNum();
        memory::BaseBufferPtr getBufferPtr(const std::string &buffer_name);
        bool setDeviceId(int id);

        virtual int64_t deserializeModule(const char* data_p, int64_t start_byte) = 0;
        virtual int64_t serializeModule(const char* data_p, int64_t start_byte) = 0;
        virtual bool init() = 0;
        virtual bool enqueueModule() = 0;
        virtual bool executeModule() = 0;

    protected:
        std::string module_name_;
        cudaStream_t stream_;
        int inputs_num_;
        int outputs_num_;
        std::vector<memory::BaseBufferPtr> input_buffers_;
        std::vector<memory::BaseBufferPtr> output_buffers_;
        int device_id_ = 0;
    private:
        std::unordered_map<std::string, memory::BaseBufferPtr> module_io_buffer_table_;
    };
    typedef std::shared_ptr<BaseModule> BaseModulePtr;

    class LayerModule : public BaseModule{
    public:
    protected:
    private:
    };
    typedef std::shared_ptr<LayerModule> LayerModulePtr;

    class BlockModule : public BaseModule{
    public:
    protected:
    private:
    };
    typedef std::shared_ptr<BlockModule> BlockModulePtr;

    class ModuleFactory{
    public:
        static void registModuleCreator(const std::string &module_type_name, std::function<BaseModulePtr()> create_func);
        static BaseModulePtr createModuleInstance(std::string &module_type_name);
    private:
        ModuleFactory(){}
        ~ModuleFactory(){}
        static std::unordered_map<std::string, std::function<BaseModulePtr()>> module_type_table_;
    };

}

#endif