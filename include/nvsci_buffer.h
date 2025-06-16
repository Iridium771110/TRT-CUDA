#ifndef NVSCI_BUFFER_H
#define NVSCI_BUFFER_H

#include <iostream>
#include <memory>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cudla.h"
#include "nvscibuf.h"
#include "nvscierror.h"
#include <unordered_map>

namespace nvscibuffer{

void enableCudaDriveApi();

class GeneralBuffer{
    public:
    GeneralBuffer() = delete;
    GeneralBuffer(const GeneralBuffer&) = delete;
    GeneralBuffer(GeneralBuffer &&) = delete;
    GeneralBuffer & operator= (const GeneralBuffer &) = delete;
    GeneralBuffer & operator= (GeneralBuffer &&) = delete;
    explicit GeneralBuffer(const cudlaDevHandle &dla_handle, const NvSciBufModule &sci_buf_module, int byte_size, int align_byte);
    ~GeneralBuffer();
    void releaseGeneralBuffer(const cudlaDevHandle &dla_handle);

    uint64_t* getDlaPtr(){ return dla_registered_buf_p_;}
    void* getCpuPtr(){ return cpu_registered_buf_p_;}
    void* getGpuPtr(){ return gpu_registered_buf_p_;} //单p存在内部释放后外部不感知的缺陷，双p访问方式麻烦存在外部释放的危险

    private:
    NvSciBufObj buffer_object_;
    NvSciBufAttrList unreconciled_attr_list_;
    NvSciBufAttrList reconciled_attr_list_;
    NvSciBufAttrList conflict_attr_list_;
    NvSciBufType buf_type_;
    uint64_t buf_size_;
    uint64_t align_byte_;
    bool cpu_access_flag_;
    NvSciBufAttrValAccessPerm perm_;
    CUuuid   uuid_;
    int num_attr_ = 6;
    NvSciBufAttrKeyValuePair raw_buf_attrs_[6];
    cudlaExternalMemoryHandleDesc cudla_ext_mem_desc_;
    cudaExternalMemoryHandleDesc cuda_ext_mem_handle_desc_;
    cudaExternalMemory_t cuda_ext_mem_;
    cudaExternalMemoryBufferDesc cuda_ext_mem_buf_desc_;

    uint64_t* dla_registered_buf_p_ = nullptr;
    void* cpu_registered_buf_p_ = nullptr;
    void* gpu_registered_buf_p_ = nullptr;
    NvSciError sci_ret_;
    void checkNvSciErr(NvSciError sci_ret, std::string msg){
        if (sci_ret != NvSciError_Success) std::cout<<"err-code "<<sci_ret<<": "<<msg<<std::endl;
    }
};
typedef std::shared_ptr<GeneralBuffer> GeneralBufferPtr;

class GeneralBufferPool{
    public:
    explicit GeneralBufferPool();
    GeneralBufferPool(GeneralBufferPool &&) = delete;
    GeneralBufferPool(const GeneralBufferPool &) = delete;
    GeneralBufferPool & operator= (GeneralBufferPool &&) = delete;
    GeneralBufferPool & operator= (const GeneralBufferPool &) = delete;
    ~GeneralBufferPool();
    void createBuffer(const cudlaDevHandle &dla_handle, const std::string &name, int byte_size, int align_byte = 128);
    void releaseBuffer(const std::string &name, const cudlaDevHandle &dla_handle);
    void clearBufferPool(const cudlaDevHandle &dla_handle);
    GeneralBufferPtr getBufferPtr(const std::string &name);
    private:
    NvSciBufModule sci_buf_module_ = nullptr;
    NvSciError sci_ret_;
    std::unordered_map<std::string, GeneralBufferPtr> sci_buffer_map_;
    void checkNvSciErr(NvSciError sci_ret, std::string msg){
        if (sci_ret != NvSciError_Success) std::cout<<"err-code "<<sci_ret<<": "<<msg<<std::endl;
    }
};

}

#endif