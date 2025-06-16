#include "nvsci_buffer.h"

namespace nvscibuffer{

void enableCudaDriveApi(){
    CUresult cu_ret = cuInit(0);
    const char* msg = nullptr;
    cuGetErrorString(cu_ret, &msg);
    std::cout<<"cu drive api init: "<<msg<<std::endl;
}

GeneralBuffer::GeneralBuffer(const cudlaDevHandle &dla_handle, const NvSciBufModule &sci_buf_module, int byte_size, int align_byte){
    buf_type_ = NvSciBufType_RawBuffer;
    buf_size_ = byte_size;
    align_byte_ = align_byte;
    cpu_access_flag_ = true;
    perm_ = NvSciBufAccessPerm_ReadWrite;
    cuDeviceGetUuid(&uuid_, 0);
    raw_buf_attrs_[0] = NvSciBufAttrKeyValuePair({NvSciBufGeneralAttrKey_Types, &buf_type_, sizeof(buf_type_)});
    raw_buf_attrs_[1] = NvSciBufAttrKeyValuePair({NvSciBufRawBufferAttrKey_Size, &buf_size_, sizeof(buf_size_)});
    raw_buf_attrs_[2] = NvSciBufAttrKeyValuePair({NvSciBufRawBufferAttrKey_Align, &align_byte_, sizeof(align_byte_)});
    raw_buf_attrs_[3] = NvSciBufAttrKeyValuePair({NvSciBufGeneralAttrKey_NeedCpuAccess, &cpu_access_flag_, sizeof(cpu_access_flag_)});//allow cpu access
    raw_buf_attrs_[4] = NvSciBufAttrKeyValuePair({NvSciBufGeneralAttrKey_RequiredPerm, &perm_, sizeof(perm_)});
    raw_buf_attrs_[5] = NvSciBufAttrKeyValuePair({NvSciBufGeneralAttrKey_GpuId, &uuid_, sizeof(uuid_)});//for gpu must have

    sci_ret_ = NvSciBufAttrListCreate(sci_buf_module, &unreconciled_attr_list_);
    sci_ret_ = NvSciBufAttrListSetAttrs(unreconciled_attr_list_, raw_buf_attrs_, num_attr_);
    sci_ret_ = NvSciBufAttrListReconcile(&unreconciled_attr_list_, 1, &reconciled_attr_list_, &conflict_attr_list_);
    sci_ret_ = NvSciBufObjAlloc(reconciled_attr_list_, &buffer_object_);
    sci_ret_ = NvSciBufObjGetCpuPtr(buffer_object_, &cpu_registered_buf_p_);
    
    memset(&cudla_ext_mem_desc_, 0, sizeof(cudlaExternalMemoryHandleDesc));
    cudla_ext_mem_desc_.extBufObject = (void*)buffer_object_;
    cudla_ext_mem_desc_.size         = buf_size_;
    cudlaImportExternalMemory(dla_handle, &cudla_ext_mem_desc_, &dla_registered_buf_p_, 0);

    memset(&cuda_ext_mem_handle_desc_, 0, sizeof(cuda_ext_mem_handle_desc_));
    cuda_ext_mem_handle_desc_.type                  = cudaExternalMemoryHandleTypeNvSciBuf;
    cuda_ext_mem_handle_desc_.handle.nvSciBufObject = buffer_object_;
    cuda_ext_mem_handle_desc_.size                  = cudla_ext_mem_desc_.size;
    cudaImportExternalMemory(&cuda_ext_mem_, &cuda_ext_mem_handle_desc_);
    memset(&cuda_ext_mem_buf_desc_, 0, sizeof(cuda_ext_mem_buf_desc_));
    cuda_ext_mem_buf_desc_.offset = 0;
    cuda_ext_mem_buf_desc_.size   = cudla_ext_mem_desc_.size;
    cudaExternalMemoryGetMappedBuffer(&gpu_registered_buf_p_, cuda_ext_mem_, &cuda_ext_mem_buf_desc_);
}
GeneralBuffer::~GeneralBuffer(){
    dla_registered_buf_p_ = nullptr;
    cpu_registered_buf_p_ = nullptr;
}

void GeneralBuffer::releaseGeneralBuffer(const cudlaDevHandle &dla_handle){//handle必须和create时的一致
    cudaDestroyExternalMemory(cuda_ext_mem_);
    gpu_registered_buf_p_ = nullptr;
    cudlaMemUnregister(dla_handle, dla_registered_buf_p_);
    NvSciBufObjFree(buffer_object_);
    NvSciBufAttrListFree(conflict_attr_list_);
    NvSciBufAttrListFree(reconciled_attr_list_);
    NvSciBufAttrListFree(unreconciled_attr_list_);
}


GeneralBufferPool::GeneralBufferPool(){
    sci_ret_ = NvSciBufModuleOpen(&sci_buf_module_);
    checkNvSciErr(sci_ret_, "failed sci buf module open");
}
GeneralBufferPool::~GeneralBufferPool(){
    NvSciBufModuleClose(sci_buf_module_);
    sci_buf_module_ = nullptr;
}

void GeneralBufferPool::createBuffer(const cudlaDevHandle &dla_handle, const std::string &name, int byte_size, int align_byte){
    GeneralBufferPtr buffer_p = std::make_shared<GeneralBuffer>(dla_handle, sci_buf_module_, byte_size, align_byte);
    sci_buffer_map_.emplace(name, buffer_p);
    buffer_p = nullptr;
}
void GeneralBufferPool::releaseBuffer(const std::string &name, const cudlaDevHandle &dla_handle){
    GeneralBufferPtr buffer_p = sci_buffer_map_.at(name);
    buffer_p->releaseGeneralBuffer(dla_handle);
    sci_buffer_map_.erase(name);
    buffer_p = nullptr;
}
void GeneralBufferPool::clearBufferPool(const cudlaDevHandle &dla_handle){
    for (std::unordered_map<std::string, GeneralBufferPtr>::iterator iter = sci_buffer_map_.begin(); iter != sci_buffer_map_.end(); iter++){
        iter->second->releaseGeneralBuffer(dla_handle);
        iter->second = nullptr;
    }
    sci_buffer_map_.clear();
}
GeneralBufferPtr GeneralBufferPool::getBufferPtr(const std::string &name){
    return sci_buffer_map_.at(name);
}

}