#ifndef NVSCI_SYNCHRONIZE_H
#define NVSCI_SYNCHRONIZE_H

#include <iostream>
#include <memory>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cudla.h"
#include "nvscisync.h"
#include "nvscierror.h"
#include <unordered_map>

namespace nvscisync{

class GeneralSyncObj{
    public:
    GeneralSyncObj() = delete;
    GeneralSyncObj(GeneralSyncObj &&) = delete;
    GeneralSyncObj(const GeneralSyncObj &) = delete;
    GeneralSyncObj & operator= (GeneralSyncObj &&) = delete;
    GeneralSyncObj & operator= (const GeneralSyncObj &) = delete;
    explicit GeneralSyncObj(const cudlaDevHandle &dla_handle, const NvSciSyncModule &sync_module);
    ~GeneralSyncObj();
    void setCudaStream(cudaStream_t stream){
        stream_ = stream;
    }
    NvSciError dlaWaitCpuSync(){
        return NvSciSyncObjSignal(dla_wait_cpu_event_obj_);
    }
    NvSciError cpuWaitDlaSync(){
        return NvSciSyncFenceWait(reinterpret_cast<NvSciSyncFence*>(dla_signal_cpu_events_p_->eofFences[0].fence), dla_signal_cpu_event_context_, -1);
    }
    cudaError dlaWaitGpuSync(){
        return cudaSignalExternalSemaphoresAsync(&cuda_signal_dla_sema_, &cuda_signal_dla_sema_param_, 1, stream_);
    }
    cudaError gpuWaitDlaSync(){
        return cudaWaitExternalSemaphoresAsync(&cuda_wait_dla_sema_, &cuda_wait_dla_sema_param_, 1, stream_);
    }

    cudlaWaitEvents* getDlaWaitCpuEventPtr() {return dla_wait_cpu_events_p_;} 
    cudlaSignalEvents* getDlaSignalCpuEventPtr() {return dla_signal_cpu_events_p_;} 
    cudlaWaitEvents* getDlaWaitGpuEventPtr() {return dla_wait_cuda_events_p_;} 
    cudlaSignalEvents* getDlaSignalGpuEventPtr() {return dla_signal_cuda_events_p_;} 
    NvSciSyncObj* getDlaWaitCpuEventObjPtr() {return &dla_wait_cpu_event_obj_;}
    NvSciSyncCpuWaitContext* getDlaSignalCpuEventContextPtr() {return &dla_signal_cpu_event_context_;}
    cudaExternalSemaphore_t* getCudaSignalDlaSemaPtr() {return &cuda_signal_dla_sema_;}
    cudaExternalSemaphore_t* getCudaWaitDlaSemaPtr() {return &cuda_wait_dla_sema_;}
    cudaExternalSemaphoreSignalParams* getCudaSignalDlaSemaParamPtr() {return &cuda_signal_dla_sema_param_;}
    cudaExternalSemaphoreWaitParams* getCudaWaitDlaSemaParamPtr() {return &cuda_wait_dla_sema_param_;}

    void unregisterSyncObj(const cudlaDevHandle &dla_handle);

    private:
    NvSciSyncObj dla_wait_cpu_event_obj_;
    NvSciSyncAttrList wait_cpu_event_waiter_attr_list_;
    NvSciSyncAttrList wait_cpu_event_signaler_attr_list_;
    NvSciSyncAttrList wait_cpu_event_reconciled_attr_list_;
    NvSciSyncAttrList wait_cpu_event_conflict_attr_list_;
    NvSciSyncObj dla_wait_cuda_event_obj_;
    NvSciSyncAttrList wait_cuda_event_waiter_attr_list_;
    NvSciSyncAttrList wait_cuda_event_signaler_attr_list_;
    NvSciSyncAttrList wait_cuda_event_reconciled_attr_list_;
    NvSciSyncAttrList wait_cuda_event_conflict_attr_list_;
    NvSciSyncObj dla_signal_cpu_event_obj_;
    NvSciSyncAttrList signal_cpu_event_waiter_attr_list_;
    NvSciSyncAttrList signal_cpu_event_signaler_attr_list_;
    NvSciSyncAttrList signal_cpu_event_reconciled_attr_list_;
    NvSciSyncAttrList signal_cpu_event_conflict_attr_list_;
    NvSciSyncObj dla_signal_cuda_event_obj_;
    NvSciSyncAttrList signal_cuda_event_waiter_attr_list_;
    NvSciSyncAttrList signal_cuda_event_signaler_attr_list_;
    NvSciSyncAttrList signal_cuda_event_reconciled_attr_list_;
    NvSciSyncAttrList signal_cuda_event_conflict_attr_list_;

    uint64_t* dla_wait_cpu_event_reg_p_ = nullptr;
    uint64_t* dla_signal_cpu_event_reg_p_ = nullptr;
    uint64_t* dla_wait_cuda_event_reg_p_ = nullptr;
    uint64_t* dla_signal_cuda_event_reg_p_ = nullptr;
    uint64_t** dla_cpu_devs_p_ = nullptr;
    uint64_t** dla_gpu_devs_p_ = nullptr;

    NvSciSyncFence dla_wait_cpu_pre_fence_;
    NvSciSyncFence dla_signal_cpu_eof_fence_;
    NvSciSyncFence dla_wait_cuda_pre_fence_;
    NvSciSyncFence dla_signal_cuda_eof_fence_;
    CudlaFence* dla_wait_cpu_pre_fences_p_ = nullptr;
    CudlaFence* dla_wait_cuda_pre_fences_p_ = nullptr;

    cudlaWaitEvents* dla_wait_cpu_events_p_;
    cudlaSignalEvents* dla_signal_cpu_events_p_;
    cudlaWaitEvents* dla_wait_cuda_events_p_;
    cudlaSignalEvents* dla_signal_cuda_events_p_;
    
    NvSciSyncCpuWaitContext dla_signal_cpu_event_context_;
    cudaExternalSemaphore_t cuda_signal_dla_sema_;
    cudaExternalSemaphore_t cuda_wait_dla_sema_;
    cudaExternalSemaphoreSignalParams cuda_signal_dla_sema_param_;
    cudaExternalSemaphoreWaitParams cuda_wait_dla_sema_param_;
    cudaStream_t stream_;
    cudaError cuda_ret_;
    cudlaStatus dla_ret_;
    NvSciError sci_ret_;
    void checkNvSciErr(NvSciError sci_ret, std::string msg){
        if (sci_ret != NvSciError_Success) std::cout<<"err-code "<<sci_ret<<": "<<msg<<std::endl;
    }
};
typedef std::shared_ptr<GeneralSyncObj> GeneralSyncObjPtr;

class GeneralSyncObjPool{
    public:
    explicit GeneralSyncObjPool();
    GeneralSyncObjPool(GeneralSyncObjPool &&) = delete;
    GeneralSyncObjPool(const GeneralSyncObjPool &) = delete;
    GeneralSyncObjPool & operator= (GeneralSyncObjPool &&) = delete;
    GeneralSyncObjPool & operator= (const GeneralSyncObjPool &) = delete;
    ~GeneralSyncObjPool();

    void createSyncObj(const cudlaDevHandle &dla_handle, const std::string &name);
    void releaseSyncObj(const std::string &name, const cudlaDevHandle &dla_handle);
    void clearSyncObjPool(const cudlaDevHandle &dla_handle);

    GeneralSyncObjPtr getSyncObjPtr(const std::string &name);

    private:
    NvSciSyncModule sync_module_;
    std::unordered_map<std::string, GeneralSyncObjPtr> sync_obj_map_;

    NvSciError sci_ret_;
    cudaError cuda_ret_;
    cudlaStatus dla_ret_;
    void checkNvSciErr(NvSciError sci_ret, std::string msg){
        if (sci_ret != NvSciError_Success) std::cout<<"err-code "<<sci_ret<<": "<<msg<<std::endl;
    }
};

}

#endif