#include "nvsci_synchronize.h"

namespace nvscisync{

GeneralSyncObj::GeneralSyncObj(const cudlaDevHandle &dla_handle, const NvSciSyncModule &sync_module){
    sci_ret_ = NvSciSyncAttrListCreate(sync_module, &wait_cpu_event_waiter_attr_list_);
    sci_ret_ = NvSciSyncAttrListCreate(sync_module, &wait_cpu_event_signaler_attr_list_);
    dla_ret_ = cudlaGetNvSciSyncAttributes(reinterpret_cast<uint64_t*>(wait_cpu_event_waiter_attr_list_), CUDLA_NVSCISYNC_ATTR_WAIT);
    bool cpu_signaler = true;
    NvSciSyncAttrKeyValuePair cpu_signaler_keyValue[2];
    memset(cpu_signaler_keyValue, 0, sizeof(cpu_signaler_keyValue));
    NvSciSyncAccessPerm cpu_signaler_perm = NvSciSyncAccessPerm_SignalOnly;
    cpu_signaler_keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    cpu_signaler_keyValue[0].value = (void*)(&cpu_signaler);
    cpu_signaler_keyValue[0].len = sizeof(cpu_signaler);
    cpu_signaler_keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    cpu_signaler_keyValue[1].value = (void*)(&cpu_signaler_perm);
    cpu_signaler_keyValue[1].len = sizeof(cpu_signaler_perm);
    sci_ret_ = NvSciSyncAttrListSetAttrs(wait_cpu_event_signaler_attr_list_, cpu_signaler_keyValue, 2);
    NvSciSyncAttrList wait_event_attrs[2] = {wait_cpu_event_signaler_attr_list_, wait_cpu_event_waiter_attr_list_};
    sci_ret_ = NvSciSyncAttrListReconcile(wait_event_attrs, 2, &wait_cpu_event_reconciled_attr_list_, &wait_cpu_event_conflict_attr_list_);
    sci_ret_ = NvSciSyncObjAlloc(wait_cpu_event_reconciled_attr_list_, &dla_wait_cpu_event_obj_);

    sci_ret_ = NvSciSyncAttrListCreate(sync_module, &wait_cuda_event_waiter_attr_list_);
    sci_ret_ = NvSciSyncAttrListCreate(sync_module, &wait_cuda_event_signaler_attr_list_);
    dla_ret_ = cudlaGetNvSciSyncAttributes(
                reinterpret_cast<uint64_t*>(wait_cuda_event_waiter_attr_list_), 
                CUDLA_NVSCISYNC_ATTR_SIGNAL);
    cuda_ret_ = cudaDeviceGetNvSciSyncAttributes(
                wait_cuda_event_signaler_attr_list_, 
                0, 
                cudaNvSciSyncAttrWait); //??? like a bug, inverse setting but is ok
    NvSciSyncAttrList wait_cuda_event_attrs[2] = {wait_cuda_event_signaler_attr_list_, wait_cuda_event_waiter_attr_list_};
    sci_ret_ = NvSciSyncAttrListReconcile(wait_cuda_event_attrs, 2, &wait_cuda_event_reconciled_attr_list_, &wait_cuda_event_conflict_attr_list_);
    sci_ret_ = NvSciSyncObjAlloc(wait_cuda_event_reconciled_attr_list_, &dla_wait_cuda_event_obj_);

    sci_ret_ = NvSciSyncAttrListCreate(sync_module, &signal_cpu_event_waiter_attr_list_);
    sci_ret_ = NvSciSyncAttrListCreate(sync_module, &signal_cpu_event_signaler_attr_list_);
    dla_ret_ = cudlaGetNvSciSyncAttributes(reinterpret_cast<uint64_t*>(signal_cpu_event_signaler_attr_list_), CUDLA_NVSCISYNC_ATTR_SIGNAL);
    bool cpu_waiter = true;
    NvSciSyncAttrKeyValuePair cpu_waiter_keyValue[2];
    memset(cpu_waiter_keyValue, 0, sizeof(cpu_waiter_keyValue));
    NvSciSyncAccessPerm cpu_waiter_perm = NvSciSyncAccessPerm_WaitOnly;
    cpu_waiter_keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    cpu_waiter_keyValue[0].value = (void*)(&cpu_waiter);
    cpu_waiter_keyValue[0].len = sizeof(cpu_waiter);
    cpu_waiter_keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    cpu_waiter_keyValue[1].value = (void*)(&cpu_waiter_perm);
    cpu_waiter_keyValue[1].len = sizeof(cpu_waiter_perm);
    sci_ret_ = NvSciSyncAttrListSetAttrs(signal_cpu_event_waiter_attr_list_, cpu_waiter_keyValue, 2);
    NvSciSyncAttrList signal_event_attrs[2] = {signal_cpu_event_signaler_attr_list_, signal_cpu_event_waiter_attr_list_};
    sci_ret_ = NvSciSyncAttrListReconcile(signal_event_attrs, 2, &signal_cpu_event_reconciled_attr_list_, &signal_cpu_event_conflict_attr_list_);
    sci_ret_ = NvSciSyncObjAlloc(signal_cpu_event_reconciled_attr_list_, &dla_signal_cpu_event_obj_);
    sci_ret_ = NvSciSyncCpuWaitContextAlloc(sync_module, &dla_signal_cpu_event_context_);

    sci_ret_ = NvSciSyncAttrListCreate(sync_module, &signal_cuda_event_waiter_attr_list_);
    sci_ret_ = NvSciSyncAttrListCreate(sync_module, &signal_cuda_event_signaler_attr_list_);
    dla_ret_ = cudlaGetNvSciSyncAttributes(reinterpret_cast<uint64_t*>(signal_cuda_event_signaler_attr_list_), CUDLA_NVSCISYNC_ATTR_SIGNAL);
    cuda_ret_ = cudaDeviceGetNvSciSyncAttributes(signal_cuda_event_waiter_attr_list_, 0, cudaNvSciSyncAttrWait);
    NvSciSyncAttrList signal_cuda_event_attrs[2] = {signal_cuda_event_signaler_attr_list_, signal_cuda_event_waiter_attr_list_};
    sci_ret_ = NvSciSyncAttrListReconcile(signal_cuda_event_attrs, 2, &signal_cuda_event_reconciled_attr_list_, &signal_cuda_event_conflict_attr_list_);
    sci_ret_ = NvSciSyncObjAlloc(signal_cuda_event_reconciled_attr_list_, &dla_signal_cuda_event_obj_);

    cudlaExternalSemaphoreHandleDesc sema_mem_desc = {0};
    memset(&sema_mem_desc, 0, sizeof(sema_mem_desc));
    sema_mem_desc.extSyncObject = dla_wait_cpu_event_obj_;
    dla_ret_ = cudlaImportExternalSemaphore(dla_handle, &sema_mem_desc, &dla_wait_cpu_event_reg_p_, 0);
    memset(&sema_mem_desc, 0, sizeof(sema_mem_desc));
    sema_mem_desc.extSyncObject = dla_signal_cpu_event_obj_;
    dla_ret_ = cudlaImportExternalSemaphore(dla_handle, &sema_mem_desc, &dla_signal_cpu_event_reg_p_, 0);
    memset(&sema_mem_desc, 0, sizeof(sema_mem_desc));
    sema_mem_desc.extSyncObject = dla_wait_cuda_event_obj_;
    dla_ret_ = cudlaImportExternalSemaphore(dla_handle, &sema_mem_desc, &dla_wait_cuda_event_reg_p_, 0);
    memset(&sema_mem_desc, 0, sizeof(sema_mem_desc));
    sema_mem_desc.extSyncObject = dla_signal_cuda_event_obj_;
    dla_ret_ = cudlaImportExternalSemaphore(dla_handle, &sema_mem_desc, &dla_signal_cuda_event_reg_p_, 0);
    cudaExternalSemaphoreHandleDesc cuda_ext_sem_desc;
    memset(&cuda_ext_sem_desc, 0, sizeof(cuda_ext_sem_desc));
    cuda_ext_sem_desc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    cuda_ext_sem_desc.handle.nvSciSyncObj = (void *)(dla_wait_cuda_event_obj_);
    cuda_ret_ = cudaImportExternalSemaphore(&cuda_signal_dla_sema_, &cuda_ext_sem_desc);
    memset(&cuda_ext_sem_desc, 0, sizeof(cuda_ext_sem_desc));
    cuda_ext_sem_desc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    cuda_ext_sem_desc.handle.nvSciSyncObj = (void *)(dla_signal_cuda_event_obj_);
    cuda_ret_ = cudaImportExternalSemaphore(&cuda_wait_dla_sema_, &cuda_ext_sem_desc);

    dla_wait_cpu_pre_fence_ = NvSciSyncFenceInitializer;
    NvSciSyncObjGenerateFence(dla_wait_cpu_event_obj_, &dla_wait_cpu_pre_fence_);
    dla_wait_cpu_events_p_ = (cudlaWaitEvents*)malloc(sizeof(cudlaWaitEvents));
    dla_wait_cpu_events_p_->numEvents = 1;
    dla_wait_cpu_pre_fences_p_ = (CudlaFence*)malloc(dla_wait_cpu_events_p_->numEvents * sizeof(CudlaFence));
    dla_wait_cpu_pre_fences_p_[0].fence = &dla_wait_cpu_pre_fence_;
    dla_wait_cpu_pre_fences_p_[0].type = CUDLA_NVSCISYNC_FENCE;
    dla_wait_cpu_events_p_->preFences = dla_wait_cpu_pre_fences_p_;

    dla_signal_cpu_events_p_ = (cudlaSignalEvents*)malloc(sizeof(cudlaSignalEvents));
    dla_signal_cpu_events_p_->numEvents = 1;
    dla_cpu_devs_p_ = (uint64_t**)malloc(dla_signal_cpu_events_p_->numEvents * sizeof(uint64_t*));
    dla_cpu_devs_p_[0] = dla_signal_cpu_event_reg_p_;
    dla_signal_cpu_events_p_->devPtrs = dla_cpu_devs_p_;
    dla_signal_cpu_eof_fence_ = NvSciSyncFenceInitializer;
    dla_signal_cpu_events_p_->eofFences = (CudlaFence*)malloc(dla_signal_cpu_events_p_->numEvents * sizeof(CudlaFence));
    dla_signal_cpu_events_p_->eofFences[0].fence = &dla_signal_cpu_eof_fence_;
    dla_signal_cpu_events_p_->eofFences[0].type = CUDLA_NVSCISYNC_FENCE;

    dla_wait_cuda_pre_fence_ = NvSciSyncFenceInitializer;
    NvSciSyncObjGenerateFence(dla_wait_cuda_event_obj_, &dla_wait_cuda_pre_fence_);
    dla_wait_cuda_events_p_ = (cudlaWaitEvents*)malloc(sizeof(cudlaWaitEvents));
    dla_wait_cuda_events_p_->numEvents = 1;
    dla_wait_cuda_pre_fences_p_ = (CudlaFence*)malloc(dla_wait_cuda_events_p_->numEvents * sizeof(CudlaFence));
    dla_wait_cuda_pre_fences_p_[0].fence = &dla_wait_cuda_pre_fence_;
    dla_wait_cuda_pre_fences_p_[0].type = CUDLA_NVSCISYNC_FENCE;
    dla_wait_cuda_events_p_->preFences = dla_wait_cuda_pre_fences_p_;
    memset(&cuda_signal_dla_sema_param_, 0, sizeof(cuda_signal_dla_sema_param_));
    cuda_signal_dla_sema_param_.params.nvSciSync.fence = (void *)(&dla_wait_cuda_pre_fence_);
    cuda_signal_dla_sema_param_.flags = 0;

    dla_signal_cuda_events_p_ = (cudlaSignalEvents*)malloc(sizeof(cudlaSignalEvents));
    dla_signal_cuda_events_p_->numEvents = 1;
    dla_gpu_devs_p_ = (uint64_t**)malloc(dla_signal_cuda_events_p_->numEvents * sizeof(uint64_t*));
    dla_gpu_devs_p_[0] = dla_signal_cuda_event_reg_p_;
    dla_signal_cuda_events_p_->devPtrs = dla_gpu_devs_p_;
    dla_signal_cuda_eof_fence_ = NvSciSyncFenceInitializer;
    dla_signal_cuda_events_p_->eofFences = (CudlaFence*)malloc(dla_signal_cuda_events_p_->numEvents * sizeof(CudlaFence));
    dla_signal_cuda_events_p_->eofFences[0].fence = &dla_signal_cuda_eof_fence_;
    dla_signal_cuda_events_p_->eofFences[0].type = CUDLA_NVSCISYNC_FENCE;
    memset(&cuda_wait_dla_sema_param_, 0, sizeof(cuda_wait_dla_sema_param_));
    cuda_wait_dla_sema_param_.params.nvSciSync.fence = (void *)(&dla_signal_cuda_eof_fence_);
    cuda_wait_dla_sema_param_.flags = 0;
}
GeneralSyncObj::~GeneralSyncObj(){
    free(dla_cpu_devs_p_);
    free(dla_gpu_devs_p_);
    dla_cpu_devs_p_ = nullptr;
    dla_gpu_devs_p_ = nullptr;
    free(dla_wait_cpu_pre_fences_p_);
    free(dla_signal_cpu_events_p_->eofFences);
    free(dla_wait_cuda_pre_fences_p_);
    free(dla_signal_cuda_events_p_->eofFences);
    dla_wait_cpu_pre_fences_p_ = nullptr;
    dla_wait_cuda_pre_fences_p_ = nullptr;
    dla_wait_cpu_events_p_->preFences = nullptr;
    dla_signal_cpu_events_p_->eofFences = nullptr;
    dla_wait_cuda_events_p_->preFences = nullptr;
    dla_signal_cuda_events_p_->eofFences = nullptr;
    NvSciSyncFenceClear(&dla_wait_cpu_pre_fence_);
    NvSciSyncFenceClear(&dla_signal_cpu_eof_fence_);
    NvSciSyncFenceClear(&dla_wait_cuda_pre_fence_);
    NvSciSyncFenceClear(&dla_signal_cuda_eof_fence_);
    free(dla_wait_cpu_events_p_);
    free(dla_signal_cpu_events_p_);
    free(dla_wait_cuda_events_p_);
    free(dla_signal_cuda_events_p_);
    dla_wait_cpu_events_p_ = nullptr;
    dla_signal_cpu_events_p_ = nullptr;
    dla_wait_cuda_events_p_ = nullptr;
    dla_signal_cuda_events_p_ = nullptr;
    NvSciSyncObjFree(dla_wait_cpu_event_obj_);
    NvSciSyncAttrListFree(wait_cpu_event_waiter_attr_list_);
    NvSciSyncAttrListFree(wait_cpu_event_signaler_attr_list_);
    NvSciSyncAttrListFree(wait_cpu_event_reconciled_attr_list_);
    NvSciSyncAttrListFree(wait_cpu_event_conflict_attr_list_);
    NvSciSyncObjFree(dla_signal_cpu_event_obj_);
    NvSciSyncAttrListFree(signal_cpu_event_waiter_attr_list_);
    NvSciSyncAttrListFree(signal_cpu_event_signaler_attr_list_);
    NvSciSyncAttrListFree(signal_cpu_event_reconciled_attr_list_);
    NvSciSyncAttrListFree(signal_cpu_event_conflict_attr_list_);
    NvSciSyncObjFree(dla_wait_cuda_event_obj_);
    NvSciSyncAttrListFree(wait_cuda_event_waiter_attr_list_);
    NvSciSyncAttrListFree(wait_cuda_event_signaler_attr_list_);
    NvSciSyncAttrListFree(wait_cuda_event_reconciled_attr_list_);
    NvSciSyncAttrListFree(wait_cuda_event_conflict_attr_list_);
    NvSciSyncObjFree(dla_signal_cuda_event_obj_);
    NvSciSyncAttrListFree(signal_cuda_event_waiter_attr_list_);
    NvSciSyncAttrListFree(signal_cuda_event_signaler_attr_list_);
    NvSciSyncAttrListFree(signal_cuda_event_reconciled_attr_list_);
    NvSciSyncAttrListFree(signal_cuda_event_conflict_attr_list_);
}

void GeneralSyncObj::unregisterSyncObj(const cudlaDevHandle &dla_handle){
    dla_ret_ = cudlaMemUnregister(dla_handle, dla_wait_cpu_event_reg_p_);
    dla_ret_ = cudlaMemUnregister(dla_handle, dla_signal_cpu_event_reg_p_);
    dla_ret_ = cudlaMemUnregister(dla_handle, dla_wait_cuda_event_reg_p_);
    dla_ret_ = cudlaMemUnregister(dla_handle, dla_signal_cuda_event_reg_p_);
}

GeneralSyncObjPool::GeneralSyncObjPool(){
    sci_ret_ = NvSciSyncModuleOpen(&sync_module_);
}
GeneralSyncObjPool::~GeneralSyncObjPool(){
    NvSciSyncModuleClose(sync_module_);
}

void GeneralSyncObjPool::createSyncObj(const cudlaDevHandle &dla_handle, const std::string &name){
    GeneralSyncObjPtr sync_obj_p = std::make_shared<GeneralSyncObj>(dla_handle, sync_module_);
    sync_obj_map_.emplace(name, sync_obj_p);
    sync_obj_p = nullptr;
}
void GeneralSyncObjPool::releaseSyncObj(const std::string &name, const cudlaDevHandle &dla_handle){
    GeneralSyncObjPtr sync_obj_p = sync_obj_map_.at(name);
    sync_obj_p->unregisterSyncObj(dla_handle);
    sync_obj_map_.erase(name);
    sync_obj_p = nullptr;
}
void GeneralSyncObjPool::clearSyncObjPool(const cudlaDevHandle &dla_handle){
    for (std::unordered_map<std::string, GeneralSyncObjPtr>::iterator iter = sync_obj_map_.begin(); iter != sync_obj_map_.end(); iter++){
        iter->second->unregisterSyncObj(dla_handle);
        iter->second = nullptr;
    }
    sync_obj_map_.clear();
}
GeneralSyncObjPtr GeneralSyncObjPool::getSyncObjPtr(const std::string &name){
    return sync_obj_map_.at(name);
}

}