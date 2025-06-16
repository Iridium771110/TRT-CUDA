#include "common.h"
#include "nvsci_buffer.h"
#include "nvsci_synchronize.h"

int main(){
    nvscibuffer::enableCudaDriveApi();
    std::string eng_path = "../dla_test.eng";
    cudaError cuda_ret;
    cudlaDevHandle dev_handle;
    cudlaStatus dla_ret;
    NvSciError sci_ret;
    dla_ret = cudlaCreateDevice(0, &dev_handle, CUDLA_STANDALONE);
    std::vector<char> model_data = common::loadDataFromFile<char, char>(eng_path);
   
    cudlaModule module_handle;
    dla_ret = cudlaModuleLoadFromMemory(dev_handle, reinterpret_cast<uint8_t*>(model_data.data()), model_data.size(), &module_handle, 0);
    if (dla_ret != cudlaSuccess){
        std::cout<<"failed load dla model to dla device"<<std::endl;
    }

    cudlaModuleAttribute dla_model_attr;
    dla_ret = cudlaModuleGetAttributes(module_handle, CUDLA_NUM_INPUT_TENSORS, &dla_model_attr);
    dla_ret = cudlaModuleGetAttributes(module_handle, CUDLA_NUM_OUTPUT_TENSORS, &dla_model_attr);
    int num_dla_input = dla_model_attr.numInputTensors;
    int num_dla_output = dla_model_attr.numOutputTensors;
    std::cout<<num_dla_input<<' '<<num_dla_output<<std::endl;
    std::vector<cudlaModuleTensorDescriptor> dla_input_tensor_desc(num_dla_input);
    std::vector<cudlaModuleTensorDescriptor> dla_output_tensor_desc(num_dla_output);
    dla_model_attr.inputTensorDesc = dla_input_tensor_desc.data();
    dla_ret = cudlaModuleGetAttributes(module_handle, CUDLA_INPUT_TENSOR_DESCRIPTORS, &dla_model_attr); 
    dla_model_attr.outputTensorDesc = dla_output_tensor_desc.data();
    dla_ret = cudlaModuleGetAttributes(module_handle, CUDLA_OUTPUT_TENSOR_DESCRIPTORS, &dla_model_attr);
    if (dla_ret != cudlaSuccess){
        std::cout<<"failed to get dla model attr info"<<std::endl;
    } std::cout<<"finish to get dla model attr info"<<std::endl;

    uint64_t input_byte_size = dla_input_tensor_desc[0].size;
    uint64_t output_byte_size = dla_output_tensor_desc[0].size;

    nvscibuffer::GeneralBufferPool buffer_pool;
    buffer_pool.createBuffer(dev_handle, "input_0", input_byte_size);
    buffer_pool.createBuffer(dev_handle, "output_0", output_byte_size);
    nvscibuffer::GeneralBufferPtr input_buffer_p = buffer_pool.getBufferPtr("input_0");
    nvscibuffer::GeneralBufferPtr output_buffer_p = buffer_pool.getBufferPtr("output_0");
    uint64_t* input_dla_p = input_buffer_p->getDlaPtr();
    uint64_t* output_dla_p = output_buffer_p->getDlaPtr();
    void* input_cpu_p = input_buffer_p->getCpuPtr();
    void* input_gpu_p = input_buffer_p->getGpuPtr();
    void* output_cpu_p = output_buffer_p->getCpuPtr();
    void* output_gpu_p = output_buffer_p->getGpuPtr();
    nvscisync::GeneralSyncObjPool sync_obj_pool;
    sync_obj_pool.createSyncObj(dev_handle, "sync_obj_0");
    nvscisync::GeneralSyncObjPtr sync_obj_p = sync_obj_pool.getSyncObjPtr("sync_obj_0");
    cudlaWaitEvents* dla_wait_cpu_events_p = sync_obj_p->getDlaWaitCpuEventPtr();
    cudlaSignalEvents* dla_signal_cpu_events_p = sync_obj_p->getDlaSignalCpuEventPtr();
    cudlaWaitEvents* dla_wait_cuda_events_p = sync_obj_p->getDlaWaitGpuEventPtr();
    cudlaSignalEvents* dla_signal_cuda_events_p = sync_obj_p->getDlaSignalGpuEventPtr();
    NvSciSyncObj* dla_wait_cpu_event_obj_p = sync_obj_p->getDlaWaitCpuEventObjPtr();
    NvSciSyncCpuWaitContext* dla_signal_cpu_event_context_p = sync_obj_p->getDlaSignalCpuEventContextPtr();
    cudaExternalSemaphore_t* cuda_signal_dla_sema_p = sync_obj_p->getCudaSignalDlaSemaPtr();
    cudaExternalSemaphore_t* cuda_wait_dla_sema_p = sync_obj_p->getCudaWaitDlaSemaPtr();
    cudaExternalSemaphoreSignalParams* cuda_signal_dla_sema_param_p = sync_obj_p->getCudaSignalDlaSemaParamPtr();
    cudaExternalSemaphoreWaitParams* cuda_wait_dla_sema_param_p = sync_obj_p->getCudaWaitDlaSemaParamPtr();


    std::string input_data_path = "../dla_test_input.bin";
    std::string output_ref_path = "../dla_test_output.bin";
    std::vector<char> input_data = common::loadDataFromFile<char, char>(input_data_path);
    std::vector<char> output_ref = common::loadDataFromFile<char, char>(output_ref_path);
    std::vector<half> input_data_f16(4*512*512);
    for (int i = 0; i < 4*512*512; i++) input_data_f16[i] = static_cast<half>(reinterpret_cast<float*>(input_data.data())[i]);
    std::cout<<" finish input data transcript "<<std::endl;

    cudlaTask dla_task;
    dla_task.moduleHandle = module_handle;
    dla_task.outputTensor = &output_dla_p; //should a ** -- theoretically is an array of buffer
    dla_task.numOutputTensors = 1;
    dla_task.numInputTensors = 1;
    dla_task.inputTensor = &input_dla_p; //should a ** -- theoretically is an array of buffer
    dla_task.waitEvents = dla_wait_cpu_events_p;
    dla_task.signalEvents = dla_signal_cpu_events_p;  

    std::memset(input_cpu_p, 0, sizeof(half)*input_data_f16.size());
    std::memset(output_cpu_p, 0, sizeof(half)*32*16*32);
    std::memcpy(input_cpu_p, input_data_f16.data(), sizeof(half)*input_data_f16.size());
    dla_ret = cudlaSubmitTask(dev_handle, &dla_task, 1, NULL, 0);
    if (dla_ret != cudlaSuccess){
        std::cout<<" failed to submit task to dla"<<std::endl;
    } else std::cout<<" finish submit task to dla "<<std::endl;
    sci_ret = NvSciSyncObjSignal(*dla_wait_cpu_event_obj_p);
    sci_ret = NvSciSyncFenceWait(reinterpret_cast<NvSciSyncFence*>(dla_signal_cpu_events_p->eofFences[0].fence), *dla_signal_cpu_event_context_p, -1);
    if (sci_ret != NvSciError_Success){
        std::cout<<" failed to wait dla signal"<<std::endl;
    } else std::cout<<" finish wait dla signal "<<std::endl;

    int last_dim = 16;
    std::vector<float> max_err_cpu(last_dim, 0.0f);
    std::vector<float> rec_ref_cpu(last_dim, 0.0f);
    std::vector<float> rec_res_cpu(last_dim, 0.0f);
    std::vector<float> ref_sum_cpu(last_dim, 0.0f);
    std::vector<float> err_sum_cpu(last_dim, 0.0f);
    for (int n = 0; n <32*16; n++){
        for (int c = 0; c < last_dim; c++){
            float ref = reinterpret_cast<float*>(output_ref.data())[n*16+c];
            float res = reinterpret_cast<half*>(output_cpu_p)[n*32 + c];
            float err = std::abs(ref - res);
            if (err > max_err_cpu[c]){
                max_err_cpu[c] = err;
                rec_ref_cpu[c] = ref;
                rec_res_cpu[c] = res;
            }
            ref_sum_cpu[c] += std::abs(ref);
            err_sum_cpu[c] += err;
        }
    }
    for (int i = 0; i < 16; i++) std::cout<<i<<": res="<<rec_res_cpu[i]<<", ref="<<rec_ref_cpu[i]<<", max_err="<<max_err_cpu[i]
                                            <<", ref_sum="<<ref_sum_cpu[i]<<", err_sum="<<err_sum_cpu[i]<<std::endl;
    std::cout<< "finish cpu-dla test"<<std::endl;


    dla_task.moduleHandle = module_handle;
    dla_task.outputTensor = &output_dla_p; //should a ** -- theoretically is an array of buffer
    dla_task.numOutputTensors = 1;
    dla_task.numInputTensors = 1;
    dla_task.inputTensor = &input_dla_p; //should a ** -- theoretically is an array of buffer
    dla_task.waitEvents = dla_wait_cuda_events_p;
    dla_task.signalEvents = dla_signal_cuda_events_p;  

for (int loop = 0; loop <= 3; loop++){
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemsetAsync(input_gpu_p, 0, 4*512*512*sizeof(half), stream);
    cudaMemsetAsync(output_gpu_p, 0, 32*16*32*sizeof(half), stream);
    cudaMemcpyAsync(input_gpu_p, input_data_f16.data(), 4*512*512*sizeof(half), cudaMemcpyHostToDevice, stream);
    // cudaStreamSynchronize(stream);
    cuda_ret = cudaSignalExternalSemaphoresAsync(cuda_signal_dla_sema_p, cuda_signal_dla_sema_param_p, 1, stream);
    if (cuda_ret != cudaSuccess){
        std::cout<<" failed to cuda signal for dla"<<std::endl;
    } else std::cout<<" finish cuda signal for dla "<<std::endl;
    dla_ret = cudlaSubmitTask(dev_handle, &dla_task, 1, NULL, 0);
    if (dla_ret != cudlaSuccess){
        std::cout<<" failed to submit task to dla"<<std::endl;
    } else std::cout<<" finish submit task to dla "<<std::endl;
    cuda_ret = cudaWaitExternalSemaphoresAsync(cuda_wait_dla_sema_p, cuda_wait_dla_sema_param_p, 1, stream);
    if (cuda_ret != cudaSuccess){
        std::cout<<" failed to cuda wait for dla"<<std::endl;
    } else std::cout<<" finish cuda wait for dla "<<std::endl;

    std::vector<half> output_res(32*16*32);
    cudaMemcpyAsync(output_res.data(), output_gpu_p, sizeof(half)*32*16*32, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::vector<float> max_err(last_dim, 0.0f);
    std::vector<float> rec_ref(last_dim, 0.0f);
    std::vector<float> rec_res(last_dim, 0.0f);
    std::vector<float> ref_sum(last_dim, 0.0f);
    std::vector<float> err_sum(last_dim, 0.0f);
    for (int n = 0; n <32*16; n++){
        for (int c = 0; c < last_dim; c++){
            float ref = reinterpret_cast<float*>(output_ref.data())[n*16+c];
            float res = output_res[n*32 + c];
            float err = std::abs(ref - res);
            if (err > max_err[c]){
                max_err[c] = err;
                rec_ref[c] = ref;
                rec_res[c] = res;
            }
            ref_sum[c] += std::abs(ref);
            err_sum[c] += err;
        }
    }
    for (int i = 0; i < 16; i++) std::cout<<i<<": res="<<rec_res[i]<<", ref="<<rec_ref[i]<<", max_err="<<max_err[i]
                                            <<", ref_sum="<<ref_sum[i]<<", err_sum="<<err_sum[i]<<std::endl;
}
    std::cout<< "finish gpu-dla test"<<std::endl;

    buffer_pool.clearBufferPool(dev_handle);
    sync_obj_pool.clearSyncObjPool(dev_handle);

    return 0;
}