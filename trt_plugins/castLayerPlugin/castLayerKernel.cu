#include "castLayerPlugin.h"

namespace custom_plugin_kernel{
    template<typename InType, typename OutType, typename LoadType, typename WriteType, int tile_element_num>
    __global__ void castKernel(const void* input_p, void* output_p, int tile_num){
        //B=256, t->1*tile, B->256*tile
        int tile_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (tile_id >= tile_num) return;

        LoadType data_load = reinterpret_cast<const LoadType*>(input_p)[tile_id];
        InType* data_load_p = reinterpret_cast<InType*>(&data_load);
        WriteType data_write;
        OutType* data_write_p = reinterpret_cast<OutType*>(&data_write);
        #pragma unroll(tile_element_num)
        for (int i = 0; i < tile_element_num; i++) data_write_p[i] = static_cast<OutType>(data_load_p[i]);
        reinterpret_cast<WriteType*>(output_p)[tile_id] = data_write;
    }

    template<typename InType, typename OutType>
    int customCastKernelExector(const void* input_p, void* output_p, int element_num, cudaStream_t stream){
        int total_in_size = sizeof(InType) * element_num;
        int total_out_size = sizeof(OutType) * element_num;
        int tile_num;
        //情况过多，没想到好办法，仅处理几种特殊情况
        
        if (sizeof(InType) == 4 && sizeof(OutType) == 4 && element_num % 4 == 0){ 
            // std::cout<<element_num<<' '<<sizeof(InType)<<' '<<sizeof(OutType)<<' '<<(element_num/4 + 255)/256<<std::endl;
            (castKernel<InType, OutType, float4, float4, 4>)<<<(element_num/4 + 255)/256, 256, 0, stream>>>(input_p, output_p, element_num/4);}
        else if (sizeof(InType) == 4 && sizeof(OutType) == 2 && element_num % 4 == 0) 
            (castKernel<InType, OutType, float4, float2, 4>)<<<(element_num/4 + 255)/256, 256, 0, stream>>>(input_p, output_p, element_num/4);
        else if (sizeof(InType) == 2 && sizeof(OutType) == 4 && element_num % 4 == 0) 
            (castKernel<InType, OutType, float2, float4, 4>)<<<(element_num/4 + 255)/256, 256, 0, stream>>>(input_p, output_p, element_num/4);
        else if (sizeof(InType) == 4 && sizeof(OutType) == 1 && element_num % 4 == 0) 
            (castKernel<InType, OutType, float4, float1, 4>)<<<(element_num/4 + 255)/256, 256, 0, stream>>>(input_p, output_p, element_num/4);
        else if (sizeof(InType) == 1 && sizeof(OutType) == 4 && element_num % 4 == 0) 
            (castKernel<InType, OutType, float1, float4, 4>)<<<(element_num/4 + 255)/256, 256, 0, stream>>>(input_p, output_p, element_num/4);
        else if (sizeof(InType) == 2 && sizeof(OutType) == 2 && element_num % 8 == 0) 
            (castKernel<InType, OutType, float4, float4, 8>)<<<(element_num/8 + 255)/256, 256, 0, stream>>>(input_p, output_p, element_num/8);
        else if (sizeof(InType) == 2 && sizeof(OutType) == 1 && element_num % 8 == 0) 
            (castKernel<InType, OutType, float4, float2, 8>)<<<(element_num/8 + 255)/256, 256, 0, stream>>>(input_p, output_p, element_num/8);
        else if (sizeof(InType) == 1 && sizeof(OutType) == 2 && element_num % 8 == 0) 
            (castKernel<InType, OutType, float2, float4, 8>)<<<(element_num/8 + 255)/256, 256, 0, stream>>>(input_p, output_p, element_num/8);
        else{
            tile_num = element_num;
            dim3 dimBlock(256);
            dim3 dimGrid((tile_num + 255) / 256);
            (castKernel<InType, OutType, InType, OutType, 1>)<<<dimGrid, dimBlock, 0, stream>>>(input_p, output_p, tile_num);
        }
        // cudaDeviceSynchronize();
        // std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;
        return 0;
    }

    // template int customCastKernelExector<float, float>(const void* input_p, void* output_p, int element_num, cudaStream_t stream);
}

std::unordered_map<int32_t, custom_plugin_kernel::customCastKernelExectorFuncPtr> nvinfer1::plugin::CastLayerPlugin::executor_map_ = {
    {convertNvDType2Key(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kFLOAT), custom_plugin_kernel::customCastKernelExector<float, float>},
    {convertNvDType2Key(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT32), custom_plugin_kernel::customCastKernelExector<float, int32_t>},
    {convertNvDType2Key(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kHALF), custom_plugin_kernel::customCastKernelExector<float, half>},
    {convertNvDType2Key(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), custom_plugin_kernel::customCastKernelExector<float, int8_t>},
    {convertNvDType2Key(nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT), custom_plugin_kernel::customCastKernelExector<half, float>},
    {convertNvDType2Key(nvinfer1::DataType::kHALF, nvinfer1::DataType::kHALF), custom_plugin_kernel::customCastKernelExector<half, half>},
    // {convertNvDType2Key(nvinfer1::DataType::kHALF, nvinfer1::DataType::kINT8), custom_plugin_kernel::customCastKernelExector<half, int8_t>},
    {convertNvDType2Key(nvinfer1::DataType::kHALF, nvinfer1::DataType::kINT32), custom_plugin_kernel::customCastKernelExector<half, int32_t>},
    {convertNvDType2Key(nvinfer1::DataType::kINT32, nvinfer1::DataType::kINT8), custom_plugin_kernel::customCastKernelExector<int32_t, int8_t>},
    {convertNvDType2Key(nvinfer1::DataType::kINT32, nvinfer1::DataType::kHALF), custom_plugin_kernel::customCastKernelExector<int32_t, half>},
    {convertNvDType2Key(nvinfer1::DataType::kINT32, nvinfer1::DataType::kFLOAT), custom_plugin_kernel::customCastKernelExector<int32_t, float>},
    {convertNvDType2Key(nvinfer1::DataType::kINT32, nvinfer1::DataType::kINT32), custom_plugin_kernel::customCastKernelExector<int32_t, int32_t>},
    {convertNvDType2Key(nvinfer1::DataType::kINT8, nvinfer1::DataType::kFLOAT), custom_plugin_kernel::customCastKernelExector<int8_t, float>},
    {convertNvDType2Key(nvinfer1::DataType::kINT8, nvinfer1::DataType::kINT32), custom_plugin_kernel::customCastKernelExector<int8_t, int32_t>},
    {convertNvDType2Key(nvinfer1::DataType::kINT8, nvinfer1::DataType::kHALF), custom_plugin_kernel::customCastKernelExector<int8_t, half>},
    {convertNvDType2Key(nvinfer1::DataType::kINT8, nvinfer1::DataType::kINT8), custom_plugin_kernel::customCastKernelExector<int8_t, int8_t>}
};

