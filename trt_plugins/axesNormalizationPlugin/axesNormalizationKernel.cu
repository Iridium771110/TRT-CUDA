#include "axesNormalizationPlugin.h"

namespace custom_plugin_kernel{
    //----------------------------instancenorm----------------------
    //精细调整应当考虑tail维的大小情况，并且对不同的维数广播情况做出对应处理
    //因此目前只对BCT的三维形式进行支持-》batch，channel，tail。并且scale、bias认为只有channel维，norm只对tail维
    //目前仅考虑batch处理batchchannel以及(warp处理batchchannel 再议)
    //内部运算使用float，因此要求scale、bias为float
    template<typename dataType, typename tileType, int tile_ele_num>
    __global__ void tailBCTNormalizationScaleChannelBlockCommonKernel(const char* input_p, char* output_p, const float* scale_p, const float* bias_p,
                                                        const int channels, const int block_tile_num, const int intern_block_round, const float eps){
        //B=128t = 4warps, B->1batchchannel
        int channel_id = blockIdx.x % channels;
        int batch_id = blockIdx.x / channels;
        __shared__ float sh_sum[128];
        __shared__ float sh_sqr_sum[128];
        __shared__ float sh_mean[32];
        __shared__ float sh_mul_fac[32];
        __shared__ float sh_bias[32];

        float th_sum = 0.0f;
        int block_ele_num = tile_ele_num * block_tile_num;
        float th_sqr_sum = 0.0f;
        const dataType* input_data_type_p = reinterpret_cast<const dataType*>(input_p);
        const tileType* block_input_tile_p = reinterpret_cast<const tileType*>(input_data_type_p + blockIdx.x*block_ele_num);
        for (int i = 0; i < intern_block_round; i++){
            int th_tile_id = static_cast<int>(threadIdx.x) + i*128;
            float load_fac = static_cast<float>(max(0, min(block_tile_num - th_tile_id, 1)));
            th_tile_id = min(th_tile_id, block_tile_num - 1);
            dataType reg_data[tile_ele_num];
            reinterpret_cast<tileType*>(reg_data)[0] = block_input_tile_p[th_tile_id];
            #pragma unroll(tile_ele_num)
            for (int j = 0; j < tile_ele_num; j++){
                float tmp_data = static_cast<float>(reg_data[j]) * load_fac;
                th_sum += tmp_data;
                th_sqr_sum += tmp_data * tmp_data;
            } 
        }

        #pragma unroll(5)
        for (int i = 0; i < 5; i++){
            th_sum += __shfl_xor_sync(0xffffffff, th_sum, 0x1 << i, 32);
            th_sqr_sum += __shfl_xor_sync(0xffffffff, th_sqr_sum, 0x1 << i, 32);
        }
        int warp_id = threadIdx.x / 32;
        int warp_tx = threadIdx.x % 32;
        //因为B=128,最后是4个有效值
        sh_sum[warp_tx*4 + (warp_id ^ (warp_tx / 8))] = th_sum;
        sh_sqr_sum[warp_tx*4 + (warp_id ^ (warp_tx / 8))] = th_sqr_sum;
        __syncthreads();
        if (threadIdx.x < 32){
            float reg_scale = scale_p[channel_id];
            sh_bias[threadIdx.x] = bias_p[channel_id];
            float tmp_sum[4];
            reinterpret_cast<float4*>(tmp_sum)[0] = reinterpret_cast<float4*>(sh_sum)[threadIdx.x];
            #pragma unroll(3)
            for (int i = 1; i < 4; i++) tmp_sum[0] += tmp_sum[i];
            float tmp_mean = tmp_sum[0] / static_cast<float>(block_ele_num);
            sh_mean[threadIdx.x] = tmp_mean;
            reinterpret_cast<float4*>(tmp_sum)[0] = reinterpret_cast<float4*>(sh_sqr_sum)[threadIdx.x];
            #pragma unroll(3)
            for (int i = 1; i < 4; i++) tmp_sum[0] += tmp_sum[i];
            float tmp_sqr_mean = tmp_sum[0] / static_cast<float>(block_ele_num);
            float tmp_var = tmp_sqr_mean - tmp_mean * tmp_mean;
            sh_mul_fac[threadIdx.x] = reg_scale / sqrt(tmp_var + eps);
            // sh_mul_fac[threadIdx.x] = reg_scale * rsqrt(tmp_var + eps);
        }
        __syncthreads();

        dataType* output_data_type_p = reinterpret_cast<dataType*>(output_p);
        tileType* block_output_tile_p = reinterpret_cast<tileType*>(output_data_type_p + blockIdx.x*block_ele_num);
        float reg_mean = sh_mean[0]; // 可检查32个值是否一致
        float reg_mul_fac = sh_mul_fac[0];
        float reg_bias = sh_bias[0];
        for (int i = 0; i < intern_block_round; i++){
            int th_tile_id = static_cast<int>(threadIdx.x) + i*128;
            th_tile_id = min(th_tile_id, block_tile_num - 1);
            dataType reg_data[tile_ele_num];
            reinterpret_cast<tileType*>(reg_data)[0] = block_input_tile_p[th_tile_id];
            #pragma unroll(tile_ele_num)
            for (int j = 0; j < tile_ele_num; j++) reg_data[j] = static_cast<dataType>((static_cast<float>(reg_data[j]) - reg_mean) * reg_mul_fac + reg_bias);
            block_output_tile_p[th_tile_id] = reinterpret_cast<tileType*>(reg_data)[0];
        }
    }

    typedef void (*tailBCTNormalizationScaleChannelBlockCommonKernelPtr)(const char*, char*, const float*, const float*,
                                                                            const int, const int, const int, const float);
    static std::unordered_map<std::string, tailBCTNormalizationScaleChannelBlockCommonKernelPtr> tail_instance_normalization_kernel_map = {
        {"1-float-float4", tailBCTNormalizationScaleChannelBlockCommonKernel<float, float4, 4>},
        {"1-half-float4", tailBCTNormalizationScaleChannelBlockCommonKernel<half, float4, 8>},
        {"1-float-float2", tailBCTNormalizationScaleChannelBlockCommonKernel<float, float2, 2>},
        {"1-half-float2", tailBCTNormalizationScaleChannelBlockCommonKernel<half, float2, 4>},
        {"1-float-float", tailBCTNormalizationScaleChannelBlockCommonKernel<float, float, 1>},
        {"1-half-half", tailBCTNormalizationScaleChannelBlockCommonKernel<half, half, 1>}
    };

    //----------------------------layernorm----------------------
    //精细调整应当考虑tail维的大小情况，并且对不同的维数广播情况做出对应处理
    //因此目前只对BCT的三维形式进行支持-》batch，channel，tail。并且scale、bias认为只有channel维，norm只对tail维
    //目前仅考虑batch处理batchchannel以及(warp处理batchchannel 再议)
    //内部运算使用float，因此要求scale、bias为float
    template<typename dataType, typename tileType, int tile_ele_num, typename scaletileType, int scale_load_round>
    __global__ void tailBCTNormalizationScaleTailWarpCommonKernel(const char* input_p, char* output_p, const float* scale_p, const float* bias_p,
                                                        const int valid_warp_num, const int warp_tile_num, const int intern_warp_round, 
                                                        const int scale_dim, const float eps){
        //B=256t = 8warps, w->1batchchannel
        //暂时对scale、bias只做单体载入处理
        int gl_warp_id = threadIdx.x / 32 + blockIdx.x * 8;
        int block_warp_id = threadIdx.x / 32;
        int warp_tx = threadIdx.x % 32;

        float th_sum = 0.0f;
        int warp_ele_num = tile_ele_num * warp_tile_num;
        float th_sqr_sum = 0.0f;
        const dataType* input_data_type_p = reinterpret_cast<const dataType*>(input_p);
        const tileType* warp_input_tile_p = reinterpret_cast<const tileType*>(input_data_type_p + gl_warp_id*warp_ele_num);

        if (gl_warp_id < valid_warp_num){
            for (int i = 0; i < intern_warp_round; i++){
                int th_tile_id = warp_tx + i*32;
                float load_fac = static_cast<float>(max(0, min(warp_tile_num - th_tile_id, 1)));
                th_tile_id = min(th_tile_id, warp_tile_num - 1);
                dataType reg_data[tile_ele_num];
                reinterpret_cast<tileType*>(reg_data)[0] = warp_input_tile_p[th_tile_id];
                #pragma unroll(tile_ele_num)
                for (int j = 0; j < tile_ele_num; j++){
                    float tmp_data = static_cast<float>(reg_data[j]) * load_fac;
                    th_sum += tmp_data;
                    th_sqr_sum += tmp_data * tmp_data;
                } 
            }
            #pragma unroll(5)
            for (int i = 0; i < 5; i++){
                th_sum += __shfl_xor_sync(0xffffffff, th_sum, 0x1 << i, 32);
                th_sqr_sum += __shfl_xor_sync(0xffffffff, th_sqr_sum, 0x1 << i, 32);
            }
        }
        float th_mean = th_sum / static_cast<float>(warp_ele_num);
        float th_var = th_sqr_sum / static_cast<float>(warp_ele_num) - th_mean * th_mean;
        float th_mul_fac = 1.0f / sqrt(th_var + eps);

        dataType* output_data_type_p = reinterpret_cast<dataType*>(output_p);
        tileType* warp_output_tile_p = reinterpret_cast<tileType*>(output_data_type_p + gl_warp_id*warp_ele_num);
        // __shared__ float sh_scale[256*tile_ele_num];
        // __shared__ float sh_bias[256*tile_ele_num];__syncthreads();
        // #pragma unroll(scale_load_round)
        // for (int r = 0; r < scale_load_round; r++){
        //     reinterpret_cast<scaletileType*>(sh_scale)[] = reinterpret_cast<scaletileType*>(scale_p)[th_tile_id*scale_load_round + r];
        //     reinterpret_cast<scaletileType*>(sh_bias)[] = reinterpret_cast<scaletileType*>(bias_p)[th_tile_id*scale_load_round + r];
        // } 
        // __syncthreads();
        for (int i = 0; i < intern_warp_round; i++){
            int th_tile_id = warp_tx + i*32;
            th_tile_id = min(th_tile_id, warp_tile_num - 1);
            dataType reg_data[tile_ele_num];
            reinterpret_cast<tileType*>(reg_data)[0] = warp_input_tile_p[th_tile_id];
            #pragma unroll(tile_ele_num)
            for (int j = 0; j < tile_ele_num; j++){
                int scale_bias_id = th_tile_id * tile_ele_num + j;
                float reg_scale = scale_p[scale_bias_id];
                float reg_bias = bias_p[scale_bias_id];
                reg_data[j] = static_cast<dataType>((static_cast<float>(reg_data[j]) - th_mean) * reg_scale * th_mul_fac + reg_bias);
            } 
            warp_output_tile_p[th_tile_id] = reinterpret_cast<tileType*>(reg_data)[0];
        }
    }

    typedef void (*tailBCTNormalizationScaleTailWarpCommonKernelPtr)(const char*, char*, const float*, const float*,
                                                                            const int, const int, const int, const int, const float);
    static std::unordered_map<std::string, tailBCTNormalizationScaleTailWarpCommonKernelPtr> tail_layer_normalization_kernel_map = {
        {"2-float-float4", tailBCTNormalizationScaleTailWarpCommonKernel<float, float4, 4, float4, 1>},
        {"2-half-float4", tailBCTNormalizationScaleTailWarpCommonKernel<half, float4, 8, float4, 2>},
        {"2-float-float2", tailBCTNormalizationScaleTailWarpCommonKernel<float, float2, 2, float2, 1>},
        {"2-half-float2", tailBCTNormalizationScaleTailWarpCommonKernel<half, float2, 4, float4, 1>},
        {"2-float-float", tailBCTNormalizationScaleTailWarpCommonKernel<float, float, 1, float , 1>},
        {"2-half-half", tailBCTNormalizationScaleTailWarpCommonKernel<half, half, 1, float , 1>}
    };

    static std::unordered_map<std::string, int> data_type_size_table = {
        {"float", 4},
        {"half", 2}
    };
    
    int customTailNormalizationKernelExector(const void* input_p, void* output_p, const float* scale_p, const float* bias_p, float eps,
                                            std::vector<int> &shape, std::string &data_type, int scale_dim_id, cudaStream_t stream){
        //设这里的shape应当是已经完成整理的3维BCT
        int dim_tail = shape[2];
        std::string func_name = std::to_string(scale_dim_id) + "-" + data_type + "-";
        if (data_type_size_table.count(data_type) == 0){
            std::cout << "unsupported data type yet in tail-normalization-kernel, use float or half"<<std::endl;
            return -2;
        }
        int ele_size = data_type_size_table.at(data_type);
        int aligned_size = dim_tail * ele_size % 16;
        int tile_size;
        if (aligned_size == 0){
            func_name = func_name + "float4";
            tile_size = 16;
        } 
        else if (aligned_size == 8){
            func_name = func_name + "float2";
            tile_size = 8;
        } 
        else {
            func_name = func_name + data_type;
            tile_size = ele_size;
        }
        int tile_ele_num = tile_size / ele_size;
        int unit_tile_num = dim_tail / tile_ele_num;
        int deal_unit_num = shape[0] * shape[1];

        if (scale_dim_id == 1){
            dim3 dimBlock(128);
            dim3 dimGrid(deal_unit_num);
            int channels = shape[1];
            int intern_block_round = (unit_tile_num + 127) / 128;
            tail_instance_normalization_kernel_map.at(func_name)<<<dimGrid, dimBlock, 0, stream>>>(
                                                        reinterpret_cast<const char*>(input_p), reinterpret_cast<char*>(output_p), scale_p, bias_p,
                                                        channels, unit_tile_num, intern_block_round, eps);
        }
        else if (scale_dim_id == 2){
            dim3 dimBlock(256);
            dim3 dimGrid((deal_unit_num + 7) / 8);
            int tail_dim = shape[2];
            int intern_warp_round = (unit_tile_num + 31) / 32;
            tail_layer_normalization_kernel_map.at(func_name)<<<dimGrid, dimBlock, 0, stream>>>(
                                                        reinterpret_cast<const char*>(input_p), reinterpret_cast<char*>(output_p), scale_p, bias_p,
                                                        deal_unit_num, unit_tile_num, intern_warp_round, tail_dim, eps);
        }
        else{
            std::cout << "AxesNormalization scale dim failed, excepts 1 for layer norm or 2 for instance norm, but get " << scale_dim_id << std::endl;
            return -2;
        }
        
        return 0;
    }
}
