#include "kornia_perspective_module.h"

namespace module{

__global__ void korniaPerspectiveFillZeroNoAlignNearestBCHWKernel(float* dst, float* src, float* M, int channel, int src_h, int src_w, int dst_h, int dst_w){
    //tmp: block->batch B=128T
    //nearest, fill zero, align false
    //dst size % 128 == 0
    int batch_id = blockIdx.x;
    int warp_id = threadIdx.x / 32;
    int warp_tx = threadIdx.x % 32;
    __shared__ float sh_src_norm_trans_dst_norm[12];
    float* M_p = M + batch_id * 9;
    if (warp_id == 0 && warp_tx == 0){
        float src_pix_trans_src_norm[9] = {
            (static_cast<float>(src_w) - 1.0f) / 2.0f, 0.0f, (static_cast<float>(src_w) - 1.0f) / 2.0f,
            0.0f, (static_cast<float>(src_h) - 1.0f) / 2.0f, (static_cast<float>(src_h) - 1.0f) / 2.0f,
            0.0f, 0.0f, 1.0f
        };
        float reg_M[9];
        #pragma unroll(9)
        for (int i = 0; i < 9; i++) reg_M[i] = M_p[i];
        float tmp[9];
        #pragma unroll(3)
        for (int m = 0; m < 3; m++){
            tmp[m*3] = reg_M[m*3]*src_pix_trans_src_norm[0];
            tmp[m*3 + 1] = reg_M[m*3 + 1]*src_pix_trans_src_norm[4];
            tmp[m*3 + 2] = reg_M[m*3]*src_pix_trans_src_norm[2] + reg_M[m*3 + 1]*src_pix_trans_src_norm[5] + reg_M[m*3 + 2]*src_pix_trans_src_norm[8];
        }
        float dst_norm_trans_dst_pix[9] = {
            2.0f / (static_cast<float>(dst_w) - 1.0f), 0.0f, -1.0f,
            0.0f, 2.0f / (static_cast<float>(dst_h) - 1.0f), -1.0f,
            0.0f, 0.0f, 1.0f
        };
        #pragma unroll(3)
        for (int n = 0; n < 3; n++){
            reg_M[n] = dst_norm_trans_dst_pix[0]*tmp[n] + dst_norm_trans_dst_pix[2]*tmp[n + 6];
            reg_M[n + 3] = dst_norm_trans_dst_pix[4]*tmp[n + 3] + dst_norm_trans_dst_pix[5]*tmp[n + 6];
            reg_M[n + 6] = dst_norm_trans_dst_pix[8]*tmp[n + 6];
        }
        // if (blockIdx.x == 0){
        //     printf("%.6f, %.6f, %.6f,\n%.6f, %.6f, %.6f,\n%.6f, %.6f, %.6f,\n", 
        //             reg_M[0], reg_M[1], reg_M[2],
        //             reg_M[3], reg_M[4], reg_M[5],
        //             reg_M[6], reg_M[7], reg_M[8]);
        // }
        float diterminate = reg_M[0] * (reg_M[4] * reg_M[8] - reg_M[5] * reg_M[7])
                            - reg_M[3] * (reg_M[1] * reg_M[8] - reg_M[2] * reg_M[7])
                            + reg_M[6] * (reg_M[1] * reg_M[5] - reg_M[2] * reg_M[4]);
        sh_src_norm_trans_dst_norm[0] = (reg_M[4] * reg_M[8] - reg_M[5] * reg_M[7]) / diterminate;
        sh_src_norm_trans_dst_norm[1] = (reg_M[2] * reg_M[7] - reg_M[1] * reg_M[8]) / diterminate;
        sh_src_norm_trans_dst_norm[2] = (reg_M[1] * reg_M[5] - reg_M[2] * reg_M[4]) / diterminate;
        sh_src_norm_trans_dst_norm[3] = (reg_M[5] * reg_M[6] - reg_M[3] * reg_M[8]) / diterminate;
        sh_src_norm_trans_dst_norm[4] = (reg_M[0] * reg_M[8] - reg_M[2] * reg_M[6]) / diterminate;
        sh_src_norm_trans_dst_norm[5] = (reg_M[3] * reg_M[2] - reg_M[0] * reg_M[5]) / diterminate;
        sh_src_norm_trans_dst_norm[6] = (reg_M[3] * reg_M[7] - reg_M[4] * reg_M[6]) / diterminate;
        sh_src_norm_trans_dst_norm[7] = (reg_M[1] * reg_M[6] - reg_M[0] * reg_M[7]) / diterminate;
        sh_src_norm_trans_dst_norm[8] = (reg_M[0] * reg_M[4] - reg_M[3] * reg_M[1]) / diterminate;

        // if (blockIdx.x == 0){
        //     printf("%.6f, %.6f, %.6f,\n%.6f, %.6f, %.6f,\n%.6f, %.6f, %.6f,\n", 
        //             sh_src_norm_trans_dst_norm[0], sh_src_norm_trans_dst_norm[1], sh_src_norm_trans_dst_norm[2],
        //             sh_src_norm_trans_dst_norm[3], sh_src_norm_trans_dst_norm[4], sh_src_norm_trans_dst_norm[5],
        //             sh_src_norm_trans_dst_norm[6], sh_src_norm_trans_dst_norm[7], sh_src_norm_trans_dst_norm[8]);
        // }
    }
    __syncthreads();

    int thread_grid_num = dst_h * dst_w / 128;
    float reg_src_norm_trans_dst_norm[9];
    float* reg_dst = dst + batch_id*channel*dst_h*dst_w;
    float* reg_src = src + batch_id*channel*src_h*src_w;
    #pragma unroll(9)
    for (int i = 0; i < 9; i++) reg_src_norm_trans_dst_norm[i] = sh_src_norm_trans_dst_norm[i];
    float eps = 1e-8;
    
    for (int group_id = 0; group_id < thread_grid_num; group_id++){
        int th_dst_id = group_id * 128 + threadIdx.x;
        int th_dst_h = th_dst_id / dst_w;
        int th_dst_w = th_dst_id % dst_w;
        float th_grid_x = (static_cast<float>(th_dst_w) / static_cast<float>(dst_w - 1) - 0.5f) * 2.0f;
        float th_grid_y = (static_cast<float>(th_dst_h) / static_cast<float>(dst_h - 1) - 0.5f) * 2.0f;
        float z_fac = th_grid_x*reg_src_norm_trans_dst_norm[6] + th_grid_y*reg_src_norm_trans_dst_norm[7] + reg_src_norm_trans_dst_norm[8];
        float th_grid_xdz = 1.0f;
        float th_grid_ydz = 1.0f;
        if (z_fac > eps){
            th_grid_xdz = (th_grid_x*reg_src_norm_trans_dst_norm[0] + th_grid_y*reg_src_norm_trans_dst_norm[1] + reg_src_norm_trans_dst_norm[2]) / z_fac;
            th_grid_ydz = (th_grid_x*reg_src_norm_trans_dst_norm[3] + th_grid_y*reg_src_norm_trans_dst_norm[4] + reg_src_norm_trans_dst_norm[5]) / z_fac;
        }
        // if (blockIdx.x == 0 && group_id == 110 && warp_id == 0){
        //     printf("thx%d: %.6f, %.6f\n",warp_tx, th_grid_xdz, th_grid_ydz);
        // }

        for (int c = 0; c < channel; c++){
            float tmp_fea = 0.0f;
            if (th_grid_xdz >= -1.0f && th_grid_xdz <= 1.0f && th_grid_ydz >= -1.0f && th_grid_ydz <= 1.0f){
                int th_src_w = static_cast<int>((th_grid_xdz + 1.0f) / 2.0f * static_cast<float>(src_w) - 0.5f + 0.5f);
                int th_src_h = static_cast<int>((th_grid_ydz + 1.0f) / 2.0f * static_cast<float>(src_h) - 0.5f + 0.5f);
                if (th_src_h >= src_h) th_src_h = src_h - 1;
                if (th_src_h < 0) th_src_h = 0;
                if (th_src_w >= src_w) th_src_w = src_w - 1;
                if (th_src_w < 0) th_src_w = 0;
                int th_src_id = th_src_h * src_w + th_src_w;
                tmp_fea = reg_src[th_src_id + c*src_h*src_w];
            }
            reg_dst[th_dst_id + c*dst_h*dst_w] = tmp_fea;
        }
    }
}

void gpuKorniaPerspectiveFillZeroNoAlignNearestBCHW(float* dst, float* src, float* M, int batch, int channel, int src_h, int src_w, int dst_h, int dst_w, cudaStream_t stream){
    dim3 dimGrid(batch);
    dim3 dimBlock(128);
    korniaPerspectiveFillZeroNoAlignNearestBCHWKernel<<<dimGrid, dimBlock, 0, stream>>>(dst, src, M, channel, src_h, src_w, dst_h, dst_w);
}

}