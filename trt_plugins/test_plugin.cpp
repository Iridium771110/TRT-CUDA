#include "axesNormalizationPlugin/axesNormalizationPlugin.h"
#include <iostream>
#include "common.h"
#include "mem_buffer.h"

void cpuNorm(float* input_p, float* output_p, float* scale_p, float* bias_p, float eps, int channel, int batch, int num){
    for (int b = 0; b < batch; b++){
        for (int c = 0; c < channel; c++){
            float sum = 0.0f;
            int idx_base = b * channel*num + c*num;
            for (int n = 0; n < num; n++){
                sum += input_p[idx_base + n];
            }
            float mean = sum / static_cast<float>(num);
            float var = 0.0f;
            for (int n = 0; n < num; n++){
                var += (input_p[idx_base + n] - mean) * (input_p[idx_base + n] - mean);
            }
            var = sqrt(eps + var / static_cast<float>(num));
            for (int n = 0; n < num; n++){
                output_p[idx_base + n] = (input_p[idx_base + n] - mean) / var * scale_p[c] + bias_p[c];
            }
            if (c == 0) std::cout<<scale_p[c] / var<<' '<<mean<<' '<<bias_p[c]<<std::endl;
        }
    }
    std::cout<<output_p[2300]<<std::endl;
}

int main(){
    int channel = 32;
    int batch = 1;
    int tail_dim = 40*60;
    float eps = 1e-5f;
    std::vector<float> scale(channel, 1.0f);
    std::vector<float> bias(channel, 0.0f);
    std::vector<float> input(batch*channel*tail_dim);
    std::vector<float> output(batch*channel*tail_dim);
    // common::randInit<float>(scale.data(), scale.size());
    // common::randInit<float>(bias.data(), bias.size());
    common::randInit<float>(input.data(), input.size());
    std::vector<float> output_ref(batch*channel*tail_dim);

    cpuNorm(input.data(), output_ref.data(), scale.data(), bias.data(), eps, channel, batch, tail_dim);

    memory::GPUBuffer scale_d("scale", sizeof(float)*scale.size());
    memory::GPUBuffer bias_d("bias", sizeof(float)*bias.size());
    memory::GPUBuffer input_d("input", sizeof(float)*input.size());
    memory::GPUBuffer output_d("output", sizeof(float)*output.size());
    scale_d.copyFromCPU(scale.data(), sizeof(float)*scale.size());
    bias_d.copyFromCPU(bias.data(), sizeof(float)*bias.size());
    input_d.copyFromCPU(input.data(), sizeof(float)*input.size());
    output_d.setZeros();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::vector<int> shape = {batch, channel, tail_dim};
    std::string type_str = "float";
    cudaStreamSynchronize(stream);
    custom_plugin_kernel::customTailNormalizationKernelExector(input_d.getDataPtr<void>(), output_d.getDataPtr<void>(), 
                                                                scale_d.getDataPtr<float>(), bias_d.getDataPtr<float>(),
                                                                eps, shape, type_str, 1, stream);
    cudaStreamSynchronize(stream);
    output_d.copyToCPU(output.data(), sizeof(float)*output.size());
    common::maxErrCheck<float, float>(output.data(), output.data(), batch, channel, tail_dim);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout<<prop.maxThreadsPerBlock<<std::endl;

    std::vector<float> check_data = common::loadDataFromFile<float, float>("../test_data/-upsampling_layer-conv-conv.0-Conv_output_0.bin");
    double check_sum = 0.0f;
    for (int i = 0; i < check_data.size(); i++) check_sum += check_data[i];
    std::cout<<check_sum<<std::endl;
    check_sum = 0.0f;
    for (int i = 0; i < check_data.size(); i++) check_sum += std::abs(check_data[i]);
    std::cout<<check_sum<<std::endl;
    common::checkDimSum<float>(check_data.data(), 8, check_data.size() / 8);
    nvinfer1::Weights test;
    std::cout<<&test<<' '<<&test.type<<' '<<&test.values<<' '<<&test.count<<std::endl;
    double a = -259233 + -272013 + -271656 + -257311;
    double b = 706116 + 704399 + 704570 + 682239;
    std::cout<<a<<' '<<b<<std::endl;

    std::vector<float> trans_check_data = check_data;
    for (int i = 0; i < 512; i++){
        for (int j = 0; j < 42*84; j++){
            trans_check_data[j*512 + i] = check_data[i*42*84 + j];
        }
    }
    common::checkDimSum<float>(trans_check_data.data(), 8, trans_check_data.size() / 8);
    return 0;
}