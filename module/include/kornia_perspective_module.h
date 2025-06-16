#ifndef MODULE_KORNIA_PERSPECTIVE_MODULE_H
#define MODULE_KORNIA_PERSPECTIVE_MODULE_H

#include "base_module.h"

namespace module{

void gpuKorniaPerspectiveFillZeroNoAlignNearestBCHW(float* dst, float* src, float* M, 
                                                    int batch, int channel, int src_h, int src_w, int dst_h, int dst_w, 
                                                    cudaStream_t stream);

class KorniaPerspectiveLayer : public LayerModule{
    public:
        virtual int64_t deserializeModule(const char* data_p, int64_t start_byte) final;
        virtual int64_t serializeModule(const char* data_p, int64_t start_byte) final;
        virtual bool init() final;
        virtual bool enqueueModule() final;
        virtual bool executeModule() final;
    private:
        static bool registed_kornia_perspective_layer_;
};

}

#endif