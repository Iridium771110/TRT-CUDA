#ifndef MEM_BUFFER_H
#define MEM_BUFFER_H

#include "common.h"

namespace memory{

class BaseBuffer{
public:
    BaseBuffer(){}
    virtual ~BaseBuffer(){}

    template <typename T>
    T* getDataPtr();
    int64_t getDataByteSize();
    std::string getBufferName();

    virtual bool copyToCPU(void* dst_p, const int64_t &byte_size) = 0;
    virtual bool copyFromCPU(const void* src_p, const int64_t &byte_size) = 0;
    virtual bool copyToGPU(void* dst_p, const int64_t &byte_size) = 0;
    virtual bool copyFromGPU(const void* src_p, const int64_t &byte_size) = 0;
    virtual bool saveToFile(std::string &file_name) = 0;
    virtual bool loadFromFile(std::string &file_name) = 0;
    virtual bool setZeros() = 0;
protected:
    void* data_ptr_ = nullptr;
    void** data_ptr_addr_ = nullptr;
    int64_t byte_size_;
    bool ptr_owner_ = false;
    std::string buffer_name_;
};

class CPUBuffer : public BaseBuffer{
public:
    CPUBuffer() = delete;
    CPUBuffer(CPUBuffer &&) = delete;
    CPUBuffer(const CPUBuffer &) = delete;
    CPUBuffer & operator= (CPUBuffer &&) = delete;
    CPUBuffer & operator= (const CPUBuffer &) = delete;
    CPUBuffer(const std::string &name, const int64_t &byte_size);
    CPUBuffer(const std::string &name, void** src_ptr_addr, const int64_t &byte_size);
    ~CPUBuffer();

    bool copyToCPU(void* dst_p, const int64_t &byte_size) final;
    bool copyFromCPU(const void* src_p, const int64_t &byte_size) final;
    bool copyToGPU(void* dst_p, const int64_t &byte_size) final;
    bool copyFromGPU(const void* src_p, const int64_t &byte_size) final;
    bool saveToFile(std::string &file_name) final;
    bool loadFromFile(std::string &file_name) final;
    bool setZeros() final;
private:
};

class GPUBuffer : public BaseBuffer{
public:
    GPUBuffer() = delete;
    GPUBuffer(GPUBuffer &&) = delete;
    GPUBuffer(const GPUBuffer &) = delete;
    GPUBuffer & operator= (GPUBuffer &&) = delete;
    GPUBuffer & operator= (const GPUBuffer &) = delete;
    GPUBuffer(const std::string &name, const int64_t &byte_size);
    GPUBuffer(const std::string &name, void** src_ptr_addr, const int64_t &byte_size);
    ~GPUBuffer();

    bool copyToCPU(void* dst_p, const int64_t &byte_size) final;
    bool copyFromCPU(const void* src_p, const int64_t &byte_size) final;
    bool copyToGPU(void* dst_p, const int64_t &byte_size) final;
    bool copyFromGPU(const void* src_p, const int64_t &byte_size) final;
    bool saveToFile(std::string &file_name) final;
    bool loadFromFile(std::string &file_name) final;
    bool setZeros() final;

    bool copyToCPUStreamAsync(void* dst_p, const int64_t &byte_size, cudaStream_t stream);
    bool copyFromCPUStreamAsync(const void* src_p, const int64_t &byte_size, cudaStream_t stream);
    bool copyToGPUStreamAsync(void* dst_p, const int64_t &byte_size, cudaStream_t stream);
    bool copyFromGPUStreamAsync(const void* src_p, const int64_t &byte_size, cudaStream_t stream);
    bool setZerosStreamAsync(cudaStream_t stream);
private:
    // cudaStream_t stream_;
};

typedef std::shared_ptr<BaseBuffer> BaseBufferPtr;
typedef std::shared_ptr<CPUBuffer> CPUBufferPtr;
typedef std::shared_ptr<GPUBuffer> GPUBufferPtr;

class Tensor{
    //CPU 直接用 vector 即可
public:
    Tensor() = delete;
    Tensor(Tensor &&) = delete;
    Tensor(const Tensor &) = delete;
    Tensor & operator= (Tensor &&) = delete;
    Tensor & operator= (const Tensor &) = delete;
    Tensor(const std::string &name, const std::string &data_type, const int64_t &byte_size);
    Tensor(const std::string &name, const std::string &data_type, const GPUBufferPtr &buffer);
    Tensor(const std::string &name, const std::string &data_type, const int64_t &byte_size, const std::vector<int> &shape);
    Tensor(const std::string &name, const std::string &data_type, const GPUBufferPtr &buffer, const std::vector<int> &shape);
    ~Tensor() = default;

    GPUBufferPtr getDataBufferPtr();
    std::vector<int> getShape();
    int getElementTypeSize();
    std::string getElementType();
    int getElementNum();
    std::string getName();
    bool reshape(const std::vector<int> &shape);

private:
    const std::string name_;
    const std::string data_type_;
    const int element_type_size_;
    int element_num_;
    std::vector<int> shape_;
    GPUBufferPtr data_buffer_p_;
};

typedef std::shared_ptr<Tensor> TensorPtr;

}

#endif