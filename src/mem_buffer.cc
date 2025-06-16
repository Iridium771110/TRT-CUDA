#include "mem_buffer.h"
#include <fstream>

namespace memory{

#define CHECK_PTR(ptr) {\
    if (ptr == nullptr){\
        std::cout<<buffer_name_<<": data ptr has been released or not generated"<<std::endl;\
        return false;\
    }}
#define CHECK_FILE(file) {\
    if (!file.good()){\
        std::cout<<buffer_name_<<": failed to open file "<<file_name<<std::endl;\
        return false;\
    }}
#define CHECK_EQUAL(aim, src) {\
    if (src != aim){\
        std::cout<<buffer_name_<<": unconsistent byte size with buffer, expect "<<aim<<" but get "<<src<<std::endl;\
        return false;\
    }}


template <typename T>
T* BaseBuffer::getDataPtr(){
    return reinterpret_cast<T*>(data_ptr_);
}
template void* BaseBuffer::getDataPtr<void>();
template float* BaseBuffer::getDataPtr<float>();
template char* BaseBuffer::getDataPtr<char>();
template half* BaseBuffer::getDataPtr<half>();

int64_t BaseBuffer::getDataByteSize(){
    return byte_size_;
}

std::string BaseBuffer::getBufferName(){
    return buffer_name_;
}

CPUBuffer::CPUBuffer(const std::string &name, const int64_t &byte_size){
    buffer_name_ = name;
    byte_size_ = byte_size;
    data_ptr_ = malloc(byte_size_);
    data_ptr_addr_ = &data_ptr_;
    ptr_owner_ = true;
}
CPUBuffer::CPUBuffer(const std::string &name, void** src_ptr_addr, const int64_t &byte_size){
    buffer_name_ = name;
    byte_size_ = byte_size;
    data_ptr_addr_ = src_ptr_addr;
    data_ptr_ = *data_ptr_addr_;
    ptr_owner_ = false;
}
CPUBuffer::~CPUBuffer(){
    if (ptr_owner_) free(data_ptr_);
    data_ptr_ = nullptr;
    data_ptr_addr_ = nullptr;
    ptr_owner_ = false;
}

bool CPUBuffer::copyToCPU(void* dst_p, const int64_t &byte_size){
    CHECK_PTR(data_ptr_)
    CHECK_EQUAL(byte_size_, byte_size)
    memcpy(dst_p, data_ptr_, byte_size_);
    return true;
}
bool CPUBuffer::copyFromCPU(const void* src_p, const int64_t &byte_size){
    CHECK_PTR(data_ptr_)
    CHECK_EQUAL(byte_size_, byte_size)
    memcpy(data_ptr_, src_p, byte_size_);
    return true;
}
bool CPUBuffer::copyToGPU(void* dst_p, const int64_t &byte_size){
    CHECK_PTR(data_ptr_)
    CHECK_EQUAL(byte_size_, byte_size)
    cudaMemcpy(dst_p, data_ptr_, byte_size_, cudaMemcpyHostToDevice);
    return true;
}
bool CPUBuffer::copyFromGPU(const void* src_p, const int64_t &byte_size){
    CHECK_PTR(data_ptr_)
    CHECK_EQUAL(byte_size_, byte_size)
    cudaMemcpy(data_ptr_, src_p, byte_size_, cudaMemcpyDeviceToHost);
    return true;
}
bool CPUBuffer::saveToFile(std::string &file_name){
    CHECK_PTR(data_ptr_)
    std::ofstream file(file_name, std::ios::binary);
    CHECK_FILE(file)
    file.seekp(0, file.beg);
    file.write(reinterpret_cast<char*>(data_ptr_), byte_size_);
    file.close();
    return true;
}
bool CPUBuffer::loadFromFile(std::string &file_name){
    CHECK_PTR(data_ptr_)
    std::ifstream file(file_name, std::ios::binary);
    CHECK_FILE(file)
    file.seekg(0, file.end);
    int64_t end_byte = file.tellg();
    file.seekg(0, file.beg);
    int64_t beg_byte = file.tellg();
    int64_t byte_length = end_byte - beg_byte;
    CHECK_EQUAL(byte_size_, byte_length)
    file.read(reinterpret_cast<char*>(data_ptr_), byte_size_);
    file.close();
    return true;
}
bool CPUBuffer::setZeros(){
    CHECK_PTR(data_ptr_)
    std::memset(data_ptr_, 0, byte_size_);
    return true;
}

GPUBuffer::GPUBuffer(const std::string &name, const int64_t &byte_size){
    buffer_name_ = name;
    byte_size_ = byte_size;
    cudaMalloc(reinterpret_cast<void**>(&data_ptr_), byte_size_);
    data_ptr_addr_ = &data_ptr_;
    ptr_owner_ = true;
}
GPUBuffer::GPUBuffer(const std::string &name, void** src_ptr_addr, const int64_t &byte_size){
    buffer_name_ = name;
    byte_size_ = byte_size;
    data_ptr_addr_ = src_ptr_addr;
    data_ptr_ = *data_ptr_addr_;
    ptr_owner_ = false;
}
GPUBuffer::~GPUBuffer(){
    if (ptr_owner_) cudaFree(data_ptr_);
    data_ptr_ = nullptr;
    data_ptr_addr_ = nullptr;
    ptr_owner_ = false;
}

bool GPUBuffer::copyToCPU(void* dst_p, const int64_t &byte_size){
    CHECK_PTR(data_ptr_)
    CHECK_EQUAL(byte_size_, byte_size)
    cudaMemcpy(dst_p, data_ptr_, byte_size_, cudaMemcpyDeviceToHost);
    return true;
}
bool GPUBuffer::copyFromCPU(const void* src_p, const int64_t &byte_size){
    CHECK_PTR(data_ptr_)
    CHECK_EQUAL(byte_size_, byte_size)
    cudaMemcpy(data_ptr_, src_p, byte_size_, cudaMemcpyHostToDevice);
    return true;
}
bool GPUBuffer::copyToGPU(void* dst_p, const int64_t &byte_size){
    CHECK_PTR(data_ptr_)
    CHECK_EQUAL(byte_size_, byte_size)
    cudaMemcpy(dst_p, data_ptr_, byte_size_, cudaMemcpyDeviceToDevice);
    return true;
}
bool GPUBuffer::copyFromGPU(const void* src_p, const int64_t &byte_size){
    CHECK_PTR(data_ptr_)
    CHECK_EQUAL(byte_size_, byte_size)
    cudaMemcpy(data_ptr_, src_p, byte_size_, cudaMemcpyDeviceToDevice);
    return true;
}
bool GPUBuffer::saveToFile(std::string &file_name){
    CHECK_PTR(data_ptr_)
    std::vector<char> data_transfer(byte_size_);
    cudaMemcpy(data_transfer.data(), data_ptr_, byte_size_, cudaMemcpyDeviceToHost);
    std::ofstream file(file_name, std::ios::binary);
    CHECK_FILE(file)
    file.seekp(0, file.beg);
    file.write(data_transfer.data(), byte_size_);
    file.close();
    return true;
}
bool GPUBuffer::loadFromFile(std::string &file_name){
    CHECK_PTR(data_ptr_)
    std::ifstream file(file_name, std::ios::binary);
    CHECK_FILE(file)
    file.seekg(0, file.end);
    int64_t end_byte = file.tellg();
    file.seekg(0, file.beg);
    int64_t beg_byte = file.tellg();
    int64_t byte_length = end_byte - beg_byte;
    CHECK_EQUAL(byte_size_, byte_length)
    std::vector<char> data_transfer(byte_size_);
    file.read(reinterpret_cast<char*>(data_transfer.data()), byte_size_);
    file.close();
    cudaMemcpy(data_ptr_, data_transfer.data(), byte_size_, cudaMemcpyHostToDevice);
    return true;
}
bool GPUBuffer::setZeros(){
    CHECK_PTR(data_ptr_)
    cudaMemset(data_ptr_, 0, byte_size_);
    return 0;
}

bool GPUBuffer::copyToCPUStreamAsync(void* dst_p, const int64_t &byte_size, cudaStream_t stream){
    CHECK_PTR(data_ptr_)
    CHECK_EQUAL(byte_size_, byte_size)
    cudaMemcpyAsync(dst_p, data_ptr_, byte_size_, cudaMemcpyDeviceToHost, stream);
    return true;
}
bool GPUBuffer::copyFromCPUStreamAsync(const void* src_p, const int64_t &byte_size, cudaStream_t stream){
    CHECK_PTR(data_ptr_)
    CHECK_EQUAL(byte_size_, byte_size)
    cudaMemcpyAsync(data_ptr_, src_p, byte_size_, cudaMemcpyHostToDevice, stream);
    return true;
}
bool GPUBuffer::copyToGPUStreamAsync(void* dst_p, const int64_t &byte_size, cudaStream_t stream){
    CHECK_PTR(data_ptr_)
    CHECK_EQUAL(byte_size_, byte_size)
    cudaMemcpyAsync(dst_p, data_ptr_, byte_size_, cudaMemcpyDeviceToDevice, stream);
    return true;
}
bool GPUBuffer::copyFromGPUStreamAsync(const void* src_p, const int64_t &byte_size, cudaStream_t stream){
    CHECK_PTR(data_ptr_)
    CHECK_EQUAL(byte_size_, byte_size)
    cudaMemcpyAsync(data_ptr_, src_p, byte_size_, cudaMemcpyDeviceToDevice, stream);
    return true;
}
bool GPUBuffer::setZerosStreamAsync(cudaStream_t stream){
    CHECK_PTR(data_ptr_)
    cudaMemsetAsync(data_ptr_, 0, byte_size_, stream);
    return 0;
}

#define GETTYPESIZE(a)  [](std::string a_type){if (a_type == "float" || a_type == "int32") return 4;\
                        else if (a_type == "float16") return 2; \
                        else if (a_type == "int8") return 1; \
                        else std::cout<<"unsupported type: "<<a_type<<std::endl;\
                        return 0;}(a)

Tensor::Tensor(const std::string &name, const std::string &data_type, const int64_t &byte_size)
        :name_(name),
        data_type_(data_type),
        element_type_size_(GETTYPESIZE(data_type)){
    element_num_ = byte_size / element_type_size_;
    shape_ = std::vector<int>({element_num_});
    data_buffer_p_ = std::make_shared<GPUBuffer>(name, byte_size);
}
Tensor::Tensor(const std::string &name, const std::string &data_type, const GPUBufferPtr &buffer)
        :name_(name),
        data_type_(data_type),
        element_type_size_(GETTYPESIZE(data_type)){
    shape_ = std::vector<int>({element_num_});
    data_buffer_p_ = buffer;
    element_num_ = buffer->getDataByteSize() / element_type_size_;
}
Tensor::Tensor(const std::string &name, const std::string &data_type, const int64_t &byte_size, const std::vector<int> &shape)
        :name_(name),
        data_type_(data_type),
        element_type_size_(GETTYPESIZE(data_type)){
    element_num_ = byte_size / element_type_size_;
    shape_ = shape;
    data_buffer_p_ = std::make_shared<GPUBuffer>(name, byte_size);
}
Tensor::Tensor(const std::string &name, const std::string &data_type, const GPUBufferPtr &buffer, const std::vector<int> &shape)
        :name_(name),
        data_type_(data_type),
        element_type_size_(GETTYPESIZE(data_type)){
    shape_ = shape;
    data_buffer_p_ = buffer;
    element_num_ = buffer->getDataByteSize() / element_type_size_;
}
GPUBufferPtr Tensor::getDataBufferPtr(){
    return data_buffer_p_;
}
std::vector<int> Tensor::getShape(){
    return shape_;
}
int Tensor::getElementTypeSize(){
    return element_type_size_;
}
std::string Tensor::getElementType(){
    return data_type_;
}
int Tensor::getElementNum(){
    return element_num_;
}
std::string Tensor::getName(){
    return name_;
}
bool Tensor::reshape(const std::vector<int> &shape){
    int new_num = 0;
    for (int i = 0; i < shape.size(); i++) new_num *= shape[i];
    if (new_num == element_num_){
        shape_ = shape;
        return true;
    }
    else{
        std::cout<<"Tensor "<<name_<<" reshape has inconsistent number of element, failed"<<std::endl;
        return false;
    }
}

}