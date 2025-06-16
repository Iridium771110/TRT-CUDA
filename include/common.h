#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <string.h>
#include <memory>
#include <unordered_map>
#include <cuda_runtime_api.h>
#include <fstream>
#include <cuda_fp16.h>
#include <nlohmann/json.hpp>

namespace common{

    template <typename Src, typename Aim>
    std::vector<Aim> loadDataFromFile(std::string file_path);

    template <typename ResT, typename RefT>
    void maxErrCheck(const ResT* res, const RefT* ref, int head, int check, int tail);

    bool fileExist(std::string file_name);

    template<typename T>
    bool randInit(T* data_p, int length);

    template <typename T>
    void checkDimSum(T* data_p, int dim, int dim_ele_num);
}


#endif