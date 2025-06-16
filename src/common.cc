#include "common.h"

namespace common{
    template <typename Src, typename Aim>
    std::vector<Aim> loadDataFromFile(std::string file_path){
        std::ifstream file(file_path, std::ios::binary);
        if (!file.good()){
            std::cout<<"file abnormal, check path or is broken: " << file_path<<std::endl;
            return std::vector<Aim>(0);
        } 
        file.seekg(0, file.end);
        int64_t end_byte = file.tellg();
        file.seekg(0, file.beg);
        int64_t beg_byte = file.tellg();
        int64_t byte_length = end_byte - beg_byte;
        int64_t num_element = byte_length / sizeof(Src);
        std::vector<Aim> ret(num_element);
        std::vector<Src> load(num_element);
        file.read(reinterpret_cast<char*>(load.data()), byte_length);
        for (int i = 0; i < num_element; i++) ret[i] = static_cast<Aim>(load[i]);
        return ret;
    }
    template std::vector<float> loadDataFromFile<float, float>(std::string file_path);
    template std::vector<char> loadDataFromFile<char, char>(std::string file_path);

    template <typename ResT, typename RefT>
    void maxErrCheck(const ResT* res, const RefT* ref, int head, int check, int tail){
        RefT sum_res_abs = static_cast<RefT>(0.0f);
        RefT sum_ref_abs = static_cast<RefT>(0.0f);
        std::vector<RefT> max_err(check, static_cast<RefT>(0));
        std::vector<RefT> err_ref(check, static_cast<RefT>(0));
        std::vector<RefT> err_res(check, static_cast<RefT>(0));
        std::vector<RefT> err_sum(check, static_cast<RefT>(0));
        for (int h = 0; h < head; h++){
            for (int c = 0; c < check; c++){
                for (int t = 0; t < tail; t++){
                    int idx = h*check*tail + c*tail + t;
                    sum_ref_abs += std::abs(static_cast<RefT>(ref[idx]));
                    sum_res_abs += std::abs(static_cast<RefT>(res[idx]));
                    RefT err = std::abs(static_cast<RefT>(res[idx] - ref[idx]));
                    err_sum[c] += err;
                    if (err > max_err[c]){
                        max_err[c] = err;
                        err_ref[c] = ref[idx];
                        err_res[c] = res[idx];
                    }
                }
            }
        }
        std::cout << "check shape (head, check, tail): " << head << ' ' << check << ' ' << tail << std::endl;
        std::cout << "ref: " << sum_ref_abs << " res: " << sum_res_abs << std::endl;
        for (int i = 0; i < check; i++) std::cout <<i<<" check dim, max err: "<< max_err[i]<<" rele. res: "<< err_res[i]<<
                                                " rele. ref: "<< err_ref[i]<<" sum err: "<< err_sum[i]<<std::endl;
    }
    template void maxErrCheck<float, float>(const float* res, const float* ref, int head, int check, int tail);

    bool fileExist(std::string file_name){
        std::ifstream file(file_name);
        if (!file.good()) return false;
        file.close();
        return true;
    }

    template<typename T>
    bool randInit(T* data_p, int length){
        for (int i = 0; i < length; i++){
            double tmp = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
            data_p[i] = static_cast<T>(tmp);
        }
        return true;
    }
    template bool randInit<float>(float* data_p, int length);

    template <typename T>
    void checkDimSum(T* data_p, int dim, int dim_ele_num){
        std::vector<double> dim_sum(dim, 0.0f);
        std::vector<double> dim_abs_sum(dim, 0.0f);
        for (int i = 0; i < dim; i++){
            for (int j = 0; j < dim_ele_num; j++){
                dim_sum[i] += data_p[i*dim_ele_num + j];
                dim_abs_sum[i] += std::abs(data_p[i*dim_ele_num + j]);
            }
        }
        for (int i = 0; i < dim; i++) std::cout<< i <<" sum: " << dim_sum[i] << " abs sum: " << dim_abs_sum[i] << std::endl;
    }
    template void checkDimSum<float>(float* data_p, int dim, int dim_ele_num);
}
