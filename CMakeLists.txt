set(CMAKE_BUILD_TYPE RELEASE)

set(RUNNING_ENV_PLATFORM "x86_64" CACHE STRING "select platform")
set(NVSCI_HEADER_PATH "" CACHE STRING "l4t nvsci include path")
option(DLA_TEST_MODE "whether build dla test" OFF)
message(RUNNING_ENV_PLATFORM ${RUNNING_ENV_PLATFORM})
if (RUNNING_ENV_PLATFORM STREQUAL "x86_64")
    add_definitions(-Dx86_64)
    set(CMAKE_CUDA_ARCHITECTURES 89)
    set(CMAKE_CUDA_FLAGS "-Xcompiler -fPIC ${CMAKE_CUDA_FLAGS} -std=c++17\
                    -gencode arch=compute_89,code=compute_89")
elseif(RUNNING_ENV_PLATFORM STREQUAL "jetson_orin")
    if (NVSCI_HEADER_PATH STREQUAL "")
        message(FATAL_ERROR "l4t nvsci include path must be setted, e.g. -DNVSCI_HEADER_PATH=/Linux_for_Tegra/nvsci_include/")
    endif()
    add_definitions(-DJETSON_ORIN)
    set(DLA_TEST_MODE ON)
    set(CMAKE_CUDA_ARCHITECTURES 87)
    set(CMAKE_CUDA_FLAGS "-Xcompiler -fPIC ${CMAKE_CUDA_FLAGS} -std=c++17\
                    -gencode arch=compute_87,code=compute_87")
else()
    message(FATAL_ERROR "platform only support x86_64 or jetson_orin, set RUNNING_ENV_PLATFORM to one of them, default is x86_64" )
endif()


cmake_minimum_required(VERSION 3.16)
project(backbone_test)
enable_language(CXX CUDA)
# add_compile_options(-Werror=return-type)

if (RUNNING_ENV_PLATFORM STREQUAL "x86_64")
    # list(APPEND CMAKE_PREFIX_PATH "/home/dong/WS/libtorch")
    set(BASE_INCLUDES /usr/local/cuda-12.4/include
                    /usr/local/TensorRT-10.5.0.18/include)
    set(BASE_LINK_LIBS /usr/local/cuda-12.4/lib64/libcudart.so
                        /usr/local/TensorRT-10.5.0.18/lib/libnvinfer.so)
    set(PLUGIN_INCLUDES /usr/local/cuda-12.4/include/
                        /usr/local/TensorRT-10.5.0.18/include/)
    set(PLUGIN_LINK_LIBS /usr/local/cuda-12.4/lib64/libcudart.so
                        /usr/local/TensorRT-10.5.0.18/lib/libnvinfer.so
                        /usr/local/TensorRT-10.5.0.18/lib/libnvinfer_plugin.so)
elseif(RUNNING_ENV_PLATFORM STREQUAL "jetson_orin")
    # list(APPEND CMAKE_PREFIX_PATH "/home/yjy/.local/lib/python3.8/site-packages/torch/")
    set(BASE_INCLUDES /usr/local/cuda-11.4/include/
                    /usr/include/aarch64-linux-gnu/)
    set(BASE_LINK_LIBS /usr/local/cuda-11.4/lib64/libcudart.so
                        /usr/local/cuda-11.4/lib64/libcudla.so
                        /usr/lib/aarch64-linux-gnu/tegra/libcuda.so
                        /usr/lib/aarch64-linux-gnu/libnvinfer.so)
    set(PLUGIN_INCLUDES /usr/local/cuda-11.4/include/
                    /usr/include/aarch64-linux-gnu/)
    set(PLUGIN_LINK_LIBS /usr/local/cuda-11.4/lib64/libcudart.so
                        /usr/lib/aarch64-linux-gnu/libnvinfer.so
                        /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so)
    set(NVSCI_INCLUDE ${NVSCI_HEADER_PATH})
    set(NVSCI_LIBS /usr/lib/aarch64-linux-gnu/tegra/libnvscibuf.so
                    /usr/lib/aarch64-linux-gnu/tegra/libnvscisync.so)
    set(NVSCI_SRC src/nvsci_buffer.cc
                    src/nvsci_synchronize.cc)
endif()
message(RUNNING_ENV_PLATFORM "=${RUNNING_ENV_PLATFORM}")
message(CMAKE_PREFIX_PATH "=${CMAKE_PREFIX_PATH}")
# find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")

set(TRT_PLUGIN_FILE trt_plugins/common/checkMacrosPlugin.cpp  
                    trt_plugins/common/plugin.cpp  
                    trt_plugins/layerNormPlugin/layerNormKernel.cu
                    trt_plugins/layerNormPlugin/layerNormPlugin.cpp
                    trt_plugins/castLayerPlugin/castLayerKernel.cu
                    trt_plugins/castLayerPlugin/castLayerPlugin.cpp
                    trt_plugins/axesNormalizationPlugin/axesNormalizationKernel.cu
                    trt_plugins/axesNormalizationPlugin/axesNormalizationPlugin.cpp)
set(TRT_PLUGIN_LINK_LIB ${PLUGIN_LINK_LIBS})
set(TRT_PLUGIN_INCLUDES_PATH trt_plugins/common/
                        trt_plugins/layerNormPlugin/
                        ${PLUGIN_INCLUDES})
add_library(trt_plugin SHARED ${TRT_PLUGIN_FILE})
target_include_directories(trt_plugin PUBLIC ${TRT_PLUGIN_INCLUDES_PATH})
target_link_libraries(trt_plugin PUBLIC ${TRT_PLUGIN_LINK_LIB})
target_link_options(trt_plugin PUBLIC "LINKER:-no-as-needed")

set(SRC_FILE ${NVSCI_SRC}
            src/common.cc
            src/mem_buffer.cc
            engine/src/base_eng.cc
            # engine/src/torch_eng.cc
            engine/src/trt_eng.cc
            module/src/base_module.cc
            module/src/kornia_perspective_module.cc
            module/src/kornia_perspective_module.cu)
set(INCLUDE_PATHS include/
                module/include/
                engine/include/
                ${BASE_INCLUDES}
                ${NVSCI_INCLUDE})
set(LINK_LIBS ${BASE_LINK_LIBS}
                ${NVSCI_LIBS})
add_library(engine_base SHARED ${SRC_FILE})
target_include_directories(engine_base PUBLIC ${INCLUDE_PATHS} 
                                            # ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(engine_base PUBLIC ${LINK_LIBS} 
                                        # ${TORCH_LIBRARIES} 
                                        trt_plugin
)

add_executable(trt_test exec_app/trt_test.cpp)
target_include_directories(trt_test PUBLIC ${INCLUDE_PATHS})
target_link_libraries(trt_test PUBLIC engine_base trt_plugin)


add_executable(plugin_test trt_plugins/test_plugin.cpp)
target_include_directories(plugin_test PUBLIC ${INCLUDE_PATHS} ${TRT_PLUGIN_INCLUDES_PATH})
target_link_libraries(plugin_test PUBLIC engine_base trt_plugin)

if (DLA_TEST_MODE)
    add_executable(dla_test exec_app/trt_dla_test.cpp)
    target_include_directories(dla_test PUBLIC ${INCLUDE_PATHS})
    target_link_libraries(dla_test PUBLIC engine_base)
endif()