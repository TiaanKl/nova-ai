cmake_minimum_required(VERSION 3.18)
project(nova_dcnv2_plugin LANGUAGES C CXX CUDA)

# Add custom FindTensorRT.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(nova_native STATIC
    ${CMAKE_SOURCE_DIR}/../kernels/Nova_DCNv2.cu
    ${CMAKE_SOURCE_DIR}/../kernels/Nova_Preprocess.cu
)

target_link_libraries(nova_native PRIVATE CUDA::cudart)

set_target_properties(nova_native PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

if (MSVC)
    target_compile_definitions(nova_native PRIVATE
        CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
        NOVA_KERNELS_EXPORTS
    )
    target_compile_options(nova_native PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_87,code=compute_87 -gencode=arch=compute_87,code=sm_87>
    )
endif()

target_include_directories(nova_native PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/../kernels
)

set(SOURCES
    nova_dcnv2_plugin.cpp
    nova_dcnv2_plugin_creator.cpp
)

add_library(nova_dcnv2_plugin SHARED ${SOURCES})

target_include_directories(nova_dcnv2_plugin PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/../kernels
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(nova_dcnv2_plugin PRIVATE
    TensorRT::nvinfer
    TensorRT::nvinfer_plugin
    TensorRT::nvonnxparser
    nova_native
)

# Remove "lib" prefix on Windows DLLs
set_target_properties(nova_dcnv2_plugin PROPERTIES PREFIX "")

install(TARGETS nova_dcnv2_plugin DESTINATION ${CMAKE_BINARY_DIR})
