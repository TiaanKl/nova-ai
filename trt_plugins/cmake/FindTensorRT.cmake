# FindTensorRT.cmake — TensorRT 10.x Windows ZIP layout

set(_TENSORRT_HINTS
    ${TENSORRT_ROOT}
    $ENV{TENSORRT_ROOT}
    "C:/SDKs/tensorrt"
)

# Includes
find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATH_SUFFIXES include
    HINTS ${_TENSORRT_HINTS}
)

# Library directory
find_path(TENSORRT_LIBRARY_DIR
    NAMES nvinfer_10.lib
    PATH_SUFFIXES lib
    HINTS ${_TENSORRT_HINTS}
)

# TensorRT 10 library names
set(_TENSORRT_LIB_NAMES
    nvinfer_10
    nvinfer_plugin_10
    nvonnxparser_10
)

foreach(lib ${_TENSORRT_LIB_NAMES})
    find_library(TENSORRT_${lib}_LIBRARY
        NAMES ${lib}
        HINTS ${TENSORRT_LIBRARY_DIR}
    )
    list(APPEND TENSORRT_LIBRARIES ${TENSORRT_${lib}_LIBRARY})
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    TensorRT
    REQUIRED_VARS
        TENSORRT_INCLUDE_DIR
        TENSORRT_LIBRARY_DIR
        TENSORRT_LIBRARIES
)

if(TensorRT_FOUND)
    message(STATUS "TensorRT 10.x found:")
    message(STATUS "  Includes: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "  Libraries: ${TENSORRT_LIBRARY_DIR}")

    add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION "${TENSORRT_nvinfer_10_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
    )

    add_library(TensorRT::nvinfer_plugin UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer_plugin PROPERTIES
        IMPORTED_LOCATION "${TENSORRT_nvinfer_plugin_10_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
    )

    add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvonnxparser PROPERTIES
        IMPORTED_LOCATION "${TENSORRT_nvonnxparser_10_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
    )
endif()
