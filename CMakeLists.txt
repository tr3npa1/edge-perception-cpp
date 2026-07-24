cmake_minimum_required(VERSION 3.24)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

project(
    edge_perception_cpp
    VERSION 1.1.0
    DESCRIPTION "Portable C++17 YOLO26M inference with optional NVIDIA acceleration"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CTest)
include(CheckIPOSupported)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(
        CMAKE_BUILD_TYPE
        Release
        CACHE STRING
        "Build type: Debug, Release, RelWithDebInfo, or MinSizeRel"
        FORCE
    )
    set_property(
        CACHE CMAKE_BUILD_TYPE
        PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel
    )
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

foreach(EDGE_CONFIG Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${EDGE_CONFIG}" EDGE_CONFIG_UPPER)
    set(
        "CMAKE_RUNTIME_OUTPUT_DIRECTORY_${EDGE_CONFIG_UPPER}"
        "${CMAKE_BINARY_DIR}/bin/${EDGE_CONFIG}"
    )
    set(
        "CMAKE_LIBRARY_OUTPUT_DIRECTORY_${EDGE_CONFIG_UPPER}"
        "${CMAKE_BINARY_DIR}/lib/${EDGE_CONFIG}"
    )
    set(
        "CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${EDGE_CONFIG_UPPER}"
        "${CMAKE_BINARY_DIR}/lib/${EDGE_CONFIG}"
    )
endforeach()

# =============================================================================
# Build options
# =============================================================================

option(
    EDGE_ENABLE_TENSORRT
    "Build the generic native TensorRT .engine backend"
    OFF
)

option(
    EDGE_ENABLE_CUDA_PIPELINE
    "Build the fused CUDA/TensorRT pipeline instead of its portable stub"
    OFF
)

option(
    EDGE_BUILD_TESTS
    "Build and register C++ and Python regression tests"
    ON
)

option(
    EDGE_BUILD_PYTHON_TESTS
    "Register pytest parity regression tests with CTest"
    ON
)

option(
    EDGE_COPY_RUNTIME_DLLS
    "Copy discovered Windows runtime DLLs beside executables"
    ON
)

option(
    EDGE_ENABLE_IPO
    "Enable interprocedural optimization where supported"
    ON
)

option(
    EDGE_WARNINGS_AS_ERRORS
    "Treat compiler warnings as errors"
    OFF
)

option(
    EDGE_ENABLE_SANITIZERS
    "Enable supported address/undefined-behavior sanitizers"
    OFF
)

option(
    EDGE_ENABLE_NATIVE_ARCH
    "Compile for the current CPU rather than a portable baseline"
    OFF
)

option(
    EDGE_ENABLE_FAST_MATH
    "Enable CUDA fast-math for an explicitly separate performance build"
    OFF
)

set(
    ONNXRUNTIME_ROOT
    ""
    CACHE PATH
    "Root of an extracted ONNX Runtime C/C++ package"
)

set(
    ONNXRUNTIME_RUNTIME_DIR
    ""
    CACHE PATH
    "Optional directory containing ONNX Runtime provider DLLs"
)

set(
    TENSORRT_ROOT
    ""
    CACHE PATH
    "Root of the TensorRT C++ SDK"
)

set(
    EDGE_EXTRA_RUNTIME_DLLS
    ""
    CACHE STRING
    "Optional semicolon-separated Windows DLL paths copied beside executables"
)

# Environment-variable aliases keep presets and CI portable without embedding
# machine-specific absolute paths in source control.
if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{EDGE_ONNXRUNTIME_ROOT})
    set(
        ONNXRUNTIME_ROOT
        "$ENV{EDGE_ONNXRUNTIME_ROOT}"
        CACHE PATH
        "Root of an extracted ONNX Runtime C/C++ package"
        FORCE
    )
endif()

if(
    NOT ONNXRUNTIME_RUNTIME_DIR
    AND DEFINED ENV{EDGE_ONNXRUNTIME_RUNTIME_DIR}
)
    set(
        ONNXRUNTIME_RUNTIME_DIR
        "$ENV{EDGE_ONNXRUNTIME_RUNTIME_DIR}"
        CACHE PATH
        "Optional directory containing ONNX Runtime provider DLLs"
        FORCE
    )
endif()

if(NOT TENSORRT_ROOT AND DEFINED ENV{EDGE_TENSORRT_ROOT})
    set(
        TENSORRT_ROOT
        "$ENV{EDGE_TENSORRT_ROOT}"
        CACHE PATH
        "Root of the TensorRT C++ SDK"
        FORCE
    )
endif()

if(NOT CUDAToolkit_ROOT AND DEFINED ENV{EDGE_CUDA_ROOT})
    set(
        CUDAToolkit_ROOT
        "$ENV{EDGE_CUDA_ROOT}"
        CACHE PATH
        "CUDA Toolkit root"
        FORCE
    )
endif()

if(NOT OpenCV_DIR AND DEFINED ENV{EDGE_OPENCV_DIR})
    set(
        OpenCV_DIR
        "$ENV{EDGE_OPENCV_DIR}"
        CACHE PATH
        "Directory containing OpenCVConfig.cmake"
        FORCE
    )
endif()

if(NOT EDGE_EXTRA_RUNTIME_DLLS AND DEFINED ENV{EDGE_EXTRA_RUNTIME_DLLS})
    set(
        EDGE_EXTRA_RUNTIME_DLLS
        "$ENV{EDGE_EXTRA_RUNTIME_DLLS}"
        CACHE STRING
        "Additional runtime DLLs"
        FORCE
    )
endif()

if(EDGE_ENABLE_CUDA_PIPELINE AND NOT EDGE_ENABLE_TENSORRT)
    message(
        FATAL_ERROR
        "EDGE_ENABLE_CUDA_PIPELINE=ON requires EDGE_ENABLE_TENSORRT=ON."
    )
endif()

# CMake selects the CUDA compiler while enable_language(CUDA) runs. Resolve it
# from the same portable EDGE_CUDA_ROOT contract used by the presets before
# enabling the language, rather than relying on a machine-global PATH entry.
if(EDGE_ENABLE_CUDA_PIPELINE AND NOT CMAKE_CUDA_COMPILER)
    if(DEFINED ENV{EDGE_CUDA_ROOT})
        if(WIN32)
            set(EDGE_NVCC_CANDIDATE "$ENV{EDGE_CUDA_ROOT}/bin/nvcc.exe")
        else()
            set(EDGE_NVCC_CANDIDATE "$ENV{EDGE_CUDA_ROOT}/bin/nvcc")
        endif()

        if(EXISTS "${EDGE_NVCC_CANDIDATE}")
            set(
                CMAKE_CUDA_COMPILER
                "${EDGE_NVCC_CANDIDATE}"
                CACHE FILEPATH
                "CUDA compiler selected from EDGE_CUDA_ROOT"
                FORCE
            )
        endif()
    endif()
endif()

if(
    EDGE_ENABLE_CUDA_PIPELINE
    AND NOT CMAKE_CUDA_HOST_COMPILER
    AND DEFINED ENV{EDGE_MSVC_COMPILER}
    AND EXISTS "$ENV{EDGE_MSVC_COMPILER}"
)
    set(
        CMAKE_CUDA_HOST_COMPILER
        "$ENV{EDGE_MSVC_COMPILER}"
        CACHE FILEPATH
        "CUDA host compiler selected from EDGE_MSVC_COMPILER"
        FORCE
    )
endif()

if(EDGE_ENABLE_CUDA_PIPELINE)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

# =============================================================================
# Dependencies
# =============================================================================

find_package(Threads REQUIRED)

find_package(
    OpenCV
    REQUIRED
    COMPONENTS
        core
        imgproc
        imgcodecs
        videoio
)

find_package(onnxruntime CONFIG QUIET)

if(TARGET onnxruntime::onnxruntime)
    set(EDGE_ONNXRUNTIME_TARGET onnxruntime::onnxruntime)
elseif(TARGET onnxruntime)
    add_library(onnxruntime::onnxruntime ALIAS onnxruntime)
    set(EDGE_ONNXRUNTIME_TARGET onnxruntime::onnxruntime)
else()
    if(NOT ONNXRUNTIME_ROOT)
        message(
            FATAL_ERROR
            "ONNX Runtime was not found.\n"
            "Set ONNXRUNTIME_ROOT to an extracted C/C++ release package."
        )
    endif()

    cmake_path(ABSOLUTE_PATH ONNXRUNTIME_ROOT NORMALIZE)

    find_path(
        ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_ROOT}/include"
        NO_DEFAULT_PATH
        REQUIRED
    )

    find_library(
        ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        HINTS
            "${ONNXRUNTIME_ROOT}/lib"
            "${ONNXRUNTIME_ROOT}/lib64"
        NO_DEFAULT_PATH
        REQUIRED
    )

    if(WIN32)
        find_file(
            ONNXRUNTIME_DLL
            NAMES onnxruntime.dll
            HINTS
                "${ONNXRUNTIME_ROOT}/bin"
                "${ONNXRUNTIME_ROOT}/lib"
            NO_DEFAULT_PATH
            REQUIRED
        )

        add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
        set_target_properties(
            onnxruntime::onnxruntime
            PROPERTIES
                IMPORTED_IMPLIB "${ONNXRUNTIME_LIBRARY}"
                IMPORTED_LOCATION "${ONNXRUNTIME_DLL}"
                INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
        )
    else()
        add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED GLOBAL)
        set_target_properties(
            onnxruntime::onnxruntime
            PROPERTIES
                IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
        )
    endif()

    set(EDGE_ONNXRUNTIME_TARGET onnxruntime::onnxruntime)
endif()

set(EDGE_ONNXRUNTIME_RUNTIME_DLLS)

if(WIN32)
    if(ONNXRUNTIME_RUNTIME_DIR)
        cmake_path(ABSOLUTE_PATH ONNXRUNTIME_RUNTIME_DIR NORMALIZE)
        file(
            GLOB EDGE_ONNXRUNTIME_RUNTIME_DLLS
            CONFIGURE_DEPENDS
            "${ONNXRUNTIME_RUNTIME_DIR}/onnxruntime*.dll"
        )
    elseif(ONNXRUNTIME_ROOT)
        file(
            GLOB EDGE_ONNXRUNTIME_RUNTIME_DLLS
            CONFIGURE_DEPENDS
            "${ONNXRUNTIME_ROOT}/bin/onnxruntime*.dll"
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime*.dll"
        )
    endif()
endif()

if(EDGE_ENABLE_TENSORRT)
    if(NOT TENSORRT_ROOT)
        message(
            FATAL_ERROR
            "EDGE_ENABLE_TENSORRT=ON requires TENSORRT_ROOT."
        )
    endif()

    cmake_path(ABSOLUTE_PATH TENSORRT_ROOT NORMALIZE)
    find_package(CUDAToolkit REQUIRED)

    find_path(
        TENSORRT_INCLUDE_DIR
        NAMES NvInfer.h
        HINTS
            "${TENSORRT_ROOT}/include"
            "${TENSORRT_ROOT}/include/x86_64-linux-gnu"
        NO_DEFAULT_PATH
        REQUIRED
    )

    find_library(
        TENSORRT_NVINFER_LIBRARY
        NAMES nvinfer_11 nvinfer_10 nvinfer
        HINTS
            "${TENSORRT_ROOT}/lib"
            "${TENSORRT_ROOT}/lib64"
            "${TENSORRT_ROOT}/lib/x64"
        NO_DEFAULT_PATH
        REQUIRED
    )

    find_library(
        TENSORRT_PLUGIN_LIBRARY
        NAMES nvinfer_plugin_11 nvinfer_plugin_10 nvinfer_plugin
        HINTS
            "${TENSORRT_ROOT}/lib"
            "${TENSORRT_ROOT}/lib64"
            "${TENSORRT_ROOT}/lib/x64"
        NO_DEFAULT_PATH
        REQUIRED
    )

    if(WIN32)
        find_file(
            TENSORRT_NVINFER_DLL
            NAMES nvinfer_11.dll nvinfer_10.dll nvinfer.dll
            HINTS
                "${TENSORRT_ROOT}/bin"
                "${TENSORRT_ROOT}/lib"
            NO_DEFAULT_PATH
            REQUIRED
        )

        find_file(
            TENSORRT_PLUGIN_DLL
            NAMES
                nvinfer_plugin_11.dll
                nvinfer_plugin_10.dll
                nvinfer_plugin.dll
            HINTS
                "${TENSORRT_ROOT}/bin"
                "${TENSORRT_ROOT}/lib"
            NO_DEFAULT_PATH
            REQUIRED
        )

        find_file(
            TENSORRT_ONNXPARSER_DLL
            NAMES
                nvonnxparser_11.dll
                nvonnxparser_10.dll
                nvonnxparser.dll
            HINTS
                "${TENSORRT_ROOT}/bin"
                "${TENSORRT_ROOT}/lib"
            NO_DEFAULT_PATH
        )

        add_library(TensorRT::nvinfer SHARED IMPORTED GLOBAL)
        set_target_properties(
            TensorRT::nvinfer
            PROPERTIES
                IMPORTED_IMPLIB "${TENSORRT_NVINFER_LIBRARY}"
                IMPORTED_LOCATION "${TENSORRT_NVINFER_DLL}"
                INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
        )

        add_library(TensorRT::nvinfer_plugin SHARED IMPORTED GLOBAL)
        set_target_properties(
            TensorRT::nvinfer_plugin
            PROPERTIES
                IMPORTED_IMPLIB "${TENSORRT_PLUGIN_LIBRARY}"
                IMPORTED_LOCATION "${TENSORRT_PLUGIN_DLL}"
                INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
        )
    else()
        add_library(TensorRT::nvinfer UNKNOWN IMPORTED GLOBAL)
        set_target_properties(
            TensorRT::nvinfer
            PROPERTIES
                IMPORTED_LOCATION "${TENSORRT_NVINFER_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
        )

        add_library(TensorRT::nvinfer_plugin UNKNOWN IMPORTED GLOBAL)
        set_target_properties(
            TensorRT::nvinfer_plugin
            PROPERTIES
                IMPORTED_LOCATION "${TENSORRT_PLUGIN_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
        )
    endif()
endif()

# =============================================================================
# Compiler policy
# =============================================================================

add_library(edge_project_options INTERFACE)
target_compile_features(edge_project_options INTERFACE cxx_std_17)

if(MSVC)
    target_compile_options(
        edge_project_options
        INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CXX>:/permissive->
            $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
    )

    if(EDGE_WARNINGS_AS_ERRORS)
        target_compile_options(
            edge_project_options
            INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:/WX>
        )
    endif()

    if(EDGE_ENABLE_NATIVE_ARCH)
        target_compile_options(
            edge_project_options
            INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
        )
    endif()

    if(EDGE_ENABLE_SANITIZERS)
        target_compile_options(
            edge_project_options
            INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:/fsanitize=address>
        )
    endif()
else()
    target_compile_options(
        edge_project_options
        INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
            $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>
            $<$<COMPILE_LANGUAGE:CXX>:-Wsign-conversion>
            $<$<COMPILE_LANGUAGE:CXX>:-Wshadow>
    )

    if(EDGE_WARNINGS_AS_ERRORS)
        target_compile_options(
            edge_project_options
            INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:-Werror>
        )
    endif()

    if(EDGE_ENABLE_NATIVE_ARCH)
        target_compile_options(
            edge_project_options
            INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:-march=native>
        )
    endif()

    if(EDGE_ENABLE_SANITIZERS)
        target_compile_options(
            edge_project_options
            INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>
                $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
        )
        target_link_options(
            edge_project_options
            INTERFACE
                -fsanitize=address,undefined
        )
    endif()
endif()

if(WIN32)
    target_compile_definitions(
        edge_project_options
        INTERFACE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
    )
endif()

# =============================================================================
# Core library and CLI
# =============================================================================

set(
    EDGE_CORE_SOURCES
    src/image_processor.cpp
    src/inference_engine.cpp
    src/postprocess.cpp
    src/benchmark.cpp
    src/gpu_pipeline_common.cpp
)

if(EDGE_ENABLE_CUDA_PIPELINE)
    list(APPEND EDGE_CORE_SOURCES src/gpu_pipeline.cu)
else()
    list(APPEND EDGE_CORE_SOURCES src/gpu_pipeline_stub.cpp)
endif()

add_library(edge_perception_core STATIC ${EDGE_CORE_SOURCES})
add_library(edge_perception::core ALIAS edge_perception_core)

target_include_directories(
    edge_perception_core
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_include_directories(
    edge_perception_core
    SYSTEM
    PUBLIC
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    edge_perception_core
    PUBLIC
        edge_project_options
        Threads::Threads
        ${EDGE_ONNXRUNTIME_TARGET}
        ${OpenCV_LIBS}
)

target_compile_definitions(
    edge_perception_core
    PUBLIC
        EDGE_PROJECT_VERSION="${PROJECT_VERSION}"
        EDGE_WITH_TENSORRT=$<BOOL:${EDGE_ENABLE_TENSORRT}>
        EDGE_WITH_CUDA_PIPELINE=$<BOOL:${EDGE_ENABLE_CUDA_PIPELINE}>
)

if(EDGE_ENABLE_TENSORRT)
    target_link_libraries(
        edge_perception_core
        PUBLIC
            TensorRT::nvinfer
            TensorRT::nvinfer_plugin
            CUDA::cudart
    )
endif()

if(EDGE_ENABLE_CUDA_PIPELINE)
    set_target_properties(
        edge_perception_core
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    target_compile_options(
        edge_perception_core
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    if(EDGE_ENABLE_FAST_MATH)
        target_compile_options(
            edge_perception_core
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        )
    endif()
endif()

add_executable(edge_perception src/main.cpp)
target_link_libraries(edge_perception PRIVATE edge_perception::core)

if(UNIX AND NOT APPLE)
    set_target_properties(
        edge_perception
        PROPERTIES
            BUILD_RPATH_USE_ORIGIN ON
            INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
    )
elseif(APPLE)
    set_target_properties(
        edge_perception
        PROPERTIES
            BUILD_RPATH_USE_ORIGIN ON
            INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}"
    )
endif()

# =============================================================================
# Automated tests
# =============================================================================

if(EDGE_BUILD_TESTS)
    find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)

    set(
        EDGE_GENERATED_TEST_MODEL
        "${CMAKE_CURRENT_BINARY_DIR}/tests/fixtures/tiny_detector.onnx"
    )

    add_custom_command(
        OUTPUT "${EDGE_GENERATED_TEST_MODEL}"
        COMMAND
            "${CMAKE_COMMAND}" -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/tests/fixtures"
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/generate_test_model.py"
            --output "${EDGE_GENERATED_TEST_MODEL}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/generate_test_model.py"
        COMMENT "Generating deterministic ONNX Runtime integration-test model"
        VERBATIM
    )

    add_custom_target(
        edge_generate_test_model
        DEPENDS "${EDGE_GENERATED_TEST_MODEL}"
    )

    add_executable(
        edge_perception_tests
        tests/test_inference.cpp
    )
    add_dependencies(edge_perception_tests edge_generate_test_model)
    target_link_libraries(
        edge_perception_tests
        PRIVATE edge_perception::core
    )

    add_test(
        NAME edge_unit_tests
        COMMAND edge_perception_tests --unit
    )
    set_tests_properties(
        edge_unit_tests
        PROPERTIES
            LABELS "unit;cpu"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(
        NAME edge_ort_cpu_integration
        COMMAND
            edge_perception_tests
            --integration
            --model "${EDGE_GENERATED_TEST_MODEL}"
    )
    set_tests_properties(
        edge_ort_cpu_integration
        PROPERTIES
            LABELS "integration;cpu;onnxruntime"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(
        NAME edge_auto_backend_cli
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_auto_backend.py"
            --executable "$<TARGET_FILE:edge_perception>"
            --model "${EDGE_GENERATED_TEST_MODEL}"
            --output-dir
            "${CMAKE_CURRENT_BINARY_DIR}/tests/auto_backend_output"
    )
    set_tests_properties(
        edge_auto_backend_cli
        PROPERTIES
            LABELS "integration;cpu;cli;fallback"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    if(EDGE_BUILD_PYTHON_TESTS)
        add_test(
            NAME edge_python_parity_unit
            COMMAND
                "${Python3_EXECUTABLE}" -m pytest
                -q
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_parity.py"
                -m "not integration"
        )
        set_tests_properties(
            edge_python_parity_unit
            PROPERTIES
                LABELS "unit;python;parity"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
        )
    endif()
endif()

# =============================================================================
# Release optimization
# =============================================================================

if(EDGE_ENABLE_IPO)
    check_ipo_supported(
        RESULT EDGE_IPO_SUPPORTED
        OUTPUT EDGE_IPO_ERROR
        LANGUAGES CXX
    )

    if(EDGE_IPO_SUPPORTED)
        if(NOT EDGE_ENABLE_CUDA_PIPELINE)
            set_property(
                TARGET edge_perception_core
                PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
            )
            set_property(
                TARGET edge_perception_core
                PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE
            )
        endif()

        set_property(
            TARGET edge_perception
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
        set_property(
            TARGET edge_perception
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE
        )

        if(EDGE_BUILD_TESTS)
            set_property(
                TARGET edge_perception_tests
                PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
            )
        endif()
    else()
        message(WARNING "IPO/LTO unavailable: ${EDGE_IPO_ERROR}")
    endif()
endif()

# =============================================================================
# Runtime DLL deployment
# =============================================================================

function(edge_copy_runtime_dlls EDGE_TARGET)
    if(NOT WIN32 OR NOT EDGE_COPY_RUNTIME_DLLS)
        return()
    endif()

    add_custom_command(
        TARGET "${EDGE_TARGET}"
        POST_BUILD
        COMMAND
            "${CMAKE_COMMAND}" -E copy_if_different
            "$<TARGET_RUNTIME_DLLS:${EDGE_TARGET}>"
            "$<TARGET_FILE_DIR:${EDGE_TARGET}>"
        COMMAND_EXPAND_LISTS
        VERBATIM
    )

    foreach(EDGE_RUNTIME_DLL IN LISTS EDGE_ONNXRUNTIME_RUNTIME_DLLS)
        add_custom_command(
            TARGET "${EDGE_TARGET}"
            POST_BUILD
            COMMAND
                "${CMAKE_COMMAND}" -E copy_if_different
                "${EDGE_RUNTIME_DLL}"
                "$<TARGET_FILE_DIR:${EDGE_TARGET}>"
            VERBATIM
        )
    endforeach()

    foreach(EDGE_RUNTIME_DLL IN LISTS EDGE_EXTRA_RUNTIME_DLLS)
        if(NOT EXISTS "${EDGE_RUNTIME_DLL}")
            message(
                FATAL_ERROR
                "EDGE_EXTRA_RUNTIME_DLLS contains a missing file: "
                "${EDGE_RUNTIME_DLL}"
            )
        endif()

        add_custom_command(
            TARGET "${EDGE_TARGET}"
            POST_BUILD
            COMMAND
                "${CMAKE_COMMAND}" -E copy_if_different
                "${EDGE_RUNTIME_DLL}"
                "$<TARGET_FILE_DIR:${EDGE_TARGET}>"
            VERBATIM
        )
    endforeach()

    if(EDGE_ENABLE_TENSORRT)
        foreach(
            EDGE_TENSORRT_DLL
            IN ITEMS
                "${TENSORRT_NVINFER_DLL}"
                "${TENSORRT_PLUGIN_DLL}"
                "${TENSORRT_ONNXPARSER_DLL}"
        )
            if(EDGE_TENSORRT_DLL AND EXISTS "${EDGE_TENSORRT_DLL}")
                add_custom_command(
                    TARGET "${EDGE_TARGET}"
                    POST_BUILD
                    COMMAND
                        "${CMAKE_COMMAND}" -E copy_if_different
                        "${EDGE_TENSORRT_DLL}"
                        "$<TARGET_FILE_DIR:${EDGE_TARGET}>"
                    VERBATIM
                )
            endif()
        endforeach()
    endif()
endfunction()

edge_copy_runtime_dlls(edge_perception)

if(EDGE_BUILD_TESTS)
    edge_copy_runtime_dlls(edge_perception_tests)
endif()

# =============================================================================
# Installation
# =============================================================================

install(
    TARGETS
        edge_perception
        edge_perception_core
        edge_project_options
    EXPORT edge_perception_targets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
)

install(
    EXPORT edge_perception_targets
    NAMESPACE edge_perception::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/edge_perception"
)

# =============================================================================
# Configuration summary
# =============================================================================

message(STATUS "")
message(STATUS "============================================================")
message(STATUS "edge-perception-cpp ${PROJECT_VERSION}")
message(STATUS "============================================================")
message(STATUS "CMake generator:             ${CMAKE_GENERATOR}")
message(STATUS "C++ compiler:                ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build type:                  ${CMAKE_BUILD_TYPE}")
message(STATUS "Build tests:                 ${EDGE_BUILD_TESTS}")
message(STATUS "Python tests:                ${EDGE_BUILD_PYTHON_TESTS}")
message(STATUS "Native TensorRT backend:     ${EDGE_ENABLE_TENSORRT}")
message(STATUS "Fused CUDA pipeline:         ${EDGE_ENABLE_CUDA_PIPELINE}")
if(EDGE_ENABLE_NATIVE_ARCH)
    message(STATUS "Portable CPU baseline:       OFF")
else()
    message(STATUS "Portable CPU baseline:       ON")
endif()
message(STATUS "IPO/LTO requested:           ${EDGE_ENABLE_IPO}")
message(STATUS "CUDA fast-math:              ${EDGE_ENABLE_FAST_MATH}")
message(STATUS "OpenCV version:              ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime target:         ${EDGE_ONNXRUNTIME_TARGET}")

if(ONNXRUNTIME_ROOT)
    message(STATUS "ONNX Runtime root:           ${ONNXRUNTIME_ROOT}")
endif()

if(EDGE_ENABLE_TENSORRT)
    message(STATUS "TensorRT root:               ${TENSORRT_ROOT}")
    message(STATUS "CUDA Toolkit version:        ${CUDAToolkit_VERSION}")
endif()

if(EDGE_ENABLE_CUDA_PIPELINE)
    message(STATUS "CUDA architectures:          ${CMAKE_CUDA_ARCHITECTURES}")
endif()

message(STATUS "============================================================")
message(STATUS "")
