cmake_minimum_required(VERSION 3.24)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

project(
    edge_perception_cpp
    VERSION 0.1.0
    DESCRIPTION "Native C++ inference and benchmarking for YOLO26M on BDD100K"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CTest)
include(CheckIPOSupported)

# =============================================================================
# Build defaults
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Give single-configuration generators such as Ninja a sensible default.
# Visual Studio is a multi-configuration generator and ignores this branch.
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

# Keep binaries in predictable directories for every generator.
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
# Project options
# =============================================================================

option(
    EDGE_ENABLE_TENSORRT
    "Build the native TensorRT .engine inference backend"
    OFF
)

option(
    EDGE_BUILD_TESTS
    "Build and register the C++ inference test executable"
    ON
)

option(
    EDGE_COPY_RUNTIME_DLLS
    "Copy discovered runtime DLLs beside Windows executables after building"
    ON
)

option(
    EDGE_ENABLE_IPO
    "Enable interprocedural optimization in Release-like builds when supported"
    ON
)

option(
    EDGE_WARNINGS_AS_ERRORS
    "Treat compiler warnings as errors"
    OFF
)

option(
    EDGE_ENABLE_SANITIZERS
    "Enable AddressSanitizer and UndefinedBehaviorSanitizer in supported builds"
    OFF
)

option(
    EDGE_ENABLE_NATIVE_ARCH
    "Compile for the current machine's CPU instead of a portable CPU baseline"
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
    "Optional directory containing ONNX Runtime and execution-provider DLLs"
)

set(
    TENSORRT_ROOT
    ""
    CACHE PATH
    "Root of the TensorRT C++ SDK; required when EDGE_ENABLE_TENSORRT=ON"
)

set(
    EDGE_EXTRA_RUNTIME_DLLS
    ""
    CACHE STRING
    "Optional semicolon-separated list of additional Windows runtime DLLs to copy"
)

# =============================================================================
# Dependency discovery: threads and OpenCV
# =============================================================================

find_package(Threads REQUIRED)

find_package(
    OpenCV
    REQUIRED
    COMPONENTS
        core
        imgproc
        imgcodecs
)

# =============================================================================
# Dependency discovery: ONNX Runtime
# =============================================================================

# Prefer an installed CMake package when one is available. Official extracted
# archives do not always ship a usable package configuration, so a robust
# ONNXRUNTIME_ROOT fallback is provided below.
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
            "Pass the extracted C/C++ package root with:\n"
            "  -DONNXRUNTIME_ROOT=\"C:/path/to/onnxruntime\"\n"
            "The directory must contain include/ and lib/ or bin/."
        )
    endif()

    cmake_path(ABSOLUTE_PATH ONNXRUNTIME_ROOT NORMALIZE)

    find_path(
        ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS
            "${ONNXRUNTIME_ROOT}/include"
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

        add_library(
            onnxruntime::onnxruntime
            SHARED
            IMPORTED
            GLOBAL
        )

        set_target_properties(
            onnxruntime::onnxruntime
            PROPERTIES
                IMPORTED_IMPLIB "${ONNXRUNTIME_LIBRARY}"
                IMPORTED_LOCATION "${ONNXRUNTIME_DLL}"
                INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
        )

    else()
        add_library(
            onnxruntime::onnxruntime
            UNKNOWN
            IMPORTED
            GLOBAL
        )

        set_target_properties(
            onnxruntime::onnxruntime
            PROPERTIES
                IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
        )
    endif()

    set(EDGE_ONNXRUNTIME_TARGET onnxruntime::onnxruntime)
endif()

# Locate ONNX Runtime execution-provider DLLs that are dynamically loaded and
# therefore may not appear in the normal linker dependency graph.
set(EDGE_ONNXRUNTIME_RUNTIME_DLLS)

if(WIN32)
    if(ONNXRUNTIME_RUNTIME_DIR)
        cmake_path(
            ABSOLUTE_PATH
            ONNXRUNTIME_RUNTIME_DIR
            NORMALIZE
        )

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

# =============================================================================
# Dependency discovery: native TensorRT backend
# =============================================================================

if(EDGE_ENABLE_TENSORRT)
    if(NOT TENSORRT_ROOT)
        message(
            FATAL_ERROR
            "EDGE_ENABLE_TENSORRT=ON requires the TensorRT C++ SDK.\n"
            "Pass its root with:\n"
            "  -DTENSORRT_ROOT=\"C:/path/to/TensorRT\"\n"
            "The Python tensorrt package alone is not sufficient for C++ "
            "headers and import libraries."
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
        NAMES
            nvinfer_11
            nvinfer_10
            nvinfer
        HINTS
            "${TENSORRT_ROOT}/lib"
            "${TENSORRT_ROOT}/lib64"
            "${TENSORRT_ROOT}/lib/x64"
        NO_DEFAULT_PATH
        REQUIRED
    )

    find_library(
        TENSORRT_PLUGIN_LIBRARY
        NAMES
            nvinfer_plugin_11
            nvinfer_plugin_10
            nvinfer_plugin
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
            NAMES
                nvinfer_11.dll
                nvinfer_10.dll
                nvinfer.dll
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

        add_library(
            TensorRT::nvinfer
            SHARED
            IMPORTED
            GLOBAL
        )

        set_target_properties(
            TensorRT::nvinfer
            PROPERTIES
                IMPORTED_IMPLIB "${TENSORRT_NVINFER_LIBRARY}"
                IMPORTED_LOCATION "${TENSORRT_NVINFER_DLL}"
                INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
        )

        add_library(
            TensorRT::nvinfer_plugin
            SHARED
            IMPORTED
            GLOBAL
        )

        set_target_properties(
            TensorRT::nvinfer_plugin
            PROPERTIES
                IMPORTED_IMPLIB "${TENSORRT_PLUGIN_LIBRARY}"
                IMPORTED_LOCATION "${TENSORRT_PLUGIN_DLL}"
                INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
        )

    else()
        add_library(
            TensorRT::nvinfer
            UNKNOWN
            IMPORTED
            GLOBAL
        )

        set_target_properties(
            TensorRT::nvinfer
            PROPERTIES
                IMPORTED_LOCATION "${TENSORRT_NVINFER_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
        )

        add_library(
            TensorRT::nvinfer_plugin
            UNKNOWN
            IMPORTED
            GLOBAL
        )

        set_target_properties(
            TensorRT::nvinfer_plugin
            PROPERTIES
                IMPORTED_LOCATION "${TENSORRT_PLUGIN_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
        )
    endif()
endif()

# =============================================================================
# Shared compiler policy target
# =============================================================================

add_library(
    edge_project_options
    INTERFACE
)

target_compile_features(
    edge_project_options
    INTERFACE
        cxx_std_17
)

if(MSVC)
    target_compile_options(
        edge_project_options
        INTERFACE
            /W4
            /permissive-
            /EHsc
            /Zc:__cplusplus
    )

    if(EDGE_WARNINGS_AS_ERRORS)
        target_compile_options(
            edge_project_options
            INTERFACE
                /WX
        )
    endif()

    if(EDGE_ENABLE_NATIVE_ARCH)
        target_compile_options(
            edge_project_options
            INTERFACE
                /arch:AVX2
        )
    endif()

    if(EDGE_ENABLE_SANITIZERS)
        target_compile_options(
            edge_project_options
            INTERFACE
                /fsanitize=address
        )
    endif()

else()
    target_compile_options(
        edge_project_options
        INTERFACE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
            -Wshadow
    )

    if(EDGE_WARNINGS_AS_ERRORS)
        target_compile_options(
            edge_project_options
            INTERFACE
                -Werror
        )
    endif()

    if(EDGE_ENABLE_NATIVE_ARCH)
        target_compile_options(
            edge_project_options
            INTERFACE
                -march=native
        )
    endif()

    if(EDGE_ENABLE_SANITIZERS)
        target_compile_options(
            edge_project_options
            INTERFACE
                -fsanitize=address,undefined
                -fno-omit-frame-pointer
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
# Core inference library
# =============================================================================

add_library(
    edge_perception_core
    STATIC
        src/image_processor.cpp
        src/inference_engine.cpp
        src/postprocess.cpp
        src/benchmark.cpp
)

add_library(
    edge_perception::core
    ALIAS
    edge_perception_core
)

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

# =============================================================================
# Command-line application
# =============================================================================

add_executable(
    edge_perception
    src/main.cpp
)

target_link_libraries(
    edge_perception
    PRIVATE
        edge_perception::core
)

# Keep Linux/macOS build-tree execution convenient and make installed binaries
# search for bundled shared libraries relative to themselves.
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
# Optional C++ tests
# =============================================================================

if(EDGE_BUILD_TESTS)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_inference.cpp")
        message(
            FATAL_ERROR
            "EDGE_BUILD_TESTS=ON, but tests/test_inference.cpp does not exist "
            "yet.\n"
            "Until that final test file is created, configure with:\n"
            "  -DEDGE_BUILD_TESTS=OFF"
        )
    endif()

    add_executable(
        edge_perception_tests
        tests/test_inference.cpp
    )

    target_link_libraries(
        edge_perception_tests
        PRIVATE
            edge_perception::core
    )

    add_test(
        NAME edge_perception_tests
        COMMAND edge_perception_tests
    )

    set_tests_properties(
        edge_perception_tests
        PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
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
        set_property(
            TARGET edge_perception_core
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )

        set_property(
            TARGET edge_perception_core
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE
        )

        set_property(
            TARGET edge_perception
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )

        set_property(
            TARGET edge_perception
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE
        )

    else()
        message(
            WARNING
            "Interprocedural optimization is unavailable: ${EDGE_IPO_ERROR}"
        )
    endif()
endif()

# =============================================================================
# Windows runtime DLL deployment
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
        COMMENT "Copying linked runtime DLLs for ${EDGE_TARGET}"
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
            COMMENT "Copying ${EDGE_RUNTIME_DLL}"
            VERBATIM
        )
    endforeach()

    foreach(EDGE_RUNTIME_DLL IN LISTS EDGE_EXTRA_RUNTIME_DLLS)
        if(NOT EXISTS "${EDGE_RUNTIME_DLL}")
            message(
                FATAL_ERROR
                "EDGE_EXTRA_RUNTIME_DLLS contains a missing file:\n"
                "  ${EDGE_RUNTIME_DLL}"
            )
        endif()

        add_custom_command(
            TARGET "${EDGE_TARGET}"
            POST_BUILD
            COMMAND
                "${CMAKE_COMMAND}" -E copy_if_different
                "${EDGE_RUNTIME_DLL}"
                "$<TARGET_FILE_DIR:${EDGE_TARGET}>"
            COMMENT "Copying additional runtime DLL ${EDGE_RUNTIME_DLL}"
            VERBATIM
        )
    endforeach()
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

if(WIN32)
    if(EDGE_ONNXRUNTIME_RUNTIME_DLLS)
        install(
            FILES ${EDGE_ONNXRUNTIME_RUNTIME_DLLS}
            DESTINATION "${CMAKE_INSTALL_BINDIR}"
            OPTIONAL
        )
    endif()

    if(EDGE_EXTRA_RUNTIME_DLLS)
        install(
            FILES ${EDGE_EXTRA_RUNTIME_DLLS}
            DESTINATION "${CMAKE_INSTALL_BINDIR}"
            OPTIONAL
        )
    endif()

    if(EDGE_ENABLE_TENSORRT)
        install(
            FILES
                "${TENSORRT_NVINFER_DLL}"
                "${TENSORRT_PLUGIN_DLL}"
            DESTINATION "${CMAKE_INSTALL_BINDIR}"
        )
    endif()
endif()

# =============================================================================
# Configuration summary
# =============================================================================

message(STATUS "")
message(STATUS "============================================================")
message(STATUS "edge-perception-cpp configuration")
message(STATUS "============================================================")
message(STATUS "Project version:             ${PROJECT_VERSION}")
message(STATUS "CMake generator:             ${CMAKE_GENERATOR}")
message(
    STATUS
    "C++ compiler:                "
    "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
)
message(STATUS "C++ standard:                ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type:                  ${CMAKE_BUILD_TYPE}")
message(STATUS "Build tests:                 ${EDGE_BUILD_TESTS}")
message(STATUS "Native TensorRT backend:     ${EDGE_ENABLE_TENSORRT}")
message(STATUS "Copy Windows runtime DLLs:   ${EDGE_COPY_RUNTIME_DLLS}")
message(STATUS "IPO/LTO requested:           ${EDGE_ENABLE_IPO}")
message(STATUS "Warnings as errors:          ${EDGE_WARNINGS_AS_ERRORS}")
message(STATUS "Sanitizers:                  ${EDGE_ENABLE_SANITIZERS}")
message(STATUS "Native CPU architecture:     ${EDGE_ENABLE_NATIVE_ARCH}")
message(STATUS "")
message(STATUS "OpenCV version:              ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories:  ${OpenCV_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime target:         ${EDGE_ONNXRUNTIME_TARGET}")

if(ONNXRUNTIME_ROOT)
    message(STATUS "ONNX Runtime root:           ${ONNXRUNTIME_ROOT}")
endif()

if(ONNXRUNTIME_RUNTIME_DIR)
    message(
        STATUS
        "ONNX Runtime runtime dir:    ${ONNXRUNTIME_RUNTIME_DIR}"
    )
endif()

if(EDGE_ENABLE_TENSORRT)
    message(STATUS "TensorRT root:               ${TENSORRT_ROOT}")
    message(STATUS "TensorRT include directory:  ${TENSORRT_INCLUDE_DIR}")
    message(
        STATUS
        "TensorRT nvinfer library:    ${TENSORRT_NVINFER_LIBRARY}"
    )
    message(
        STATUS
        "TensorRT plugin library:     ${TENSORRT_PLUGIN_LIBRARY}"
    )
    message(STATUS "CUDA Toolkit version:        ${CUDAToolkit_VERSION}")
endif()

message(
    STATUS
    "Executable output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)
message(STATUS "============================================================")
message(STATUS "")