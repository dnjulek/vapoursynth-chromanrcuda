cmake_minimum_required(VERSION 3.20)

project(ChromaNRCuda LANGUAGES CXX CUDA)

add_library(chromanrcuda_kernel OBJECT src/kernel.cu)

set_target_properties(chromanrcuda_kernel PROPERTIES
    CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

add_library(chromanrcuda_source OBJECT src/chromanrcuda.cpp)

target_include_directories(chromanrcuda_source PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(PkgConfig QUIET MODULE)

if(PKG_CONFIG_FOUND)
    pkg_search_module(VS vapoursynth)

    if(VS_FOUND)
        message(STATUS "Found VapourSynth r${VS_VERSION}")
        target_include_directories(chromanrcuda_source PRIVATE ${VS_INCLUDE_DIRS})
    endif()
endif()

if(NOT VS_FOUND)
    if(EXISTS "C:/Program Files/VapourSynth/sdk/include/vapoursynth")
        set(VS_INCLUDE_DIR "C:/Program Files/VapourSynth/sdk/include/vapoursynth" CACHE PATH "Path to VapourSynth headers")
    else()
        set(VS_INCLUDE_DIR "" CACHE PATH "Path to VapourSynth headers")
    endif()

    if(VS_INCLUDE_DIR STREQUAL "")
        message(WARNING "VapourSynth not found")
    endif()

    target_include_directories(chromanrcuda_source PRIVATE ${VS_INCLUDE_DIR})
endif()

set_target_properties(chromanrcuda_source PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

add_library(chromanrcuda SHARED)
target_link_libraries(chromanrcuda chromanrcuda_source chromanrcuda_kernel)

if(VS_FOUND)
    cmake_path(APPEND install_dir ${VS_LIBDIR} vapoursynth)
    install(TARGETS chromanrcuda LIBRARY DESTINATION ${install_dir})
else()
    if(WIN32)
        install(TARGETS chromanrcuda RUNTIME)
    else()
        install(TARGETS chromanrcuda LIBRARY)
    endif()
endif()
