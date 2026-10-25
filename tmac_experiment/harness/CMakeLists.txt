cmake_minimum_required(VERSION 3.14)
project(tmac_baseline_harness LANGUAGES C CXX)

if(NOT LLAMA_SOURCE)
    message(FATAL_ERROR "Set LLAMA_SOURCE to the pinned llama.cpp submodule")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_SOURCE} llama)

add_executable(tmac-operator operator.cpp)
target_link_libraries(tmac-operator PRIVATE ggml)
target_include_directories(tmac-operator PRIVATE ${LLAMA_SOURCE}/ggml/include)
target_compile_features(tmac-operator PRIVATE cxx_std_17)

add_executable(tmac-logits logits.cpp)
target_link_libraries(tmac-logits PRIVATE llama)
target_compile_features(tmac-logits PRIVATE cxx_std_17)
