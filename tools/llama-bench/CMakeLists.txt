# llama-bench-impl: benchmark logic, reusable by app

set(TARGET llama-bench-impl)

add_library(${TARGET} llama-bench.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-bench executable

set(TARGET llama-bench)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-bench-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(BUILD_TESTING)
    add_test(NAME test-llama-bench-q4-kernel-valid COMMAND llama-bench --q4-kernel auto --version)
    add_test(NAME test-llama-bench-q4-kernel-invalid COMMAND llama-bench --q4-kernel invalid)
    set_tests_properties(test-llama-bench-q4-kernel-invalid PROPERTIES WILL_FAIL TRUE)
    add_test(NAME test-llama-bench-q4-kernel-repack-conflict COMMAND llama-bench --q4-kernel 4x4 --repack 0)
    set_tests_properties(test-llama-bench-q4-kernel-repack-conflict PROPERTIES WILL_FAIL TRUE)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
