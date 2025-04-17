project(msit VERSION 1.0.0 LANGUAGES CXX C)
cmake_minimum_required(VERSION 3.14)

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Python3 ${PYTHON_VERSION} EXACT COMPONENTS Development)
include_directories(${Python3_INCLUDE_DIRS})

add_library(msit_c SHARED)

# 安全编译选项
target_compile_options(msit_c PRIVATE "-Wall")
target_compile_options(msit_c PRIVATE "-fPIC")
target_compile_options(msit_c PRIVATE "-fstack-protector-all")
target_compile_options(msit_c PRIVATE "-ftrapv")
target_compile_options(msit_c PRIVATE "-fstack-check")
target_compile_options(msit_c PRIVATE "-D_FORTIFY_SOURCE=2")

target_link_options(msit_c PRIVATE "-Wl,-z,relor")
target_link_options(msit_c PRIVATE "-Wl,-z,now")
target_link_options(msit_c PRIVATE "-Wl,-z,noexecstack")
target_link_options(msit_c PRIVATE "-Wl,--disable-new-dtags")
target_link_options(msit_c PRIVATE "-s")

target_link_libraries(msit_c PUBLIC dl)
target_link_libraries(msit_c PUBLIC pthread)
target_link_libraries(msit_c PUBLIC ${cpython_LIBRARIES})

if(DEFINED BUILD_TYPE AND "${BUILD_TYPE}" STREQUAL "debug")
    target_compile_options(msit_c PRIVATE "-O0")
    target_compile_options(msit_c PRIVATE "-g")
    target_compile_definitions(msit_c PRIVATE __DEBUG__)
else()
    target_compile_options(msit_c PRIVATE "-O2")
endif()

target_include_directories(msit_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE SOURCES "*.cpp")
target_sources(msit_c PRIVATE ${SOURCES})

install(TARGETS msit_c LIBRARY DESTINATION lib)
