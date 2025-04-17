cmake_minimum_required(VERSION 3.28)
project(NNFS)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(extern/eigen)
add_subdirectory(extern/Catch2)

add_library(NNFS STATIC
        activation_function.cpp
        layer.cpp
        network.cpp
        loss_function.cpp
        data_loader.cpp
        utils.cpp
        sgd.cpp
        adam.cpp
)

target_include_directories(NNFS PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(NNFS PUBLIC Eigen3::Eigen)

add_executable(test test.cpp)

target_link_libraries(test PRIVATE NNFS Catch2::Catch2WithMain)
