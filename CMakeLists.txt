cmake_minimum_required(VERSION 3.28)
project(neural_network_from_scratch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(extern/eigen)
add_subdirectory(extern/Catch2)

add_executable(test
        activation_function.cpp
        layer.cpp
        network.cpp
        loss_function.cpp
        data_loader.cpp
        utils.cpp
        sgd.cpp
        adam.cpp
        test.cpp
)

target_link_libraries(test PRIVATE Eigen3::Eigen)
target_link_libraries(test PRIVATE Catch2::Catch2WithMain)
