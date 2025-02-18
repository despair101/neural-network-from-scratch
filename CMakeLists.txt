cmake_minimum_required(VERSION 3.28)
project(neural_network_from_scratch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(neural_network_from_scratch main.cpp
        activation_function.cpp
        layer.cpp
        network.cpp
        loss_function.cpp)
target_link_libraries(neural_network_from_scratch Eigen3::Eigen)


