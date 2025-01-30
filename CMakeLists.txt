cmake_minimum_required(VERSION 3.28)
project(neural_network_from_scratch)

set(CMAKE_CXX_STANDARD 20)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)


add_executable(neural_network_from_scratch main.cpp
        utils.h
        activation_function.h
        activation_function.cpp
        layer.cpp
        layer.h
        network.cpp
        network.h
        loss_function.cpp
        loss_function.h
        linear_function.cpp
        linear_function.h
        function.cpp
        function.h)
target_link_libraries(neural_network_from_scratch Eigen3::Eigen)


