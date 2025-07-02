cmake_minimum_required(VERSION 3.29)
project(proyecto-final-grupo-5)

set(CMAKE_CXX_STANDARD 20)

add_executable(untitled3 main.cpp
        tensor.h
        nn_interfaces.h
        nn_dense.h
        nn_activation.h
        nn_loss.h
        neural_network.h
        nn_optimizer.h)
