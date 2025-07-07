cmake_minimum_required(VERSION 3.29)
project(proyecto-final-grupo-5)

set(CMAKE_CXX_STANDARD 20)

add_executable(untitled3
        #main.cpp
        #include/algebra/tensor.h
        #include/nn/neural_network.h
        #include/nn/activation.h
        #include/nn/optimizer.h
        #include/nn/loss.h
        #include/nn/interfaces.h
        #include/nn/dense.h
        #src/utec/agent/PonAgent.cpp
        #src/utec/agent/EnvGym.cpp
        #include/agent/PongAgent.h
        #include/agent/EnvGym.h
        tests/test_agent_env.cpp
        src/utec/agent/EnvGym.cpp
        src/utec/agent/PonAgent.cpp
        src/utec/concurrent_queue.h
        src/utec/thread_pool.h
        include/nn/parallel_inference.h
)
