cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

set(PROJECT_ROOT /home/sim/pytorchcpp)
set(CUDNN_INCLUDE_DIR /usr/lib/cuda/include)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" ${PROJECT_ROOT}/libtorch/share/cmake)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(user.o user_space.cpp)
target_include_directories(user.o PUBLIC ${PROJECT_ROOT}/libtorch/include)
target_link_libraries(user.o "${TORCH_LIBRARIES}")
set_property(TARGET user.o PROPERTY CXX_STANDARD 14)
