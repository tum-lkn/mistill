cmake_minimum_required(VERSION 3.16)
project(makeHnsa)

#set(PROJECT_ROOT /home/mininet/dc-emulation/cpp)
set(PROJECT_ROOT /home/patrick/Documents/GitHub/lkn/dc-mininet-emulation/cpp)
# set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" /home/developer/libtorch/share/cmake)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" ${PROJECT_ROOT}/libtorch/share/cmake)
find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_CXX_STANDARD 14)

add_executable(makeHnsa main.cpp)
target_include_directories(makeHnsa PUBLIC ${PROJECT_ROOT}/libtorch/include)
target_link_libraries(makeHnsa "${TORCH_LIBRARIES}")
set_property(TARGET makeHnsa PROPERTY CXX_STANDARD 14)
