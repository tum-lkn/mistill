cmake_minimum_required(VERSION 3.23)
project(avxnn)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512cd -O2")

add_executable(avxnn avx.cpp)
# target_link_libraries(avxnn pthread)
add_executable(chronobench chronobench.cpp)
