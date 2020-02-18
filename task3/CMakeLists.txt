project(task3)

cmake_minimum_required (VERSION 2.6)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -Wall -Wpedantic -O3 --std=c++0x")
set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,readability-*,modernize-*,performance-*,portability-*,cppcoreguidelines-*,-modernize-use-trailing-return-type)

file(GLOB SOURCES "*.cpp")

add_executable(task3 ${SOURCES})
