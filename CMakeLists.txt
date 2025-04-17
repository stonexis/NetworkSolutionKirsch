cmake_minimum_required(VERSION 3.14)
project(genstressfield)

# Default to Release build if none is set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Force optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ZLIB REQUIRED)

add_executable(${PROJECT_NAME}
  main.cpp
  cnpy.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}      # for gendata.hpp and cnpy.h
)

target_link_libraries(${PROJECT_NAME} PRIVATE ZLIB::ZLIB)
