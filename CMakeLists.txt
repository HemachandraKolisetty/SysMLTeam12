cmake_minimum_required(VERSION 3.15)
project(SysMLTeam12)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11 package
find_package(pybind11 REQUIRED)

# Add the executable
add_executable(main main.cpp)

# Link pybind11 to the executable
target_link_libraries(main PRIVATE pybind11::embed)

# Include Python directories
find_package(Python3 REQUIRED COMPONENTS Development)
target_include_directories(main PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(main PRIVATE ${Python3_LIBRARIES})