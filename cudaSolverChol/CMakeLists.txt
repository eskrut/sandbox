project(cudaSolverChol)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)

find_package(CUDA REQUIRED)

find_package(sbf REQUIRED)

cuda_add_executable(${PROJECT_NAME} ${SRC_LIST})
include_directories(${SBF_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${SBF_LIBRARIES} cublas cusparse)
