cmake_minimum_required(VERSION 3.18..3.21)

project(orb_gpu CXX CUDA)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_EXPORT_COMPILE_COMMANDS On)
set(CMAKE_CUDA_ARCHITECTURES 80)

# project specific flags

set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV)

add_executable(orb_gpu src/main.cpp src/fast.cu src/scaling.cu src/accum.cu
                       src/gaussian_blur.cu)
target_include_directories(
  orb_gpu PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                 $<INSTALL_INTERFACE:include> ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orb_gpu PUBLIC CUDA::cudart CUDA::cuda_driver
                                     ${OpenCV_LIBS})
