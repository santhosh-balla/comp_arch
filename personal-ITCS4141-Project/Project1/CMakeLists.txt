cmake_minimum_required(VERSION 3.6)
project( OpenCVExample LANGUAGES C CXX )

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package( OpenCV REQUIRED )

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

add_executable( DisplayImage DisplayImage.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )

add_executable( FilteringOpenCV FilteringOpenCV.cpp )
target_link_libraries( FilteringOpenCV ${OpenCV_LIBS} )

add_executable( Filtering Filtering.cpp )
target_link_libraries( Filtering ${OpenCV_LIBS} )

if(CMAKE_CUDA_COMPILER)
    set_source_files_properties(Filtering.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(Filtering PRIVATE ENABLE_CUDA=1)
    set_target_properties(Filtering PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()