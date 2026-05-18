cmake_minimum_required(VERSION 3.5)
project(rgbd_to_pointcloud)

SET(CMAKE_BUILD_TYPE Release)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

add_compile_options(-w)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(realsense2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include 
    ${OpenCV_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS} 
    ${Eigen3_INCLUDE_DIRS} 
    ${TBB_INCLUDE_DIRS}
)

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(mylib SHARED ${SRCS})
target_link_libraries(
    mylib 
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} 
    ${Eigen3_LIBRARIES} 
    TBB::tbb 
    realsense2::realsense2
)

# add_executable(vis Example/vis.cpp)
# add_executable(rs_align Example/rs_align.cpp)
# add_executable(mask Example/testMaskVis.cpp)
add_executable(sem_edge Example/testSemEdge.cpp)

# target_link_libraries(vis mylib)
# target_link_libraries(rs_align mylib)
# target_link_libraries(mask mylib)
target_link_libraries(sem_edge mylib)