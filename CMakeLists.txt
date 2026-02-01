cmake_minimum_required(VERSION 3.16)
project(DelaunaySurfaceReconstruction CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CGAL REQUIRED)
find_package(SCIP REQUIRED)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

aux_source_directory (src SRC_LIST)

include_directories(include)

add_executable(delaunay_surface_reconstruction ${SRC_LIST})
target_include_directories(delaunay_surface_reconstruction PRIVATE ${SCIP_INCLUDE_DIRS})
target_link_libraries(delaunay_surface_reconstruction PRIVATE CGAL::CGAL ${SCIP_LIBRARIES})
target_compile_definitions(delaunay_surface_reconstruction PRIVATE USE_SCIP=1)
