cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)

project(MetaRay)

include_directories(include)
include_directories(external/glad/include)
include_directories(external/partio/src/lib)
include_directories(external/spdlog/include)

# aux_source_directory(src/MTR MTR_SRCS)
# aux_source_directory(src SRCS)
# set(SPECIFIC_OUTPUT_DIR "${CMAKE_BUILD_TYPE}")

set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin")

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX) 
    set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/${CMAKE_BUILD_TYPE}")
    set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/${CMAKE_BUILD_TYPE}")
endif()

# message(${LIBRARY_OUTPUT_PATH})
file(COPY ${SHADERS} DESTINATION ${EXECUTABLE_OUTPUT_DIRECTORY}${CMAKE_BUILD_TYPE}/shaders)

add_subdirectory(external/glad)
add_subdirectory(external/partio)
add_subdirectory(external/spdlog)

add_subdirectory(src/MPM)