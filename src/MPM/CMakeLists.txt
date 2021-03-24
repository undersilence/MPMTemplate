project(MaterialPointMethod)

if(MSVC)
    # Force to always compile with W4
    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    # Update if necessary
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
endif()

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    add_definitions(-DTBB_USE_DEBUG)
    add_definitions(-DTBB_USE_THREADING_TOOLS)
endif()

# add_definitions(-DDISTRIBUTE)
# message("cmake flag ${CMAKE_CXX_FLAGS}") 
add_executable(MPM main.cpp mpm_sim.cpp mpm_utils.cpp mpm_defs.cpp mpm_mtl.cpp)

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(MPM PRIVATE Eigen3::Eigen)

find_package(TBB CONFIG REQUIRED)
target_link_libraries(MPM PRIVATE TBB::tbb)

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    target_link_libraries(MPM PRIVATE TBB::tbb_debug)
endif()

target_link_libraries(MPM PRIVATE partio)
target_link_libraries(MPM PRIVATE spdlog)