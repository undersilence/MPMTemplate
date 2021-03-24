project(glad)
include_directories(include)
add_library(glad STATIC src/glad.c)

target_link_libraries(glad PRIVATE dl)