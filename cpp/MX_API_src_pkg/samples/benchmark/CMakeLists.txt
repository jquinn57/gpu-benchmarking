cmake_minimum_required(VERSION 3.10)

get_filename_component(app_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} 
                    ${MX_API_DIR}/include             
                    )
                    
file(GLOB local_src
    "*.c"
    "*.cpp"
	)

add_executable(${app_name} ${local_src})
#target_compile_options(${app_name}  PRIVATE -Wall -Wextra -g)

target_link_libraries(${app_name}  /usr/lib/x86_64-linux-gnu/libmx_accl.so  ${OpenCV_LIBS} onnxruntime ) 

file(COPY "../models/yolov7Tiny416/" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
