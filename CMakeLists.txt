cmake_minimum_required(VERSION 3.16)

project(HalideZoon)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_BUILD_TYPE "Release")

option(WITH_LIBPNG "exist libpng and libz for halide image_io" ON)
option(WITH_LIBJPEG "exist libjpeg for halide image_io" ON)
option(WITH_OPENCV "exist opencv" ON)

find_package(Halide REQUIRED)

set(HALIDE_ZOON_UTIL_LIBS "")
if(WITH_LIBPNG)
    add_definitions(-DHAVE_LIBPNG -DHAVE_LIBZ)
    include_directories(${PNG_INCLUDE_DIRS})
    message(STATUS PNG: ${PNG_LIBRARIES})
    list(APPEND HALIDE_ZOON_UTIL_LIBS ${PNG_LIBRARIES})
else()
    add_definitions(-DHALIDE_NO_PNG)
endif()

if(WITH_LIBJPEG)
    add_definitions(-DHAVE_LIBJPEG)
    include_directories(${JPEG_INCLUDE_DIRS})
    message(STATUS JPEG: ${JPEG_LIBRARIES})
    list(APPEND HALIDE_ZOON_UTIL_LIBS ${JPEG_LIBRARIES})
else()
    add_definitions(-DHALIDE_NO_JPEG)
endif()

if(WITH_OPENCV)
    find_package(OpenCV REQUIRED)
    add_definitions(-DHAVE_OPENCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS OPENCV: ${OpenCV_LIBS})
    list(APPEND HALIDE_ZOON_UTIL_LIBS ${OpenCV_LIBS})
endif()

file(GLOB FOLDERS ${CMAKE_CURRENT_SOURCE_DIR}/source/*)
FOREACH(FOLDER ${FOLDERS})
  IF(IS_DIRECTORY ${FOLDER})
    STRING(REGEX REPLACE  ".+/(.+)\\.*" "\\1" PROJECT_NAME ${FOLDER})
    FILE(GLOB FOLDER_SRC ${FOLDER}/*.cpp)

    add_executable(${PROJECT_NAME} ${FOLDER_SRC})
    target_link_libraries(${PROJECT_NAME} PRIVATE Halide::Halide ${HALIDE_ZOON_UTIL_LIBS})

    MESSAGE(STATUS ====== Project: ${PROJECT_NAME} ======)
    MESSAGE(STATUS Folder: ${FOLDER})
    MESSAGE(STATUS Source: ${FOLDER_SRC})
  ENDIF()
ENDFOREACH()

message(STATUS Copy Config Files.)
configure_file(images/gray.png images/gray.png COPYONLY)
configure_file(images/rgb.png images/rgb.png COPYONLY)

message(STATUS Done.)
