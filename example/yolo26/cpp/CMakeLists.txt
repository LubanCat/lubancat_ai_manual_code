cmake_minimum_required(VERSION 3.10)

project(rknn_yolo26_demo)

function(buildtarget)
  # From https://stackoverflow.com/questions/23327687/how-to-write-a-cmake-function-with-more-than-one-parameter-groups
  cmake_parse_arguments(
    PARSED_ARGS # prefix of output variables
    "" # list of names of the boolean arguments (only defined ones will be true)
    "NAME" # list of names of mono-valued arguments
    "INCS;SRCS;DEPS;DEFS" # list of names of multi-valued arguments (output variables are lists)
    ${ARGN} # arguments of the function to parse, here we take the all original ones
  )
  # note: if it remains unparsed arguments, here, they can be found in variable PARSED_ARGS_UNPARSED_ARGUMENTS
  if(NOT PARSED_ARGS_NAME)
    message(FATAL_ERROR "You must provide a name")
  endif(NOT PARSED_ARGS_NAME)
  project(${PARSED_ARGS_NAME})

  include_directories(${PARSED_ARGS_INCS})
  add_executable(${PROJECT_NAME} ${PARSED_ARGS_SRCS})
  target_link_libraries(${PROJECT_NAME} ${PARSED_ARGS_DEPS})

  if(PARSED_ARGS_DEFS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ${PARSED_ARGS_DEFS})
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
  endif()

  install(TARGETS ${PROJECT_NAME} DESTINATION .)
endfunction(buildtarget)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

set(rknpu_yolo26_det_file rknpu2/yolo26_det.cc)

if (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103")
    add_definitions(-DRV1106_1103)
    set(rknpu_yolo26_det_file rknpu2/yolo26_det_rv1106_1103.cc)
endif()

#dma
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/allocator/dma)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/allocator/drm)
# RGA2 only support under 4G memory
if (ENABLE_DMA32 STREQUAL "TRUE")
    add_definitions(-DDMA_ALLOC_DMA32)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

# yolo26_det_demo
buildtarget(NAME yolo26_det_demo
    INCS ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRKNNRT_INCLUDES} ${LIBTIMER_INCLUDES}
    SRCS yolo26_det_demo.cc postprocess_det.cc ${rknpu_yolo26_det_file}
    DEPS imageutils fileutils imagedrawing ${LIBRKNNRT} dl
)

# Currently rv1106/rv1103 not support yolo26 seg/pose/obb, only support yolo26 det
if (NOT (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103"))

# opencv
# set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
find_package(OpenCV REQUIRED)

# yolo26_seg_demo
buildtarget(NAME yolo26_seg_demo
    INCS ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRKNNRT_INCLUDES} ${LIBTIMER_INCLUDES}
    SRCS yolo26_seg_demo.cc postprocess_seg.cc rknpu2/yolo26_seg.cc
    DEPS imageutils fileutils imagedrawing ${LIBRKNNRT} ${OpenCV_LIBS} dl
)

# yolo26_pose_demo
buildtarget(NAME yolo26_pose_demo
    INCS ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRKNNRT_INCLUDES} ${LIBTIMER_INCLUDES}
    SRCS yolo26_pose_demo.cc postprocess_pose.cc rknpu2/yolo26_pose.cc
    DEPS imageutils fileutils imagedrawing ${LIBRKNNRT} ${OpenCV_LIBS} dl
)

# yolo26_obb_demo
buildtarget(NAME yolo26_obb_demo
    INCS ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRKNNRT_INCLUDES} ${LIBTIMER_INCLUDES}
    SRCS yolo26_obb_demo.cc postprocess_obb.cc rknpu2/yolo26_obb.cc
    DEPS imageutils fileutils imagedrawing ${LIBRKNNRT} ${OpenCV_LIBS} dl
)
endif()

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/bus.jpg DESTINATION model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)
