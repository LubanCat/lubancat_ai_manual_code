cmake_minimum_required(VERSION 3.10)

project(rknn_yolov5seg_demo)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# 从系统中查找opencv
find_package(OpenCV REQUIRED)

# only RGA on rv1106 and rk3588 support handle
if (TARGET_SOC STREQUAL "rk3588")
    add_definitions(-DLIBRGA_IM2D_HANDLE)
endif()

# RGA2 only support under 4G memory
if (ENABLE_DMA32 STREQUAL "TRUE")
    add_definitions(-DDMA_ALLOC_DMA32)
endif()

# support rknn zero copy api
if (ENABLE_ZERO_COPY STREQUAL "TRUE")
    add_definitions(-DENABLE_ZERO_COPY)
endif()

# dma
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/allocator/dma)

# drm
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/allocator/drm)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../utils/ utils.out)

# rknn_yolov5seg_demo (image)
add_executable(rknn_yolov5seg_demo
    yolov5seg_demo.cc
    postprocess.cc
    rknpu2/yolov5_seg.cc
    image_utils.c
)

target_link_libraries(rknn_yolov5seg_demo
    ${LIBRGA}
    ${OpenCV_LIBS}
    ${LIBRKNNRT}
    -ldl
)

# yolov5seg_videocapture_demo
add_executable(yolov5seg_videocapture_demo
    yolov5seg_videocapture_demo.cc
    postprocess.cc
    rknpu2/yolov5_seg.cc
    image_utils.c
)

target_link_libraries(yolov5seg_videocapture_demo
    ${LIBRGA}
    ${OpenCV_LIBS}
    ${LIBRKNNRT}
    -ldl
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRGA_INCLUDES}
    ${LIBRKNNRT_INCLUDES}
)

install(TARGETS rknn_yolov5seg_demo DESTINATION .)
install(TARGETS yolov5seg_videocapture_demo DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/bus.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/coco_80_labels_list.txt DESTINATION model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)