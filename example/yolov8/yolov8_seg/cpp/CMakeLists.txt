cmake_minimum_required(VERSION 3.10)

project(rknn_yolov8_seg_demo)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# 从系统查找opencv
# set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
find_package(OpenCV REQUIRED)

# 查找自行编译安装的OpenCV，请修改下面的OpenCV_DIR路径
# set(OpenCV_DIR /path/opencv/lib/cmake/opencv4)
# find_package(OpenCV REQUIRED)
# file(GLOB OpenCV_FILES "${OpenCV_DIR}/../../libopencv*")
# install(PROGRAMS ${OpenCV_FILES} DESTINATION lib)

# only RGA on rv1106 and rk3588 support handle
if (TARGET_SOC STREQUAL "rk3588")
    add_definitions(-DLIBRGA_IM2D_HANDLE)
endif()

# RGA2 only support under 4G memory
if (ENABLE_DMA32 STREQUAL "TRUE")
    add_definitions(-DDMA_ALLOC_DMA32)
endif()

#dma
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/allocator/dma)

#drm
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/allocator/drm)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

# rknn_yolov8_seg_demo
add_executable(${PROJECT_NAME}
    main.cc
    postprocess.cc
    rknpu2/yolov8_seg.cc
    image_utils.c
)

target_link_libraries(${PROJECT_NAME}
    ${LIBRGA}
    ${OpenCV_LIBS}
    ${LIBRKNNRT}
    dl
)

# yolov8seg_videocapture_demo
add_executable(yolov8seg_videocapture_demo
    yolov8seg_videocapture_demo.cc
    postprocess.cc
    rknpu2/yolov8_seg.cc
    image_utils.c
)

target_link_libraries(yolov8seg_videocapture_demo
    ${LIBRGA}
    ${OpenCV_LIBS}
    ${LIBRKNNRT}
    dl
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRGA_INCLUDES}
    ${LIBRKNNRT_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(TARGETS yolov8seg_videocapture_demo DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/bus.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/coco_80_labels_list.txt DESTINATION model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)
