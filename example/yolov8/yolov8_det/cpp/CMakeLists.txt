cmake_minimum_required(VERSION 3.10)

project(rknn_yolov8_demo)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# 从系统查找opencv
# set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
find_package(OpenCV REQUIRED)

# rga
set(RGA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/librga)
set(LIBRGA ${RGA_PATH}/${CMAKE_SYSTEM_NAME}/aarch64/librga.a)
set(RGA_INCLUDES ${RGA_PATH}/include)
install(PROGRAMS ${RGA_PATH}/${CMAKE_SYSTEM_NAME}/aarch64/librga.so DESTINATION lib)

# only RGA on rv1106 and rk3588 support handle
if (TARGET_SOC STREQUAL "rk3588")
    add_definitions(-DLIBRGA_IM2D_HANDLE)
endif()

# RGA2 only support under 4G memory
if (ENABLE_DMA32 STREQUAL "TRUE")
    add_definitions(-DDMA_ALLOC_DMA32)
endif()

# dma
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/allocator/dma)

# rknn runtime
if (TARGET_SOC STREQUAL "rk3588" OR TARGET_SOC STREQUAL "rk356x")
    set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/rknpu2)
    set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/aarch64/librknnrt.so)
    set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/include)
endif()
install(PROGRAMS ${LIBRKNNRT} DESTINATION lib)
set(LIBRKNNRT ${LIBRKNNRT})

# rknn_yolov8_demo(image)
add_executable(${PROJECT_NAME}
    src/main.cc
    src/postprocess.cc
    src/yolov8.cc
    src/image_utils.c
)

target_link_libraries(${PROJECT_NAME}
    ${LIBRGA}
    ${OpenCV_LIBS}
    ${LIBRKNNRT}
    dl
)

# yolov8_vediocapture_demo
add_executable(yolov8_videocapture_demo
    src/yolov8_videocapture_demo.cc
    src/postprocess.cc
    src/yolov8.cc
    src/image_utils.c
)

target_link_libraries(yolov8_videocapture_demo
    ${LIBRGA}
    ${OpenCV_LIBS}
    ${LIBRKNNRT}
    dl
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${RGA_INCLUDES}
    ${LIBRKNNRT_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(TARGETS yolov8_videocapture_demo DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/bus.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/coco_80_labels_list.txt DESTINATION model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)
